"""P37 student observation: P29 combat facts plus the inverse kill field.

The actor remains 801 -> 1024 -> 1024 -> 1024 -> 18. The first 667 columns
are exactly P29. The final 134 columns replace the obsolete P36 coin course:
120 guidance-map values, 9 current field/chain facts, and 5 route/aim facts.

P35/P36 replay must not be loaded even though it is also 801 columns wide:
the final 134 columns have different semantics.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

from training.killfield_teacher import (
    DEFAULT_BOUNCES,
    DEFAULT_FLIGHT_FRAMES,
    DEFAULT_RAYS,
    FIELD_LEVELS,
    GUIDANCE_DISTANCE_DECAY,
    HUNT_CHAIN_MAX_EXPONENT,
    HUNT_CHAIN_WINDOW_FRAMES,
    HuntChainState,
    InverseDensityFieldBuilder,
    _angle_delta,
    _cell,
)
from training.score_distill import build_net
from training.survival_frontier_rl import MAP_H, MAP_W
from training.survival_opportunity_rl import P29_OBS_DIM, opportunity_observation
from training.survival_rl_warmstart import WarmStartActorCritic


KILLFIELD_MAP_DIM = MAP_W * MAP_H
KILLFIELD_STATE_DIM = 9
KILLFIELD_ROUTE_DIM = 5
KILLFIELD_EXTRA_DIM = (
    KILLFIELD_MAP_DIM + KILLFIELD_STATE_DIM + KILLFIELD_ROUTE_DIM)
P37_OBS_DIM = P29_OBS_DIM + KILLFIELD_EXTRA_DIM

assert P29_OBS_DIM == 667
assert KILLFIELD_EXTRA_DIM == 134
assert P37_OBS_DIM == 801


def _finite_flight(field, cell):
    if not field.in_bounds(cell):
        return float(field.max_flight_frames)
    value = float(field.min_frames[cell])
    return value if math.isfinite(value) else float(field.max_flight_frames)


def _local_vector(game, tank, cell, normalizer):
    world_x = (cell[0] + 0.5) * game.scale
    world_y = (cell[1] + 0.5) * game.scale
    delta_x, delta_y = world_x - tank.x, world_y - tank.y
    heading = math.radians(tank.rotation) - math.pi / 2.0
    cosine, sine = math.cos(heading), math.sin(heading)
    forward = (delta_x * cosine + delta_y * sine) / game.scale
    right = (-delta_x * sine + delta_y * cosine) / game.scale
    return (
        float(np.clip(forward / normalizer, -1.0, 1.0)),
        float(np.clip(right / normalizer, -1.0, 1.0)),
    )


class KillFieldFeatureState:
    """Cached field, observable hunt-chain state, and 134 student features."""

    def __init__(self, ray_count=DEFAULT_RAYS,
                 max_bounces=DEFAULT_BOUNCES,
                 max_flight_frames=DEFAULT_FLIGHT_FRAMES):
        self.ray_count = int(ray_count)
        self.max_bounces = int(max_bounces)
        self.max_flight_frames = int(max_flight_frames)
        self.reset()

    def reset(self):
        self.game = None
        self.round_number = None
        self.builder = None
        self.field_cache = {}
        self.field = None
        self.chain = HuntChainState()
        self.last_chain_gain = 0.0
        self.previous_cell = None
        self.previous_target = None
        self.field_builds = 0
        self.field_build_seconds = 0.0

    def ensure_field(self, game):
        if game is not self.game or game.round_number != self.round_number:
            self.game = game
            self.round_number = game.round_number
            self.builder = InverseDensityFieldBuilder(
                game, self.ray_count, self.max_bounces,
                self.max_flight_frames)
            self.field_cache = {}
            self.chain = HuntChainState()
            self.last_chain_gain = 0.0
            self.previous_cell = _cell(game, game.tanks[0])
            self.previous_target = None
        target = _cell(game, game.tanks[1])
        if target not in self.field_cache:
            started = time.perf_counter()
            self.field_cache[target] = self.builder.build(target)
            self.field_build_seconds += time.perf_counter() - started
            self.field_builds += 1
        self.field = self.field_cache[target]
        if self.previous_target is None:
            self.previous_target = target
        return self.field

    def adopt_teacher(self, game, teacher):
        """Mirror the teacher's already-updated public state for collection."""
        self.game = game
        self.round_number = game.round_number
        self.builder = teacher._builder
        self.field_cache = teacher._field_cache
        self.field = teacher.field
        self.chain = teacher.chain.clone()
        self.last_chain_gain = float(teacher.last_chain_gain)
        self.previous_cell = _cell(game, game.tanks[0])
        self.previous_target = self.field.target_cell

    def advance(self, game, frames=1):
        """Advance chain state after real frames and return the new reward."""
        old_cell = self.previous_cell
        old_target = self.previous_target
        self.chain.advance(frames)
        field = self.ensure_field(game)
        current_cell = _cell(game, game.tanks[0])
        target = field.target_cell
        gain = 0.0 if old_cell is None else self.chain.collect_ascent(
            field, old_cell, current_cell, target_stable=(old_target == target))
        self.last_chain_gain = float(gain)
        self.previous_cell = current_cell
        self.previous_target = target
        return self.last_chain_gain

    @staticmethod
    def _source_quality(field, cell):
        count = field.count_at(cell)
        if count <= 0 or field.max_count <= 0:
            return 0.0
        density = math.log1p(count) / math.log1p(field.max_count)
        flight = _finite_flight(field, cell)
        timely = math.exp(-flight / max(field.max_flight_frames, 1))
        return density * (0.5 + 0.5 * timely)

    def _best_source(self, game, field, current):
        candidates = []
        for item in game.reachable:
            source = (int(item["x"]), int(item["y"]))
            quality = self._source_quality(field, source)
            if quality <= 0.0:
                continue
            distances = game.dist_map(source[0], source[1])
            if distances is None:
                continue
            distance = distances[current[0]][current[1]]
            if distance is None or distance != distance:
                continue
            envelope = quality * math.exp(
                -GUIDANCE_DISTANCE_DECAY * float(distance))
            candidates.append((
                -envelope, float(distance), source[1], source[0]))
        if not candidates:
            return None, 0.0
        best = min(candidates)
        return (best[3], best[2]), best[1]

    @staticmethod
    def _next_cell(game, current, target):
        if target is None or current == target:
            return current
        distances = game.dist_map(target[0], target[1])
        if distances is None:
            return target
        current_distance = distances[current[0]][current[1]]
        if current_distance is None or current_distance != current_distance:
            return target
        choices = []
        for x, y in ((current[0] - 1, current[1]),
                     (current[0] + 1, current[1]),
                     (current[0], current[1] - 1),
                     (current[0], current[1] + 1)):
            if not (0 <= x < len(game.maze)
                    and 0 <= y < len(game.maze[0])):
                continue
            distance = distances[x][y]
            if (distance is not None and distance == distance
                    and distance < current_distance):
                choices.append((float(distance), y, x))
        if not choices:
            return target
        best = min(choices)
        return best[2], best[1]

    def features(self, game):
        field = self.ensure_field(game)
        tank = game.tanks[0]
        current = _cell(game, tank)

        # Engine arrays are [x, y]; previous 120-cell student maps are [y, x].
        guidance_map = np.zeros((MAP_H, MAP_W), dtype=np.float32)
        width = min(field.guidance.shape[0], MAP_W)
        height = min(field.guidance.shape[1], MAP_H)
        guidance_map[:height, :width] = field.guidance[:width, :height].T
        # The exponential reward is collectible only once per
        # (Laika cell, player cell).  Magnitude remains B(c); sign exposes the
        # otherwise hidden one-shot ledger without adding another 120 inputs.
        for target, cell in self.chain.collected:
            if (target == field.target_cell
                    and 0 <= cell[0] < width and 0 <= cell[1] < height):
                guidance_map[cell[1], cell[0]] *= -1.0

        heading = math.radians(tank.rotation) - math.pi / 2.0
        best_aim, concentration = field.best_aim_at(current, heading)
        flight = _finite_flight(field, current)
        state = np.asarray([
            field.guidance_at(current),
            field.relative_success_at(current),
            field.tier_at(current) / max(FIELD_LEVELS, 1),
            field.value_at(current) / float(2 ** (FIELD_LEVELS - 1)),
            np.clip(flight / max(field.max_flight_frames, 1), 0.0, 1.0),
            concentration,
            self.chain.count / max(HUNT_CHAIN_MAX_EXPONENT, 1),
            self.chain.timer / max(HUNT_CHAIN_WINDOW_FRAMES, 1),
            self.last_chain_gain / float(2 ** HUNT_CHAIN_MAX_EXPONENT),
        ], dtype=np.float32)

        source, distance = self._best_source(game, field, current)
        waypoint = self._next_cell(game, current, source)
        waypoint_forward, waypoint_right = _local_vector(
            game, tank, waypoint, 2.0)
        if best_aim is None:
            aim_cosine = aim_sine = 0.0
        else:
            delta = _angle_delta(best_aim, heading)
            aim_cosine, aim_sine = math.cos(delta), math.sin(delta)
        route = np.asarray([
            waypoint_forward,
            waypoint_right,
            np.clip(distance / 20.0, 0.0, 1.0),
            aim_cosine,
            aim_sine,
        ], dtype=np.float32)
        result = np.concatenate([guidance_map.ravel(), state, route])
        if result.shape != (KILLFIELD_EXTRA_DIM,):
            raise RuntimeError(
                f"P37 feature width {result.shape}, expected "
                f"({KILLFIELD_EXTRA_DIM},)")
        return result


def killfield_observation(encoder, game, ledger, econ, frontier, analyzer,
                          field_state):
    """Return the 801-vector and the P29 diagnostics used by trainers."""
    base, metrics, fire_facts = opportunity_observation(
        encoder, game, ledger, econ, frontier, analyzer)
    observation = np.concatenate([base, field_state.features(game)])
    if observation.shape != (P37_OBS_DIM,):
        raise RuntimeError(
            f"P37 observation width {observation.shape}, expected "
            f"({P37_OBS_DIM},)")
    return observation, metrics, fire_facts


def load_expanded_warmstart(path, device):
    """Expand a P29 actor; new P37 columns start at exactly zero."""
    payload = torch.load(path, map_location=device, weights_only=True)
    source_dim = int(payload.get("in_dim", P29_OBS_DIM))
    if source_dim != P29_OBS_DIM:
        raise ValueError(
            "P37 warm-start must be the 667-input P29 actor; "
            f"got {source_dim}. P35/P36 actors have incompatible tail semantics.")
    source = payload["state_dict"]
    model = WarmStartActorCritic(in_dim=P37_OBS_DIM).to(device)
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.weight[:, :P29_OBS_DIM].copy_(source["0.weight"])
        model.fc1.bias.copy_(source["0.bias"])
        model.fc2.weight.copy_(source["2.weight"])
        model.fc2.bias.copy_(source["2.bias"])
        model.fc3.weight.copy_(source["4.weight"])
        model.fc3.bias.copy_(source["4.bias"])
        model.actor.weight.copy_(source["6.weight"])
        model.actor.bias.copy_(source["6.bias"])
        nn.init.orthogonal_(model.value.weight, gain=1.0)
        nn.init.zeros_(model.value.bias)
    return model


def verify_warmstart(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    reference = build_net(P29_OBS_DIM).to(device)
    reference.load_state_dict(payload["state_dict"])
    base = torch.randn(8, P29_OBS_DIM, device=device)
    extra = torch.randn(8, KILLFIELD_EXTRA_DIM, device=device)
    with torch.no_grad():
        expected = reference(base)
        actual, _ = model(torch.cat([base, extra], dim=1))
    difference = float((expected - actual).abs().max())
    if difference != 0.0:
        raise RuntimeError(f"P37 warm-start mismatch: {difference}")
    return difference
