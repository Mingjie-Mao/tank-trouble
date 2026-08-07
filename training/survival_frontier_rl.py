"""P27: observable frontier curriculum with retained success trajectories.

This fixes two structural problems in P26:
1. every reward-bearing exploration state is part of the observation;
2. successful cap trajectories survive later on-policy updates through BC replay.

The iter03 530-wide actor is expanded to 656 inputs.  New first-layer columns
start at zero, so the initial actor is exactly iter03 before learning.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.mpc_agent import CANDIDATES
from training.score_distill import build_net
from training.survival_distill_v2 import OBS_DIM, legacy_econ
from training.survival_expert_iter_530 import (
    DECIDE_EVERY,
    apply_action,
    build_observation,
)
from training.survival_rl_warmstart import WarmStartActorCritic


HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_WARMSTART = os.path.join(MODELS_DIR, "p24r530_iter03.pt")
DEFAULT_ACTOR = os.path.join(MODELS_DIR, "p28_dense_frontier_actor.pt")
DEFAULT_CHECKPOINT = os.path.join(MODELS_DIR, "p28_dense_frontier_checkpoint.pt")
DEFAULT_SUCCESS = os.path.join(HERE, "p28_dense_frontier_success.npz")

MAP_W = 12
MAP_H = 10
VISITED_DIM = MAP_W * MAP_H
TARGET_DIM = 6
FRONTIER_DIM = VISITED_DIM + TARGET_DIM
FRONTIER_OBS_DIM = OBS_DIM + FRONTIER_DIM
FPS = 25
REWARD_SCALE = 50.0
SURVIVAL_REWARD_PER_FRAME = 0.0
UNIQUE_CELL_REWARD = 0.30
FRONTIER_PROGRESS_REWARD = 0.25
TERMINAL_PENALTY = 20.0
SUCCESS_CONTEXT_STEPS = 8


def cell_of(game, tank):
    return int(tank.x // game.scale), int(tank.y // game.scale)


def cell_distance(game, start, end):
    distance_map = game.dist_map(start[0], start[1])
    if distance_map is None:
        return None
    end_x, end_y = end
    if 0 <= end_x < len(distance_map) \
            and 0 <= end_y < len(distance_map[end_x]):
        value = distance_map[end_x][end_y]
        if value is not None and value == value:
            return float(value)
    return None


class FrontierState:
    """Permanent visited map plus one latched, observable frontier target."""

    def __init__(self, game, dense=True):
        self.dense = dense
        self.visited = np.zeros((MAP_H, MAP_W), dtype=np.float32)
        self.current_cell = cell_of(game, game.tanks[0])
        self._mark(self.current_cell)
        self.target = None
        self._choose_target(game)

    def _valid(self, game, cell):
        cell_x, cell_y = cell
        return 0 <= cell_x < min(len(game.maze), MAP_W) \
            and 0 <= cell_y < min(len(game.maze[0]), MAP_H)

    def _mark(self, cell):
        cell_x, cell_y = cell
        if 0 <= cell_x < MAP_W and 0 <= cell_y < MAP_H:
            self.visited[cell_y, cell_x] = 1.0

    def _is_visited(self, cell):
        cell_x, cell_y = cell
        return bool(self.visited[cell_y, cell_x])

    def _choose_target(self, game):
        candidates = []
        width = min(len(game.maze), MAP_W)
        height = min(len(game.maze[0]), MAP_H)
        for cell_y in range(height):
            for cell_x in range(width):
                cell = (cell_x, cell_y)
                if self._is_visited(cell):
                    continue
                distance = cell_distance(game, self.current_cell, cell)
                if distance is not None:
                    candidates.append((distance, cell_y, cell_x))
        self.target = None if not candidates else (
            min(candidates)[2], min(candidates)[1])

    def observe_position(self, game):
        """Advance the frontier state and return (BFS delta, first visit)."""
        new_cell = cell_of(game, game.tanks[0])
        if new_cell == self.current_cell or not self._valid(game, new_cell):
            return 0.0, False

        progress = 0.0
        if self.target is not None:
            before = cell_distance(game, self.current_cell, self.target)
            after = cell_distance(game, new_cell, self.target)
            if before is not None and after is not None:
                progress = before - after

        first_visit = not self._is_visited(new_cell)
        if first_visit:
            self._mark(new_cell)
        self.current_cell = new_cell
        if self.target == new_cell or self.target is None:
            self._choose_target(game)
        return progress, first_visit

    def _next_cell(self, game, current_cell=None, target=None):
        current_cell = current_cell or cell_of(game, game.tanks[0])
        target = self.target if target is None else target
        if target is None or current_cell == target:
            return target
        current_distance = cell_distance(game, current_cell, target)
        if current_distance is None:
            return target
        candidates = []
        cell_x, cell_y = current_cell
        for neighbor in ((cell_x - 1, cell_y), (cell_x + 1, cell_y),
                         (cell_x, cell_y - 1), (cell_x, cell_y + 1)):
            if not self._valid(game, neighbor):
                continue
            distance = cell_distance(game, neighbor, target)
            if distance is not None and distance < current_distance:
                candidates.append((distance, neighbor[1], neighbor[0]))
        if not candidates:
            return target
        best = min(candidates)
        return best[2], best[1]

    def continuous_distance(self, game, target=None):
        """Continuous path proxy to a latched target; decreases every frame."""
        target = self.target if target is None else target
        if target is None:
            return 0.0
        tank = game.tanks[0]
        current = cell_of(game, tank)
        steps = cell_distance(game, current, target)
        if steps is None:
            return 0.0
        waypoint = self._next_cell(game, current, target)
        if waypoint is None:
            return steps
        waypoint_x = (waypoint[0] + 0.5) * game.scale
        waypoint_y = (waypoint[1] + 0.5) * game.scale
        local_distance = math.hypot(
            tank.x - waypoint_x, tank.y - waypoint_y) / game.scale
        return max(steps - 1.0, 0.0) + local_distance

    def features(self, game):
        features = np.zeros(FRONTIER_DIM, dtype=np.float32)
        features[:VISITED_DIM] = self.visited.ravel()
        tank = game.tanks[0]
        if self.target is None:
            return features

        target_x, target_y = self.target
        world_x = (target_x + 0.5) * game.scale
        world_y = (target_y + 0.5) * game.scale
        delta_x, delta_y = world_x - tank.x, world_y - tank.y
        forward_angle = math.radians(tank.rotation) - math.pi / 2
        cosine, sine = math.cos(forward_angle), math.sin(forward_angle)
        forward = (delta_x * cosine + delta_y * sine) / game.scale
        right = (-delta_x * sine + delta_y * cosine) / game.scale
        distance = cell_distance(game, self.current_cell, self.target)
        waypoint = self._next_cell(game)
        waypoint_x = (waypoint[0] + 0.5) * game.scale
        waypoint_y = (waypoint[1] + 0.5) * game.scale
        waypoint_delta_x = waypoint_x - tank.x
        waypoint_delta_y = waypoint_y - tank.y
        waypoint_forward = (
            waypoint_delta_x * cosine + waypoint_delta_y * sine) / game.scale
        waypoint_right = (
            -waypoint_delta_x * sine + waypoint_delta_y * cosine) / game.scale
        tail = [
            np.clip(forward / 12.0, -1.0, 1.0),
            np.clip(right / 12.0, -1.0, 1.0),
            min((distance or 0.0) / 20.0, 1.0),
            self.visited.mean(),
            np.clip(waypoint_forward / 2.0, -1.0, 1.0),
            np.clip(waypoint_right / 2.0, -1.0, 1.0),
        ]
        if not self.dense:
            tail[-2:] = [
                target_x / max(MAP_W - 1, 1),
                target_y / max(MAP_H - 1, 1),
            ]
        features[VISITED_DIM:] = tail
        return features


def frontier_observation(encoder, game, ledger, econ, frontier):
    base = build_observation(encoder, game, ledger, econ)
    return np.concatenate([base, frontier.features(game)])


def load_expanded_warmstart(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if int(payload.get("in_dim", OBS_DIM)) != OBS_DIM:
        raise ValueError("P27 warm-start must be a 530-input Replica actor")
    source = payload["state_dict"]
    model = WarmStartActorCritic(in_dim=FRONTIER_OBS_DIM).to(device)
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.weight[:, :OBS_DIM].copy_(source["0.weight"])
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


def verify_expanded_warmstart(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    reference = build_net(OBS_DIM).to(device)
    reference.load_state_dict(payload["state_dict"])
    base = torch.randn(8, OBS_DIM, device=device)
    frontier = torch.randn(8, FRONTIER_DIM, device=device)
    with torch.no_grad():
        expected = reference(base)
        actual, _ = model(torch.cat([base, frontier], dim=1))
    difference = (expected - actual).abs().max().item()
    if difference != 0.0:
        raise RuntimeError(f"expanded warm-start mismatch: {difference}")
    return difference


class FrontierSurvivalEnv:
    def __init__(self, seed, cap_seconds, start_pool,
                 terminal_penalty=TERMINAL_PENALTY):
        from training.tt_gym_env import TankTroubleGym

        self.seed_rng = random.Random(seed)
        self.encoder = TankTroubleGym(
            seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
        self.econ = dict(
            legacy_econ(), cap=int(cap_seconds * FPS), start=float(start_pool))
        self.terminal_penalty = terminal_penalty
        self.game = None
        self.ledger = None
        self.frontier = None
        self.episode_return = 0.0
        self.shaping_total = 0.0

    def reset(self):
        from tank_trouble_original.game import Game
        from training.survival_mode import Ledger

        self.game = Game(
            seed=self.seed_rng.randrange(1 << 30), ai_enabled=True,
            invincible={1}, hit_immunity_frames={1: 0})
        self.ledger = Ledger(self.game, self.econ)
        self.frontier = FrontierState(self.game)
        self.episode_return = 0.0
        self.shaping_total = 0.0
        return frontier_observation(
            self.encoder, self.game, self.ledger, self.econ, self.frontier)

    def step(self, action_index):
        pool_before = self.ledger.pool
        hits_before = self.ledger.hits
        shaping = 0.0
        frontier_progress = 0.0
        first_visit_event = False
        end = "alive"
        action = CANDIDATES[int(action_index)]
        for _ in range(DECIDE_EVERY):
            latched_target = self.frontier.target
            distance_before = self.frontier.continuous_distance(
                self.game, latched_target)
            apply_action(self.game, action)
            events = self.game.step()
            end = self.ledger.on_frame(self.game, events)
            if end != "death":
                shaping += SURVIVAL_REWARD_PER_FRAME
                distance_after = self.frontier.continuous_distance(
                    self.game, latched_target)
                progress = distance_before - distance_after
                _, first_visit = self.frontier.observe_position(self.game)
                if progress > 0 and self.game.tanks[0].hit_something:
                    progress = 0.0
                shaping += FRONTIER_PROGRESS_REWARD * progress
                shaping += UNIQUE_CELL_REWARD * float(first_visit)
                frontier_progress += progress
                first_visit_event = first_visit_event or first_visit
            if end != "alive":
                break

        reward = (self.ledger.pool - pool_before) / REWARD_SCALE + shaping
        failure_adjustment = self.terminal_penalty \
            - self.econ["start"] / REWARD_SCALE
        if end == "death":
            reward -= self.ledger.pool / REWARD_SCALE + failure_adjustment
        elif end == "drain":
            reward -= failure_adjustment
        elif end == "cap":
            reward += self.econ["start"] / REWARD_SCALE
        self.episode_return += reward
        self.shaping_total += shaping

        done = end != "alive"
        observation = None if done else frontier_observation(
            self.encoder, self.game, self.ledger, self.econ, self.frontier)
        info = {
            "frontier_progress": frontier_progress,
            "first_visit": first_visit_event,
            "hit": self.ledger.hits > hits_before,
        }
        if done:
            info.update({
                "end": end,
                "return": self.episode_return,
                "frames": self.ledger.frames,
                "hits": self.ledger.hits,
                "stuck_frames": self.ledger.stuck_frames,
                "style": self.ledger.style,
                "unique_cells": int(self.frontier.visited.sum()),
                "shaping": self.shaping_total,
                "pool": self.ledger.pool,
            })
        return observation, float(reward), done, info


class SuccessReplay:
    def __init__(self, path, max_steps=50_000):
        self.path = path
        self.max_steps = max_steps
        self.observations = np.empty((0, FRONTIER_OBS_DIM), dtype=np.float32)
        self.actions = np.empty(0, dtype=np.int64)
        if path and os.path.exists(path):
            data = np.load(path)
            if data["observations"].shape[1] == FRONTIER_OBS_DIM:
                self.observations = data["observations"].astype(np.float32)
                self.actions = data["actions"].astype(np.int64)

    def __len__(self):
        return len(self.actions)

    def add(self, trajectory):
        if not trajectory:
            return
        observations = np.stack([item[0] for item in trajectory])
        actions = np.asarray([item[1] for item in trajectory], dtype=np.int64)
        self.observations = np.concatenate([self.observations, observations])
        self.actions = np.concatenate([self.actions, actions])
        if len(self) > self.max_steps:
            self.observations = self.observations[-self.max_steps:]
            self.actions = self.actions[-self.max_steps:]

    def sample(self, count, device):
        if not len(self):
            return None
        indices = np.random.randint(0, len(self), size=min(count, len(self)))
        observations = torch.as_tensor(
            self.observations[indices], dtype=torch.float32, device=device)
        actions = torch.as_tensor(
            self.actions[indices], dtype=torch.long, device=device)
        return observations, actions

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez_compressed(
            self.path, observations=self.observations, actions=self.actions)


def success_keyframes(trajectory, context_steps=SUCCESS_CONTEXT_STEPS):
    """Keep only actions that led into frontier progress/hit events."""
    useful = [index for index, item in enumerate(trajectory) if item[2]]
    if not useful:
        return []
    selected = set()
    for index in useful:
        selected.update(range(max(0, index - context_steps), index + 1))
    return [trajectory[index] for index in sorted(selected)]


def distribution(model, observations, temperature):
    actor_scores, values = model(observations)
    return Categorical(logits=actor_scores / temperature), values


def collect_rollout(model, envs, observations, pending, replay, args, device):
    buffers = {key: [] for key in (
        "observations", "actions", "log_probs", "rewards", "dones", "values")}
    episodes = []
    successes = 0
    for _ in range(args.rollout_steps):
        observation_tensor = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=device)
        with torch.no_grad():
            dist, values = distribution(model, observation_tensor, args.temperature)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        rewards = np.empty(len(envs), dtype=np.float32)
        dones = np.empty(len(envs), dtype=np.float32)
        next_observations = []
        for index, env in enumerate(envs):
            action = actions[index].item()
            pending[index].append([
                observations[index].copy(), action, False])
            next_obs, reward, done, info = env.step(action)
            pending[index][-1][2] = bool(
                info["frontier_progress"] > 1e-4
                or info["first_visit"] or info["hit"])
            rewards[index] = reward
            dones[index] = float(done)
            if done:
                episodes.append(info)
                if info["end"] == "cap":
                    replay.add(success_keyframes(pending[index]))
                    successes += 1
                pending[index] = []
                next_obs = env.reset()
            next_observations.append(next_obs)

        buffers["observations"].append(np.stack(observations))
        buffers["actions"].append(actions.cpu().numpy())
        buffers["log_probs"].append(log_probs.cpu().numpy())
        buffers["rewards"].append(rewards)
        buffers["dones"].append(dones)
        buffers["values"].append(values.cpu().numpy())
        observations = next_observations

    with torch.no_grad():
        final_tensor = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=device)
        _, final_values = model(final_tensor)
    rewards = np.stack(buffers["rewards"])
    dones = np.stack(buffers["dones"])
    values = np.stack(buffers["values"])
    advantages = np.zeros_like(rewards)
    last_advantage = np.zeros(len(envs), dtype=np.float32)
    next_values = final_values.cpu().numpy()
    for step in reversed(range(args.rollout_steps)):
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + args.gamma * next_values * nonterminal \
            - values[step]
        last_advantage = delta + args.gamma * args.gae_lambda \
            * nonterminal * last_advantage
        advantages[step] = last_advantage
        next_values = values[step]
    rollout = {
        "observations": np.concatenate(buffers["observations"]),
        "actions": np.concatenate(buffers["actions"]),
        "log_probs": np.concatenate(buffers["log_probs"]),
        "advantages": advantages.reshape(-1),
        "returns": (advantages + values).reshape(-1),
        "values": values.reshape(-1),
    }
    return rollout, observations, pending, episodes, successes


def update_ppo(model, optimizer, rollout, replay, args, device,
               value_only=False):
    observations = torch.as_tensor(
        rollout["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(
        rollout["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(
        rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(
        rollout["returns"], dtype=torch.float32, device=device)
    old_values = torch.as_tensor(
        rollout["values"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    totals = dict(policy=0.0, value=0.0, entropy=0.0, bc=0.0, kl=0.0, n=0)
    size = len(observations)
    for _ in range(args.epochs):
        order = torch.randperm(size, device=device)
        for start in range(0, size, args.minibatch):
            indices = order[start:start + args.minibatch]
            if value_only:
                with torch.no_grad():
                    hidden = model.features(observations[indices])
                    actor_scores = model.actor(hidden)
                dist = Categorical(logits=actor_scores / args.temperature)
                values = model.value(hidden.detach()).squeeze(-1)
            else:
                dist, values = distribution(
                    model, observations[indices], args.temperature)
            new_log_probs = dist.log_prob(actions[indices])
            log_ratio = new_log_probs - old_log_probs[indices]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(
                ratio * advantages[indices],
                torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                * advantages[indices]).mean()
            clipped_values = old_values[indices] + torch.clamp(
                values - old_values[indices], -args.value_clip, args.value_clip)
            value_loss = 0.5 * torch.max(
                (values - returns[indices]).pow(2),
                (clipped_values - returns[indices]).pow(2)).mean()
            entropy = dist.entropy().mean()
            bc_loss = torch.zeros((), device=device)
            if not value_only:
                success_batch = replay.sample(args.bc_batch, device)
                if success_batch is not None:
                    success_obs, success_actions = success_batch
                    success_scores, _ = model(success_obs)
                    bc_loss = nn.functional.cross_entropy(
                        success_scores / args.temperature, success_actions)
            loss = args.value_coef * value_loss if value_only else \
                policy_loss + args.value_coef * value_loss \
                - args.entropy_coef * entropy + args.bc_coef * bc_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = ((ratio - 1) - log_ratio).mean()
            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += entropy.item()
            totals["bc"] += bc_loss.item()
            totals["kl"] += approximate_kl.item()
            totals["n"] += 1
        if not value_only and totals["kl"] / totals["n"] > args.target_kl:
            break
    count = max(totals.pop("n"), 1)
    return {key: value / count for key, value in totals.items()}


def summarize(episodes):
    if not episodes:
        return {"episodes": 0}
    frames = sum(item["frames"] for item in episodes)
    hits = sum(item["hits"] for item in episodes)
    return {
        "episodes": len(episodes),
        "cap_pct": sum(x["end"] == "cap" for x in episodes) / len(episodes),
        "death_pct": sum(x["end"] == "death" for x in episodes) / len(episodes),
        "drain_pct": sum(x["end"] == "drain" for x in episodes) / len(episodes),
        "seconds_per_hit": frames / FPS / max(hits, 1),
        "mean_cells": float(np.mean([x["unique_cells"] for x in episodes])),
        "mean_return": float(np.mean([x["return"] for x in episodes])),
        "stuck_pct": sum(x["stuck_frames"] for x in episodes) / max(frames, 1),
    }


def greedy_probe(model, cap_seconds, start_pool, terminal_penalty, seed, device):
    """One fixed deterministic episode to expose train/deploy mismatch."""
    env = FrontierSurvivalEnv(
        seed, cap_seconds, start_pool, terminal_penalty)
    observation = env.reset()
    while True:
        tensor = torch.as_tensor(
            observation[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            scores, _ = model(tensor)
        observation, _, done, info = env.step(scores.argmax(1).item())
        if done:
            return info


def parse_stages(specification):
    stages = []
    for item in specification.split(","):
        cap, start, updates = item.split(":")
        stages.append((int(cap), float(start), int(updates)))
    return stages


def save_models(model, optimizer, replay, args, update, steps, stage, metrics):
    actor_payload = {
        "state_dict": model.actor_state_dict(),
        "in_dim": FRONTIER_OBS_DIM,
        "version": "p28_dense_frontier_rl",
        "frontier_dim": FRONTIER_DIM,
        "econ": dict(legacy_econ(), cap=stage[0] * FPS, start=stage[1]),
        "update": update,
        "total_steps": steps,
        "success_steps": len(replay),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(args.actor), exist_ok=True)
    torch.save(actor_payload, args.actor)
    root, extension = os.path.splitext(args.actor)
    torch.save(actor_payload, f"{root}_iter{update:02d}{extension}")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "update": update,
        "total_steps": steps,
        "stage": stage,
    }, args.checkpoint)
    replay.save()


def train_command(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    model = load_expanded_warmstart(args.warmstart, device)
    difference = verify_expanded_warmstart(model, args.warmstart, device)
    print(f"expanded warm-start exact: max_diff={difference:.1f}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    replay = SuccessReplay(args.success_replay, args.max_success_steps)
    update = 0
    total_steps = 0
    started = time.time()
    for stage_index, stage in enumerate(parse_stages(args.stages), start=1):
        cap_seconds, start_pool, updates = stage
        envs = [FrontierSurvivalEnv(
            args.seed + stage_index * 100_003 + index * 1009,
            cap_seconds, start_pool, args.terminal_penalty)
                for index in range(args.envs)]
        observations = [env.reset() for env in envs]
        pending = [[] for _ in envs]
        print(f"stage {stage_index}: cap={cap_seconds}s start={start_pool:g} "
              f"updates={updates}", flush=True)
        for _ in range(updates):
            update += 1
            rollout, observations, pending, episodes, successes = \
                collect_rollout(
                    model, envs, observations, pending, replay, args, device)
            total_steps += args.envs * args.rollout_steps
            value_only = update <= args.value_warmup_updates
            losses = update_ppo(
                model, optimizer, rollout, replay, args, device, value_only)
            metrics = {**losses, **summarize(episodes)}
            greedy = greedy_probe(
                model, cap_seconds, start_pool, args.terminal_penalty,
                args.seed + 90_000_000 + stage_index, device)
            metrics.update({
                "greedy_end": greedy["end"],
                "greedy_cells": greedy["unique_cells"],
                "greedy_hits": greedy["hits"],
                "greedy_frames": greedy["frames"],
            })
            print(
                f"update {update} steps={total_steps} "
                f"episodes={metrics.get('episodes', 0)} "
                f"cap={metrics.get('cap_pct', 0):.1%} "
                f"cells={metrics.get('mean_cells', 0):.1f} "
                f"hit_s={metrics.get('seconds_per_hit', math.inf):.1f} "
                f"new_success={successes} replay={len(replay)} "
                f"greedy={greedy['end']}/{greedy['unique_cells']}cells "
                f"bc={losses['bc']:.3f} kl={losses['kl']:.4f} "
                f"elapsed={time.time()-started:.0f}s", flush=True)
            save_models(
                model, optimizer, replay, args, update, total_steps,
                stage, metrics)
    print(f"actor: {args.actor}\nsuccess replay: {args.success_replay}", flush=True)


class FrontierRLPolicy:
    name = "p27_observable_frontier"

    def __init__(self, model_path):
        from training.tt_gym_env import TankTroubleGym

        payload = torch.load(model_path, weights_only=True)
        self.network = build_net(int(payload["in_dim"]))
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.dense_frontier = payload.get("version") == \
            "p28_dense_frontier_rl"
        self.econ = payload.get("econ", legacy_econ())
        self.encoder = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
        self.game = None
        self.round_number = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.frontier = None
        self.last_action = (1, 1, 0)

    def reset(self):
        self.game = None
        self.round_number = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.frontier = None
        self.last_action = (1, 1, 0)

    def _dict(self):
        throttle, turn, fire = self.last_action
        return {
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": fire == 1,
        }

    def act_ctx(self, game, ledger):
        if not game.tanks[0].alive:
            return {}
        if game is not self.context_game \
                or game.round_number != self.context_round:
            self.context_game = game
            self.context_round = game.round_number
            self.context_step = 0
            self.frontier = FrontierState(
                game, dense=self.dense_frontier)
            self.last_action = (1, 1, 0)
        else:
            self.frontier.observe_position(game)
        if self.context_step % DECIDE_EVERY == 0:
            observation = frontier_observation(
                self.encoder, game, ledger, self.econ, self.frontier)
            with torch.no_grad():
                scores = self.network(torch.as_tensor(observation)[None])[0]
            self.last_action = CANDIDATES[int(scores.argmax())]
        self.context_step += 1
        return self._dict()

    def act(self, game):
        from training.survival_mode import Ledger

        if game is not self.game or game.round_number != self.round_number:
            self.game = game
            self.round_number = game.round_number
            self.ledger = Ledger(game, self.econ)
        else:
            end = self.ledger.on_frame(game, game.events)
            if end in ("drain", "cap"):
                self.ledger = Ledger(game, self.econ)
        return self.act_ctx(game, self.ledger)


def smoke_command(args):
    device = torch.device(args.device)
    model = load_expanded_warmstart(args.warmstart, device)
    difference = verify_expanded_warmstart(model, args.warmstart, device)
    cap_seconds, start_pool, _ = parse_stages(args.stages)[0]
    env = FrontierSurvivalEnv(
        args.seed, cap_seconds, start_pool, args.terminal_penalty)
    observation = env.reset()
    for _ in range(args.smoke_steps):
        with torch.no_grad():
            scores, _ = model(torch.as_tensor(observation)[None])
        observation, _, done, info = env.step(scores.argmax(1).item())
        if done:
            print(json.dumps(info, ensure_ascii=False), flush=True)
            observation = env.reset()
    print(f"smoke ok: obs={observation.shape} warmstart_diff={difference:.1f}",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "train"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--warmstart", default=DEFAULT_WARMSTART)
        subparser.add_argument("--actor", default=DEFAULT_ACTOR)
        subparser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
        subparser.add_argument("--success-replay", default=DEFAULT_SUCCESS)
        subparser.add_argument("--stages", default="12:120:4")
        subparser.add_argument("--terminal-penalty", type=float,
                               default=TERMINAL_PENALTY)
        subparser.add_argument("--seed", type=int, default=27_000_001)
        subparser.add_argument("--device", default="cpu")
    smoke = subparsers.choices["smoke"]
    smoke.add_argument("--smoke-steps", type=int, default=64)
    smoke.set_defaults(func=smoke_command)

    train = subparsers.choices["train"]
    train.add_argument("--envs", type=int, default=8)
    train.add_argument("--rollout-steps", type=int, default=256)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--minibatch", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--temperature", type=float, default=0.05)
    train.add_argument("--gamma", type=float, default=0.995)
    train.add_argument("--gae-lambda", type=float, default=0.97)
    train.add_argument("--clip", type=float, default=0.1)
    train.add_argument("--value-clip", type=float, default=0.2)
    train.add_argument("--value-coef", type=float, default=0.05)
    train.add_argument("--entropy-coef", type=float, default=0.003)
    train.add_argument("--bc-coef", type=float, default=0.05)
    train.add_argument("--bc-batch", type=int, default=256)
    train.add_argument("--max-success-steps", type=int, default=50_000)
    train.add_argument("--max-grad-norm", type=float, default=0.5)
    train.add_argument("--target-kl", type=float, default=0.02)
    train.add_argument("--value-warmup-updates", type=int, default=1)
    train.add_argument("--torch-threads", type=int, default=4)
    train.set_defaults(func=train_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
