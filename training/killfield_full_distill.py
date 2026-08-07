"""Full P37 single-network score distillation with DAgger correction.

The deployed policy contains one MLP that maps a 914-dimensional factual
observation to all 18 action scores.  It performs no MPC rollout at runtime.
A four-frame maximum macro executor only holds the network's last decision;
an observed collision or exact firing opportunity asks the same network to
decide again.

Pipeline:
  1. teacher-distribution bootstrap with complete paired score landscapes;
  2. train the first expanded score network;
  3. run the student on its own states and have the P37 teacher relabel them;
  4. aggregate and retrain for each DAgger correction round.
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.laika import LaikaAI
from training.killfield_distill import (
    DistillLedger,
    OUTCOME_WEIGHT,
    SCORE_SCALE,
    _classify,
    _forced_fire_scores,
)
from training.killfield_student import (
    P37_OBS_DIM,
    KillFieldFeatureState,
    killfield_observation,
)
from training.killfield_teacher import (
    COMMIT_MOVE_FRAMES,
    COMMIT_TURN_FRAMES,
    DEFAULT_BOUNCES,
    DEFAULT_FLIGHT_FRAMES,
    GOOD_FIRE_BONUS,
    NO_EFFECT_REPEAT_PENALTY,
    KillFieldTeacher,
    _alignment,
    _cell,
    density_rollout,
)
from training.mpc_agent import CANDIDATES, make_sandbox
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.score_distill import build_net
from training.survival_distill_v2 import legacy_econ
from training.survival_expert_iter_530 import apply_action
from training.survival_frontier_rl import FrontierState


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "killfield_full_data")
DEFAULT_MODEL = os.path.join(HERE, "models", "p37_killfield_full.pt")
DEFAULT_WARMSTART = os.path.join(
    HERE, "models", "p37_killfield_student_short.pt")

MOVE_OPTIONS = [(throttle, turn)
                for throttle in (0, 1, 2) for turn in (0, 1, 2)]
ACTION_PREVIEW_FACTS = 10
ACTION_PREVIEW_DIM = len(MOVE_OPTIONS) * ACTION_PREVIEW_FACTS
DECISION_STATE_DIM = 23
P37_FULL_OBS_DIM = P37_OBS_DIM + ACTION_PREVIEW_DIM + DECISION_STATE_DIM
FORCED_FIRE_INDEX = CANDIDATES.index((1, 1, 1))

assert ACTION_PREVIEW_DIM == 90
assert P37_FULL_OBS_DIM == 914


def _factor_action(action):
    features = np.zeros(8, dtype=np.float32)
    throttle, turn, fire = action
    features[int(throttle)] = 1.0
    features[3 + int(turn)] = 1.0
    features[6 + int(bool(fire))] = 1.0
    return features


def action_preview_features(game, field, chain, horizon):
    """Nine movement-conditioned physical field projections.

    These are facts, not teacher values.  They expose the exact quantities
    that P37's leaf calculation would otherwise require the MLP to reconstruct
    from a wall bitmap and a compressed aim field.
    """
    result = np.zeros(ACTION_PREVIEW_DIM, dtype=np.float32)
    start_cell = _cell(game, game.tanks[0])
    start_guidance = field.guidance_at(start_cell)
    start_x, start_y = game.tanks[0].x, game.tanks[0].y
    start_rotation = game.tanks[0].rotation

    for option, (throttle, turn) in enumerate(MOVE_OPTIONS):
        sandbox = make_sandbox(game, "L1", rng_seed=0)
        opponent = sandbox.tanks[1]
        opponent.forward = opponent.backup = False
        opponent.turn_left = opponent.turn_right = False
        opponent.fire = False
        me = sandbox.tanks[0]
        apply_action(sandbox, (throttle, turn, 0))
        simulated_chain = chain.clone()
        previous_cell = start_cell
        chain_gain = 0.0
        peak_value = field.value_at(start_cell)
        for _ in range(horizon):
            sandbox.step()
            simulated_chain.advance()
            current_cell = _cell(sandbox, me)
            peak_value = max(peak_value, field.value_at(current_cell))
            if current_cell != previous_cell:
                chain_gain += simulated_chain.collect_ascent(
                    field, previous_cell, current_cell)
                previous_cell = current_cell
            if not me.alive:
                break

        end_cell = _cell(sandbox, me)
        alignment, _, concentration = _alignment(field, sandbox, me)
        displacement = math.hypot(me.x - start_x, me.y - start_y)
        rotation_delta = abs(
            (me.rotation - start_rotation + 180.0) % 360.0 - 180.0)
        values = [
            field.guidance_at(end_cell),
            np.clip(field.guidance_at(end_cell) - start_guidance, -1.0, 1.0),
            field.value_at(end_cell) / 64.0,
            peak_value / 64.0,
            field.relative_success_at(end_cell),
            alignment,
            concentration,
            np.clip(chain_gain / 127.0, 0.0, 1.0),
            np.clip(displacement / max(game.scale * 2.0, 1e-6), 0.0, 1.0),
            np.clip(rotation_delta / 180.0, 0.0, 1.0),
        ]
        begin = option * ACTION_PREVIEW_FACTS
        result[begin:begin + ACTION_PREVIEW_FACTS] = values
    return result


def decision_state_features(game, state):
    effect = np.asarray([
        np.clip(state.last_displacement / max(game.scale, 1e-6), 0.0, 1.0),
        np.clip(state.last_rotation_delta / 11.25, 0.0, 1.0),
        float(state.failed_translation),
        float(state.failed_turn),
        float(state.action_no_effect),
        np.clip(state.no_effect_frames / 8.0, 0.0, 1.0),
    ], dtype=np.float32)
    macro = np.concatenate([
        np.asarray([
            np.clip(state.observed_commit_remaining
                    / max(COMMIT_MOVE_FRAMES, 1), 0.0, 1.0)
        ], dtype=np.float32),
        _factor_action(state.observed_committed_action),
    ])
    result = np.concatenate([
        _factor_action(state.previous_action), effect, macro])
    if result.shape != (DECISION_STATE_DIM,):
        raise RuntimeError(f"decision-state width mismatch: {result.shape}")
    return result


def full_observation(encoder, game, ledger, econ, frontier, analyzer,
                     field_state, decision_state, horizon):
    base, metrics, fire_facts = killfield_observation(
        encoder, game, ledger, econ, frontier, analyzer, field_state)
    preview = action_preview_features(
        game, field_state.field, field_state.chain, horizon)
    state = decision_state_features(game, decision_state)
    observation = np.concatenate([base, preview, state])
    if observation.shape != (P37_FULL_OBS_DIM,):
        raise RuntimeError(
            f"full P37 observation {observation.shape}, "
            f"expected ({P37_FULL_OBS_DIM},)")
    return observation, metrics, fire_facts


class TeacherDecisionView:
    """Read the teacher state exactly as it stood before its latest choice."""

    def __init__(self, teacher):
        self.previous_action = teacher.observed_previous_action
        self.last_displacement = teacher.last_displacement
        self.last_rotation_delta = teacher.last_rotation_delta
        self.failed_translation = teacher.failed_translation
        self.failed_turn = teacher.failed_turn
        self.action_no_effect = teacher.action_no_effect
        self.no_effect_frames = teacher.no_effect_frames
        self.observed_commit_remaining = teacher.observed_commit_remaining
        self.observed_committed_action = teacher.observed_committed_action


class RuntimeDecisionState:
    """Observable action-effect history and macro state for deployment."""

    def __init__(self, game):
        tank = game.tanks[0]
        self.previous_action = (1, 1, 0)
        self.last_displacement = 0.0
        self.last_rotation_delta = 0.0
        self.failed_translation = False
        self.failed_turn = False
        self.action_no_effect = False
        self.no_effect_frames = 0
        self.commit_remaining = 0
        self.committed_action = (1, 1, 0)
        self.observed_commit_remaining = 0
        self.observed_committed_action = self.committed_action
        self.pose = (tank.x, tank.y, tank.rotation)
        self.frame = game.frame

    def observe(self, game):
        tank = game.tanks[0]
        if game.frame == self.frame:
            return
        self.last_displacement = math.hypot(
            tank.x - self.pose[0], tank.y - self.pose[1])
        self.last_rotation_delta = abs(
            (tank.rotation - self.pose[2] + 180.0) % 360.0 - 180.0)
        requested_translation = self.previous_action[0] != 1
        requested_turn = self.previous_action[1] != 1
        moved = self.last_displacement > max(1e-4, game.scale * 1e-4)
        turned = self.last_rotation_delta > 1e-3
        self.failed_translation = requested_translation and not moved
        self.failed_turn = requested_turn and not turned
        self.action_no_effect = (
            (requested_translation or requested_turn)
            and not moved and not turned)
        self.no_effect_frames = (
            self.no_effect_frames + 1 if self.action_no_effect else 0)
        if self.action_no_effect or tank.hit_something:
            self.commit_remaining = 0
        self.observed_commit_remaining = self.commit_remaining
        self.observed_committed_action = self.committed_action

    def record(self, game, action, planned):
        self.previous_action = action
        if planned:
            self.committed_action = action
            self.commit_remaining = (
                0 if action[2] else
                COMMIT_MOVE_FRAMES if action[0] != 1 else
                COMMIT_TURN_FRAMES if action[1] != 1 else 0)
        else:
            self.commit_remaining = max(0, self.commit_remaining - 1)
        tank = game.tanks[0]
        self.pose = (tank.x, tank.y, tank.rotation)
        self.frame = game.frame


def _save_shard(path, rows):
    if rows:
        X, Y, A, W = zip(*rows)
    else:
        X, Y, A, W = [], [], [], []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32).reshape(-1, P37_FULL_OBS_DIM),
        Y=np.asarray(Y, dtype=np.float32).reshape(-1, len(CANDIDATES)),
        A=np.asarray(A, dtype=np.int64),
        W=np.asarray(W, dtype=np.float32),
    )


def _teacher_worker(job):
    (worker, rounds, seed0, phase_dir, settings) = job
    import torch
    from tank_trouble_original.game import Game
    from training.tt_gym_env import TankTroubleGym

    torch.set_num_threads(1)
    encoder = TankTroubleGym(
        seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
    rows, stats = [], {}
    for episode in range(rounds):
        seed = seed0 + episode
        game = Game(seed=seed, ai_enabled=True)
        teacher = KillFieldTeacher(
            seed=seed ^ 0x37D31517,
            ray_count=settings["rays"],
            max_bounces=settings["bounces"],
            max_flight_frames=settings["flight_frames"],
            horizon=settings["horizon"], hold=settings["hold"])
        ledger = DistillLedger(game, settings["cap_frames"])
        econ = dict(
            legacy_econ(), cap=settings["cap_frames"], start=0.0)
        frontier = FrontierState(game, dense=True)
        analyzer = OpportunityAnalyzer360(game)
        field_state = KillFieldFeatureState(
            settings["rays"], settings["bounces"],
            settings["flight_frames"])
        episode_rows, hits, died = [], [], False
        for _ in range(settings["cap_frames"]):
            teacher.act(game)
            if teacher.last_decision_kind in ("plan", "forced_fire"):
                field_state.adopt_teacher(game, teacher)
                view = TeacherDecisionView(teacher)
                observation, _, _ = full_observation(
                    encoder, game, ledger, econ, frontier, analyzer,
                    field_state, view, settings["horizon"])
                scores = teacher.last_scores
                if scores is None:
                    scores = _forced_fire_scores(teacher, game)
                episode_rows.append((
                    observation,
                    np.clip(scores / SCORE_SCALE, -2.0, 2.0),
                    teacher.last_action_index,
                ))
            apply_action(game, teacher.last_action)
            events = game.step()
            hits.extend((event[1], event[2]) for event in events
                        if event[0] == "hit")
            ledger.advance(game)
            frontier.observe_position(game)
            if game.alive_count <= 1:
                died = True
                break
        if died:
            for _ in range(80):
                events = game.step()
                hits.extend((event[1], event[2]) for event in events
                            if event[0] == "hit")
                if any(event[0] == "round_end" for event in events):
                    break
        outcome = _classify(game, hits, timed_out=not died)
        weight = OUTCOME_WEIGHT[outcome]
        rows.extend((x, y, a, weight) for x, y, a in episode_rows)
        stats[outcome] = stats.get(outcome, 0) + 1
    path = os.path.join(phase_dir, f"shard_{worker:02d}.npz")
    _save_shard(path, rows)
    return path, stats, len(rows)


def _paired_teacher_scores(game, field, chain, seed, settings,
                           decision_state):
    scores = np.asarray([
        density_rollout(
            game, action, field, seed, chain,
            settings["horizon"], settings["hold"])
        for action in CANDIDATES
    ], dtype=np.float32)
    if decision_state.action_no_effect:
        failed = decision_state.previous_action[:2]
        for index, action in enumerate(CANDIDATES):
            if action[:2] == failed:
                scores[index] -= NO_EFFECT_REPEAT_PENALTY
    me = game.tanks[0]
    if (me.alive and game.tanks[1].alive and me.trigger_released
            and game.weapon_ready(me)
            and LaikaAI(game, me).check_bullet_path(
                me.rotation).get("result") == "HIT"):
        scores[FORCED_FIRE_INDEX] = max(
            float(scores.max()) + GOOD_FIRE_BONUS, GOOD_FIRE_BONUS)
    return scores


def _dagger_worker(job):
    (worker, rounds, seed0, phase_dir, model_path, settings) = job
    import torch
    from tank_trouble_original.game import Game

    torch.set_num_threads(1)
    rng = random.Random(seed0 ^ (worker * 104729 + 0xDA66E2))
    policy = KillFieldFullPolicy(model_path, **settings)
    rows, stats = [], {}
    for episode in range(rounds):
        game = Game(seed=seed0 + episode, ai_enabled=True)
        policy.reset()
        hits, died = [], False
        for _ in range(settings["cap_frames"]):
            action_dict = policy.act(game)
            if policy.last_planned:
                scores = _paired_teacher_scores(
                    game, policy.field, policy.field_state.chain,
                    rng.randrange(1 << 30), settings,
                    policy.decision_state)
                rows.append((
                    policy.last_observation.copy(),
                    np.clip(scores / SCORE_SCALE, -2.0, 2.0),
                    int(np.argmax(scores)),
                    1.0,
                ))
            apply_action(game, policy.last_action)
            events = game.step()
            hits.extend((event[1], event[2]) for event in events
                        if event[0] == "hit")
            if game.alive_count <= 1:
                died = True
                break
        if died:
            for _ in range(80):
                events = game.step()
                hits.extend((event[1], event[2]) for event in events
                            if event[0] == "hit")
                if any(event[0] == "round_end" for event in events):
                    break
        outcome = _classify(game, hits, timed_out=not died)
        stats[outcome] = stats.get(outcome, 0) + 1
    path = os.path.join(phase_dir, f"shard_{worker:02d}.npz")
    _save_shard(path, rows)
    return path, stats, len(rows)


def _phase_jobs(rounds, workers, seed, phase_dir, settings,
                worker_function, model_path=None):
    base, remainder = divmod(rounds, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + int(worker < remainder)
        if count <= 0:
            continue
        common = (worker, count, seed + offset, phase_dir)
        jobs.append(
            common + ((settings,) if model_path is None
                      else (model_path, settings)))
        offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(worker_function, jobs)
    stats, samples, paths = {}, 0, []
    for path, worker_stats, count in results:
        paths.append(path)
        samples += count
        for key, value in worker_stats.items():
            stats[key] = stats.get(key, 0) + value
    print(
        f"  complete: {samples} samples / {time.time()-started:.1f}s / "
        f"rollout outcomes={stats}", flush=True)
    return paths


def collect_teacher(args, settings):
    phase_dir = os.path.join(args.data_dir, "teacher")
    print(f"===== teacher bootstrap: {args.teacher_rounds} rounds =====",
          flush=True)
    return _phase_jobs(
        args.teacher_rounds, args.workers, args.seed, phase_dir,
        settings, _teacher_worker)


def collect_dagger(args, settings, iteration, model_path):
    phase_dir = os.path.join(args.data_dir, f"dagger_{iteration:02d}")
    seed = args.seed + 10_000_000 + iteration * 1_000_000
    print(
        f"===== DAgger {iteration}: {args.rounds_per_dagger} student rounds =====",
        flush=True)
    return _phase_jobs(
        args.rounds_per_dagger, args.workers, seed, phase_dir,
        settings, _dagger_worker, model_path)


def load_all(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*", "shard_*.npz")))
    arrays = {key: [] for key in ("X", "Y", "A", "W")}
    for path in paths:
        data = np.load(path)
        if data["X"].shape[1] != P37_FULL_OBS_DIM:
            raise ValueError(f"bad full-P37 shard: {path} {data['X'].shape}")
        for key in arrays:
            arrays[key].append(data[key])
    if not arrays["X"]:
        raise RuntimeError("no full-P37 data")
    return {key: np.concatenate(values) for key, values in arrays.items()}


def _load_expanded_network(path, device):
    import torch

    payload = torch.load(path, map_location=device, weights_only=True)
    source_dim = int(payload["in_dim"])
    if source_dim > P37_FULL_OBS_DIM:
        raise ValueError(f"cannot shrink {source_dim} to {P37_FULL_OBS_DIM}")
    source = payload["state_dict"]
    network = build_net(P37_FULL_OBS_DIM).to(device)
    with torch.no_grad():
        network[0].weight.zero_()
        network[0].weight[:, :source_dim].copy_(source["0.weight"])
        network[0].bias.copy_(source["0.bias"])
        for index in (2, 4, 6):
            network[index].weight.copy_(source[f"{index}.weight"])
            network[index].bias.copy_(source[f"{index}.bias"])
    return network


def train_aggregate(args, settings, iteration, source_path):
    import torch
    import torch.nn.functional as functional

    data = load_all(args.data_dir)
    device = torch.device(args.device)
    network = _load_expanded_network(source_path, device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    X = torch.as_tensor(data["X"], device=device)
    Y = torch.as_tensor(data["Y"], device=device)
    A = torch.as_tensor(data["A"], device=device)
    W = torch.as_tensor(data["W"], device=device)
    n = len(X)
    epochs = args.bootstrap_epochs if iteration == 0 else args.dagger_epochs
    print(
        f"===== train iter {iteration}: {n} aggregate samples / "
        f"{epochs} epochs =====", flush=True)
    started = time.time()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, device=device)
        total, batches = 0.0, 0
        for begin in range(0, n, args.batch):
            index = order[begin:begin + args.batch]
            prediction = network(X[index])
            regression = functional.smooth_l1_loss(
                prediction, Y[index], reduction="none").mean(dim=1)
            classification = functional.cross_entropy(
                prediction / args.action_temperature,
                A[index], reduction="none")
            weight = W[index]
            loss = (weight * (
                regression + args.action_coef * classification)).sum() \
                / weight.sum().clamp_min(1e-6)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(
            f"  epoch {epoch:02d}/{epochs} "
            f"loss={total/max(batches,1):.5f} "
            f"elapsed={time.time()-started:.1f}s", flush=True)

    root, extension = os.path.splitext(args.model)
    path = f"{root}_iter{iteration:02d}{extension}"
    payload = {
        "state_dict": network.state_dict(),
        "in_dim": P37_FULL_OBS_DIM,
        "version": "p37_killfield_full_dagger",
        "iteration": iteration,
        "samples": n,
        "settings": settings,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    torch.save(payload, args.model)
    print(f"  saved: {path} and {args.model}", flush=True)
    return args.model


class KillFieldFullPolicy:
    """One score network; no teacher scores or MPC at runtime."""

    name = "P37 完整蒸馏单网络"

    def __init__(self, model_path=DEFAULT_MODEL, rays=512, bounces=2,
                 flight_frames=75, horizon=36, hold=8, cap_frames=500,
                 **_ignored):
        import torch
        from training.tt_gym_env import TankTroubleGym

        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if int(payload.get("in_dim", 0)) != P37_FULL_OBS_DIM:
            raise ValueError(
                f"full P37 policy requires {P37_FULL_OBS_DIM} inputs")
        self.torch = torch
        self.network = build_net(P37_FULL_OBS_DIM)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.encoder = TankTroubleGym(
            seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
        self.rays = int(rays)
        self.bounces = int(bounces)
        self.flight_frames = int(flight_frames)
        self.horizon = int(horizon)
        self.hold = int(hold)
        self.cap_frames = int(cap_frames)
        self.reset()

    def reset(self):
        self.game = None
        self.round_number = None
        self.ledger = None
        self.frontier = None
        self.analyzer = None
        self.field_state = None
        self.decision_state = None
        self.field = None
        self.last_frame = None
        self.last_action = (1, 1, 0)
        self.last_observation = None
        self.last_planned = False

    def _start_round(self, game):
        self.game = game
        self.round_number = game.round_number
        self.ledger = DistillLedger(game, self.cap_frames)
        self.econ = dict(
            legacy_econ(), cap=self.cap_frames, start=0.0)
        self.frontier = FrontierState(game, dense=True)
        self.analyzer = OpportunityAnalyzer360(game)
        self.field_state = KillFieldFeatureState(
            self.rays, self.bounces, self.flight_frames)
        self.field_state.ensure_field(game)
        self.decision_state = RuntimeDecisionState(game)
        self.last_frame = game.frame

    @staticmethod
    def _action_dict(action):
        throttle, turn, fire = action
        return {
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": fire == 1,
        }

    def act(self, game):
        self.last_planned = False
        if not game.tanks[0].alive:
            return {}
        if not game.tanks[1].alive:
            state = self.decision_state
            action = ((1, 1, 0) if state is None
                      else state.committed_action)
            self.last_action = (action[0], action[1], 0)
            return self._action_dict(self.last_action)
        if (game is not self.game
                or game.round_number != self.round_number):
            self._start_round(game)
        elapsed = max(0, game.frame - self.last_frame)
        if elapsed:
            self.ledger.advance(game, elapsed)
            self.frontier.observe_position(game)
            self.field_state.advance(game, elapsed)
            self.decision_state.observe(game)
            self.last_frame = game.frame
        self.field = self.field_state.ensure_field(game)

        me = game.tanks[0]
        exact_hit = (
            me.trigger_released and game.weapon_ready(me)
            and LaikaAI(game, me).check_bullet_path(
                me.rotation).get("result") == "HIT")
        if exact_hit:
            self.decision_state.commit_remaining = 0
        if (self.decision_state.commit_remaining > 0
                and not self.decision_state.action_no_effect
                and not me.hit_something and not exact_hit):
            action = self.decision_state.committed_action
            self.last_action = action
            self.decision_state.record(game, action, planned=False)
            return self._action_dict(action)

        self.decision_state.observed_commit_remaining = \
            self.decision_state.commit_remaining
        self.decision_state.observed_committed_action = \
            self.decision_state.committed_action
        observation, _, _ = full_observation(
            self.encoder, game, self.ledger, self.econ,
            self.frontier, self.analyzer, self.field_state,
            self.decision_state, self.horizon)
        with self.torch.no_grad():
            scores = self.network(
                self.torch.as_tensor(observation).unsqueeze(0))[0]
        action = CANDIDATES[int(scores.argmax())]
        self.last_observation = observation
        self.last_action = action
        self.last_planned = True
        self.decision_state.record(game, action, planned=True)
        return self._action_dict(action)

    def telemetry(self):
        state = self.field_state
        mean_build = 0.0 if state is None else (
            state.field_build_seconds / max(state.field_builds, 1))
        decision = self.decision_state
        return {
            "field_builds": 0 if state is None else state.field_builds,
            "mean_field_build_seconds": mean_build,
            "cached_target_cells": 0 if state is None else len(state.field_cache),
            "hunt_chain": 0 if state is None else state.chain.count,
            "hunt_chain_timer": 0 if state is None else state.chain.timer,
            "hunt_chain_total": 0.0,
            "last_chain_gain": 0.0 if state is None else state.last_chain_gain,
            "no_effect_frames": 0 if decision is None else decision.no_effect_frames,
            "no_effect_events": 0,
        }


def settings_from_args(args):
    return {
        "rays": args.rays,
        "bounces": args.bounces,
        "flight_frames": args.flight_frames,
        "horizon": args.horizon,
        "hold": args.hold,
        "cap_frames": args.cap_frames,
    }


def pipeline(args):
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    settings = settings_from_args(args)
    collect_teacher(args, settings)
    model_path = train_aggregate(
        args, settings, iteration=0, source_path=args.warmstart)
    for iteration in range(1, args.dagger_rounds + 1):
        collect_dagger(args, settings, iteration, model_path)
        model_path = train_aggregate(
            args, settings, iteration=iteration, source_path=model_path)
    print(
        f"===== full P37 pipeline complete (no policy evaluation): "
        f"{model_path} =====", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pipeline"])
    parser.add_argument("--teacher-rounds", type=int, default=96)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--rounds-per-dagger", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=38_000_000)
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--bounces", type=int, default=2)
    parser.add_argument("--flight-frames", type=int, default=75)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--hold", type=int, default=8)
    parser.add_argument("--cap-frames", type=int, default=500)
    parser.add_argument("--bootstrap-epochs", type=int, default=14)
    parser.add_argument("--dagger-epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--action-coef", type=float, default=0.04)
    parser.add_argument("--action-temperature", type=float, default=0.08)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmstart", default=DEFAULT_WARMSTART)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    pipeline(args)


if __name__ == "__main__":
    main()
