"""P38 fast privileged distillation for the P37 kill-field teacher.

The P37 deployed observation still performs two online searches: a cached
inverse kill-field build and movement-conditioned physics previews.  P38 keeps
both computations on the offline teacher side only.  The deployed network sees
cheap physical facts plus a small spatial maze tensor, while auxiliary heads
learn to reconstruct the privileged field and preview targets during training.

Unlike P37 collection, an episode does not terminate at the first destroyed
tank.  If Laika dies first, a dedicated survival teacher keeps planning until
the original round_end scoring frame, so the student receives labels for the
75 live post-kill frames in which existing bullets can still cause a double
death.
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

from tank_trouble_original import constants as C
from tank_trouble_original.laika import LaikaAI
from training.killfield_distill import (
    DistillLedger,
    OUTCOME_WEIGHT,
    SCORE_SCALE,
    _classify,
    _forced_fire_scores,
)
from training.killfield_full_distill import (
    ACTION_PREVIEW_DIM,
    DECISION_STATE_DIM,
    RuntimeDecisionState,
    _paired_teacher_scores,
    action_preview_features,
    decision_state_features,
)
from training.killfield_student import (
    KILLFIELD_EXTRA_DIM,
    KillFieldFeatureState,
)
from training.killfield_teacher import (
    COMMIT_MOVE_FRAMES,
    COMMIT_TURN_FRAMES,
    DEFAULT_BOUNCES,
    DEFAULT_FLIGHT_FRAMES,
    KillFieldTeacher,
)
from training.mpc_agent import CANDIDATES, make_sandbox
from training.score_distill import extra_bullets
from training.survival_distill_v2 import bind_env, legacy_econ
from training.survival_expert_iter_530 import apply_action
from training.survival_frontier_rl import (
    FRONTIER_DIM,
    MAP_H,
    MAP_W,
    FrontierState,
)
from training.tt_gym_env import obs_dim


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "killfield_fast_data")
DEFAULT_MODEL = os.path.join(HERE, "models", "p38_killfield_fast.pt")

RAW_PHYSICS_DIM = obs_dim(True, True)
EXTRA_BULLET_DIM = 24
FAST_FIRE_DIM = 6
PHASE_DIM = 4
FAST_VECTOR_DIM = (
    RAW_PHYSICS_DIM + EXTRA_BULLET_DIM + FRONTIER_DIM
    + FAST_FIRE_DIM + DECISION_STATE_DIM + PHASE_DIM)
FAST_MAP_CHANNELS = 8
POST_KILL_DECIDE_EVERY = 2
POST_KILL_FIRE_PENALTY = 3000.0

assert RAW_PHYSICS_DIM == 125
assert FRONTIER_DIM == 126
assert FAST_VECTOR_DIM == 308


def _fast_fire_facts(game):
    """One current firing ray, not a fan or action-conditioned search."""
    me, enemy = game.tanks
    weapon_ready = bool(game.weapon_ready(me))
    trigger_ready = bool(me.trigger_released)
    result = None
    closest = float("inf")
    if me.alive and enemy.alive and weapon_ready and trigger_ready:
        assessment = LaikaAI(game, me).check_bullet_path(me.rotation)
        result = assessment.get("result")
        closest = float(assessment.get("closest", float("inf")))
    proximity = 0.0 if not math.isfinite(closest) else max(
        0.0, 1.0 - closest / max(game.scale, 1e-6))
    return np.asarray([
        float(weapon_ready),
        float(trigger_ready),
        float(result == "HIT"),
        float(result == "SUICIDE"),
        proximity,
        float(enemy.alive),
    ], dtype=np.float32)


def phase_features(game):
    post_kill = bool(game.tanks[0].alive and not game.tanks[1].alive)
    live_remaining = 0.0
    if post_kill and not game.frozen:
        live_remaining = np.clip(
            (game.end_count - C.NUMBEROFFRAMESFROZEN)
            / max(C.NUMBEROFFRAMESBEFOREEND
                  - C.NUMBEROFFRAMESFROZEN, 1),
            0.0, 1.0)
    return np.asarray([
        float(game.tanks[1].alive),
        float(post_kill),
        float(live_remaining),
        float(game.frozen),
    ], dtype=np.float32)


def fast_vector_observation(encoder, game, ledger, frontier, decision_state):
    """308 cheap features with no sandbox or inverse-field construction."""
    bind_env(encoder, game, ledger.frames)
    vector = np.concatenate([
        encoder._obs(),
        extra_bullets(game),
        frontier.features(game),
        _fast_fire_facts(game),
        decision_state_features(game, decision_state),
        phase_features(game),
    ]).astype(np.float32, copy=False)
    if vector.shape != (FAST_VECTOR_DIM,):
        raise RuntimeError(
            f"P38 vector shape {vector.shape}, expected ({FAST_VECTOR_DIM},)")
    return vector


def fast_spatial_observation(game, frontier):
    """Maze, entities, exploration, and bullet occupancy on a 10x12 grid."""
    spatial = np.zeros(
        (FAST_MAP_CHANNELS, MAP_H, MAP_W), dtype=np.float32)
    spatial[0:2] = 1.0  # solid padding outside the generated maze
    width = min(len(game.maze), MAP_W)
    height = min(len(game.maze[0]), MAP_H)
    for x in range(width):
        for y in range(height):
            spatial[0, y, x] = float(game.maze[x][y][1])
            spatial[1, y, x] = float(game.maze[x][y][2])
            spatial[2, y, x] = 1.0

    for channel, tank in ((3, game.tanks[0]), (4, game.tanks[1])):
        x, y = int(tank.x // game.scale), int(tank.y // game.scale)
        if 0 <= x < MAP_W and 0 <= y < MAP_H:
            spatial[channel, y, x] = 1.0
    spatial[5] = frontier.visited

    for bullet in game.bullets:
        if bullet.removed:
            continue
        x, y = int(bullet.x // game.scale), int(bullet.y // game.scale)
        if not (0 <= x < MAP_W and 0 <= y < MAP_H):
            continue
        channel = 6 if bullet.owner is game.tanks[0] else 7
        urgency = 1.0 - np.clip(
            bullet.lifetime / max(float(C.BULLETLIFETIME), 1.0), 0.0, 1.0)
        spatial[channel, y, x] = max(
            spatial[channel, y, x], 0.25 + 0.75 * urgency)
    return spatial


def post_kill_survival_scores(game, horizon=75):
    """Exact offline survival labels for the original live-bullet window.

    Each movement is held in a sandbox until scoring freeze or the requested
    horizon.  The real teacher replans every two frames, so this is a receding
    horizon controller rather than a claim that one command is globally best.
    """
    scores = np.full(len(CANDIDATES), -1e9, dtype=np.float32)
    remaining = max(
        1, game.end_count - C.NUMBEROFFRAMESFROZEN)
    rollout_frames = min(int(horizon), int(remaining))
    for move_index, (throttle, turn) in enumerate(
            (action[:2] for action in CANDIDATES[::2])):
        sandbox = make_sandbox(game, "L1", rng_seed=0)
        me = sandbox.tanks[0]
        start_x, start_y = me.x, me.y
        action = (throttle, turn, 0)
        apply_action(sandbox, action)
        min_clearance = 8.0
        survived = True
        elapsed = 0
        for elapsed in range(rollout_frames):
            events = sandbox.step()
            if not me.alive:
                survived = False
                break
            if sandbox.bullets:
                clearance = min(
                    math.hypot(b.x - me.x, b.y - me.y)
                    for b in sandbox.bullets) / max(sandbox.scale, 1e-6)
                min_clearance = min(min_clearance, clearance)
            if sandbox.frozen or any(
                    event[0] == "round_end" for event in events):
                break

        if survived:
            displacement = math.hypot(me.x - start_x, me.y - start_y) \
                / max(sandbox.scale, 1e-6)
            control_cost = (
                0.20 * float(throttle != 1)
                + 0.10 * float(turn != 1))
            score = (SCORE_SCALE + 40.0 * min(min_clearance, 8.0)
                     + 0.5 * min(displacement, 8.0) - control_cost)
        else:
            score = -SCORE_SCALE + 8.0 * elapsed
        no_fire_index = move_index * 2
        scores[no_fire_index] = score
        scores[no_fire_index + 1] = score - POST_KILL_FIRE_PENALTY
    return scores


def build_fast_network():
    import torch.nn as nn

    class FastKillFieldNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.map_encoder = nn.Sequential(
                nn.Conv2d(FAST_MAP_CHANNELS, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * MAP_H * MAP_W, 256), nn.ReLU(),
            )
            self.vector_encoder = nn.Sequential(
                nn.Linear(FAST_VECTOR_DIM, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(768, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
            )
            self.score_head = nn.Linear(1024, len(CANDIDATES))
            self.field_head = nn.Linear(1024, KILLFIELD_EXTRA_DIM)
            self.preview_head = nn.Linear(1024, ACTION_PREVIEW_DIM)
            self.survival_head = nn.Linear(1024, 1)

        def forward(self, vector, spatial):
            latent = self.fusion(__import__("torch").cat([
                self.vector_encoder(vector), self.map_encoder(spatial)
            ], dim=1))
            return (
                self.score_head(latent),
                self.field_head(latent),
                self.preview_head(latent),
                self.survival_head(latent).squeeze(1),
            )

    return FastKillFieldNet()


def _save_shard(path, rows):
    keys = ("V", "M", "Y", "A", "W", "F", "P", "FM", "S", "PH", "G")
    if rows:
        columns = dict(zip(keys, zip(*rows)))
    else:
        columns = {key: [] for key in keys}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        V=np.asarray(columns["V"], dtype=np.float32).reshape(
            -1, FAST_VECTOR_DIM),
        M=np.asarray(columns["M"], dtype=np.float32).reshape(
            -1, FAST_MAP_CHANNELS, MAP_H, MAP_W),
        Y=np.asarray(columns["Y"], dtype=np.float32).reshape(
            -1, len(CANDIDATES)),
        A=np.asarray(columns["A"], dtype=np.int64),
        W=np.asarray(columns["W"], dtype=np.float32),
        F=np.asarray(columns["F"], dtype=np.float32).reshape(
            -1, KILLFIELD_EXTRA_DIM),
        P=np.asarray(columns["P"], dtype=np.float32).reshape(
            -1, ACTION_PREVIEW_DIM),
        FM=np.asarray(columns["FM"], dtype=np.float32),
        S=np.asarray(columns["S"], dtype=np.float32),
        PH=np.asarray(columns["PH"], dtype=np.uint8),
        G=np.asarray(columns["G"], dtype=np.int64),
    )


def _teacher_worker(job):
    worker, rounds, seed0, phase_dir, settings = job
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
            seed=seed ^ 0x38FA571,
            ray_count=settings["rays"],
            max_bounces=settings["bounces"],
            max_flight_frames=settings["flight_frames"],
            horizon=settings["horizon"], hold=settings["hold"])
        ledger = DistillLedger(game, settings["cap_frames"])
        frontier = FrontierState(game, dense=True)
        field_state = KillFieldFeatureState(
            settings["rays"], settings["bounces"],
            settings["flight_frames"])
        decision_state = RuntimeDecisionState(game)
        episode_rows = []
        hits = []
        first_death = False
        round_ended = False
        post_action = (1, 1, 0)
        post_commit = 0

        max_total = settings["cap_frames"] + 170
        for _ in range(max_total):
            if game.tanks[0].alive:
                decision_state.observe(game)

            planned = False
            scores = None
            field_label = np.zeros(KILLFIELD_EXTRA_DIM, dtype=np.float32)
            preview_label = np.zeros(ACTION_PREVIEW_DIM, dtype=np.float32)
            field_mask = 0.0

            if game.tanks[0].alive and game.tanks[1].alive:
                teacher.act(game)
                action = teacher.last_action
                planned = teacher.last_decision_kind in (
                    "plan", "forced_fire")
                if planned:
                    field_state.adopt_teacher(game, teacher)
                    field_label = field_state.features(game)
                    preview_label = action_preview_features(
                        game, field_state.field, field_state.chain,
                        settings["horizon"])
                    field_mask = 1.0
                    scores = teacher.last_scores
                    if scores is None:
                        scores = _forced_fire_scores(teacher, game)
            elif (game.tanks[0].alive and not game.tanks[1].alive
                  and not game.frozen):
                if (post_commit <= 0 or decision_state.action_no_effect
                        or game.tanks[0].hit_something):
                    scores = post_kill_survival_scores(
                        game, settings["post_kill_horizon"])
                    action = CANDIDATES[int(np.argmax(scores))]
                    post_action = (action[0], action[1], 0)
                    post_commit = POST_KILL_DECIDE_EVERY - 1
                    action = post_action
                    planned = True
                else:
                    action = post_action
                    post_commit -= 1
            else:
                action = (1, 1, 0)

            if planned:
                vector = fast_vector_observation(
                    encoder, game, ledger, frontier, decision_state)
                spatial = fast_spatial_observation(game, frontier)
                episode_rows.append([
                    vector, spatial,
                    np.clip(scores / SCORE_SCALE, -2.0, 2.0),
                    CANDIDATES.index(action),
                    field_label, preview_label, field_mask,
                    int(not game.tanks[1].alive),
                ])

            if game.tanks[0].alive:
                decision_state.record(game, action, planned=planned)
            apply_action(game, action)
            events = game.step()
            hits.extend((event[1], event[2]) for event in events
                        if event[0] == "hit")
            ledger.advance(game)
            frontier.observe_position(game)
            if game.alive_count <= 1:
                first_death = True
            if any(event[0] == "round_end" for event in events):
                round_ended = True
                break
            if not first_death and ledger.frames >= settings["cap_frames"]:
                break

        outcome = _classify(
            game, hits, timed_out=not first_death)
        survived = float(
            round_ended and game.tanks[0].alive
            and not game.tanks[1].alive)
        for (vector, spatial, score, action, field_label,
             preview_label, field_mask, post_phase) in episode_rows:
            weight = (2.5 if post_phase else OUTCOME_WEIGHT[outcome])
            rows.append((
                vector, spatial, score, action, weight,
                field_label, preview_label, field_mask,
                survived, post_phase, seed,
            ))
        stats[outcome] = stats.get(outcome, 0) + 1

    path = os.path.join(phase_dir, f"shard_{worker:02d}.npz")
    _save_shard(path, rows)
    post_samples = sum(int(row[-2]) for row in rows)
    return path, stats, len(rows), post_samples


def collect_teacher(args, settings):
    phase_dir = os.path.join(args.data_dir, "teacher")
    base, remainder = divmod(args.teacher_rounds, args.workers)
    jobs, offset = [], 0
    for worker in range(args.workers):
        count = base + int(worker < remainder)
        if count:
            jobs.append((
                worker, count, args.seed + offset, phase_dir, settings))
            offset += count
    print(
        f"===== P38 collect: {args.teacher_rounds} rounds / "
        f"{len(jobs)} workers =====", flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_teacher_worker, jobs)
    stats, total, post = {}, 0, 0
    for _, worker_stats, count, post_count in results:
        total += count
        post += post_count
        for key, value in worker_stats.items():
            stats[key] = stats.get(key, 0) + value
    print(
        f"  collected {total} decisions ({post} post-kill) / "
        f"{time.time()-started:.1f}s / outcomes={stats}", flush=True)


def _dagger_worker(job):
    (worker, rounds, seed0, phase_dir, model_path, settings) = job
    import torch
    from tank_trouble_original.game import Game

    torch.set_num_threads(1)
    rng = random.Random(seed0 ^ (worker * 104729 + 0xDA6638))
    policy = KillFieldFastPolicy(model_path, cap_frames=settings["cap_frames"])
    rows, stats = [], {}
    for episode in range(rounds):
        seed = seed0 + episode
        game = Game(seed=seed, ai_enabled=True)
        policy.reset()
        field_state = KillFieldFeatureState(
            settings["rays"], settings["bounces"],
            settings["flight_frames"])
        field_state.ensure_field(game)
        episode_rows = []
        hits = []
        first_death = False
        round_ended = False
        max_total = settings["cap_frames"] + 170
        for _ in range(max_total):
            action_dict = policy.act(game)
            if policy.last_planned and policy.last_observation is not None:
                vector, spatial = policy.last_observation
                post_phase = int(not game.tanks[1].alive)
                if post_phase:
                    scores = post_kill_survival_scores(
                        game, settings["post_kill_horizon"])
                    field_label = np.zeros(
                        KILLFIELD_EXTRA_DIM, dtype=np.float32)
                    preview_label = np.zeros(
                        ACTION_PREVIEW_DIM, dtype=np.float32)
                    field_mask = 0.0
                else:
                    field = field_state.ensure_field(game)
                    scores = _paired_teacher_scores(
                        game, field, field_state.chain,
                        rng.randrange(1 << 30), settings,
                        policy.decision_state)
                    field_label = field_state.features(game)
                    preview_label = action_preview_features(
                        game, field, field_state.chain,
                        settings["horizon"])
                    field_mask = 1.0
                episode_rows.append([
                    vector.copy(), spatial.copy(),
                    np.clip(scores / SCORE_SCALE, -2.0, 2.0),
                    int(np.argmax(scores)), field_label, preview_label,
                    field_mask, post_phase,
                ])

            apply_action(game, policy.last_action)
            events = game.step()
            hits.extend((event[1], event[2]) for event in events
                        if event[0] == "hit")
            if game.tanks[1].alive:
                field_state.advance(game)
            if game.alive_count <= 1:
                first_death = True
            if any(event[0] == "round_end" for event in events):
                round_ended = True
                break
            if not first_death and game.frame >= settings["cap_frames"]:
                break

        outcome = _classify(game, hits, timed_out=not first_death)
        survived = float(
            round_ended and game.tanks[0].alive
            and not game.tanks[1].alive)
        for (vector, spatial, score, action, field_label,
             preview_label, field_mask, post_phase) in episode_rows:
            weight = 2.5 if post_phase else OUTCOME_WEIGHT[outcome]
            rows.append((
                vector, spatial, score, action, weight,
                field_label, preview_label, field_mask,
                survived, post_phase, seed,
            ))
        stats[outcome] = stats.get(outcome, 0) + 1

    path = os.path.join(phase_dir, f"shard_{worker:02d}.npz")
    _save_shard(path, rows)
    post_samples = sum(int(row[-2]) for row in rows)
    return path, stats, len(rows), post_samples


def collect_dagger(args, settings, iteration, model_path):
    phase_dir = os.path.join(args.data_dir, f"dagger_{iteration:02d}")
    seed0 = args.seed + 10_000_000 + iteration * 1_000_000
    base, remainder = divmod(args.rounds_per_dagger, args.workers)
    jobs, offset = [], 0
    for worker in range(args.workers):
        count = base + int(worker < remainder)
        if count:
            jobs.append((
                worker, count, seed0 + offset, phase_dir,
                model_path, settings))
            offset += count
    print(
        f"===== P38 DAgger {iteration}: {args.rounds_per_dagger} rounds / "
        f"{len(jobs)} workers =====", flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_dagger_worker, jobs)
    stats, total, post = {}, 0, 0
    for _, worker_stats, count, post_count in results:
        total += count
        post += post_count
        for key, value in worker_stats.items():
            stats[key] = stats.get(key, 0) + value
    print(
        f"  relabeled {total} decisions ({post} post-kill) / "
        f"{time.time()-started:.1f}s / outcomes={stats}", flush=True)


def load_all(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*", "shard_*.npz")))
    keys = ("V", "M", "Y", "A", "W", "F", "P", "FM", "S", "PH", "G")
    arrays = {key: [] for key in keys}
    for path in paths:
        data = np.load(path)
        if data["V"].shape[1] != FAST_VECTOR_DIM:
            raise ValueError(f"bad P38 shard {path}: {data['V'].shape}")
        for key in keys:
            arrays[key].append(data[key])
    if not arrays["V"]:
        raise RuntimeError("no P38 training data")
    return {key: np.concatenate(value) for key, value in arrays.items()}


def train(args, settings, iteration=0, source_model=None):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    data = load_all(args.data_dir)
    n = len(data["V"])
    generator = np.random.default_rng(args.seed)
    # Split by complete episodes, not random frames from the same trajectory.
    # Otherwise adjacent teacher states leak into validation and overstate
    # generalization to new maps/seeds.
    groups = generator.permutation(np.unique(data["G"]))
    val_group_count = max(1, int(len(groups) * args.validation_fraction))
    post_groups = generator.permutation(np.unique(data["G"][data["PH"] > 0]))
    chosen = []
    if len(post_groups):
        chosen.append(int(post_groups[0]))
    chosen.extend(
        int(group) for group in groups
        if int(group) not in chosen)
    val_groups = np.asarray(chosen[:val_group_count], dtype=np.int64)
    is_val = np.isin(data["G"], val_groups)
    val_index = np.flatnonzero(is_val)
    train_index = np.flatnonzero(~is_val)
    val_count = len(val_index)
    device = torch.device(args.device)
    tensors = {
        key: torch.as_tensor(value, device=device)
        for key, value in data.items()
    }
    network = build_fast_network().to(device)
    if source_model is not None:
        payload = torch.load(
            source_model, map_location=device, weights_only=True)
        network.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    def batch_loss(index):
        scores, field, preview, survival = network(
            tensors["V"][index], tensors["M"][index])
        score_loss = functional.smooth_l1_loss(
            scores, tensors["Y"][index], reduction="none").mean(1)
        # Distill the complete score landscape.  Hard argmax labels are noisy
        # when several controls are effectively tied (especially after a kill
        # with no incoming bullet), whereas the teacher distribution preserves
        # that indifference.
        teacher_distribution = functional.softmax(
            tensors["Y"][index] / args.action_temperature, dim=1)
        action_loss = -(
            teacher_distribution * functional.log_softmax(
                scores / args.action_temperature, dim=1)).sum(1)
        field_loss = functional.smooth_l1_loss(
            field, tensors["F"][index], reduction="none").mean(1)
        preview_loss = functional.smooth_l1_loss(
            preview, tensors["P"][index], reduction="none").mean(1)
        auxiliary = tensors["FM"][index] * (
            args.field_coef * field_loss
            + args.preview_coef * preview_loss)
        survival_loss = functional.binary_cross_entropy_with_logits(
            survival, tensors["S"][index], reduction="none")
        combined = (score_loss + args.action_coef * action_loss
                    + auxiliary + args.survival_coef * survival_loss)
        movement_scores = tensors["Y"][index, ::2]
        movement_spread = (
            movement_scores.max(1).values
            - movement_scores.min(1).values)
        critical_weight = 1.0 + args.critical_coef * movement_spread.clamp(
            0.0, 1.0)
        weight = tensors["W"][index] * critical_weight
        return (combined * weight).sum() / weight.sum().clamp_min(1e-6)

    def validation_metrics():
        network.eval()
        with torch.inference_mode():
            index = torch.as_tensor(val_index, device=device)
            scores, field, preview, survival = network(
                tensors["V"][index], tensors["M"][index])
            target = tensors["A"][index]
            correct = scores.argmax(1) == target
            phase = tensors["PH"][index].bool()
            chosen = scores.argmax(1)
            teacher_scores = tensors["Y"][index]
            regret = (teacher_scores.max(1).values
                      - teacher_scores.gather(
                          1, chosen[:, None]).squeeze(1))
            overall = correct.float().mean().item()
            pre = correct[~phase].float().mean().item() if (~phase).any() else 0.0
            post = correct[phase].float().mean().item() if phase.any() else 0.0
            pre_regret = regret[~phase].mean().item() if (~phase).any() else 0.0
            post_regret = regret[phase].mean().item() if phase.any() else 0.0
            catastrophic = (regret > 0.25).float().mean().item()
            mse = functional.mse_loss(scores, tensors["Y"][index]).item()
            privileged = tensors["FM"][index].bool()
            field_mae = functional.l1_loss(
                field[privileged], tensors["F"][index][privileged]).item() \
                if privileged.any() else 0.0
            preview_mae = functional.l1_loss(
                preview[privileged], tensors["P"][index][privileged]).item() \
                if privileged.any() else 0.0
            survival_accuracy = (
                (survival.sigmoid() >= 0.5)
                == tensors["S"][index].bool()).float().mean().item()
        network.train()
        return (mse, overall, pre, post, pre_regret, post_regret,
                catastrophic, field_mae, preview_mae, survival_accuracy)

    post_count = int(np.asarray(data["PH"], dtype=np.int64).sum())
    print(
        f"===== P38 train: {len(train_index)} train / {val_count} val / "
        f"{post_count} post-kill =====", flush=True)
    started = time.time()
    network.train()
    best_state = None
    best_key = (-1.0, -float("inf"))
    best_epoch = 0
    if source_model is not None:
        baseline = validation_metrics()
        baseline_mse = baseline[0]
        baseline_pre, baseline_post = baseline[2], baseline[3]
        baseline_pre_regret, baseline_post_regret = baseline[4], baseline[5]
        baseline_catastrophic = baseline[6]
        baseline_regret = 0.5 * (
            baseline_pre_regret + baseline_post_regret) \
            if baseline_post > 0.0 else baseline_pre_regret
        best_key = (
            -baseline_regret, -baseline_catastrophic, -baseline_mse)
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in network.state_dict().items()
        }
        print(
            f"  epoch 00 baseline pre={baseline_pre:.1%} "
            f"post={baseline_post:.1%} regret={baseline_regret:.5f} "
            f"cat={baseline_catastrophic:.1%}", flush=True)
    for epoch in range(1, args.epochs + 1):
        order = generator.permutation(train_index)
        total, batches = 0.0, 0
        for begin in range(0, len(order), args.batch):
            index = torch.as_tensor(
                order[begin:begin + args.batch], device=device)
            loss = batch_loss(index)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        (mse, top1, pre, post, pre_regret, post_regret,
         catastrophic, field_mae, preview_mae,
         survival_accuracy) = validation_metrics()
        # Teacher score landscapes frequently contain near ties.  Exact argmax
        # agreement can call an equally valuable action "wrong", so checkpoint
        # selection minimizes phase-balanced teacher regret; top-1 is retained
        # as a diagnostic only.
        balanced = 0.5 * (pre + post) if post > 0.0 else pre
        balanced_regret = 0.5 * (pre_regret + post_regret) \
            if post > 0.0 else pre_regret
        key = (-balanced_regret, -catastrophic, -mse)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
            }
        print(
            f"  epoch {epoch:02d}/{args.epochs} "
            f"loss={total/max(batches,1):.5f} val_mse={mse:.5f} "
            f"top1={top1:.1%} pre={pre:.1%} post={post:.1%} "
            f"regret={pre_regret:.4f}/{post_regret:.4f} "
            f"cat={catastrophic:.1%} "
            f"field_mae={field_mae:.3f} preview_mae={preview_mae:.3f} "
            f"survival={survival_accuracy:.1%} "
            f"elapsed={time.time()-started:.1f}s", flush=True)

    if best_state is None:
        raise RuntimeError("P38 training produced no checkpoint")
    network.load_state_dict(best_state)
    print(
        f"  selected epoch {best_epoch} with balanced regret "
        f"{-best_key[0]:.5f}", flush=True)

    payload = {
        "state_dict": network.state_dict(),
        "version": "p38_fast_privileged_distill",
        "vector_dim": FAST_VECTOR_DIM,
        "map_channels": FAST_MAP_CHANNELS,
        "samples": n,
        "post_kill_samples": post_count,
        "selected_epoch": best_epoch,
        "settings": settings,
        "iteration": iteration,
    }
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    root, extension = os.path.splitext(args.model)
    iteration_path = f"{root}_iter{iteration:02d}{extension}"
    torch.save(payload, iteration_path)
    torch.save(payload, args.model)
    print(f"  saved {iteration_path} and {args.model}", flush=True)


class KillFieldFastPolicy:
    """P38 deployment policy: one network, no field or sandbox search."""

    name = "P38 快速特权蒸馏网络"

    def __init__(self, model_path=DEFAULT_MODEL, cap_frames=500, **_ignored):
        import torch
        from training.tt_gym_env import TankTroubleGym

        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if int(payload.get("vector_dim", 0)) != FAST_VECTOR_DIM:
            raise ValueError("P38 model vector dimension mismatch")
        self.torch = torch
        self.network = build_fast_network()
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.encoder = TankTroubleGym(
            seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
        self.cap_frames = int(cap_frames)
        self.reset()
        with torch.inference_mode():
            self.network(
                torch.zeros(1, FAST_VECTOR_DIM),
                torch.zeros(1, FAST_MAP_CHANNELS, MAP_H, MAP_W))

    def reset(self):
        self.game = None
        self.round_number = None
        self.ledger = None
        self.frontier = None
        self.decision_state = None
        self.last_frame = None
        self.last_action = (1, 1, 0)
        self.last_observation = None
        self.last_planned = False
        self.decision_seconds = []

    def _start_round(self, game):
        self.game = game
        self.round_number = game.round_number
        self.ledger = DistillLedger(game, self.cap_frames)
        self.frontier = FrontierState(game, dense=True)
        self.decision_state = RuntimeDecisionState(game)
        self.last_frame = game.frame

    def _observation(self, game):
        """观测构造点。子类可覆写以追加特权输入（如 P40 的击杀场）。"""
        return (
            fast_vector_observation(
                self.encoder, game, self.ledger,
                self.frontier, self.decision_state),
            fast_spatial_observation(game, self.frontier),
        )

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
        self.last_observation = None
        if not game.tanks[0].alive:
            return {}
        if (game is not self.game
                or game.round_number != self.round_number):
            self._start_round(game)
        elapsed = max(0, game.frame - self.last_frame)
        if elapsed:
            self.ledger.advance(game, elapsed)
            self.frontier.observe_position(game)
            self.decision_state.observe(game)
            self.last_frame = game.frame
        if game.frozen:
            self.last_action = (1, 1, 0)
            return self._action_dict(self.last_action)

        me, enemy = game.tanks
        exact_hit = (
            enemy.alive and me.trigger_released and game.weapon_ready(me)
            and LaikaAI(game, me).check_bullet_path(
                me.rotation).get("result") == "HIT")
        if exact_hit:
            action = (1, 1, 1)
            self.decision_state.record(game, action, planned=True)
            self.last_action = action
            self.last_planned = True
            return self._action_dict(action)

        max_commit = (POST_KILL_DECIDE_EVERY - 1
                      if not enemy.alive else None)
        if max_commit is not None:
            self.decision_state.commit_remaining = min(
                self.decision_state.commit_remaining, max_commit)
        if (self.decision_state.commit_remaining > 0
                and not self.decision_state.action_no_effect
                and not me.hit_something):
            action = self.decision_state.committed_action
            if not enemy.alive:
                action = (action[0], action[1], 0)
            self.decision_state.record(game, action, planned=False)
            self.last_action = action
            return self._action_dict(action)

        started = time.perf_counter()
        vector, spatial = self._observation(game)
        with self.torch.inference_mode():
            scores = self.network(
                self.torch.as_tensor(vector).unsqueeze(0),
                self.torch.as_tensor(spatial).unsqueeze(0))[0][0]
            if not enemy.alive:
                scores = scores.clone()
                scores[1::2] = -float("inf")
            action = CANDIDATES[int(scores.argmax())]
        self.decision_seconds.append(time.perf_counter() - started)
        self.last_observation = (vector, spatial)
        self.last_action = action
        self.last_planned = True
        self.decision_state.record(game, action, planned=True)
        if not enemy.alive:
            self.decision_state.commit_remaining = min(
                self.decision_state.commit_remaining,
                POST_KILL_DECIDE_EVERY - 1)
        return self._action_dict(action)

    def telemetry(self):
        timings = np.asarray(self.decision_seconds, dtype=np.float64)
        return {
            "decisions": len(timings),
            "mean_decision_ms": 0.0 if not len(timings) else 1000 * timings.mean(),
            "p95_decision_ms": 0.0 if not len(timings) else 1000 * np.percentile(timings, 95),
            "field_builds": 0,
            "sandbox_previews": 0,
            "no_effect_frames": 0 if self.decision_state is None
            else self.decision_state.no_effect_frames,
        }


def settings_from_args(args):
    return {
        "rays": args.rays,
        "bounces": args.bounces,
        "flight_frames": args.flight_frames,
        "horizon": args.horizon,
        "hold": args.hold,
        "post_kill_horizon": args.post_kill_horizon,
        "cap_frames": args.cap_frames,
    }


def pipeline(args):
    settings = settings_from_args(args)
    collect_teacher(args, settings)
    train(args, settings, iteration=0)
    for iteration in range(1, args.dagger_rounds + 1):
        collect_dagger(args, settings, iteration, args.model)
        train(args, settings, iteration=iteration, source_model=args.model)
    print(
        f"===== P38 pipeline complete (no gameplay evaluation): "
        f"{args.model} =====", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pipeline", "collect", "train"])
    parser.add_argument("--teacher-rounds", type=int, default=96)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--rounds-per-dagger", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=38_500_000)
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--bounces", type=int, default=DEFAULT_BOUNCES)
    parser.add_argument("--flight-frames", type=int,
                        default=DEFAULT_FLIGHT_FRAMES)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--hold", type=int, default=8)
    parser.add_argument("--post-kill-horizon", type=int, default=75)
    parser.add_argument("--cap-frames", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--action-coef", type=float, default=0.05)
    parser.add_argument("--field-coef", type=float, default=0.15)
    parser.add_argument("--preview-coef", type=float, default=0.10)
    parser.add_argument("--survival-coef", type=float, default=0.10)
    parser.add_argument("--critical-coef", type=float, default=2.0,
                        help="extra weight for high-stakes movement landscapes")
    parser.add_argument("--action-temperature", type=float, default=0.08)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-model", default=None,
                        help="optional P38 checkpoint to continue from")
    parser.add_argument("--iteration", type=int, default=0,
                        help="checkpoint suffix used by the train command")
    args = parser.parse_args()
    settings = settings_from_args(args)
    if args.command == "pipeline":
        pipeline(args)
    elif args.command == "collect":
        collect_teacher(args, settings)
    else:
        train(
            args, settings, iteration=args.iteration,
            source_model=args.source_model)


if __name__ == "__main__":
    main()
