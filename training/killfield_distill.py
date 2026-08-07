"""Short P37 score-distillation pipeline and playback policy.

This follows the successful P21 pattern: collect the teacher's complete
18-action score landscape with paired rollout randomness, then regress scores
instead of only copying argmax labels.  Samples are taken only on macro-action
planning frames; the student uses the same small commitment wrapper at runtime.
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.killfield_student import (
    P37_OBS_DIM,
    KillFieldFeatureState,
    killfield_observation,
    load_expanded_warmstart,
)
from training.killfield_teacher import (
    COMMIT_MOVE_FRAMES,
    COMMIT_TURN_FRAMES,
    DEFAULT_BOUNCES,
    DEFAULT_FLIGHT_FRAMES,
    GOOD_FIRE_BONUS,
    KillFieldTeacher,
)
from training.mpc_agent import CANDIDATES
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.score_distill import build_net
from training.survival_distill_v2 import legacy_econ
from training.survival_expert_iter_530 import apply_action
from training.survival_frontier_rl import FrontierState


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "killfield_distill_short")
DEFAULT_MODEL = os.path.join(HERE, "models", "p37_killfield_student_short.pt")
DEFAULT_WARMSTART = os.path.join(
    HERE, "models", "p29c_feature_only_actor_iter02.pt")
SCORE_SCALE = 12_000.0
FORCED_FIRE_INDEX = CANDIDATES.index((1, 1, 1))


class DistillLedger:
    """Minimal observable ledger needed by the inherited P29 input block."""

    def __init__(self, game, cap):
        self.pool = 0.0
        self.frames = 0
        self.cap = int(cap)
        self.visited = {_cell(game): 0}

    def advance(self, game, frames=1):
        self.frames += int(frames)
        self.visited[_cell(game)] = self.frames


def _cell(game):
    tank = game.tanks[0]
    return int(tank.x // game.scale), int(tank.y // game.scale)


def _classify(game, hits, timed_out):
    if timed_out:
        return "timeout"
    alive = [tank.alive for tank in game.tanks]
    if alive == [True, False]:
        enemy_hits = [item for item in hits if item[1] == 1]
        attacker = enemy_hits[-1][0] if enemy_hits else None
        return "active_win" if attacker == 0 else "opponent_self_win"
    if alive == [False, True]:
        own_hits = [item for item in hits if item[1] == 0]
        attacker = own_hits[-1][0] if own_hits else None
        return "enemy_loss" if attacker == 1 else "self_loss"
    return "double"


OUTCOME_WEIGHT = {
    "active_win": 1.0,
    "opponent_self_win": 0.65,
    "double": 0.30,
    "enemy_loss": 0.40,
    "self_loss": 0.30,
    "timeout": 0.45,
}


def _forced_fire_scores(teacher, game):
    # Inspect the paired score landscape without consuming the teacher's next
    # rollout seed, then encode its explicit exact-hit override.
    rng_state = teacher.rng.getstate()
    scores = teacher.scores(game)
    teacher.rng.setstate(rng_state)
    scores = scores.copy()
    scores[FORCED_FIRE_INDEX] = max(
        float(scores.max()) + GOOD_FIRE_BONUS, GOOD_FIRE_BONUS)
    return scores


def _worker(job):
    (worker, rounds, seed0, data_dir, rays, bounces, flight_frames,
     horizon, hold, cap_frames, epsilon) = job
    import torch
    from tank_trouble_original.game import Game
    from training.tt_gym_env import TankTroubleGym

    torch.set_num_threads(1)
    rng = random.Random(seed0 ^ (worker * 104729 + 0x37D1))
    encoder = TankTroubleGym(
        seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
    observations, labels, actions, weights, kinds = [], [], [], [], []
    stats = {}

    for episode in range(rounds):
        seed = seed0 + episode
        game = Game(seed=seed, ai_enabled=True)
        teacher = KillFieldTeacher(
            seed=seed ^ 0x37D31517, ray_count=rays,
            max_bounces=bounces, max_flight_frames=flight_frames,
            horizon=horizon, hold=hold)
        ledger = DistillLedger(game, cap_frames)
        econ = dict(legacy_econ(), cap=cap_frames, start=0.0)
        frontier = FrontierState(game, dense=True)
        analyzer = OpportunityAnalyzer360(game)
        field_state = KillFieldFeatureState(
            ray_count=rays, max_bounces=bounces,
            max_flight_frames=flight_frames)
        episode_rows = []
        hits = []
        died = False

        for _ in range(cap_frames):
            action_dict = teacher.act(game)
            if teacher.last_decision_kind in ("plan", "forced_fire"):
                field_state.adopt_teacher(game, teacher)
                observation, _, _ = killfield_observation(
                    encoder, game, ledger, econ, frontier, analyzer,
                    field_state)
                scores = teacher.last_scores
                if scores is None:
                    scores = _forced_fire_scores(teacher, game)
                kind = int(teacher.last_decision_kind == "forced_fire")
                episode_rows.append((
                    observation,
                    np.clip(scores / SCORE_SCALE, -2.0, 2.0),
                    teacher.last_action_index,
                    kind,
                ))

            executed = teacher.last_action
            if (teacher.last_decision_kind == "plan"
                    and rng.random() < epsilon):
                executed = (rng.randrange(3), rng.randrange(3), 0)
                teacher.committed_action = executed
            apply_action(game, executed)
            events = game.step()
            for event in events:
                if event[0] == "hit":
                    hits.append((event[1], event[2]))
            ledger.advance(game)
            frontier.observe_position(game)
            if game.alive_count <= 1:
                died = True
                break

        if died:
            # Keep the original live-bullet double-death window, but collect no
            # terminal frames and perform no further expensive MPC calls.
            for _ in range(80):
                events = game.step()
                for event in events:
                    if event[0] == "hit":
                        hits.append((event[1], event[2]))
                if any(event[0] == "round_end" for event in events):
                    break
        outcome = _classify(game, hits, timed_out=not died)
        outcome_weight = OUTCOME_WEIGHT[outcome]
        for observation, score, action, kind in episode_rows:
            observations.append(observation)
            labels.append(score)
            actions.append(action)
            kinds.append(kind)
            weights.append(outcome_weight * (1.5 if kind else 1.0))
        stats[outcome] = stats.get(outcome, 0) + 1

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"shard_{worker:02d}.npz")
    np.savez_compressed(
        path,
        X=np.asarray(observations, dtype=np.float32),
        Y=np.asarray(labels, dtype=np.float32),
        A=np.asarray(actions, dtype=np.int64),
        W=np.asarray(weights, dtype=np.float32),
        K=np.asarray(kinds, dtype=np.uint8),
    )
    return path, stats, len(observations)


def collect(args):
    base, remainder = divmod(args.rounds, args.workers)
    jobs, offset = [], 0
    for worker in range(args.workers):
        count = base + int(worker < remainder)
        if count <= 0:
            continue
        jobs.append((
            worker, count, args.seed + offset, args.data_dir,
            args.rays, args.bounces, args.flight_frames,
            args.horizon, args.hold, args.cap_frames, args.epsilon))
        offset += count
    print(
        f"P37 collect: {args.rounds} rounds / {len(jobs)} workers / "
        f"{args.rays} rays / cap {args.cap_frames}", flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    paths, stats, samples = [], {}, 0
    for path, worker_stats, count in results:
        paths.append(path)
        samples += count
        for key, value in worker_stats.items():
            stats[key] = stats.get(key, 0) + value
    print(
        f"P37 collect complete: {samples} macro samples / "
        f"{time.time()-started:.1f}s / outcomes={stats}", flush=True)
    return paths


def load_shards(paths):
    arrays = {key: [] for key in ("X", "Y", "A", "W", "K")}
    for path in paths:
        data = np.load(path)
        if data["X"].shape[1] != P37_OBS_DIM:
            raise ValueError(f"bad P37 shard shape: {path} {data['X'].shape}")
        for key in arrays:
            arrays[key].append(data[key])
    return {key: np.concatenate(value) for key, value in arrays.items()}


def train_short(paths, args):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    data = load_shards(paths)
    device = torch.device(args.device)
    warm = load_expanded_warmstart(args.warmstart, device)
    network = build_net(P37_OBS_DIM).to(device)
    network.load_state_dict(warm.actor_state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    X = torch.as_tensor(data["X"], device=device)
    Y = torch.as_tensor(data["Y"], device=device)
    A = torch.as_tensor(data["A"], device=device)
    W = torch.as_tensor(data["W"], device=device)
    n = len(X)
    print(
        f"P37 train: {n} samples / {args.epochs} epochs / "
        f"warmstart={os.path.basename(args.warmstart)}", flush=True)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        total = 0.0
        batches = 0
        for begin in range(0, n, args.batch):
            index = order[begin:begin + args.batch]
            prediction = network(X[index])
            regression = functional.smooth_l1_loss(
                prediction, Y[index], reduction="none").mean(dim=1)
            classification = functional.cross_entropy(
                prediction / args.action_temperature,
                A[index], reduction="none")
            sample_weight = W[index]
            loss = (sample_weight * (
                regression + args.action_coef * classification)).sum() \
                / sample_weight.sum().clamp_min(1e-6)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(
            f"  epoch {epoch:02d}/{args.epochs} loss={total/max(batches,1):.5f} "
            f"elapsed={time.time()-started:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    torch.save({
        "state_dict": network.state_dict(),
        "in_dim": P37_OBS_DIM,
        "version": "p37_killfield_score_distill_short",
        "teacher": {
            "rays": args.rays,
            "bounces": args.bounces,
            "flight_frames": args.flight_frames,
            "horizon": args.horizon,
            "hold": args.hold,
        },
        "samples": n,
        "epochs": args.epochs,
    }, args.model)
    print(f"P37 model saved: {args.model}", flush=True)


class KillFieldStudentPolicy:
    """P37 distilled score network with the teacher's macro executor."""

    name = "P37 击杀场蒸馏学生"

    def __init__(self, model_path=DEFAULT_MODEL, ray_count=512,
                 max_bounces=DEFAULT_BOUNCES,
                 max_flight_frames=DEFAULT_FLIGHT_FRAMES,
                 cap_frames=750):
        import torch
        from training.tt_gym_env import TankTroubleGym

        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if int(payload.get("in_dim", 0)) != P37_OBS_DIM:
            raise ValueError(f"P37 student requires {P37_OBS_DIM} inputs")
        self.torch = torch
        self.network = build_net(P37_OBS_DIM)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.encoder = TankTroubleGym(
            seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
        self.ray_count = int(ray_count)
        self.max_bounces = int(max_bounces)
        self.max_flight_frames = int(max_flight_frames)
        self.cap_frames = int(cap_frames)
        self.reset()

    def reset(self):
        self.game = None
        self.round_number = None
        self.ledger = None
        self.frontier = None
        self.analyzer = None
        self.field_state = None
        self.field = None
        self.last_frame = None
        self.commit_remaining = 0
        self.committed_action = (1, 1, 0)

    def _start_round(self, game):
        self.game = game
        self.round_number = game.round_number
        self.ledger = DistillLedger(game, self.cap_frames)
        self.econ = dict(
            legacy_econ(), cap=self.cap_frames, start=0.0)
        self.frontier = FrontierState(game, dense=True)
        self.analyzer = OpportunityAnalyzer360(game)
        self.field_state = KillFieldFeatureState(
            self.ray_count, self.max_bounces, self.max_flight_frames)
        self.last_frame = game.frame
        self.commit_remaining = 0

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

    def _observe(self, game):
        elapsed = max(0, game.frame - self.last_frame)
        if elapsed:
            self.ledger.advance(game, elapsed)
            self.frontier.observe_position(game)
            self.field_state.advance(game, elapsed)
            self.last_frame = game.frame
        observation, metrics, fire_facts = killfield_observation(
            self.encoder, game, self.ledger, self.econ,
            self.frontier, self.analyzer, self.field_state)
        self.field = self.field_state.field
        return observation, metrics, fire_facts

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if not game.tanks[1].alive:
            action = getattr(self, "committed_action", (1, 1, 0))
            return self._action_dict((action[0], action[1], 0))
        if (game is not self.game
                or game.round_number != self.round_number):
            self._start_round(game)
        observation, _, fire_facts = self._observe(game)

        # This is the teacher's exact physical HIT fact, not a learned or
        # hand-tuned firing threshold.
        if fire_facts[3] > 0.5:
            self.commit_remaining = 0
            return self._action_dict((1, 1, 1))
        if self.commit_remaining > 0 and not game.tanks[0].hit_something:
            self.commit_remaining -= 1
            return self._action_dict(self.committed_action)

        with self.torch.no_grad():
            scores = self.network(
                self.torch.as_tensor(observation).unsqueeze(0))[0]
        action = CANDIDATES[int(scores.argmax())]
        if action[2] == 0:
            self.committed_action = action
            self.commit_remaining = (
                COMMIT_MOVE_FRAMES if action[0] != 1
                else COMMIT_TURN_FRAMES if action[1] != 1 else 0)
        return self._action_dict(action)

    def telemetry(self):
        state = self.field_state
        mean_build = 0.0 if state is None else (
            state.field_build_seconds / max(state.field_builds, 1))
        return {
            "field_builds": 0 if state is None else state.field_builds,
            "mean_field_build_seconds": mean_build,
            "cached_target_cells": 0 if state is None else len(state.field_cache),
            "hunt_chain": 0 if state is None else state.chain.count,
            "hunt_chain_timer": 0 if state is None else state.chain.timer,
            "hunt_chain_total": 0.0,
            "last_chain_gain": 0.0 if state is None else state.last_chain_gain,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["short-train"])
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=37_800_000)
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--bounces", type=int, default=2)
    parser.add_argument("--flight-frames", type=int, default=75)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--hold", type=int, default=8)
    parser.add_argument("--cap-frames", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--action-coef", type=float, default=0.04)
    parser.add_argument("--action-temperature", type=float, default=0.08)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmstart", default=DEFAULT_WARMSTART)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    paths = collect(args)
    train_short(paths, args)


if __name__ == "__main__":
    main()
