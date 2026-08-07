"""P24-P22 Replica-530: survival teacher with P22 joint-score DAgger."""

import argparse
import csv
import glob
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.score_distill import build_net, full_obs, train
from training.survival_distill_v2 import (
    LEDGER_DIM,
    OBS_DIM,
    bind_env,
    ledger_features,
    legacy_econ,
)


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "survival_replica_530_data")
MODELS_DIR = os.path.join(HERE, "models")
SEED_NET = os.path.join(MODELS_DIR, "p24r530_seed.pt")
BEST_NET = os.path.join(MODELS_DIR, "p24r530_best.pt")
HISTORY = os.path.join(DATA_DIR, "history.csv")
SCALE = 300.0
DECIDE_EVERY = 2
TEACHER_SEED_BASE = 13_000_000
DAGGER_SEED_BASE = 14_000_000
EVAL_SEED_BASE = 15_000_000
SURVIVAL_EVAL_SEED_BASE = 16_000_000
TEACHER_HIT_INTERVAL = 2.4
TEACHER_STUCK_PCT = 4.7
TEACHER_STYLE_RATE = 1.08


def build_observation(env, game, ledger, econ):
    bind_env(env, game, ledger.frames)
    return np.concatenate([full_obs(env), ledger_features(ledger, econ)])


def teacher_scores(game, ledger, rng_seed, econ):
    from training.mpc_agent import CANDIDATES, make_sandbox
    from training.survival_mode import survival_rollout

    scores = np.empty(len(CANDIDATES), dtype=np.float32)
    for index, action in enumerate(CANDIDATES):
        sandbox = make_sandbox(game, "L2", rng_seed=rng_seed)
        scores[index] = survival_rollout(
            sandbox, action, ledger.pool, ledger.visited, ledger.frames,
            econ=econ, style=True)
    return scores


def apply_action(game, action):
    throttle, turn, fire = action
    tank = game.tanks[0]
    tank.forward, tank.backup = throttle == 2, throttle == 0
    tank.turn_left, tank.turn_right = turn == 0, turn == 2
    tank.fire = fire == 1


def load_network(path):
    import torch

    payload = torch.load(path, weights_only=True)
    network = build_net(payload.get("in_dim", OBS_DIM))
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return network


class SurvivalReplica530Policy:
    name = "p24r530_joint_scorenet"

    def __init__(self, model_path):
        import torch
        from training.mpc_agent import CANDIDATES
        from training.tt_gym_env import TankTroubleGym

        self.torch = torch
        self.candidates = CANDIDATES
        self.network = load_network(model_path)
        self.env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
        self.econ = legacy_econ()
        self.game = None
        self.round_number = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.last_action = (1, 1, 0)

    def reset(self):
        self.game = None
        self.round_number = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.last_action = (1, 1, 0)

    def _predict(self, game, ledger):
        observation = build_observation(
            self.env, game, ledger, self.econ)
        with self.torch.no_grad():
            scores = self.network(
                self.torch.as_tensor(observation).unsqueeze(0))[0]
        self.last_action = self.candidates[int(scores.argmax())]

    def _action_dict(self):
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
        if (game is not self.context_game
                or game.round_number != self.context_round):
            self.context_game = game
            self.context_round = game.round_number
            self.context_step = 0
            self.last_action = (1, 1, 0)
        if self.context_step % DECIDE_EVERY == 0:
            self._predict(game, ledger)
        self.context_step += 1
        return self._action_dict()

    def act(self, game):
        from training.survival_mode import Ledger

        if not game.tanks[0].alive:
            return {}
        if game is not self.game or game.round_number != self.round_number:
            self.game = game
            self.round_number = game.round_number
            self.ledger = Ledger(game, self.econ)
        else:
            end = self.ledger.on_frame(game, game.events)
            if end in ("drain", "cap"):
                self.ledger = Ledger(game, self.econ)
        return self.act_ctx(game, self.ledger)


def collect_worker(job):
    (phase, worker, rounds, seed0, epsilon, actor_kind, actor_path,
     output_path) = job
    import torch
    from tank_trouble_original.game import Game
    from training.mpc_agent import CANDIDATES
    from training.survival_mode import Ledger
    from training.tt_gym_env import TankTroubleGym

    torch.set_num_threads(1)
    econ = legacy_econ()
    actor = SurvivalReplica530Policy(actor_path) \
        if actor_kind == "student" else None
    env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
    rng = random.Random(seed0 + worker * 104729 + 47)
    observations = []
    labels = []
    totals = {
        "games": 0,
        "death": 0,
        "drain": 0,
        "cap": 0,
        "hits": 0,
        "frames": 0,
        "stuck": 0,
        "style": 0.0,
        "regret": 0.0,
        "lethal": 0,
        "decisions": 0,
    }
    for episode in range(rounds):
        game = Game(seed=seed0 + episode, ai_enabled=True, invincible={1})
        ledger = Ledger(game, econ)
        if actor is not None:
            actor.reset()
        action = (1, 1, 0)
        while True:
            if ledger.frames % DECIDE_EVERY == 0:
                observation = build_observation(env, game, ledger, econ)
                scores = teacher_scores(
                    game, ledger, rng.randrange(1 << 30), econ)
                observations.append(observation)
                labels.append(scores / SCALE)
                if actor is None:
                    pick = int(scores.argmax())
                else:
                    with torch.no_grad():
                        prediction = actor.network(
                            torch.as_tensor(observation).unsqueeze(0))[0]
                    pick = int(prediction.argmax())
                    totals["regret"] += float(scores.max() - scores[pick])
                    totals["lethal"] += int(
                        scores[pick] < -150 and scores.max() > -150)
                totals["decisions"] += 1
                if rng.random() < epsilon:
                    action = (rng.randrange(3), rng.randrange(3), 0)
                else:
                    action = CANDIDATES[pick]
            apply_action(game, action)
            events = game.step()
            end = ledger.on_frame(game, events)
            if end != "alive":
                break
        totals["games"] += 1
        totals[end] += 1
        totals["hits"] += ledger.hits
        totals["frames"] += ledger.frames
        totals["stuck"] += ledger.stuck_frames
        totals["style"] += ledger.style

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        X=np.asarray(observations, dtype=np.float32),
        Y=np.asarray(labels, dtype=np.float32),
        stat_keys=np.asarray(list(totals.keys())),
        stat_values=np.asarray(list(totals.values()), dtype=np.float64),
    )
    return output_path, totals


def collect_phase(phase, rounds, workers, seed, actor_kind, actor_path,
                  epsilon, chunk_size):
    output_dir = os.path.join(DATA_DIR, phase)
    os.makedirs(output_dir, exist_ok=True)
    jobs = []
    completed = []
    for job_id, start in enumerate(range(0, rounds, chunk_size)):
        count = min(chunk_size, rounds - start)
        path = os.path.join(
            output_dir, f"shard_{start:06d}_{count:04d}.npz")
        if os.path.exists(path):
            data = np.load(path)
            if "stat_keys" in data and "stat_values" in data:
                stats = dict(zip(data["stat_keys"].tolist(),
                                 data["stat_values"].tolist()))
                if int(stats.get("games", 0)) == count:
                    completed.append((path, stats))
                    continue
        jobs.append((phase, job_id, count, seed + start, epsilon,
                     actor_kind, actor_path, path))
    started = time.time()
    total_chunks = len(completed) + len(jobs)
    print(f"===== {phase}: {rounds}局 actor={actor_kind} "
          f"chunk={chunk_size} 已完成{len(completed)}/{total_chunks} =====",
          flush=True)
    totals = {}
    samples = 0

    def add_output(path, stats):
        nonlocal samples
        samples += len(np.load(path)["X"])
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
    for path, stats in completed:
        add_output(path, stats)

    def print_progress(done):
        games = max(totals.get("games", 0), 1)
        frames = max(totals.get("frames", 0), 1)
        alive_seconds = totals.get("frames", 0) / 25.0
        decisions = max(totals.get("decisions", 0), 1)
        message = (
            f"  进度 {done}/{total_chunks}块 {int(totals.get('games', 0))}/"
            f"{rounds}局  {samples}样本  "
            f"{alive_seconds / max(totals.get('hits', 0), 1):.1f}s/中  "
            f"卡墙 {totals.get('stuck', 0) / frames:.1%}  "
            f"风格 {totals.get('style', 0) / max(alive_seconds, 1):+.2f}/s  "
            f"死{totals.get('death', 0)/games:.1%} "
            f"干{totals.get('drain', 0)/games:.1%} "
            f"满{totals.get('cap', 0)/games:.1%}")
        if actor_kind == "student":
            message += (
                f"  后悔 {totals.get('regret', 0)/decisions:.1f} "
                f"致命 {totals.get('lethal', 0)/decisions:.2%}")
        print(message, flush=True)
        if actor_kind == "teacher" and games >= 128:
            hit_interval = alive_seconds / max(totals.get("hits", 0), 1)
            stuck_rate = totals.get("stuck", 0) / frames
            style_rate = totals.get("style", 0) / max(alive_seconds, 1)
            if hit_interval > 4.0 or stuck_rate > 0.12 or style_rate < 0.30:
                print("  !! 老师遥测偏离基线，请暂停检查：期望约 "
                      "2.3s/中、4.9%卡墙、+1.04/s风格", flush=True)

    done = len(completed)
    if completed:
        print_progress(done)
    if jobs:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            for path, stats in pool.imap_unordered(collect_worker, jobs):
                add_output(path, stats)
                done += 1
                if done % max(1, workers) == 0 or done == total_chunks:
                    print_progress(done)
    games = max(totals["games"], 1)
    alive_seconds = totals["frames"] / 25.0
    decisions = max(totals["decisions"], 1)
    print(f"  {samples}样本 / {totals['games']}局 / "
          f"{time.time() - started:.0f}s", flush=True)
    print(f"  命中间隔 {alive_seconds / max(totals['hits'], 1):.1f}s  "
          f"卡墙 {totals['stuck'] / max(totals['frames'], 1):.1%}  "
          f"风格 {totals['style'] / max(alive_seconds, 1):+.2f}/s  "
          f"终局 死{totals['death']/games:.1%} "
          f"干{totals['drain']/games:.1%} 满{totals['cap']/games:.1%}",
          flush=True)
    if actor_kind == "student":
        print(f"  平均老师后悔 {totals['regret']/decisions:.1f}  "
              f"致命误判 {totals['lethal']/decisions:.2%}", flush=True)
    return samples, totals


def load_all_data():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*", "shard_*.npz")))
    observations = []
    labels = []
    for path in paths:
        data = np.load(path)
        if data["X"].shape[1] != OBS_DIM:
            print(f"  跳过维度不符: {path} {data['X'].shape}", flush=True)
            continue
        observations.append(data["X"])
        labels.append(data["Y"])
    if not observations:
        raise RuntimeError(f"没有 {OBS_DIM} 维训练数据: {DATA_DIR}")
    X = np.concatenate(observations)
    Y = np.concatenate(labels)
    print(f"  聚合 {len(paths)} shard -> {len(X)}样本", flush=True)
    return X, Y


def train_candidate(name, epochs):
    import torch

    X, Y = load_all_data()
    torch.manual_seed(0)
    network, metrics = train(X, Y, epochs=epochs)
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"p24r530_{name}.pt")
    torch.save({
        "state_dict": network.state_dict(),
        "in_dim": X.shape[1],
        "ledger_dim": LEDGER_DIM,
        "version": "p24_p22_replica_530",
    }, path)
    print(f"  保存 {path}", flush=True)
    return path, len(X), metrics


def survival_eval_worker(job):
    model_path, seed, count = job
    from training.survival_mode import run_survival

    policy = SurvivalReplica530Policy(model_path)
    econ = legacy_econ()
    return [run_survival(policy, seed + index, econ=econ)
            for index in range(count)]


def evaluate_survival(model_path, rounds, seed, workers):
    from training.survival_mode import _agg

    jobs = split_jobs(model_path, rounds, seed, workers)
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(survival_eval_worker, jobs)
    aggregate = _agg([item for part in parts for item in part])
    print(f"  生存 {aggregate['n']}局: {aggregate['hit_iv']:.1f}s/中  "
          f"卡墙 {aggregate['stuck_pct']:.1f}%  "
          f"风格 {aggregate['style_rate']:+.2f}/s  "
          f"存活 {aggregate['alive_s']:.1f}s", flush=True)
    return aggregate


def original_eval_worker(job):
    model_path, seed, count = job
    from training.evaluate import play_round_dual_engine

    policy = SurvivalReplica530Policy(model_path)
    return [play_round_dual_engine(policy, seed + index)
            for index in range(count)]


def split_jobs(model_path, rounds, seed, workers):
    base, remainder = divmod(rounds, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        count = base + (worker < remainder)
        if count:
            jobs.append((model_path, seed + offset, count))
            offset += count
    return jobs


def evaluate_original(model_path, rounds, seed, workers):
    jobs = split_jobs(model_path, rounds, seed, workers)
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = [item for part in pool.map(original_eval_worker, jobs)
                   for item in part]
    total = len(results)
    count = lambda result: sum(
        item["true_result"] == result for item in results)
    shots = sum(item["shots"] for item in results)
    metrics = {
        "n": total,
        "win_count": count("win"),
        "win": count("win") / total,
        "loss": count("loss") / total,
        "double_death": count("double_death") / total,
        "draw": count("draw") / total,
        "shots_per_game": shots / total,
        "hit_rate": sum(item["kills"] for item in results) / max(shots, 1),
    }
    print(f"  原版 {total}局: 胜 {metrics['win']:.1%}  "
          f"负 {metrics['loss']:.1%}  双亡 {metrics['double_death']:.1%}  "
          f"场均开火 {metrics['shots_per_game']:.1f}", flush=True)
    return metrics


def evaluate_imitation(model_path, probe_dir):
    import torch

    paths = sorted(glob.glob(os.path.join(probe_dir, "shard_*.npz")))
    observations = []
    labels = []
    for path in paths:
        data = np.load(path)
        if not all(key in data for key in ("X", "Y", "context_id")):
            continue
        mask = data["context_id"] == 0
        observations.append(data["X"][mask])
        repeated = data["Y"][mask]
        labels.append(repeated.mean(axis=1) if repeated.ndim == 3
                      else repeated)
    if not observations:
        print("  固定行为探针不可用，动作后悔不参与本轮晋升", flush=True)
        return {"n": 0, "regret": float("inf"), "within_5": 0.0,
                "top1": 0.0, "top3": 0.0}
    X = np.concatenate(observations)
    Y = np.concatenate(labels)
    network = load_network(model_path)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), 4096):
            prediction = network(torch.as_tensor(X[start:start + 4096]))
            predictions.append(prediction.numpy() * SCALE)
    prediction = np.concatenate(predictions)
    action = prediction.argmax(axis=1)
    rows = np.arange(len(Y))
    regret = Y.max(axis=1) - Y[rows, action]
    teacher_action = Y.argmax(axis=1)
    top3 = np.argpartition(prediction, -3, axis=1)[:, -3:]
    metrics = {
        "n": len(Y),
        "regret": float(regret.mean()),
        "within_5": float((regret <= 5.0).mean()),
        "top1": float((action == teacher_action).mean()),
        "top3": float((top3 == teacher_action[:, None]).any(axis=1).mean()),
    }
    print(f"  固定行为探针 {metrics['n']}状态: 后悔 {metrics['regret']:.1f}  "
          f"5分内 {metrics['within_5']:.1%}  "
          f"top1 {metrics['top1']:.1%}  "
          f"top3 {metrics['top3']:.1%}", flush=True)
    return metrics


def evaluate_latency(model_path, probe_dir, samples=512):
    import torch

    observations = []
    for path in sorted(glob.glob(os.path.join(probe_dir, "shard_*.npz"))):
        data = np.load(path)
        if not all(key in data for key in ("X", "context_id")):
            continue
        observations.append(data["X"][data["context_id"] == 0])
    if not observations:
        print("  固定行为探针不可用，无法测量推理延迟", flush=True)
        return {"p50_ms": float("inf"), "p95_ms": float("inf")}
    rows = np.concatenate(observations)
    rows = rows[np.arange(samples) % len(rows)]
    network = load_network(model_path)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for row in rows[:32]:
                network(torch.as_tensor(row).unsqueeze(0))
            elapsed = []
            for row in rows:
                started = time.perf_counter_ns()
                network(torch.as_tensor(row).unsqueeze(0))
                elapsed.append((time.perf_counter_ns() - started) / 1e6)
    finally:
        torch.set_num_threads(old_threads)
    metrics = {
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
    }
    print(f"  单状态网络推理: p50 {metrics['p50_ms']:.3f}ms  "
          f"p95 {metrics['p95_ms']:.3f}ms", flush=True)
    return metrics


def behavior_score(survival, imitation):
    hit_interval = survival["hit_iv"]
    hit_closeness = (min(1.0, TEACHER_HIT_INTERVAL / hit_interval)
                     if np.isfinite(hit_interval) and hit_interval > 0
                     else 0.0)
    stuck_closeness = min(
        1.0, TEACHER_STUCK_PCT / max(survival["stuck_pct"], 1e-6))
    style_closeness = float(np.exp(
        -abs(survival["style_rate"] - TEACHER_STYLE_RATE) / 2.0))
    regret_closeness = (float(np.exp(-imitation["regret"] / 50.0))
                        if np.isfinite(imitation["regret"]) else 0.0)
    imitation_closeness = (0.55 * regret_closeness
                           + 0.30 * imitation["within_5"]
                           + 0.15 * imitation["top1"])
    return float(0.60 * imitation_closeness
                 + 0.20 * hit_closeness
                 + 0.10 * stuck_closeness
                 + 0.10 * style_closeness)


def wilson_lower_bound(wins, total, z=1.96):
    if total <= 0:
        return 0.0
    rate = wins / total
    denominator = 1.0 + z * z / total
    centre = rate + z * z / (2.0 * total)
    margin = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (centre - margin) / denominator


HISTORY_COLUMNS = [
    "iteration", "time", "new_samples", "total_samples", "val_mse",
    "val_top1", "gate_seed", "new_win", "old_win", "new_hit_interval",
    "old_hit_interval", "new_regret", "old_regret", "new_behavior",
    "old_behavior", "new_latency_p95_ms", "old_latency_p95_ms",
    "win_wilson_lower", "safety_floor", "promoted", "model",
]


def append_history(row):
    os.makedirs(DATA_DIR, exist_ok=True)
    new_file = not os.path.exists(HISTORY)
    if not new_file:
        with open(HISTORY, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            old_columns = reader.fieldnames or []
        if old_columns != HISTORY_COLUMNS:
            with open(HISTORY, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
                writer.writeheader()
                for old_row in rows:
                    writer.writerow({key: old_row.get(key, "")
                                     for key in HISTORY_COLUMNS})
    with open(HISTORY, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def print_history():
    if not os.path.exists(HISTORY):
        print("(没有训练记录)")
        return
    with open(HISTORY, encoding="utf-8") as handle:
        print(handle.read(), end="")


def run_pipeline(args):
    global DATA_DIR, HISTORY
    if args.fresh:
        if os.path.isdir(args.data_dir):
            shutil.rmtree(args.data_dir)
        for path in (args.seed_net, args.best_net):
            if os.path.exists(path):
                os.remove(path)
    DATA_DIR = args.data_dir
    HISTORY = os.path.join(DATA_DIR, "history.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    collect_phase(
        "teacher", args.teacher_rounds, args.workers,
        TEACHER_SEED_BASE, "teacher", None, args.epsilon,
        args.collect_chunk)
    if not os.path.exists(args.seed_net):
        candidate, total_samples, metrics = train_candidate(
            "seed", args.epochs)
        if os.path.abspath(candidate) != os.path.abspath(args.seed_net):
            shutil.copy2(candidate, args.seed_net)
    else:
        candidate = args.seed_net
        total_samples = sum(
            len(np.load(path)["X"])
            for path in glob.glob(os.path.join(DATA_DIR, "teacher",
                                               "shard_*.npz")))
        metrics = (float("nan"), float("nan"), float("nan"))
    if not os.path.exists(args.best_net):
        shutil.copy2(candidate, args.best_net)
        print("===== 初始学生验收 =====", flush=True)
        evaluate_survival(
            args.best_net, args.survival_eval_n,
            SURVIVAL_EVAL_SEED_BASE, args.workers)
        evaluate_original(
            args.best_net, args.eval_n, EVAL_SEED_BASE, args.workers)
        append_history({
            "iteration": 0,
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "new_samples": total_samples,
            "total_samples": total_samples,
            "val_mse": round(metrics[0], 6),
            "val_top1": round(metrics[1], 6),
            "gate_seed": EVAL_SEED_BASE,
            "new_win": "",
            "old_win": "",
            "new_hit_interval": "",
            "old_hit_interval": "",
            "new_regret": "",
            "old_regret": "",
            "new_behavior": "",
            "old_behavior": "",
            "new_latency_p95_ms": "",
            "old_latency_p95_ms": "",
            "win_wilson_lower": "",
            "safety_floor": "",
            "promoted": True,
            "model": os.path.basename(args.seed_net),
        })

    for iteration in range(1, args.dagger_rounds + 1):
        phase = f"iter{iteration:02d}"
        new_samples, _ = collect_phase(
            phase, args.rounds_per_dagger, args.workers,
            DAGGER_SEED_BASE + iteration * 1_000_000,
            "student", args.best_net, args.epsilon, args.collect_chunk)
        candidate, total_samples, metrics = train_candidate(
            f"iter{iteration:02d}", args.epochs)
        gate_seed = EVAL_SEED_BASE + iteration * 10_000
        survival_seed = SURVIVAL_EVAL_SEED_BASE + iteration * 10_000
        print(f"===== 第 {iteration} 轮配对门 =====", flush=True)
        new_survival = evaluate_survival(
            candidate, args.survival_eval_n, survival_seed, args.workers)
        old_survival = evaluate_survival(
            args.best_net, args.survival_eval_n, survival_seed, args.workers)
        new_imitation = evaluate_imitation(candidate, args.behavior_probe_dir)
        old_imitation = evaluate_imitation(
            args.best_net, args.behavior_probe_dir)
        new_latency = evaluate_latency(candidate, args.behavior_probe_dir)
        old_latency = evaluate_latency(args.best_net, args.behavior_probe_dir)
        new_behavior = behavior_score(new_survival, new_imitation)
        old_behavior = behavior_score(old_survival, old_imitation)
        print(f"  行为接近分: 新 {new_behavior:.3f} / 旧 {old_behavior:.3f}",
              flush=True)
        new_original = evaluate_original(
            candidate, args.eval_n, gate_seed, args.workers)
        old_original = evaluate_original(
            args.best_net, args.eval_n, gate_seed, args.workers)
        safety_floor = args.original_safety_floor
        win_wilson_lower = wilson_lower_bound(
            new_original["win_count"], new_original["n"])
        realtime = new_latency["p95_ms"] <= args.latency_p95_ms
        clearly_stronger = (
            new_original["win"] >= safety_floor
            and win_wilson_lower > args.laika_wilson_floor)
        promoted = realtime and new_behavior > old_behavior and clearly_stronger
        if promoted:
            shutil.copy2(candidate, args.best_net)
            print(f"  行为晋升: {new_behavior:.3f} > {old_behavior:.3f}  "
                  f"p95 {new_latency['p95_ms']:.3f}ms  "
                  f"胜率 {new_original['win']:.1%}, 95%下限 "
                  f"{win_wilson_lower:.1%}", flush=True)
        else:
            print(f"  不晋升: 行为 {new_behavior:.3f}/{old_behavior:.3f}  "
                  f"p95 {new_latency['p95_ms']:.3f}/{args.latency_p95_ms:.1f}ms  "
                  f"胜率 {new_original['win']:.1%}/安全线 "
                  f"{safety_floor:.1%}, 95%下限 {win_wilson_lower:.1%}",
                  flush=True)
        append_history({
            "iteration": iteration,
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "new_samples": new_samples,
            "total_samples": total_samples,
            "val_mse": round(metrics[0], 6),
            "val_top1": round(metrics[1], 6),
            "gate_seed": gate_seed,
            "new_win": round(new_original["win"], 6),
            "old_win": round(old_original["win"], 6),
            "new_hit_interval": round(new_survival["hit_iv"], 6),
            "old_hit_interval": round(old_survival["hit_iv"], 6),
            "new_regret": round(new_imitation["regret"], 6),
            "old_regret": round(old_imitation["regret"], 6),
            "new_behavior": round(new_behavior, 6),
            "old_behavior": round(old_behavior, 6),
            "new_latency_p95_ms": round(new_latency["p95_ms"], 6),
            "old_latency_p95_ms": round(old_latency["p95_ms"], 6),
            "win_wilson_lower": round(win_wilson_lower, 6),
            "safety_floor": round(safety_floor, 6),
            "promoted": promoted,
            "model": os.path.basename(candidate),
        })

    print("===== P24-P22 Replica-530 完成 =====", flush=True)
    print_history()


def main():
    global DATA_DIR, HISTORY
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pipeline", "eval", "eval-original",
                                         "history"))
    parser.add_argument("--teacher-rounds", type=int, default=1500)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--rounds-per-dagger", type=int, default=3000)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=400)
    parser.add_argument("--survival-eval-n", type=int, default=80)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--collect-chunk", type=int, default=16)
    parser.add_argument(
        "--behavior-probe-dir",
        default=os.path.join(HERE, "survival_distillability_data"))
    parser.add_argument("--latency-p95-ms", type=float, default=5.0)
    parser.add_argument("--original-safety-floor", type=float, default=0.55)
    parser.add_argument("--laika-wilson-floor", type=float, default=0.50)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--seed-net", default=SEED_NET)
    parser.add_argument("--best-net", default=BEST_NET)
    parser.add_argument("--net", default=BEST_NET)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17_000_000)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.mode == "pipeline":
        run_pipeline(args)
    elif args.mode == "eval":
        evaluate_survival(args.net, args.n, args.seed, args.workers)
    elif args.mode == "eval-original":
        evaluate_original(args.net, args.n, args.seed, args.workers)
    else:
        DATA_DIR = args.data_dir
        HISTORY = os.path.join(DATA_DIR, "history.csv")
        print_history()


if __name__ == "__main__":
    main()
