"""Compare 408/410/530-dimensional observations on survival-teacher labels."""

import argparse
import copy
import glob
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.score_distill import FULL_OBS_DIM, build_net, full_obs
from training.survival_distill_v2 import (
    BEST_NET,
    LEDGER_DIM,
    SurvivalTwoHeadPolicy,
    _teacher_labels,
    bind_env,
    ledger_features,
    legacy_econ,
)


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "survival_distillability_data")
RESULT_PATH = os.path.join(DATA_DIR, "results.json")
CONTEXT_NAMES = ("actual", "low_pool", "sparse_cover", "dense_cover")


def _apply(game, action):
    tank = game.tanks[0]
    tank.forward = bool(action.get("forward", False))
    tank.backup = bool(action.get("backup", False))
    tank.turn_left = bool(action.get("turn_left", False))
    tank.turn_right = bool(action.get("turn_right", False))
    tank.fire = bool(action.get("fire", False))


def _contexts(game, ledger):
    current = (int(game.tanks[0].x // game.scale),
               int(game.tanks[0].y // game.scale))
    actual = copy.copy(ledger)
    low_pool = copy.copy(ledger)
    low_pool.pool = 20.0 if abs(ledger.pool - 20.0) > 5.0 else 150.0
    sparse = copy.copy(ledger)
    sparse.visited = {current: ledger.frames}
    dense = copy.copy(ledger)
    dense.visited = {
        (cell["x"], cell["y"]): ledger.frames for cell in game.reachable
    }
    return actual, low_pool, sparse, dense


def _collect_worker(job):
    worker, target_states, seed0, repeats, sample_every, model_path = job
    import torch
    from tank_trouble_original.game import Game
    from training.survival_mode import Ledger
    from training.tt_gym_env import TankTroubleGym

    torch.set_num_threads(1)
    econ = legacy_econ()
    policy = SurvivalTwoHeadPolicy(model_path)
    env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
    rng = random.Random(seed0 + worker * 104729)
    observations = []
    scores = []
    game_ids = []
    state_ids = []
    context_ids = []
    physical_states = 0
    episode = 0
    while physical_states < target_states:
        seed = seed0 + worker * 10000 + episode
        game = Game(seed=seed, ai_enabled=True, invincible={1})
        ledger = Ledger(game, econ)
        policy.reset()
        sampled_this_game = 0
        while True:
            if (ledger.frames % sample_every == 0
                    and sampled_this_game < 8
                    and physical_states < target_states):
                bind_env(env, game, ledger.frames)
                physical = full_obs(env)
                context_ledgers = _contexts(game, ledger)
                repeat_seeds = [rng.randrange(1 << 30)
                                for _ in range(repeats)]
                for context_id, context_ledger in enumerate(context_ledgers):
                    observation = np.concatenate([
                        physical, ledger_features(context_ledger, econ)
                    ])
                    repeated_scores = []
                    for rollout_seed in repeat_seeds:
                        teacher_scores, _, _ = _teacher_labels(
                            game, context_ledger, rollout_seed, econ)
                        repeated_scores.append(teacher_scores)
                    observations.append(observation)
                    scores.append(repeated_scores)
                    game_ids.append(seed)
                    state_ids.append(worker * 1_000_000 + physical_states)
                    context_ids.append(context_id)
                physical_states += 1
                sampled_this_game += 1

            action = policy.act_ctx(game, ledger)
            _apply(game, action)
            events = game.step()
            end = ledger.on_frame(game, events)
            if end != "alive":
                break
        episode += 1

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"shard_{worker}.npz")
    np.savez_compressed(
        path,
        X=np.asarray(observations, dtype=np.float32),
        Y=np.asarray(scores, dtype=np.float32),
        game_id=np.asarray(game_ids, dtype=np.int64),
        state_id=np.asarray(state_ids, dtype=np.int64),
        context_id=np.asarray(context_ids, dtype=np.uint8),
    )
    return path, physical_states, episode


def collect(total_states, workers, repeats, seed, sample_every, model_path,
            fresh=False):
    if fresh and os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    base, remainder = divmod(total_states, workers)
    jobs = [
        (worker, base + (worker < remainder), seed, repeats,
         sample_every, model_path)
        for worker in range(workers) if base + (worker < remainder)
    ]
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        outputs = pool.map(_collect_worker, jobs)
    print(f"采集完成: {sum(item[1] for item in outputs)}物理状态 / "
          f"{sum(item[2] for item in outputs)}局 / "
          f"{time.time() - started:.0f}s", flush=True)


def load_data():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "shard_*.npz")))
    if not paths:
        raise RuntimeError(f"没有探针数据: {DATA_DIR}")
    arrays = {key: [] for key in ("X", "Y", "game_id", "state_id",
                                   "context_id")}
    for path in paths:
        data = np.load(path)
        for key in arrays:
            arrays[key].append(data[key])
    return {key: np.concatenate(value) for key, value in arrays.items()}


def label_metrics(data):
    repeated = data["Y"]
    mean_scores = repeated.mean(axis=1)
    mean_best = mean_scores.argmax(axis=1)
    sample_best = repeated.argmax(axis=2)
    agreement = (sample_best == mean_best[:, None]).mean(axis=1)
    sorted_scores = np.sort(mean_scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    results = {
        "repeat_to_mean_agreement": float(agreement.mean()),
        "unanimous_rate": float((agreement == 1.0).mean()),
        "mean_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
    }
    by_context = {}
    for context_id, name in enumerate(CONTEXT_NAMES):
        mask = data["context_id"] == context_id
        by_context[name] = {
            "n": int(mask.sum()),
            "agreement": float(agreement[mask].mean()),
            "median_margin": float(np.median(margins[mask])),
        }
    results["by_context"] = by_context

    state_ids = np.unique(data["state_id"])
    changed_low = changed_sparse = changed_dense = 0
    valid = 0
    for state_id in state_ids:
        indices = np.flatnonzero(data["state_id"] == state_id)
        if len(indices) != len(CONTEXT_NAMES):
            continue
        order = indices[np.argsort(data["context_id"][indices])]
        best = mean_best[order]
        changed_low += int(best[1] != best[0])
        changed_sparse += int(best[2] != best[0])
        changed_dense += int(best[3] != best[0])
        valid += 1
    results["counterfactual_action_change"] = {
        "low_pool": changed_low / max(valid, 1),
        "sparse_cover": changed_sparse / max(valid, 1),
        "dense_cover": changed_dense / max(valid, 1),
    }
    return results


def _prediction_metrics(prediction, truth):
    predicted_action = prediction.argmax(axis=1)
    teacher_action = truth.argmax(axis=1)
    rows = np.arange(len(truth))
    regret = truth.max(axis=1) - truth[rows, predicted_action]
    top3 = np.argpartition(prediction, -3, axis=1)[:, -3:]
    return {
        "n": int(len(truth)),
        "mse": float(np.mean((prediction - truth) ** 2)),
        "top1": float(np.mean(predicted_action == teacher_action)),
        "top3": float(np.mean((top3 == teacher_action[:, None]).any(axis=1))),
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "p90_regret": float(np.percentile(regret, 90)),
        "within_5": float(np.mean(regret <= 5.0)),
        "within_15": float(np.mean(regret <= 15.0)),
    }


def train_variant(X, mean_scores, game_ids, context_ids, input_dim, epochs,
                  width, seed):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    unique_games = np.unique(game_ids)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_games)
    validation_games = set(unique_games[:max(1, len(unique_games) // 5)])
    validation = np.asarray([game in validation_games for game in game_ids])
    training = ~validation
    X_tensor = torch.as_tensor(X[:, :input_dim])
    Y_tensor = torch.as_tensor(mean_scores / 300.0)
    network = build_net(input_dim, width=width)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    batch_size = 256
    train_indices = np.flatnonzero(training)
    for _ in range(epochs):
        shuffled = train_indices[torch.randperm(len(train_indices)).numpy()]
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start:start + batch_size]
            loss = functional.mse_loss(
                network(X_tensor[indices]), Y_tensor[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    network.eval()
    val_indices = np.flatnonzero(validation)
    with torch.no_grad():
        prediction = network(X_tensor[val_indices]).numpy() * 300.0
    truth = mean_scores[val_indices]
    metrics = {
        "input_dim": input_dim,
        "validation_states": int(len(val_indices)),
    }
    metrics.update(_prediction_metrics(prediction, truth))
    metrics["by_context"] = {}
    validation_contexts = context_ids[val_indices]
    for context_id, name in enumerate(CONTEXT_NAMES):
        mask = validation_contexts == context_id
        metrics["by_context"][name] = _prediction_metrics(
            prediction[mask], truth[mask])
    return metrics


def _summarize_runs(runs):
    keys = ("top1", "top3", "mean_regret", "p90_regret", "within_5",
            "within_15")
    summary = {}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    actual_runs = [run["by_context"]["actual"] for run in runs]
    summary["actual"] = {}
    for key in keys:
        values = np.asarray([run[key] for run in actual_runs],
                            dtype=np.float64)
        summary["actual"][key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


def train_all(epochs, width, train_seeds):
    data = load_data()
    mean_scores = data["Y"].mean(axis=1)
    results = {"labels": label_metrics(data), "models": {}}
    for name, input_dim in (("physical_408", FULL_OBS_DIM),
                            ("pool_time_410", FULL_OBS_DIM + 2),
                            ("full_ledger_530", FULL_OBS_DIM + LEDGER_DIM)):
        started = time.time()
        runs = [
            train_variant(
                data["X"], mean_scores, data["game_id"],
                data["context_id"], input_dim, epochs, width, seed)
            for seed in range(train_seeds)
        ]
        summary = _summarize_runs(runs)
        results["models"][name] = {"summary": summary, "runs": runs}
        print(f"{name}: top1 {summary['top1']['mean']:.1%}  "
              f"top3 {summary['top3']['mean']:.1%}  "
              f"regret {summary['mean_regret']['mean']:.1f}  "
              f"<=5 {summary['within_5']['mean']:.1%}  "
              f"({time.time() - started:.0f}s)", flush=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "train", "all"))
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12_400_000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--model", default=BEST_NET)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.mode in ("collect", "all"):
        collect(args.states, args.workers, args.repeats, args.seed,
                args.sample_every, args.model, args.fresh)
    if args.mode in ("train", "all"):
        train_all(args.epochs, args.width, args.train_seeds)


if __name__ == "__main__":
    main()
