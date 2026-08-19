"""Collect and train clean temporal movement intent from the safe hybrid.

The temporal model never labels or owns firing. Collection keeps only rounds
that are true wins, and validation is split by round seed to prevent adjacent
frames from leaking across train and validation sets.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import Game
from training.evaluate import RoundTracker, _round_stats
from training.battle_supervision import diagnose_battle
from training.sparse_exact_safety_policy import SparseExactSafetyPolicy, parse_seeds
from training.temporal_intent_model import (
    HOLD_BINS,
    TEMPORAL_FEATURE_DIM,
    build_temporal_intent_net,
    movement_run_targets,
)
from training.tt_gym_env import TRUNCATE_FRAMES


DEFAULT_BASE_NET = "training/models/p26_amortized_mpc_iter05.pt"
DEFAULT_VALUE_NET = "training/models/p27b_risk_value_iter00.pt"


def group_split(round_seeds, validation_fraction=0.25, split_seed=2701):
    unique = np.unique(np.asarray(round_seeds, dtype=np.int64))
    if len(unique) < 2:
        raise RuntimeError("temporal training needs at least two round seeds")
    shuffled = np.random.default_rng(split_seed).permutation(unique)
    count = max(1, min(len(unique) - 1,
                       int(round(len(unique) * validation_fraction))))
    validation = set(int(value) for value in shuffled[:count])
    mask = np.asarray([int(seed) in validation for seed in round_seeds])
    return np.flatnonzero(~mask), np.flatnonzero(mask), sorted(validation)


def sequence_targets(movements):
    movements = np.asarray(movements, dtype=np.int64)
    remaining, hold = movement_run_targets(movements)
    interrupt = np.zeros(len(movements), dtype=np.float32)
    if len(movements) > 1:
        interrupt[:-1] = movements[:-1] != movements[1:]
    return remaining, hold, interrupt


def _make_teacher(base_net, value_net):
    return SparseExactSafetyPolicy(
        base_net=base_net,
        value_net=value_net,
        fire_margin=0.16,
        top_k=12,
        search_horizon=72,
        search_death_penalty=0.18,
        search_dd_penalty=0.45,
        search_kill_bonus=0.05,
        search_max_death=0.0,
        search_max_dd=0.0,
        successor_shield=True,
        successor_horizon=72,
        successor_shield_max_safe_roots=2,
        suppress_secured_fire=True,
        min_unsecured_fire_gain=2.0,
        audit_interval=6,
        proactive_interval=24,
        behavior_full_search=True,
        search_hold_frames=6,
        search_on_fire=True,
        risk_search_threshold=0.18,
        long_tail_fire_horizon=375,
        topology_assist=True,
        topology_intent_max_frames=75,
        topology_cooldown_frames=25,
        topology_pursuit_delay_frames=20,
        network_move_hold_frames=4,
        temporal_record_state=True,
        deterministic_search_seeds=True,
    )


def _collect_seed(job):
    seed, base_net, value_net, max_frames = job
    import torch

    torch.set_num_threads(1)
    game = Game(seed=seed, ai_enabled=True)
    policy = _make_teacher(base_net, value_net)
    policy.set_round_seed(seed)
    tracker = RoundTracker(game)
    rows = []
    true_result = None
    frames = 0
    started = time.time()
    while frames < max_frames or tracker.first_destroy is not None:
        controls = policy.act(game)
        sample = dict(policy.last_temporal_sample)
        tank = game.tanks[0]
        target = sample["target"]
        target_x = (target[0] + 0.5) * game.scale
        target_y = (target[1] + 0.5) * game.scale

        tank.forward = bool(controls.get("forward", False))
        tank.backup = bool(controls.get("backup", False))
        tank.turn_left = bool(controls.get("turn_left", False))
        tank.turn_right = bool(controls.get("turn_right", False))
        tank.fire = bool(controls.get("fire", False))
        tracker.pre_step()
        events = game.step()
        frames += 1
        tracker.post_step(events, 1)
        distance_after = float(np.hypot(
            tank.x - target_x, tank.y - target_y) / game.scale)
        sample["progress"] = float(np.clip(
            sample["target_distance_before"] - distance_after, -1.0, 1.0))
        sample["frame"] = frames - 1
        rows.append(sample)
        for event in events:
            if event[0] == "round_end":
                winner = event[1]
                true_result = ("win" if winner == 0 else
                               "loss" if winner == 1 else "double_death")
        if true_result:
            break

    stats = _round_stats(tracker, true_result or "draw", frames)
    result = {
        "seed": int(seed),
        "result": true_result or "draw",
        "frames": int(frames),
        "elapsed_seconds": time.time() - started,
        "rows": rows if true_result == "win" else [],
        "search_rate": float(policy.exact_searches / max(1, frames)),
    }
    result.update(stats)
    result["event_metrics"] = policy.event_tracker.summary()
    result["diagnosis"] = diagnose_battle(result)
    return result


def collect(args):
    seeds = parse_seeds(args.seed_list)
    jobs = [(seed, args.base_net, args.value_net, args.max_frames)
            for seed in seeds]
    workers = max(1, min(args.workers, len(jobs)))
    if workers == 1:
        iterator = map(_collect_seed, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers)
        iterator = pool.imap_unordered(_collect_seed, jobs)
    rounds = []
    started = time.time()
    try:
        for row in iterator:
            rounds.append(row)
            print(
                f"collect={len(rounds)}/{len(jobs)} seed={row['seed']} "
                f"result={row['result']} frames={row['frames']} "
                f"search={row['search_rate']:.1%}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    rounds.sort(key=lambda item: item["seed"])

    arrays = {key: [] for key in (
        "features", "movement", "hold", "interrupt", "progress", "weight",
        "round_seed", "frame", "risk", "category", "reason",
        "topology_active", "topology_kind", "full_search",
        "search_needed", "search_needed_mask",
    )}
    for round_row in rounds:
        rows = round_row["rows"]
        if not rows:
            continue
        movement = np.asarray([row["movement"] for row in rows],
                              dtype=np.int64)
        _, hold, interrupt = sequence_targets(movement)
        for index, row in enumerate(rows):
            weight = 1.0
            if row["topology_active"]:
                weight *= 2.0
            if row["full_search"]:
                weight *= 1.5
            if row["category"] != "standard":
                weight *= 1.5
            arrays["features"].append(row["features"])
            arrays["movement"].append(row["movement"])
            arrays["hold"].append(hold[index])
            arrays["interrupt"].append(interrupt[index])
            arrays["progress"].append(row["progress"])
            arrays["weight"].append(weight)
            arrays["round_seed"].append(round_row["seed"])
            arrays["frame"].append(row["frame"])
            arrays["risk"].append(row["risk"])
            arrays["category"].append(row["category"])
            arrays["reason"].append(row["reason"])
            arrays["topology_active"].append(row["topology_active"])
            arrays["topology_kind"].append(row["topology_kind"])
            arrays["full_search"].append(row["full_search"])
            arrays["search_needed"].append(row["search_needed_target"])
            arrays["search_needed_mask"].append(row["search_needed_mask"])

    if not arrays["features"]:
        raise RuntimeError("collection produced no clean winning sequences")
    payload = {
        "features": np.asarray(arrays["features"], dtype=np.float32),
        "movement": np.asarray(arrays["movement"], dtype=np.int64),
        "hold": np.asarray(arrays["hold"], dtype=np.int64),
        "interrupt": np.asarray(arrays["interrupt"], dtype=np.float32),
        "progress": np.asarray(arrays["progress"], dtype=np.float32),
        "weight": np.asarray(arrays["weight"], dtype=np.float32),
        "round_seed": np.asarray(arrays["round_seed"], dtype=np.int64),
        "frame": np.asarray(arrays["frame"], dtype=np.int32),
        "risk": np.asarray(arrays["risk"], dtype=np.float32),
        "category": np.asarray(arrays["category"], dtype="U32"),
        "reason": np.asarray(arrays["reason"], dtype="U16"),
        "topology_active": np.asarray(
            arrays["topology_active"], dtype=np.bool_),
        "topology_kind": np.asarray(arrays["topology_kind"], dtype="U32"),
        "full_search": np.asarray(arrays["full_search"], dtype=np.bool_),
        "search_needed": np.asarray(
            arrays["search_needed"], dtype=np.float32),
        "search_needed_mask": np.asarray(
            arrays["search_needed_mask"], dtype=np.bool_),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    report = {
        "out": args.out,
        "requested_rounds": len(seeds),
        "clean_rounds": int(len(np.unique(payload["round_seed"]))),
        "states": int(len(payload["movement"])),
        "results": dict(Counter(row["result"] for row in rounds)),
        "categories": dict(Counter(payload["category"].tolist())),
        "reasons": dict(Counter(payload["reason"].tolist())),
        "topology_frames": int(payload["topology_active"].sum()),
        "full_search_frames": int(payload["full_search"].sum()),
        "search_label_states": int(payload["search_needed_mask"].sum()),
        "search_positive_states": int(payload["search_needed"].sum()),
        "mean_search_rate": float(np.mean(
            [row["search_rate"] for row in rounds])),
        "elapsed_seconds": time.time() - started,
        "failed_rounds": [
            {key: row[key] for key in (
                "seed", "result", "frames", "death_cause", "diagnosis")}
            for row in rounds if row["result"] != "win"
        ],
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _sequence_indices(round_seed, selected):
    selected = np.asarray(selected, dtype=np.int64)
    values = round_seed[selected]
    return [selected[values == seed] for seed in np.unique(values)]


def train(args):
    import torch
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pad_sequence

    data = dict(np.load(args.data, allow_pickle=False))
    feature_dim = int(data["features"].shape[1])
    train_index, val_index, val_seeds = group_split(
        data["round_seed"], args.val_fraction, args.split_seed)
    torch.manual_seed(args.split_seed)
    model = build_temporal_intent_net(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim, layers=args.layers)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tensors = {key: torch.as_tensor(value) for key, value in data.items()
               if value.dtype.kind not in "USO"}
    train_sequences = _sequence_indices(data["round_seed"], train_index)
    val_sequences = _sequence_indices(data["round_seed"], val_index)

    def batch(sequence_list):
        lengths = torch.as_tensor([len(index) for index in sequence_list])
        mask = (torch.arange(int(lengths.max())).unsqueeze(0)
                < lengths.unsqueeze(1))
        result = {"mask": mask}
        for key in (
                "features", "movement", "hold", "interrupt", "progress",
                "weight", "topology_active", "search_needed",
                "search_needed_mask"):
            result[key] = pad_sequence(
                [tensors[key][torch.as_tensor(index)] for index in sequence_list],
                batch_first=True)
        return result

    def losses(rows):
        output = model(rows["features"])
        mask = rows["mask"]
        weight = rows["weight"][mask]
        movement = functional.cross_entropy(
            output["movement_delta"][mask], rows["movement"][mask],
            reduction="none")
        hold = functional.cross_entropy(
            output["hold"][mask], rows["hold"][mask], reduction="none")
        interrupt = functional.binary_cross_entropy_with_logits(
            output["interrupt"][mask], rows["interrupt"][mask],
            reduction="none")
        progress = functional.smooth_l1_loss(
            output["progress"][mask], rows["progress"][mask], reduction="none")
        search_mask = rows["search_needed_mask"].bool() & mask
        if search_mask.any():
            search_target = rows["search_needed"][search_mask]
            positive = search_target.sum().clamp_min(1.0)
            negative = (1.0 - search_target).sum()
            positive_weight = (negative / positive).clamp(
                min=1.0, max=args.search_positive_weight_cap)
            search = functional.binary_cross_entropy_with_logits(
                output["search_needed"][search_mask], search_target,
                reduction="none", pos_weight=positive_weight)
            search_weight = rows["weight"][search_mask]
            search = (search * search_weight).sum() / search_weight.sum()
        else:
            search = movement.sum() * 0.0
        movement_logits = output["movement_delta"]
        same = ((rows["movement"][:, 1:] == rows["movement"][:, :-1])
                & mask[:, 1:] & mask[:, :-1])
        smooth = (movement_logits[:, 1:] - movement_logits[:, :-1]).abs().mean(2)
        smooth = smooth[same].mean() if same.any() else movement.sum() * 0.0
        weighted = lambda value: (value * weight).sum() / weight.sum()
        total = (weighted(movement)
                 + args.hold_weight * weighted(hold)
                 + args.interrupt_weight * weighted(interrupt)
                 + args.progress_weight * weighted(progress)
                 + args.search_weight * search
                 + args.smooth_weight * smooth)
        return total, (
            movement, hold, interrupt, progress, search, smooth), output

    def metrics(sequence_list):
        model.eval()
        rows = batch(sequence_list)
        with torch.no_grad():
            total, parts, output = losses(rows)
            mask = rows["mask"]
            predicted = output["movement_delta"].argmax(2)
            accuracy = (predicted[mask] == rows["movement"][mask]).float().mean()
            topology = rows["topology_active"].bool() & mask
            topology_accuracy = (
                (predicted[topology] == rows["movement"][topology]).float().mean()
                if topology.any() else torch.tensor(0.0))
            features = rows["features"]
            base_score = features[:, :, :18].reshape(
                features.shape[0], features.shape[1], 9, 2).max(3).values
            base_movement = base_score.argmax(2)
            base_accuracy = (
                (base_movement[mask] == rows["movement"][mask]).float().mean())
            search_mask = rows["search_needed_mask"].bool() & mask
            search_probability = torch.sigmoid(output["search_needed"])
            search_prediction = search_probability >= args.search_threshold
            search_target = rows["search_needed"].bool()
            true_positive = (search_prediction & search_target & search_mask).sum()
            predicted_positive = (search_prediction & search_mask).sum()
            actual_positive = (search_target & search_mask).sum()
            search_recall = true_positive / actual_positive.clamp_min(1)
            search_precision = true_positive / predicted_positive.clamp_min(1)
        model.train()
        return {
            "loss": float(total),
            "movement_accuracy": float(accuracy),
            "topology_accuracy": float(topology_accuracy),
            "base_movement_accuracy": float(base_accuracy),
            "search_recall": float(search_recall),
            "search_precision": float(search_precision),
            "search_labels": int(search_mask.sum()),
            "search_positives": int(actual_positive),
            "smooth": float(parts[5]),
        }

    baseline = metrics(val_sequences)
    best = None
    best_state = None
    stale = 0
    started = time.time()
    rng = np.random.default_rng(args.split_seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_sequences))
        running = 0.0
        count = 0
        model.train()
        for start in range(0, len(order), args.batch_rounds):
            rows = batch([train_sequences[index]
                          for index in order[start:start + args.batch_rounds]])
            loss = losses(rows)[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.detach())
            count += 1
        values = metrics(val_sequences)
        improved = best is None or values["loss"] < best["loss"] - args.min_delta
        if improved:
            best = values
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch}/{args.epochs} train={running/max(1,count):.4f} "
            f"val={values['loss']:.4f} move={values['movement_accuracy']:.1%} "
            f"topology={values['topology_accuracy']:.1%} stale={stale}",
            flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "hold_bins": HOLD_BINS,
        "data": args.data,
        "validation_seeds": val_seeds,
    }, args.out)
    report = {
        "model": args.out,
        "data": args.data,
        "states": int(len(data["movement"])),
        "rounds": int(len(np.unique(data["round_seed"]))),
        "validation_seeds": val_seeds,
        "baseline_metrics": baseline,
        "best_metrics": best,
        "elapsed_seconds": time.time() - started,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--seed-list", required=True)
    collect_parser.add_argument("--workers", type=int, default=6)
    collect_parser.add_argument("--max-frames", type=int,
                                default=TRUNCATE_FRAMES)
    collect_parser.add_argument("--base-net", default=DEFAULT_BASE_NET)
    collect_parser.add_argument("--value-net", default=DEFAULT_VALUE_NET)
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--report")
    collect_parser.set_defaults(func=collect)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--report", required=True)
    train_parser.add_argument("--hidden-dim", type=int, default=192)
    train_parser.add_argument("--layers", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--min-epochs", type=int, default=8)
    train_parser.add_argument("--patience", type=int, default=6)
    train_parser.add_argument("--batch-rounds", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--val-fraction", type=float, default=0.25)
    train_parser.add_argument("--split-seed", type=int, default=2701)
    train_parser.add_argument("--hold-weight", type=float, default=0.25)
    train_parser.add_argument("--interrupt-weight", type=float, default=0.15)
    train_parser.add_argument("--progress-weight", type=float, default=0.10)
    train_parser.add_argument("--smooth-weight", type=float, default=0.02)
    train_parser.add_argument("--search-weight", type=float, default=0.75)
    train_parser.add_argument("--search-threshold", type=float, default=0.35)
    train_parser.add_argument("--search-positive-weight-cap", type=float,
                              default=12.0)
    train_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_parser.set_defaults(func=train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
