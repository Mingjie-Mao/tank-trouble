"""Train a conservative privileged safety shield from exact-teacher labels.

This model does not imitate the teacher's single selected action. It predicts
the exact teacher's safe root-action set. The P27b champion remains in control
unless the shield is confident that the champion action is unsafe and that a
high-value, non-firing replacement is safe.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.exact_teacher_residual import (  # noqa: E402
    DEFAULT_BASE_NET,
    DEFAULT_VALUE_NET,
    _csv,
    _indices_for_seeds,
    build_features,
    stratified_seed_folds,
    stratified_seed_holdout,
)
from training.mpc_agent import CANDIDATES  # noqa: E402
from training.p26_amortized_mpc import DATA_DIR, load_p26_network  # noqa: E402
from training.p27_risk_value import load_model  # noqa: E402


NONFIRE_ACTIONS = np.asarray([
    index for index, action in enumerate(CANDIDATES) if action[2] == 0
], dtype=np.int64)
CHAMPION_FEATURE_DIM = 297
POLICY_VALUE_SLICE = slice(261, 279)


def load_safety_data(phases):
    keys = (
        "X", "P", "privileged_valid", "allowed", "champion_action",
        "champion_action_allowed", "round_seed", "capture_reason", "category",
    )
    parts = []
    paths = []
    privileged_names = None
    for phase in phases:
        pattern = os.path.join(DATA_DIR, phase, "shard_*.npz")
        for path in sorted(glob.glob(pattern)):
            data = np.load(path)
            missing = [key for key in keys if key not in data.files]
            if missing:
                raise RuntimeError(f"{path} lacks safety fields: {missing}")
            names = tuple(data["privileged_feature_names"].astype(str))
            if privileged_names is None:
                privileged_names = names
            elif names != privileged_names:
                raise RuntimeError(f"privileged schema mismatch in {path}")
            parts.append({key: data[key] for key in keys})
            paths.append(path)
    if not parts:
        raise RuntimeError("no privileged safety shards found")
    joined = {
        key: np.concatenate([part[key] for part in parts]) for key in keys
    }
    if len(np.unique(joined["round_seed"])) < 10:
        raise RuntimeError("safety training needs at least ten complete seeds")
    if joined["allowed"].shape != (len(joined["X"]), len(CANDIDATES)):
        raise RuntimeError("allowed action set has the wrong shape")
    if not joined["privileged_valid"].all():
        raise RuntimeError("one or more rows lack privileged state")
    if not (np.isfinite(joined["X"]).all()
            and np.isfinite(joined["P"]).all()):
        raise RuntimeError("non-finite privileged safety input")
    return joined, paths, privileged_names


def build_safety_features(base_net, value_net, data, batch=512,
                          include_current_observation=False,
                          frame_stack=4):
    champion = build_features(
        base_net,
        value_net,
        data["X"].astype(np.float32),
        data["champion_action"].astype(np.int64),
        batch,
    )
    if champion.shape[1] != CHAMPION_FEATURE_DIM:
        raise RuntimeError(
            f"champion feature mismatch: {champion.shape[1]} != "
            f"{CHAMPION_FEATURE_DIM}")
    parts = [champion, data["P"].astype(np.float32)]
    if include_current_observation:
        if data["X"].shape[1] % frame_stack:
            raise RuntimeError("stacked observation is not divisible by frame stack")
        frame_dim = data["X"].shape[1] // frame_stack
        parts.append(data["X"][:, -frame_dim:].astype(np.float32))
    return (np.concatenate(parts, axis=1).astype(np.float32),
            champion[:, POLICY_VALUE_SLICE].copy())


def build_safety_head(in_dim, width=96, dropout=0.10):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, len(CANDIDATES)),
    )


def _fit_model(features, data, indices, mean, std, args, seed):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    model = build_safety_head(features.shape[1], args.width, args.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.as_tensor((features[indices] - mean) / std)
    target = torch.as_tensor(data["allowed"][indices].astype(np.float32))
    champion = torch.as_tensor(
        data["champion_action"][indices].astype(np.int64))
    row = torch.arange(len(indices))
    weight = torch.ones_like(target)
    weight[row, champion] = args.champion_action_weight
    dangerous = torch.as_tensor(
        (data["allowed"][indices].sum(axis=1)
         <= args.danger_root_threshold).astype(np.float32))
    weight *= 1.0 + dangerous[:, None] * (args.danger_state_weight - 1.0)

    generator = torch.Generator().manual_seed(seed + 17)
    model.train()
    for _ in range(args.epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), args.batch):
            batch_index = order[start:start + args.batch]
            logits = model(x[batch_index])
            loss = functional.binary_cross_entropy_with_logits(
                logits,
                target[batch_index],
                weight=weight[batch_index],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.eval()
    return model


def _predict(models, features, indices, mean, std):
    import torch

    x = torch.as_tensor((features[indices] - mean) / std)
    probabilities = []
    with torch.no_grad():
        for model in models:
            probabilities.append(torch.sigmoid(model(x)).numpy())
    return np.mean(probabilities, axis=0)


def choose_replacements(safe_probability, champion_action, policy_value,
                        champion_unsafe_threshold, alternative_safe_threshold,
                        safety_gap):
    """Return conservative non-fire replacements, or -1 for no override."""
    safe_probability = np.asarray(safe_probability, dtype=np.float32)
    champion_action = np.asarray(champion_action, dtype=np.int64)
    policy_value = np.asarray(policy_value, dtype=np.float32)
    rows = np.arange(len(champion_action))
    champion_safe = safe_probability[rows, champion_action]
    champion_unsafe = 1.0 - champion_safe
    replacements = np.full(len(champion_action), -1, dtype=np.int64)

    for row in range(len(champion_action)):
        if champion_unsafe[row] < champion_unsafe_threshold:
            continue
        candidates = NONFIRE_ACTIONS[
            (safe_probability[row, NONFIRE_ACTIONS]
             >= alternative_safe_threshold)
            & (safe_probability[row, NONFIRE_ACTIONS] - champion_safe[row]
               >= safety_gap)
            & (NONFIRE_ACTIONS != champion_action[row])
        ]
        if len(candidates) == 0:
            continue
        replacements[row] = int(candidates[
            np.argmax(policy_value[row, candidates])
        ])
    return replacements


def safety_override_metrics(safe_probability, policy_value, data, indices,
                            champion_unsafe_threshold,
                            alternative_safe_threshold, safety_gap):
    champion = data["champion_action"][indices].astype(np.int64)
    replacements = choose_replacements(
        safe_probability,
        champion,
        policy_value[indices],
        champion_unsafe_threshold,
        alternative_safe_threshold,
        safety_gap,
    )
    predicted = replacements >= 0
    predicted_rows = np.flatnonzero(predicted)
    champion_actually_unsafe = ~data[
        "allowed"][indices, champion].astype(np.bool_)
    replacement_actually_safe = np.zeros(len(indices), dtype=np.bool_)
    replacement_actually_safe[predicted_rows] = data["allowed"][
        indices[predicted_rows], replacements[predicted_rows]
    ].astype(np.bool_)
    correct = predicted & champion_actually_unsafe & replacement_actually_safe
    count = int(predicted.sum())
    correct_count = int(correct.sum())
    unsafe_count = int(champion_actually_unsafe.sum())
    bands = {
        int(seed) // 1000
        for seed in data["round_seed"][indices][predicted]
    }
    return {
        "champion_unsafe_threshold": float(champion_unsafe_threshold),
        "alternative_safe_threshold": float(alternative_safe_threshold),
        "safety_gap": float(safety_gap),
        "predicted_overrides": count,
        "correct_overrides": correct_count,
        "precision": correct_count / max(1, count),
        "coverage": count / max(1, len(indices)),
        "unsafe_champion_precision": int(
            (predicted & champion_actually_unsafe).sum()) / max(1, count),
        "safe_replacement_precision": int(
            (predicted & replacement_actually_safe).sum()) / max(1, count),
        "unsafe_champion_recall": correct_count / max(1, unsafe_count),
        "unnecessary_overrides": int(
            (predicted & ~champion_actually_unsafe).sum()),
        "unsafe_replacements": int(
            (predicted & ~replacement_actually_safe).sum()),
        "predicted_seed_bands": len(bands),
        "unsafe_champion_states": unsafe_count,
        "states": int(len(indices)),
    }


def calibrate(safe_probability, policy_value, data, indices, args):
    champion = data["champion_action"][indices].astype(np.int64)
    champion_unsafe = 1.0 - safe_probability[
        np.arange(len(indices)), champion]
    nonfire_safe = safe_probability[:, NONFIRE_ACTIONS].max(axis=1)
    champion_thresholds = np.unique(np.concatenate((
        np.linspace(0.55, 0.995, 38),
        np.quantile(champion_unsafe, np.linspace(0.50, 0.995, 45)),
    )))
    alternative_thresholds = np.unique(np.concatenate((
        np.linspace(0.55, 0.995, 38),
        np.quantile(nonfire_safe, np.linspace(0.50, 0.995, 45)),
    )))
    eligible = []
    diagnostic = []
    for champion_threshold in champion_thresholds:
        for alternative_threshold in alternative_thresholds:
            for safety_gap in args.safety_gaps:
                metrics = safety_override_metrics(
                    safe_probability,
                    policy_value,
                    data,
                    indices,
                    champion_threshold,
                    alternative_threshold,
                    safety_gap,
                )
                if (metrics["predicted_overrides"]
                        >= args.min_calibration_overrides
                        and metrics["coverage"] <= args.max_coverage
                        and metrics["predicted_seed_bands"] >= 2):
                    diagnostic.append(metrics)
                    if metrics["precision"] >= args.min_calibration_precision:
                        eligible.append(metrics)
    pool = eligible or diagnostic
    if not pool:
        return None, False
    best = max(pool, key=lambda item: (
        item["correct_overrides"],
        item["precision"],
        item["unsafe_champion_precision"],
        item["safe_replacement_precision"],
        -item["coverage"],
    ))
    return best, bool(eligible)


def train(args):
    import torch

    phases = _csv(args.phases)
    data, paths, privileged_names = load_safety_data(phases)
    base_net, _ = load_p26_network(args.base_net)
    value_net, value_in_dim = load_model(args.value_net)
    if value_in_dim != data["X"].shape[1]:
        raise RuntimeError("safety features require the plain P27b value head")
    features, policy_value = build_safety_features(
        base_net, value_net, data, args.feature_batch,
        args.include_current_observation, args.frame_stack)

    dev_seeds, test_seeds = stratified_seed_holdout(
        data["round_seed"], args.test_fraction, args.split_seed)
    dev_index = _indices_for_seeds(data["round_seed"], dev_seeds)
    test_index = _indices_for_seeds(data["round_seed"], test_seeds)
    folds = stratified_seed_folds(
        np.asarray(dev_seeds, dtype=np.int64), args.folds,
        args.split_seed + 1)
    dev_row_seeds = data["round_seed"][dev_index]
    oof_safe = np.zeros((len(dev_index), len(CANDIDATES)), dtype=np.float32)
    fold_reports = []

    for fold in range(args.folds):
        validation_seeds = [
            seed for seed, assignment in folds.items() if assignment == fold
        ]
        validation_local = np.flatnonzero(np.isin(
            dev_row_seeds, np.asarray(validation_seeds, dtype=np.int64)))
        training_local = np.flatnonzero(~np.isin(
            dev_row_seeds, np.asarray(validation_seeds, dtype=np.int64)))
        training_index = dev_index[training_local]
        fold_mean = features[training_index].mean(axis=0, keepdims=True)
        fold_std = features[training_index].std(
            axis=0, keepdims=True).clip(min=1e-4)
        model = _fit_model(
            features, data, training_index, fold_mean, fold_std, args,
            args.split_seed + 100 + fold)
        oof_safe[validation_local] = _predict(
            [model], features, dev_index[validation_local],
            fold_mean, fold_std)
        fold_reports.append({
            "fold": fold,
            "train_seeds": int(len(np.unique(
                data["round_seed"][training_index]))),
            "validation_seeds": sorted(int(seed) for seed in validation_seeds),
            "validation_states": int(len(validation_local)),
            "validation_unsafe_champion": int((~data[
                "champion_action_allowed"][dev_index[validation_local]]).sum()),
        })

    calibration, calibration_passed = calibrate(
        oof_safe, policy_value, data, dev_index, args)

    mean = features[dev_index].mean(axis=0, keepdims=True)
    std = features[dev_index].std(axis=0, keepdims=True).clip(min=1e-4)
    final_models = [
        _fit_model(
            features, data, dev_index, mean, std, args,
            args.split_seed + 1000 + member)
        for member in range(args.ensemble)
    ]
    test_safe = _predict(final_models, features, test_index, mean, std)
    test_metrics = None
    if calibration is not None:
        test_metrics = safety_override_metrics(
            test_safe,
            policy_value,
            data,
            test_index,
            calibration["champion_unsafe_threshold"],
            calibration["alternative_safe_threshold"],
            calibration["safety_gap"],
        )
    test_passed = bool(
        test_metrics is not None
        and test_metrics["predicted_overrides"] >= args.min_test_overrides
        and test_metrics["precision"] >= args.min_test_precision
        and test_metrics["coverage"] <= args.max_test_coverage
        and test_metrics["predicted_seed_bands"] >= 2
    )
    passed = bool(calibration_passed and test_passed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dicts": [model.state_dict() for model in final_models],
        "in_dim": int(features.shape[1]),
        "width": args.width,
        "dropout": args.dropout,
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "base_net": args.base_net,
        "value_net": args.value_net,
        "phases": phases,
        "privileged_feature_names": privileged_names,
        "include_current_observation": args.include_current_observation,
        "frame_stack": args.frame_stack,
        "calibration": calibration,
        "offline_gate_passed": passed,
    }, args.out)

    report = {
        "model": args.out,
        "decision": (
            "feasibility_pass_collect_more" if passed
            else "reject_or_adjust_before_more_collection"),
        "offline_gate_passed": passed,
        "calibration_passed": calibration_passed,
        "test_passed": test_passed,
        "states": int(len(data["X"])),
        "seeds": int(len(np.unique(data["round_seed"]))),
        "feature_dim": int(features.shape[1]),
        "privileged_dim": int(data["P"].shape[1]),
        "include_current_observation": args.include_current_observation,
        "safe_action_rate": float(data["allowed"].mean()),
        "unsafe_champion_states": int((~data[
            "champion_action_allowed"].astype(np.bool_)).sum()),
        "unsafe_champion_rate": float((~data[
            "champion_action_allowed"].astype(np.bool_)).mean()),
        "no_safe_root_states": int((data["allowed"].sum(axis=1) == 0).sum()),
        "dev_seeds": dev_seeds,
        "test_seeds": test_seeds,
        "dev_states": int(len(dev_index)),
        "test_states": int(len(test_index)),
        "folds": fold_reports,
        "calibration": calibration,
        "test": test_metrics,
        "capture_reasons": dict(Counter(
            data["capture_reason"].astype(str))),
        "categories": dict(Counter(data["category"].astype(str))),
        "paths": paths,
        "elapsed_seconds": time.time() - args.started,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", required=True)
    parser.add_argument("--base-net", default=DEFAULT_BASE_NET)
    parser.add_argument("--value-net", default=DEFAULT_VALUE_NET)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=33001)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--feature-batch", type=int, default=512)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--include-current-observation", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--champion-action-weight", type=float, default=3.0)
    parser.add_argument("--danger-state-weight", type=float, default=2.0)
    parser.add_argument("--danger-root-threshold", type=int, default=2)
    parser.add_argument("--safety-gaps", type=float, nargs="+",
                        default=(0.0, 0.10, 0.20, 0.30))
    parser.add_argument("--min-calibration-overrides", type=int, default=6)
    parser.add_argument("--min-calibration-precision", type=float, default=0.85)
    parser.add_argument("--max-coverage", type=float, default=0.05)
    parser.add_argument("--min-test-overrides", type=int, default=2)
    parser.add_argument("--min-test-precision", type=float, default=0.80)
    parser.add_argument("--max-test-coverage", type=float, default=0.06)
    args = parser.parse_args()
    args.started = time.time()
    train(args)


if __name__ == "__main__":
    main()
