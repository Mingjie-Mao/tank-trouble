"""Train a conservative residual policy from frozen exact-teacher labels.

The P27b champion remains the default action source.  This head only predicts
whether a teacher-labelled correction is reliable and which action to use.
Calibration is group-held-out by complete round seed, never by individual
frames, so adjacent observations cannot leak across train and evaluation.
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

from training.p26_amortized_mpc import DATA_DIR, load_p26_network  # noqa: E402
from training.p27_risk_value import load_model  # noqa: E402


DEFAULT_BASE_NET = "training/models/p26_amortized_mpc_iter05.pt"
DEFAULT_VALUE_NET = "training/models/p27b_risk_value_iter00.pt"


def _csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_residual_data(phases):
    keys = (
        "X", "Y_action", "action_valid", "champion_action",
        "champion_action_allowed", "teacher_override", "teacher_advantage",
        "teacher_margin", "residual_target", "round_seed", "capture_reason",
        "category",
    )
    parts = []
    paths = []
    for phase in phases:
        for path in sorted(glob.glob(
                os.path.join(DATA_DIR, phase, "shard_*.npz"))):
            data = np.load(path)
            missing = [key for key in keys if key not in data.files]
            if missing:
                raise RuntimeError(f"{path} lacks residual fields: {missing}")
            parts.append({key: data[key] for key in keys})
            paths.append(path)
    if not parts:
        raise RuntimeError("no residual shards found")
    joined = {
        key: np.concatenate([part[key] for part in parts]) for key in keys
    }
    if len(np.unique(joined["round_seed"])) < 10:
        raise RuntimeError("residual training needs at least ten complete seeds")
    return joined, paths


def stratified_seed_holdout(seeds, fraction, split_seed):
    """Hold out complete seeds while preserving each thousand-seed band."""
    rng = np.random.default_rng(split_seed)
    train = []
    test = []
    unique = np.unique(seeds.astype(np.int64))
    bands = sorted(set(int(seed) // 1000 for seed in unique))
    for band in bands:
        values = unique[unique // 1000 == band]
        values = rng.permutation(values)
        count = max(1, min(len(values) - 1,
                           int(round(len(values) * fraction))))
        test.extend(int(value) for value in values[:count])
        train.extend(int(value) for value in values[count:])
    return sorted(train), sorted(test)


def stratified_seed_folds(seeds, folds, split_seed):
    rng = np.random.default_rng(split_seed)
    assignment = {}
    unique = np.unique(seeds.astype(np.int64))
    for band in sorted(set(int(seed) // 1000 for seed in unique)):
        values = unique[unique // 1000 == band]
        for index, value in enumerate(rng.permutation(values)):
            assignment[int(value)] = index % folds
    return assignment


def _indices_for_seeds(row_seeds, selected):
    selected = set(int(seed) for seed in selected)
    return np.flatnonzero(np.asarray([
        int(seed) in selected for seed in row_seeds
    ], dtype=np.bool_))


def build_features(base_net, value_net, x, champion_action, batch=512):
    """Use frozen champion outputs, not raw geometry fitting, as features."""
    import torch

    features = []
    for start in range(0, len(x), batch):
        xb = torch.as_tensor(x[start:start + batch], dtype=torch.float32)
        actions = torch.as_tensor(
            champion_action[start:start + batch], dtype=torch.int64)
        with torch.no_grad():
            base = base_net(xb)
            value = value_net(xb)
            base_score = base["score"]
            base_aux = torch.sigmoid(base["aux"]).flatten(1)
            base_fire = torch.sigmoid(base["fire"])
            value_score = value["score"]
            value_aux_raw = torch.sigmoid(value["aux"])
            value_aux = value_aux_raw.flatten(1)
            policy_value = (
                value_score
                + 0.04 * value_aux_raw[:, :, 0]
                - 0.12 * value_aux_raw[:, :, 1]
                - 0.18 * value_aux_raw[:, :, 2]
                + 0.02 * value_aux_raw[:, :, 4]
                + 0.02 * value_aux_raw[:, :, 5]
            )
            one_hot = torch.nn.functional.one_hot(
                actions, num_classes=18).float()
            features.append(torch.cat((
                base_score, base_aux, base_fire,
                value_score, value_aux, policy_value, one_hot,
            ), dim=1).cpu().numpy())
    return np.concatenate(features).astype(np.float32)


def build_residual_head(in_dim, width=96, dropout=0.12):
    import torch.nn as nn

    class _ResidualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.ReLU(),
            )
            self.gate = nn.Linear(width, 1)
            self.action = nn.Linear(width, 18)

        def forward(self, x):
            hidden = self.encoder(x)
            return {
                "gate": self.gate(hidden).squeeze(1),
                "action": self.action(hidden),
            }

    return _ResidualHead()


def _fit_model(features, data, indices, mean, std, args, seed):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    model = build_residual_head(
        features.shape[1], args.width, args.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.as_tensor((features[indices] - mean) / std)
    target = torch.as_tensor(
        data["residual_target"][indices].astype(np.float32))
    action = torch.as_tensor(
        data["Y_action"][indices].astype(np.int64)).clamp_min(0)
    valid = torch.as_tensor(
        data["action_valid"][indices].astype(np.bool_))
    positives = float(target.sum())
    pos_weight = torch.tensor(min(
        args.max_positive_weight,
        max(1.0, (len(target) - positives) / max(1.0, positives))))
    generator = torch.Generator().manual_seed(seed + 17)
    model.train()
    for _ in range(args.epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), args.batch):
            batch_index = order[start:start + args.batch]
            output = model(x[batch_index])
            gate_loss = functional.binary_cross_entropy_with_logits(
                output["gate"], target[batch_index], pos_weight=pos_weight)
            action_loss = functional.cross_entropy(
                output["action"], action[batch_index], reduction="none",
                label_smoothing=args.label_smoothing)
            action_weight = (
                args.background_action_weight
                + target[batch_index]
                * (args.target_action_weight - args.background_action_weight)
            ) * valid[batch_index].float()
            action_loss = (
                (action_loss * action_weight).sum()
                / action_weight.sum().clamp_min(1.0))
            loss = gate_loss + args.action_loss_weight * action_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.eval()
    return model


def _predict(models, features, indices, mean, std):
    import torch

    x = torch.as_tensor((features[indices] - mean) / std)
    gate = []
    action = []
    with torch.no_grad():
        for model in models:
            output = model(x)
            gate.append(torch.sigmoid(output["gate"]).numpy())
            action.append(torch.softmax(output["action"], dim=1).numpy())
    return np.mean(gate, axis=0), np.mean(action, axis=0)


def override_metrics(gate_probability, action_probability, data, indices,
                     gate_threshold, action_threshold):
    predicted_action = action_probability.argmax(axis=1)
    confidence = action_probability.max(axis=1)
    champion = data["champion_action"][indices].astype(np.int64)
    target_action = data["Y_action"][indices].astype(np.int64)
    target = data["residual_target"][indices].astype(np.bool_)
    predicted = (
        (gate_probability >= gate_threshold)
        & (confidence >= action_threshold)
        & (predicted_action != champion)
    )
    correct = predicted & target & (predicted_action == target_action)
    count = int(predicted.sum())
    correct_count = int(correct.sum())
    target_count = int(target.sum())
    unsafe_target = target & ~data[
        "champion_action_allowed"][indices].astype(np.bool_)
    unsafe_correct = correct & unsafe_target
    seeds = data["round_seed"][indices][predicted]
    bands = set(int(seed) // 1000 for seed in seeds)
    return {
        "gate_threshold": float(gate_threshold),
        "action_threshold": float(action_threshold),
        "predicted_overrides": count,
        "correct_overrides": correct_count,
        "precision": correct_count / max(1, count),
        "coverage": count / max(1, len(indices)),
        "target_recall": correct_count / max(1, target_count),
        "unsafe_target_recall": int(unsafe_correct.sum())
        / max(1, int(unsafe_target.sum())),
        "predicted_seed_bands": len(bands),
        "target_count": target_count,
        "states": int(len(indices)),
    }


def calibrate(gate_probability, action_probability, data, indices, args):
    gate_thresholds = np.unique(np.concatenate((
        np.linspace(0.10, 0.99, 60),
        np.quantile(gate_probability, np.linspace(0.50, 0.995, 80)),
    )))
    action_confidence = action_probability.max(axis=1)
    action_thresholds = np.unique(np.concatenate((
        np.linspace(0.05, 0.90, 36),
        np.quantile(action_confidence, np.linspace(0.20, 0.99, 40)),
    )))
    eligible = []
    fallback = []
    for gate_threshold in gate_thresholds:
        for action_threshold in action_thresholds:
            metrics = override_metrics(
                gate_probability, action_probability, data, indices,
                gate_threshold, action_threshold)
            if (metrics["predicted_overrides"] >= args.min_calibration_overrides
                    and metrics["coverage"] <= args.max_coverage
                    and metrics["predicted_seed_bands"] >= 2):
                fallback.append(metrics)
                if metrics["precision"] >= args.min_calibration_precision:
                    eligible.append(metrics)
    pool = eligible or fallback
    if not pool:
        return None, False
    best = max(pool, key=lambda item: (
        item["correct_overrides"], item["precision"],
        item["unsafe_target_recall"], -item["coverage"]))
    return best, bool(eligible)


def train(args):
    import torch

    phases = _csv(args.phases)
    data, paths = load_residual_data(phases)
    base_net, _ = load_p26_network(args.base_net)
    value_net, value_in_dim = load_model(args.value_net)
    if value_in_dim != data["X"].shape[1]:
        raise RuntimeError("residual features require the plain P27b value head")
    features = build_features(
        base_net, value_net, data["X"].astype(np.float32),
        data["champion_action"].astype(np.int64), args.feature_batch)

    dev_seeds, test_seeds = stratified_seed_holdout(
        data["round_seed"], args.test_fraction, args.split_seed)
    dev_index = _indices_for_seeds(data["round_seed"], dev_seeds)
    test_index = _indices_for_seeds(data["round_seed"], test_seeds)
    mean = features[dev_index].mean(axis=0, keepdims=True)
    std = features[dev_index].std(axis=0, keepdims=True).clip(min=1e-4)

    folds = stratified_seed_folds(
        np.asarray(dev_seeds, dtype=np.int64), args.folds,
        args.split_seed + 1)
    oof_gate = np.zeros(len(dev_index), dtype=np.float32)
    oof_action = np.zeros((len(dev_index), 18), dtype=np.float32)
    dev_row_seeds = data["round_seed"][dev_index]
    fold_reports = []
    for fold in range(args.folds):
        validation_seeds = [seed for seed, value in folds.items()
                            if value == fold]
        validation_local = np.flatnonzero(np.asarray([
            int(seed) in set(validation_seeds) for seed in dev_row_seeds
        ], dtype=np.bool_))
        training_local = np.flatnonzero(~np.isin(
            dev_row_seeds, np.asarray(validation_seeds, dtype=np.int64)))
        model = _fit_model(
            features, data, dev_index[training_local], mean, std, args,
            args.split_seed + 100 + fold)
        gate, action = _predict(
            [model], features, dev_index[validation_local], mean, std)
        oof_gate[validation_local] = gate
        oof_action[validation_local] = action
        fold_reports.append({
            "fold": fold,
            "train_seeds": int(len(set(
                data["round_seed"][dev_index[training_local]]))),
            "validation_seeds": sorted(int(seed) for seed in validation_seeds),
            "validation_states": int(len(validation_local)),
            "validation_targets": int(data[
                "residual_target"][dev_index[validation_local]].sum()),
        })

    calibration, calibration_passed = calibrate(
        oof_gate, oof_action, data, dev_index, args)
    final_models = [
        _fit_model(
            features, data, dev_index, mean, std, args,
            args.split_seed + 1000 + member)
        for member in range(args.ensemble)
    ]

    test_gate, test_action = _predict(
        final_models, features, test_index, mean, std)
    test_metrics = None
    if calibration is not None:
        test_metrics = override_metrics(
            test_gate, test_action, data, test_index,
            calibration["gate_threshold"],
            calibration["action_threshold"])
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
        "calibration": calibration,
        "offline_gate_passed": passed,
    }, args.out)
    report = {
        "model": args.out,
        "decision": (
            "ready_for_online_ab" if passed
            else "reject_or_collect_more_before_online_ab"),
        "offline_gate_passed": passed,
        "calibration_passed": calibration_passed,
        "test_passed": test_passed,
        "states": int(len(data["X"])),
        "seeds": int(len(np.unique(data["round_seed"]))),
        "residual_targets": int(data["residual_target"].sum()),
        "target_rate": float(data["residual_target"].mean()),
        "positive_seeds": int(len(np.unique(
            data["round_seed"][data["residual_target"].astype(bool)]))),
        "feature_dim": int(features.shape[1]),
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
    parser.add_argument("--split-seed", type=int, default=32001)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--feature-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--max-positive-weight", type=float, default=6.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.55)
    parser.add_argument("--background-action-weight", type=float, default=0.25)
    parser.add_argument("--target-action-weight", type=float, default=4.0)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--min-calibration-overrides", type=int, default=8)
    parser.add_argument("--min-calibration-precision", type=float, default=0.85)
    parser.add_argument("--max-coverage", type=float, default=0.05)
    parser.add_argument("--min-test-overrides", type=int, default=3)
    parser.add_argument("--min-test-precision", type=float, default=0.80)
    parser.add_argument("--max-test-coverage", type=float, default=0.06)
    args = parser.parse_args()
    args.started = time.time()
    train(args)


if __name__ == "__main__":
    main()
