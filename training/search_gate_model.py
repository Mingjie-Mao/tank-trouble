"""State-aware classifier for deciding when full exact search is necessary."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.temporal_intent_pipeline import group_split


def build_search_gate_net(feature_dim, hidden_dim=256):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
    )


def threshold_report(probability, target, target_recall=0.95):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    rows = []
    for threshold in np.linspace(0.02, 0.98, 97):
        predicted = probability >= threshold
        true_positive = int((predicted & target).sum())
        recall = true_positive / max(1, int(target.sum()))
        precision = true_positive / max(1, int(predicted.sum()))
        rows.append({
            "threshold": float(threshold),
            "trigger_rate": float(predicted.mean()),
            "recall": float(recall),
            "precision": float(precision),
        })
    eligible = [row for row in rows if row["recall"] >= target_recall]
    selected = (min(eligible, key=lambda row: row["trigger_rate"])
                if eligible else max(rows, key=lambda row: row["recall"]))
    return selected, rows


class SearchGateRuntime:
    def __init__(self, path):
        import torch

        payload = torch.load(path, weights_only=False)
        self.torch = torch
        self.mean = torch.as_tensor(payload["mean"])
        self.scale = torch.as_tensor(payload["scale"])
        self.model = build_search_gate_net(
            int(payload["feature_dim"]), int(payload["hidden_dim"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.threshold = float(payload["threshold"])
        self.feature_dim = int(payload["feature_dim"])

    def predict(self, features):
        values = self.torch.as_tensor(
            np.asarray(features, dtype=np.float32)).reshape(1, -1)
        values = (values - self.mean) / self.scale
        with self.torch.no_grad():
            return float(self.torch.sigmoid(self.model(values)[0, 0]))


def train(args):
    import torch
    import torch.nn.functional as functional

    data = dict(np.load(args.data, allow_pickle=False))
    labeled = np.flatnonzero(data["search_needed_mask"])
    train_all, validation_all, validation_seeds = group_split(
        data["round_seed"], args.val_fraction, args.split_seed)
    train_index = np.intersect1d(labeled, train_all)
    validation_index = np.intersect1d(labeled, validation_all)
    if not len(train_index) or not len(validation_index):
        raise RuntimeError("search gate requires labeled train and validation data")

    features = torch.as_tensor(data["features"])
    target = torch.as_tensor(data["search_needed"]).float()
    train_features = features[train_index]
    mean = train_features.mean(0)
    scale = train_features.std(0).clamp_min(1e-3)
    normalized = (features - mean) / scale
    positive = target[train_index].sum().clamp_min(1.0)
    negative = (1.0 - target[train_index]).sum()
    positive_weight = (negative / positive).clamp(
        min=1.0, max=args.positive_weight_cap)

    torch.manual_seed(args.split_seed)
    model = build_search_gate_net(features.shape[1], args.hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def evaluate():
        model.eval()
        with torch.no_grad():
            logits = model(normalized[validation_index]).squeeze(1)
            loss = functional.binary_cross_entropy_with_logits(
                logits, target[validation_index], pos_weight=positive_weight)
            probability = torch.sigmoid(logits).numpy()
        selected, sweep = threshold_report(
            probability, data["search_needed"][validation_index],
            args.target_recall)
        model.train()
        return float(loss), selected, sweep

    best_loss = float("inf")
    best_state = None
    best_threshold = None
    best_sweep = None
    stale = 0
    index_tensor = torch.as_tensor(train_index)
    for epoch in range(1, args.epochs + 1):
        order = index_tensor[torch.randperm(len(index_tensor))]
        model.train()
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            logits = model(normalized[index]).squeeze(1)
            loss = functional.binary_cross_entropy_with_logits(
                logits, target[index], pos_weight=positive_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss, selected, sweep = evaluate()
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
            best_threshold = selected
            best_sweep = sweep
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch}/{args.epochs} val={val_loss:.4f} "
            f"threshold={selected['threshold']:.2f} "
            f"recall={selected['recall']:.1%} "
            f"trigger={selected['trigger_rate']:.1%} stale={stale}",
            flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    model.load_state_dict(best_state)
    payload = {
        "state_dict": model.state_dict(),
        "feature_dim": int(features.shape[1]),
        "hidden_dim": args.hidden_dim,
        "mean": mean,
        "scale": scale,
        "threshold": best_threshold["threshold"],
        "data": args.data,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    report = {
        "model": args.out,
        "data": args.data,
        "labeled_states": int(len(labeled)),
        "positive_states": int(data["search_needed"][labeled].sum()),
        "train_states": int(len(train_index)),
        "validation_states": int(len(validation_index)),
        "validation_seeds": validation_seeds,
        "positive_weight": float(positive_weight),
        "best_loss": best_loss,
        "selected": best_threshold,
        "threshold_sweep": best_sweep,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight-cap", type=float, default=12.0)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=2701)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
