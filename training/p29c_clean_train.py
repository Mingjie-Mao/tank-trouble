"""Train a clean P29c risk/value head from one teacher objective.

Unlike the earlier P29 run, this trainer rejects legacy score semantics,
splits validation by complete round seed, and learns normalized per-state
advantages. The model remains a small assist head and is compatible with the
existing P27 runtime policy.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.p26_amortized_mpc import AUX_DIM, AUX_NAMES, DATA_DIR  # noqa: E402
from training.p27_risk_value import (  # noqa: E402
    P29C_CONTEXT_DIM,
    P29C_CONTEXT_NAMES,
    build_p27b_net,
)


def _csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _weights(value):
    result = {}
    for item in _csv(value or ""):
        name, weight = item.split("=", 1)
        result[name.strip()] = float(weight)
    return result


def _scalar_text(data, key, default=""):
    if key not in data.files:
        return default
    value = np.asarray(data[key]).reshape(-1)
    return str(value[0]) if len(value) else default


def load_clean_data(args):
    category_weights = _weights(args.category_weights)
    arrays = []
    used = []
    for phase in _csv(args.include_phases):
        for path in sorted(glob.glob(os.path.join(
                DATA_DIR, phase, "shard_*.npz"))):
            data = np.load(path)
            required = {"X", "Y_score", "Y_aux", "W", "category",
                        "round_seed", "frame", "objective_version"}
            if not required.issubset(data.files):
                continue
            objective = _scalar_text(data, "objective_version")
            if objective != args.objective_version:
                continue
            context_dim = int(np.asarray(data.get(
                "context_dim", np.asarray([0]))).reshape(-1)[0])
            if context_dim != P29C_CONTEXT_DIM:
                continue
            x = data["X"].astype(np.float32)
            ys = data["Y_score"].astype(np.float32)
            ya = data["Y_aux"].astype(np.float32)
            if ya.ndim != 3 or ya.shape[1:] != (18, AUX_DIM):
                continue
            w = data["W"].astype(np.float32)
            categories = data["category"].astype(str)
            for category, multiplier in category_weights.items():
                w[categories == category] *= multiplier
            arrays.append((
                x,
                ys,
                ya,
                w,
                categories,
                data["round_seed"].astype(np.int64),
                data["frame"].astype(np.int32),
            ))
            used.append(path)
    if not arrays:
        raise RuntimeError(
            "no clean P29c shards matched objective/context requirements")

    joined = [np.concatenate([item[index] for item in arrays])
              for index in range(7)]
    x, ys, ya, w, categories, seeds, frames = joined

    # Keep one label per exact round/frame. Adjacent categories can identify
    # the same state; the highest-weight label is the most specific one.
    keep = {}
    for index, key in enumerate(zip(seeds.tolist(), frames.tolist())):
        previous = keep.get(key)
        if previous is None or w[index] > w[previous]:
            keep[key] = index
    indices = np.asarray(sorted(keep.values()), dtype=np.int64)
    return (
        x[indices], ys[indices], ya[indices], w[indices],
        categories[indices], seeds[indices], frames[indices], used,
    )


def group_split(seeds, val_fraction, split_seed):
    unique = np.unique(seeds)
    if len(unique) < 2:
        raise RuntimeError("P29c needs at least two round seeds for group split")
    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(unique)
    val_groups = max(1, min(len(unique) - 1,
                            int(round(len(unique) * val_fraction))))
    val_seeds = set(int(value) for value in shuffled[:val_groups])
    val_mask = np.asarray([int(value) in val_seeds for value in seeds])
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask), sorted(val_seeds)


def _weighted_samples(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def train(args):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.split_seed)
    x, ys, ya, w, categories, seeds, frames, paths = load_clean_data(args)
    train_idx, val_idx, val_seeds = group_split(
        seeds, args.val_fraction, args.split_seed)

    xt = torch.as_tensor(x[train_idx])
    yst = torch.as_tensor(ys[train_idx])
    yat = torch.as_tensor(ya[train_idx])
    wt = torch.as_tensor(w[train_idx])
    xv = torch.as_tensor(x[val_idx])
    ysv = torch.as_tensor(ys[val_idx])
    yav = torch.as_tensor(ya[val_idx])
    wv = torch.as_tensor(w[val_idx])

    net = build_p27b_net(x.shape[1], args.width)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch_loss(inputs, target_score, target_aux, weights):
        output = net(inputs)
        score_per_sample = F.smooth_l1_loss(
            output["score"], target_score, reduction="none",
            beta=args.huber_beta).mean(dim=1)
        score = _weighted_samples(score_per_sample, weights)
        aux_per_sample = F.binary_cross_entropy_with_logits(
            output["aux"], target_aux, reduction="none").mean(dim=(1, 2))
        aux = _weighted_samples(aux_per_sample, weights)
        best = target_score.argmax(dim=1)
        predicted_best = output["score"].gather(1, best.unsqueeze(1))
        rank_losses = torch.relu(
            args.rank_margin - predicted_best + output["score"])
        rank_mask = torch.ones_like(rank_losses)
        rank_mask.scatter_(1, best.unsqueeze(1), 0.0)
        rank = _weighted_samples(
            (rank_losses * rank_mask).sum(dim=1) / 17.0, weights)
        target_policy = torch.softmax(
            target_score / args.policy_temperature, dim=1)
        policy = -(target_policy * torch.log_softmax(
            output["score"] / args.policy_temperature, dim=1)).sum(dim=1)
        policy = _weighted_samples(policy, weights)
        total = (score + args.aux_weight * aux
                 + args.rank_weight * rank
                 + args.policy_weight * policy)
        return total, score, aux, rank, policy, output

    def metrics():
        net.eval()
        with torch.no_grad():
            values = batch_loss(xv, ysv, yav, wv)
            output = values[-1]
            top1 = (output["score"].argmax(1)
                    == ysv.argmax(1)).float().mean().item()
            top3 = (output["score"].topk(3, dim=1).indices
                    == ysv.argmax(1, keepdim=True)).any(1).float().mean().item()
            aux_acc = ((torch.sigmoid(output["aux"]) > 0.5)
                       == (yav > 0.5)).float().mean().item()
        net.train()
        return tuple(value.item() for value in values[:-1]) + (
            top1, top3, aux_acc)

    print(f"P29c clean data: {len(paths)} shards, {len(x)} unique states, "
          f"{len(np.unique(seeds))} rounds", flush=True)
    print(f"  train={len(train_idx)} val={len(val_idx)} "
          f"val_rounds={len(val_seeds)}", flush=True)
    for category, count in sorted(Counter(categories).items()):
        print(f"  category {category}: {count}", flush=True)

    best_state = None
    best_metrics = None
    best_epoch = 0
    best_val = float("inf")
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(xt))
        train_total = 0.0
        batches = 0
        net.train()
        for start in range(0, len(xt), args.batch):
            index = order[start:start + args.batch]
            loss = batch_loss(
                xt[index], yst[index], yat[index], wt[index])[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            train_total += loss.item()
            batches += 1
        values = metrics()
        if values[0] < best_val - args.min_delta:
            best_val = values[0]
            best_epoch = epoch
            best_metrics = values
            best_state = {key: value.detach().clone()
                          for key, value in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"  epoch {epoch}/{args.epochs} "
            f"train={train_total/max(1,batches):.4f} val={values[0]:.4f} "
            f"score={values[1]:.4f} aux={values[2]:.4f} "
            f"rank={values[3]:.4f} policy={values[4]:.4f} "
            f"top1={values[5]:.1%} top3={values[6]:.1%} "
            f"aux_acc={values[7]:.1%} stale={stale} "
            f"elapsed={time.time()-started:.0f}s",
            flush=True,
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"early stop at epoch {epoch}; best={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    net.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "in_dim": int(x.shape[1]),
        "width": int(args.width),
        "phases": _csv(args.include_phases),
        "aux_names": AUX_NAMES,
        "context_dim": P29C_CONTEXT_DIM,
        "context_names": P29C_CONTEXT_NAMES,
        "objective_version": args.objective_version,
        "best_epoch": best_epoch,
        "best_val": best_val,
    }
    torch.save(payload, args.out)
    report = {
        "model": args.out,
        "objective_version": args.objective_version,
        "states": int(len(x)),
        "rounds": int(len(np.unique(seeds))),
        "train_states": int(len(train_idx)),
        "val_states": int(len(val_idx)),
        "val_round_seeds": val_seeds,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "best_metrics": {
            "total": best_metrics[0],
            "score": best_metrics[1],
            "aux": best_metrics[2],
            "rank": best_metrics[3],
            "policy": best_metrics[4],
            "top1": best_metrics[5],
            "top3": best_metrics[6],
            "aux_acc": best_metrics[7],
        },
        "categories": dict(Counter(categories)),
        "shards": paths,
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    print(f"saved {args.out}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-phases", required=True)
    parser.add_argument("--objective-version", default="p29c_p28_ensemble_v1")
    parser.add_argument("--category-weights", default=(
        "direct_shot_loss=3.5,bounce_shot_loss=3.0,self_shot_loss=3.5,"
        "finish_window=2.5,missed_fire_window=2.2,blind_fire=2.0,"
        "stutter_stall=2.4,dead_end_stall=3.0,passive_map_control=2.2,"
        "active_pursuit_gap=2.5,long_game=2.0"))
    parser.add_argument("--out", default=(
        "training/models/p29c_clean_distill_iter00.pt"))
    parser.add_argument("--report", default=(
        "training/analysis/runs/p29c_clean_train_report.json"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.22)
    parser.add_argument("--split-seed", type=int, default=29031)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--aux-weight", type=float, default=0.12)
    parser.add_argument("--rank-weight", type=float, default=0.40)
    parser.add_argument("--rank-margin", type=float, default=0.10)
    parser.add_argument("--policy-weight", type=float, default=0.28)
    parser.add_argument("--policy-temperature", type=float, default=0.35)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
