"""P30 teacher-consensus correction policy.

P30 keeps the proven P27b policy as its default action. A small network only
learns whether a high-confidence teacher would override that action, and if so
predicts replacement movement and fire decisions separately. Ambiguous teacher
states contribute risk targets but never contribute an action target.
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

from training.mpc_agent import CANDIDATES  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    AUX_DIM,
    AUX_NAMES,
    DATA_DIR,
    build_observation,
    select_action,
    stack_observation,
)
from training.p27_risk_value import (  # noqa: E402
    P27BRiskValuePolicy,
    P29C_CONTEXT_DIM,
    _controls,
)


P30_OBJECTIVE = "p30_consensus_correction_v1"
P30_ACTION_DIM = len(CANDIDATES)
P30_MOVE_DIM = P30_ACTION_DIM // 2


def _csv(value):
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _weight_map(value):
    result = {}
    for item in _csv(value):
        name, weight = item.split("=", 1)
        result[name.strip()] = float(weight)
    return result


def _scalar_text(data, key, default=""):
    if key not in data.files:
        return default
    values = np.asarray(data[key]).reshape(-1)
    return str(values[0]) if len(values) else default


def build_p30_net(in_dim, width=512):
    import torch.nn as nn

    class _P30CorrectionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width // 2), nn.ReLU(),
            )
            hidden = width // 2
            self.override = nn.Linear(hidden, 1)
            self.move = nn.Linear(hidden, P30_MOVE_DIM)
            self.fire = nn.Linear(hidden, 1)
            self.delta = nn.Linear(hidden, 1)
            self.base_aux = nn.Linear(hidden, AUX_DIM)
            self.best_aux = nn.Linear(hidden, AUX_DIM)

        def forward(self, x):
            hidden = self.trunk(x)
            return {
                "override": self.override(hidden).squeeze(-1),
                "move": self.move(hidden),
                "fire": self.fire(hidden).squeeze(-1),
                "delta": self.delta(hidden).squeeze(-1),
                "base_aux": self.base_aux(hidden),
                "best_aux": self.best_aux(hidden),
            }

    return _P30CorrectionNet()


def _load_arrays(args):
    category_weights = _weight_map(args.category_weights)
    rows = []
    paths = []
    required = {
        "X", "Y_aux", "W", "category", "round_seed", "frame", "chosen",
        "best", "action_valid", "override_target", "teacher_confidence",
        "objective_version",
    }
    for phase in _csv(args.include_phases):
        for path in sorted(glob.glob(os.path.join(
                DATA_DIR, phase, "shard_*.npz"))):
            data = np.load(path)
            if not required.issubset(data.files):
                continue
            if _scalar_text(data, "objective_version") != args.objective_version:
                continue
            context_dim = int(np.asarray(data.get(
                "context_dim", np.asarray([0]))).reshape(-1)[0])
            if context_dim != P29C_CONTEXT_DIM:
                continue
            categories = data["category"].astype(str)
            weights = np.clip(data["W"].astype(np.float32),
                              args.min_weight, args.max_weight)
            for category, multiplier in category_weights.items():
                weights[categories == category] *= multiplier
            rows.append({
                "X": data["X"].astype(np.float32),
                "Y_aux": data["Y_aux"].astype(np.float32),
                "W": weights,
                "category": categories,
                "round_seed": data["round_seed"].astype(np.int64),
                "frame": data["frame"].astype(np.int32),
                "chosen": data["chosen"].astype(np.int64),
                "best": data["best"].astype(np.int64),
                "action_valid": data["action_valid"].astype(bool),
                "override": data["override_target"].astype(bool),
                "confidence": data["teacher_confidence"].astype(np.float32),
                "regret": data["regret"].astype(np.float32),
                "source": data["teacher_source"].astype(str),
            })
            paths.append(path)
    if not rows:
        raise RuntimeError("no P30 consensus shards matched the requested data")

    joined = {
        key: np.concatenate([row[key] for row in rows])
        for key in rows[0]
    }
    # Multiple issue detectors may save the same state. Keep the most useful
    # label, preferring a valid high-confidence action over a risk-only label.
    keep = {}
    for index, key in enumerate(zip(
            joined["round_seed"].tolist(), joined["frame"].tolist())):
        rank = (
            int(joined["action_valid"][index]),
            float(joined["confidence"][index]),
            float(joined["W"][index]),
        )
        previous = keep.get(key)
        if previous is None or rank > previous[0]:
            keep[key] = (rank, index)
    indices = np.asarray(sorted(value[1] for value in keep.values()),
                         dtype=np.int64)
    for key in joined:
        joined[key] = joined[key][indices]

    one_hot = np.eye(P30_ACTION_DIM, dtype=np.float32)[joined["chosen"]]
    joined["state_dim"] = int(joined["X"].shape[1])
    joined["X"] = np.concatenate((joined["X"], one_hot), axis=1)
    action_rows = np.arange(len(joined["chosen"]))
    joined["base_aux"] = joined["Y_aux"][action_rows, joined["chosen"]]
    joined["best_aux"] = joined["Y_aux"][action_rows, joined["best"]]
    joined["move"] = joined["best"] // 2
    joined["fire"] = joined["best"] % 2
    joined["delta_target"] = np.clip(
        joined["regret"], 0.0, args.max_delta).astype(np.float32)
    return joined, paths


def _group_split(seeds, fraction, split_seed):
    unique = np.unique(seeds)
    if len(unique) < 2:
        raise RuntimeError("P30 needs at least two complete rounds")
    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(unique)
    count = max(1, min(len(unique) - 1, int(round(len(unique) * fraction))))
    validation = set(int(value) for value in shuffled[:count])
    mask = np.asarray([int(value) in validation for value in seeds])
    return np.flatnonzero(~mask), np.flatnonzero(mask), sorted(validation)


def _weighted(values, weights, mask=None):
    if mask is not None:
        values = values[mask]
        weights = weights[mask]
    if len(values) == 0:
        return values.sum() * 0.0
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def train(args):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.split_seed)
    data, paths = _load_arrays(args)
    train_index, val_index, val_seeds = _group_split(
        data["round_seed"], args.val_fraction, args.split_seed)
    tensors = {
        key: torch.as_tensor(value)
        for key, value in data.items()
        if isinstance(value, np.ndarray) and value.dtype.kind not in "USO"
    }
    net = build_p30_net(data["X"].shape[1], args.width)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_valid = data["action_valid"][train_index]
    train_override = data["override"][train_index] & train_valid
    positives = max(1, int(train_override.sum()))
    negatives = max(1, int(train_valid.sum()) - positives)
    positive_scale = float(np.clip(
        negatives / positives, args.min_positive_scale,
        args.max_positive_scale))

    def loss_for(index):
        x = tensors["X"][index]
        weight = tensors["W"][index]
        valid = tensors["action_valid"][index].bool()
        override = tensors["override"][index].float()
        override_mask = valid & tensors["override"][index].bool()
        output = net(x)

        override_loss = F.binary_cross_entropy_with_logits(
            output["override"], override, reduction="none")
        class_weight = torch.where(
            override > 0.5,
            torch.full_like(weight, positive_scale),
            torch.ones_like(weight))
        override_loss = _weighted(
            override_loss, weight * class_weight, valid)
        move_loss = _weighted(F.cross_entropy(
            output["move"], tensors["move"][index], reduction="none"),
            weight, override_mask)
        fire_loss = _weighted(F.binary_cross_entropy_with_logits(
            output["fire"], tensors["fire"][index].float(),
            reduction="none"), weight, override_mask)
        delta_loss = _weighted(F.smooth_l1_loss(
            output["delta"], tensors["delta_target"][index],
            reduction="none", beta=args.huber_beta), weight, valid)
        base_aux_loss = _weighted(F.binary_cross_entropy_with_logits(
            output["base_aux"], tensors["base_aux"][index],
            reduction="none").mean(dim=1), weight)
        best_aux_loss = _weighted(F.binary_cross_entropy_with_logits(
            output["best_aux"], tensors["best_aux"][index],
            reduction="none").mean(dim=1), weight, valid)
        total = (
            args.override_weight * override_loss
            + args.move_weight * move_loss
            + args.fire_weight * fire_loss
            + args.delta_weight * delta_loss
            + args.base_aux_weight * base_aux_loss
            + args.best_aux_weight * best_aux_loss
        )
        return total, {
            "override": override_loss,
            "move": move_loss,
            "fire": fire_loss,
            "delta": delta_loss,
            "base_aux": base_aux_loss,
            "best_aux": best_aux_loss,
        }, output

    def metrics():
        net.eval()
        index = torch.as_tensor(val_index)
        with torch.no_grad():
            total, losses, output = loss_for(index)
            valid = tensors["action_valid"][index].bool()
            target_override = tensors["override"][index].bool()
            predicted_override = torch.sigmoid(
                output["override"]) >= args.metric_override_threshold
            correct_override = (predicted_override == target_override)[valid]
            true_positive = (predicted_override & target_override & valid).sum()
            predicted_positive = (predicted_override & valid).sum()
            actual_positive = (target_override & valid).sum()
            override_rows = valid & target_override
            move_correct = (
                output["move"].argmax(1) == tensors["move"][index])
            fire_correct = ((torch.sigmoid(output["fire"]) >= 0.5)
                            == tensors["fire"][index].bool())
            predicted_action = output["move"].argmax(1) * 2 + (
                torch.sigmoid(output["fire"]) >= 0.5).long()
            deployed_action = torch.where(
                predicted_override, predicted_action,
                tensors["chosen"][index])
            action_correct = (deployed_action == tensors["best"][index])[valid]
            result = {
                "total": float(total),
                **{name: float(value) for name, value in losses.items()},
                "override_accuracy": float(correct_override.float().mean())
                if len(correct_override) else 0.0,
                "override_precision": float(
                    true_positive / predicted_positive.clamp_min(1)),
                "override_recall": float(
                    true_positive / actual_positive.clamp_min(1)),
                "move_accuracy": float(move_correct[override_rows].float().mean())
                if override_rows.any() else 0.0,
                "fire_accuracy": float(fire_correct[override_rows].float().mean())
                if override_rows.any() else 0.0,
                "deployed_action_accuracy": float(action_correct.float().mean())
                if len(action_correct) else 0.0,
            }
        net.train()
        return result

    print(f"P30 data: {len(paths)} shards, {len(data['X'])} unique states, "
          f"{len(np.unique(data['round_seed']))} rounds", flush=True)
    print(f"  train={len(train_index)} val={len(val_index)} "
          f"valid={data['action_valid'].mean():.1%} "
          f"override={data['override'][data['action_valid']].mean():.1%} "
          f"positive_scale={positive_scale:.2f}", flush=True)
    for source, count in sorted(Counter(data["source"]).items()):
        print(f"  teacher {source}: {count}", flush=True)
    for category, count in sorted(Counter(data["category"]).items()):
        print(f"  category {category}: {count}", flush=True)

    best_state = None
    best_metrics = None
    best_epoch = 0
    best_value = float("inf")
    stale = 0
    started = time.time()
    train_tensor = torch.as_tensor(train_index)
    for epoch in range(1, args.epochs + 1):
        order = train_tensor[torch.randperm(len(train_tensor))]
        running = 0.0
        batches = 0
        net.train()
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            total, _, _ = loss_for(index)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            running += float(total.detach())
            batches += 1
        values = metrics()
        if values["total"] < best_value - args.min_delta:
            best_value = values["total"]
            best_epoch = epoch
            best_metrics = values
            best_state = {
                key: value.detach().clone()
                for key, value in net.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"  epoch {epoch}/{args.epochs} "
            f"train={running/max(1,batches):.4f} "
            f"val={values['total']:.4f} "
            f"override={values['override_accuracy']:.1%} "
            f"P/R={values['override_precision']:.1%}/"
            f"{values['override_recall']:.1%} "
            f"move={values['move_accuracy']:.1%} "
            f"fire={values['fire_accuracy']:.1%} "
            f"deployed={values['deployed_action_accuracy']:.1%} "
            f"stale={stale} elapsed={time.time()-started:.0f}s",
            flush=True,
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"early stop at {epoch}; best={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("P30 training produced no checkpoint")
    net.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "in_dim": int(data["X"].shape[1]),
        "state_dim": int(data["state_dim"]),
        "width": int(args.width),
        "objective_version": args.objective_version,
        "phases": _csv(args.include_phases),
        "aux_names": AUX_NAMES,
        "best_epoch": best_epoch,
        "best_val": best_value,
    }
    torch.save(payload, args.out)
    report = {
        "model": args.out,
        "objective_version": args.objective_version,
        "states": int(len(data["X"])),
        "rounds": int(len(np.unique(data["round_seed"]))),
        "action_valid_rate": float(data["action_valid"].mean()),
        "override_rate": float(
            data["override"][data["action_valid"]].mean()),
        "train_states": int(len(train_index)),
        "val_states": int(len(val_index)),
        "val_round_seeds": val_seeds,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "teacher_sources": dict(Counter(data["source"])),
        "categories": dict(Counter(data["category"])),
        "shards": paths,
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    print(f"saved {args.out}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def load_p30_model(path):
    import torch

    payload = torch.load(path, weights_only=False)
    net = build_p30_net(int(payload["in_dim"]), int(payload.get("width", 512)))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


class P30CorrectionPolicy(P27BRiskValuePolicy):
    name = "p30_consensus_correction"

    def __init__(self, base_net, value_net, correction_net,
                 override_threshold=0.72,
                 background_override_threshold=0.84,
                 min_predicted_gain=0.03,
                 max_override_death=0.55,
                 max_override_dd=0.50,
                 **kwargs):
        super().__init__(base_net=base_net, value_net=value_net, **kwargs)
        self.correction_net, payload = load_p30_model(correction_net)
        self.correction_state_dim = int(payload["state_dim"])
        self.override_threshold = float(override_threshold)
        self.background_override_threshold = float(
            background_override_threshold)
        self.min_predicted_gain = float(min_predicted_gain)
        self.max_override_death = float(max_override_death)
        self.max_override_dd = float(max_override_dd)
        self.correction_counts = {}

    def reset(self):
        super().reset()
        self.correction_counts = {}

    def _correction_count(self, name):
        self.correction_counts[name] = self.correction_counts.get(name, 0) + 1

    def _correction_input(self, stacked, context, action_index):
        state = np.asarray(stacked, dtype=np.float32)
        if self.correction_state_dim == len(state) + P29C_CONTEXT_DIM:
            state = np.concatenate((state, np.asarray(context, np.float32)))
        if len(state) != self.correction_state_dim:
            return None
        one_hot = np.zeros(P30_ACTION_DIM, dtype=np.float32)
        one_hot[action_index] = 1.0
        return np.concatenate((state, one_hot))

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            from training.opportunity_teacher_v2 import OpportunityAnalyzer360
            self.analyzer = OpportunityAnalyzer360(game)
            self.frames = 0
            self.history = []
            self.pos_window.clear()
            self.input_window.clear()
            self.clear_fire_frames = 0
            self.context_positions.clear()
            self.context_distances.clear()
            self.last_context.fill(0.0)

        observation, metrics = build_observation(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        self.history.append(observation)
        stacked = stack_observation(self.history, self.frame_stack)
        with self.torch.no_grad():
            base_output = self.base_net(
                self.torch.as_tensor(stacked).unsqueeze(0))
        outputs = {
            "score": base_output["score"][0].numpy(),
            "aux": base_output["aux"][0].numpy(),
            "fire": base_output["fire"][0].numpy(),
        }
        base_action = select_action(
            outputs, self.candidates, self.fire_margin,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        base_index = self.candidates.index(base_action)
        category = self._detect_category(game, _controls(base_action), metrics)
        context = self._update_context(game, metrics)
        p27 = self._p27_value(stacked, context)
        adjusted = self._adjust_outputs(
            outputs, category, p27, base_index, metrics)
        p27_action = select_action(
            adjusted, self.candidates, self.fire_margin,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        p27_index = self.candidates.index(p27_action)
        action = p27_action

        correction_input = self._correction_input(
            stacked, context, p27_index)
        if correction_input is not None:
            with self.torch.no_grad():
                correction = self.correction_net(
                    self.torch.as_tensor(correction_input).unsqueeze(0))
            probability = float(self.torch.sigmoid(
                correction["override"][0]))
            predicted_gain = float(correction["delta"][0])
            predicted_move = int(correction["move"][0].argmax())
            predicted_fire = int(self.torch.sigmoid(
                correction["fire"][0]) >= 0.5)
            predicted_index = predicted_move * 2 + predicted_fire
            best_aux = self.torch.sigmoid(
                correction["best_aux"][0]).numpy()
            threshold = (self.override_threshold if category
                         else self.background_override_threshold)
            if probability >= threshold:
                self._correction_count("triggered")
                if predicted_index == p27_index:
                    self._correction_count("same_action")
                elif predicted_gain < self.min_predicted_gain:
                    self._correction_count("low_gain")
                elif best_aux[1] > self.max_override_death:
                    self._correction_count("death_gate")
                elif best_aux[2] > self.max_override_dd:
                    self._correction_count("dd_gate")
                else:
                    action = CANDIDATES[predicted_index]
                    self._correction_count("accepted")
                    self._correction_count(category or "background_state")

        throttle, turn, fire = action
        if len(game.tanks) > 1 and not game.tanks[1].alive:
            fire = 0
        return {
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": fire == 1,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train"], nargs="?", default="train")
    parser.add_argument("--include-phases", required=True)
    parser.add_argument("--objective-version", default=P30_OBJECTIVE)
    parser.add_argument("--category-weights", default=(
        "direct_shot_loss=1.8,bounce_shot_loss=1.5,self_shot_loss=1.8,"
        "finish_window=1.4,missed_fire_window=1.3,blind_fire=1.2,"
        "stutter_stall=1.5,dead_end_stall=1.8,passive_map_control=1.4,"
        "active_pursuit_gap=1.5,long_game=1.2,background_state=0.8"))
    parser.add_argument("--out", default=(
        "training/models/p30_consensus_correction_iter00.pt"))
    parser.add_argument("--report", default=(
        "training/analysis/runs/p30_consensus_train_report.json"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.22)
    parser.add_argument("--split-seed", type=int, default=3001)
    parser.add_argument("--min-weight", type=float, default=0.25)
    parser.add_argument("--max-weight", type=float, default=8.0)
    parser.add_argument("--max-delta", type=float, default=1.5)
    parser.add_argument("--min-positive-scale", type=float, default=0.7)
    parser.add_argument("--max-positive-scale", type=float, default=4.0)
    parser.add_argument("--override-weight", type=float, default=1.0)
    parser.add_argument("--move-weight", type=float, default=0.8)
    parser.add_argument("--fire-weight", type=float, default=0.45)
    parser.add_argument("--delta-weight", type=float, default=0.25)
    parser.add_argument("--base-aux-weight", type=float, default=0.22)
    parser.add_argument("--best-aux-weight", type=float, default=0.22)
    parser.add_argument("--metric-override-threshold", type=float,
                        default=0.5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
