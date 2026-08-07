"""
P27b risk/value assist head.

This is intentionally not a replacement policy. It keeps the current P26
champion as the base policy and only adds small hard-state score nudges from a
separate value/risk head trained on stronger teacher labels.
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.maze import h_open, v_open  # noqa: E402
from training.evaluate import play_round_dual_engine  # noqa: E402
from training.mpc_agent import CANDIDATES  # noqa: E402
from training.opportunity_distill import _shot_event  # noqa: E402
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    AUX_DIM,
    AUX_NAMES,
    DATA_DIR,
    MODELS_DIR,
    build_observation,
    fire_targets,
    load_p26_network,
    select_action,
    stack_observation,
)

DEFAULT_PHASES = (
    "p26v9_strong_h72s3_970000",
    "p26v9_strong_h72s3_990000",
)
HARD_CATEGORIES = {
    "missed_fire_window",
    "missed_kill_line",
    "unsafe_fire_death",
    "double_death_risk",
    "fire_into_double_death",
    "waste_or_unsafe_fire",
    "unsafe_movement",
    "movement_value_gap",
    "stutter_stall",
    "dead_end_stall",
    "passive_map_control",
    "blind_fire",
}

P29C_CONTEXT_NAMES = (
    "move_0p5s",
    "move_1s",
    "move_2s",
    "enemy_distance",
    "approach_0p5s",
    "approach_1s",
    "open_exits",
    "dead_end_depth",
    "clear_fire_persistence",
    "direct_line",
    "reachable_line",
    "incoming_risk",
    "round_progress",
)
P29C_CONTEXT_DIM = len(P29C_CONTEXT_NAMES)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _input_tuple(inp):
    return (
        bool(inp.get("forward", False)),
        bool(inp.get("backup", False)),
        bool(inp.get("turn_left", False)),
        bool(inp.get("turn_right", False)),
        bool(inp.get("fire", False)),
    )


def _action_tuple(inp):
    return (
        2 if inp.get("forward") else 0 if inp.get("backup") else 1,
        0 if inp.get("turn_left") else 2 if inp.get("turn_right") else 1,
        1 if inp.get("fire") else 0,
    )


def _controls(action):
    throttle, turn, fire = action
    return {
        "forward": throttle == 2,
        "backup": throttle == 0,
        "turn_left": turn == 0,
        "turn_right": turn == 2,
        "fire": fire == 1,
    }


def _cell(game, tank):
    x = max(0, min(len(game.maze) - 1, int(tank.x // game.scale)))
    y = max(0, min(len(game.maze[0]) - 1, int(tank.y // game.scale)))
    return x, y


def _open_neighbors(game, x, y):
    width, height = len(game.maze), len(game.maze[0])
    count = 0
    if x > 0 and v_open(game.maze, x, y):
        count += 1
    if x < width - 1 and v_open(game.maze, x + 1, y):
        count += 1
    if y > 0 and h_open(game.maze, x, y - 1):
        count += 1
    if y < height - 1 and h_open(game.maze, x, y):
        count += 1
    return count


def _dead_end_penalty(game, x, y):
    if not getattr(game, "dead_ends", None):
        return 0.0
    value = game.dead_ends[x][y]
    return 0.0 if value is None else float(value)


def build_p27b_net(in_dim, width=512):
    import torch.nn as nn

    class _P27BNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
            )
            self.score = nn.Linear(width, 18)
            self.aux = nn.Linear(width, 18 * AUX_DIM)

        def forward(self, x):
            h = self.trunk(x)
            return {
                "score": self.score(h),
                "aux": self.aux(h).view(-1, 18, AUX_DIM),
            }

    return _P27BNet()


def _csv_items(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _weight_map(value):
    weights = {}
    for item in _csv_items(value):
        if "=" not in item:
            raise ValueError(f"expected name=weight, got {item!r}")
        name, weight = item.split("=", 1)
        weights[name.strip()] = float(weight)
    return weights


def load_value_data(phases, category_weights=None, phase_weights=None):
    phase_weights = _weight_map(phase_weights)
    category_weights = _weight_map(category_weights)
    xs, ys, ya, weights, cats, used = [], [], [], [], [], []
    for phase in phases:
        paths = sorted(glob.glob(os.path.join(DATA_DIR, phase, "shard_*.npz")))
        for path in paths:
            data = np.load(path)
            if "Y_score" not in data.files or "Y_aux" not in data.files:
                continue
            if data["Y_aux"].shape[2] != AUX_DIM:
                continue
            x = data["X"].astype(np.float32)
            y_score = data["Y_score"].astype(np.float32)
            y_aux = data["Y_aux"].astype(np.float32)
            w = data["W"].astype(np.float32) if "W" in data.files else np.ones(len(x), np.float32)
            c = (data["category"].astype(str)
                 if "category" in data.files
                 else np.asarray(["unknown"] * len(x)))
            w = w * phase_weights.get(phase, 1.0)
            for category, multiplier in category_weights.items():
                w[c == category] *= multiplier
            xs.append(x)
            ys.append(y_score)
            ya.append(y_aux)
            weights.append(w)
            cats.append(c)
            used.append(path)
    if not xs:
        raise RuntimeError("no P27b value data found")
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ya),
        np.concatenate(weights),
        np.concatenate(cats),
        used,
    )


def _weighted_mean(loss, weights):
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _rank_loss(pred, target, weights, margin):
    import torch

    best = target.argmax(dim=1)
    pred_best = pred.gather(1, best.unsqueeze(1))
    losses = torch.relu(margin - pred_best + pred)
    mask = torch.ones_like(losses)
    mask.scatter_(1, best.unsqueeze(1), 0.0)
    return _weighted_mean(losses * mask, weights)


def save_model(net, path, in_dim, width, phases):
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "in_dim": int(in_dim),
        "width": int(width),
        "phases": list(phases),
        "aux_names": AUX_NAMES,
    }, path)
    print(f"saved {path}", flush=True)


def load_model(path):
    import torch

    payload = torch.load(path, weights_only=False)
    in_dim = int(payload["in_dim"])
    width = int(payload.get("width", 512))
    net = build_p27b_net(in_dim, width)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, in_dim


def train(args):
    import torch
    import torch.nn.functional as F

    phases = _csv_items(args.include_phases) or list(DEFAULT_PHASES)
    X, Y_score, Y_aux, W, categories, paths = load_value_data(
        phases,
        category_weights=args.category_weights,
        phase_weights=args.phase_weights,
    )
    n = len(X)
    n_val = max(1, min(n - 1, int(n * args.val_frac)))
    rng = np.random.default_rng(2701)
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xt = torch.as_tensor(X[train_idx])
    Yst = torch.as_tensor(Y_score[train_idx])
    Yat = torch.as_tensor(Y_aux[train_idx])
    Wt = torch.as_tensor(W[train_idx])
    Xv = torch.as_tensor(X[val_idx])
    Ysv = torch.as_tensor(Y_score[val_idx])
    Yav = torch.as_tensor(Y_aux[val_idx])
    Wv = torch.as_tensor(W[val_idx])

    net = build_p27b_net(X.shape[1], args.width)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def batch_loss(x, ys, ya, w):
        out = net(x)
        score = _weighted_mean(
            F.mse_loss(out["score"], ys, reduction="none"), w)
        aux = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                out["aux"], ya, reduction="none"), w)
        rank = _rank_loss(out["score"], ys, w, args.rank_margin)
        return score + args.aux_weight * aux + args.rank_weight * rank, score, aux, rank, out

    def metrics():
        with torch.no_grad():
            loss, score, aux, rank, out = batch_loss(Xv, Ysv, Yav, Wv)
            top1 = (out["score"].argmax(1) == Ysv.argmax(1)).float().mean().item()
            top3 = (out["score"].topk(3, dim=1).indices
                    == Ysv.argmax(1, keepdim=True)).any(1).float().mean().item()
            aux_acc = ((torch.sigmoid(out["aux"]) > 0.5)
                       == (Yav > 0.5)).float().mean().item()
        return loss.item(), score.item(), aux.item(), rank.item(), top1, top3, aux_acc

    print(f"P27b data: {len(paths)} shards -> {n} samples", flush=True)
    for category, count in sorted(Counter(categories).items()):
        print(f"  category {category}: {count}", flush=True)
    best_state, best_val = None, float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(Xt))
        total, batches = 0.0, 0
        for start in range(0, len(Xt), args.batch):
            idx = order[start:start + args.batch]
            loss, *_ = batch_loss(Xt[idx], Yst[idx], Yat[idx], Wt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        vals = metrics()
        if vals[0] < best_val:
            best_val = vals[0]
            best_state = {k: v.detach().clone()
                          for k, v in net.state_dict().items()}
        print(f"  epoch {epoch+1}/{args.epochs} "
              f"train {total/max(1,batches):.4f} val {vals[0]:.4f} "
              f"score {vals[1]:.4f} aux {vals[2]:.4f} rank {vals[3]:.4f} "
              f"top1 {vals[4]:.1%} top3 {vals[5]:.1%} "
              f"aux_acc {vals[6]:.1%} {time.time()-t0:.0f}s",
              flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)
    save_model(net, args.out, X.shape[1], args.width, phases)
    return args.out


class P27BRiskValuePolicy:
    name = "p27b_risk_value"

    def __init__(self, base_net, value_net, fire_margin=0.16,
                 assist_margin=0.08, assist_weight=0.35, max_bonus=0.10,
                 kill_weight=0.04, death_weight=0.12,
                 double_death_weight=0.18, survive_weight=0.02,
                 risk_threshold=0.55, fire_delta_margin=0.14,
                 global_fire_risk_penalty=0.0,
                 global_fire_risk_threshold=1.10,
                 global_fire_dd_threshold=1.10,
                 low_quality_fire_penalty=0.0,
                 low_quality_fire_delta=0.0,
                 opportunity_bonus=0.0,
                 opportunity_min_line=0.76,
                 opportunity_max_risk=0.25,
                 opportunity_max_danger=0.45,
                 opportunity_min_fire_delta=0.08,
                 escape_bonus=0.0,
                 escape_min_gain=-0.02,
                 escape_max_danger=0.55,
                 escape_hold_frames=0,
                 stall_fire_penalty=0.0,
                 stall_window=40, stall_distance=0.22,
                 fire_window_line=0.70, fire_window_frames=8,
                 blind_fire_line=0.35, pressure_radius=0.75):
        import torch
        from training.tt_gym_env import TankTroubleGym

        self.torch = torch
        self.candidates = CANDIDATES
        self.base_net, self.frame_stack = load_p26_network(base_net)
        self.value_net, self.value_in_dim = load_model(value_net)
        self.fire_margin = float(fire_margin)
        self.assist_margin = float(assist_margin)
        self.assist_weight = float(assist_weight)
        self.max_bonus = float(max_bonus)
        self.kill_weight = float(kill_weight)
        self.death_weight = float(death_weight)
        self.double_death_weight = float(double_death_weight)
        self.survive_weight = float(survive_weight)
        self.risk_threshold = float(risk_threshold)
        self.fire_delta_margin = float(fire_delta_margin)
        self.global_fire_risk_penalty = float(global_fire_risk_penalty)
        self.global_fire_risk_threshold = float(global_fire_risk_threshold)
        self.global_fire_dd_threshold = float(global_fire_dd_threshold)
        self.low_quality_fire_penalty = float(low_quality_fire_penalty)
        self.low_quality_fire_delta = float(low_quality_fire_delta)
        self.opportunity_bonus = float(opportunity_bonus)
        self.opportunity_min_line = float(opportunity_min_line)
        self.opportunity_max_risk = float(opportunity_max_risk)
        self.opportunity_max_danger = float(opportunity_max_danger)
        self.opportunity_min_fire_delta = float(opportunity_min_fire_delta)
        self.escape_bonus = float(escape_bonus)
        self.escape_min_gain = float(escape_min_gain)
        self.escape_max_danger = float(escape_max_danger)
        self.escape_hold_frames = int(escape_hold_frames)
        self.stall_fire_penalty = float(stall_fire_penalty)
        self.stall_window = int(stall_window)
        self.stall_distance = float(stall_distance)
        self.fire_window_line = float(fire_window_line)
        self.fire_window_frames = int(fire_window_frames)
        self.blind_fire_line = float(blind_fire_line)
        self.pressure_radius = float(pressure_radius)
        self.env = TankTroubleGym(seed=0, reward_version=1,
                                  obs_traj=True, obs_nav=True)
        self.reset()

    def reset(self):
        self.game = None
        self.analyzer = None
        self.frames = 0
        self.history = []
        self.pos_window = deque(maxlen=self.stall_window)
        self.input_window = deque(maxlen=self.stall_window)
        self.clear_fire_frames = 0
        self.escape_frames_remaining = 0
        self.active_escape_category = None
        self.assist_counts = {}
        self.context_positions = deque(maxlen=51)
        self.context_distances = deque(maxlen=51)
        self.last_context = np.zeros(P29C_CONTEXT_DIM, dtype=np.float32)

    def _count(self, name):
        self.assist_counts[name] = self.assist_counts.get(name, 0) + 1

    def _detect_category(self, game, inp, metrics):
        cmd = _input_tuple(inp)
        me, enemy = game.tanks[0], game.tanks[1]
        line, reach, risk = [float(value) for value in metrics[:3]]
        self.pos_window.append((me.x, me.y))
        self.input_window.append(cmd)

        if cmd[4] and enemy.alive:
            shot = _shot_event(game)
            closest = float("inf") if shot is None else shot.get(
                "closest", float("inf"))
            result = None if shot is None else shot.get("result")
            if (line < self.blind_fire_line and result != "HIT"
                    and closest > self.pressure_radius * game.scale):
                return "blind_fire"

        if (not cmd[4]) and enemy.alive and line >= self.fire_window_line:
            self.clear_fire_frames += 1
            if self.clear_fire_frames >= self.fire_window_frames:
                self.clear_fire_frames = 0
                return "missed_fire_window"
        else:
            self.clear_fire_frames = 0

        if len(self.pos_window) == self.stall_window:
            dx = self.pos_window[-1][0] - self.pos_window[0][0]
            dy = self.pos_window[-1][1] - self.pos_window[0][1]
            displacement = math.hypot(dx, dy)
            moving_cmds = sum(any(command[:4])
                              for command in self.input_window)
            x, y = _cell(game, me)
            dead_end = _dead_end_penalty(game, x, y)
            exits = _open_neighbors(game, x, y)
            stalled = displacement < self.stall_distance * game.scale
            if stalled:
                self.pos_window.clear()
                self.input_window.clear()
                if dead_end > 0 or exits <= 1:
                    return "dead_end_stall"
                if line < 0.35 and reach < 0.55 and risk < 0.35:
                    return "passive_map_control"
                if moving_cmds >= self.stall_window // 4:
                    return "stutter_stall"
        return None

    @staticmethod
    def _history_delta(values, lag):
        if len(values) < 2:
            return 0.0
        start = max(0, len(values) - 1 - lag)
        return float(values[start] - values[-1])

    @staticmethod
    def _history_move(values, lag):
        if len(values) < 2:
            return 0.0
        start = max(0, len(values) - 1 - lag)
        dx = float(values[-1][0] - values[start][0])
        dy = float(values[-1][1] - values[start][1])
        return math.hypot(dx, dy)

    def _update_context(self, game, metrics):
        me, enemy = game.tanks[0], game.tanks[1]
        self.context_positions.append((float(me.x), float(me.y)))
        distance = math.hypot(float(enemy.x - me.x), float(enemy.y - me.y))
        self.context_distances.append(distance)
        scale = max(1.0, float(game.scale))
        x, y = _cell(game, me)
        line, reach, risk = [float(value) for value in metrics[:3]]
        self.last_context = np.asarray([
            min(4.0, self._history_move(self.context_positions, 12) / scale),
            min(6.0, self._history_move(self.context_positions, 25) / scale),
            min(10.0, self._history_move(self.context_positions, 50) / scale),
            min(2.0, distance / (10.0 * scale)),
            np.clip(self._history_delta(self.context_distances, 12)
                    / (4.0 * scale), -1.0, 1.0),
            np.clip(self._history_delta(self.context_distances, 25)
                    / (6.0 * scale), -1.0, 1.0),
            _open_neighbors(game, x, y) / 4.0,
            min(1.0, _dead_end_penalty(game, x, y) / 4.0),
            min(1.0, self.clear_fire_frames
                / max(1.0, float(self.fire_window_frames))),
            line,
            reach,
            risk,
            min(1.0, self.frames / 2500.0),
        ], dtype=np.float32)
        return self.last_context

    def _p27_value(self, stacked, context=None):
        value_input = np.asarray(stacked, dtype=np.float32)
        if self.value_in_dim == len(value_input) + P29C_CONTEXT_DIM:
            if context is None:
                context = self.last_context
            value_input = np.concatenate((
                value_input, np.asarray(context, dtype=np.float32)))
        if len(value_input) != self.value_in_dim:
            return None
        with self.torch.no_grad():
            out = self.value_net(
                self.torch.as_tensor(value_input).unsqueeze(0))
        score = out["score"][0].numpy()
        aux = _sigmoid(out["aux"][0].numpy())
        value = (score
                 + self.kill_weight * aux[:, 0]
                 - self.death_weight * aux[:, 1]
                 - self.double_death_weight * aux[:, 2]
                 + self.survive_weight * aux[:, 4]
                 + self.survive_weight * aux[:, 5])
        return score, aux, value

    def _fire_deltas(self, p27_score):
        return p27_score.reshape(9, 2)[:, 1] - p27_score.reshape(9, 2)[:, 0]

    def _apply_global_safety(self, score, p27_score, p27_aux, metrics):
        line, _, risk = [float(value) for value in metrics[:3]]
        fire_mask = np.asarray([action[2] == 1 for action in self.candidates])
        if self.global_fire_risk_penalty > 0.0:
            danger = np.maximum(p27_aux[:, 1], p27_aux[:, 2])
            risky_fire = fire_mask & (
                (p27_aux[:, 1] > self.global_fire_risk_threshold)
                | (p27_aux[:, 2] > self.global_fire_dd_threshold))
            if np.any(risky_fire):
                multiplier = 1.0 + max(0.0, risk - 0.30)
                score[risky_fire] -= self.global_fire_risk_penalty * multiplier
                self._count("global_risky_fire_penalty")
            very_risky_fire = fire_mask & (danger > max(
                self.global_fire_risk_threshold, self.global_fire_dd_threshold))
            if np.any(very_risky_fire):
                score[very_risky_fire] -= self.global_fire_risk_penalty
                self._count("global_very_risky_fire_penalty")

        if self.low_quality_fire_penalty > 0.0:
            deltas = self._fire_deltas(p27_score)
            low_quality = np.zeros(len(self.candidates), dtype=bool)
            for movement, delta in enumerate(deltas):
                if delta < self.low_quality_fire_delta and line < 0.70:
                    low_quality[movement * 2 + 1] = True
            if np.any(low_quality):
                score[low_quality] -= self.low_quality_fire_penalty
                self._count("low_quality_fire_penalty")
        return score

    def _best_safe_fire(self, p27_score, p27_aux, p27_value, metrics):
        line, _, risk = [float(value) for value in metrics[:3]]
        if line < self.opportunity_min_line or risk > self.opportunity_max_risk:
            return None
        best_index, best_value = None, -float("inf")
        deltas = self._fire_deltas(p27_score)
        for movement, delta in enumerate(deltas):
            index = movement * 2 + 1
            danger = max(float(p27_aux[index, 1]), float(p27_aux[index, 2]))
            if danger > self.opportunity_max_danger:
                continue
            if float(delta) < self.opportunity_min_fire_delta:
                continue
            value = float(p27_value[index])
            if value > best_value:
                best_value = value
                best_index = index
        return best_index

    def _best_escape_move(self, p27_aux, p27_value, base_value):
        best_index, best_gain = None, -float("inf")
        for index, action in enumerate(self.candidates):
            if action[2] == 1:
                continue
            if action[0] == 1 and action[1] == 1:
                continue
            danger = max(float(p27_aux[index, 1]), float(p27_aux[index, 2]))
            if danger > self.escape_max_danger:
                continue
            gain = float(p27_value[index] - base_value)
            if gain < self.escape_min_gain:
                continue
            if gain > best_gain:
                best_gain = gain
                best_index = index
        return best_index, best_gain

    def _adjust_outputs(self, outputs, category, p27, base_index, metrics):
        if p27 is None:
            return outputs
        p27_score, p27_aux, p27_value = p27
        adjusted = dict(outputs)
        score = outputs["score"].copy()
        score = self._apply_global_safety(score, p27_score, p27_aux, metrics)
        if category not in HARD_CATEGORIES:
            adjusted["score"] = score
            adjusted["aux"] = outputs["aux"]
            adjusted["fire"] = outputs["fire"]
            return adjusted
        line, _, risk = [float(value) for value in metrics[:3]]
        base_value = float(p27_value[base_index])
        order = np.argsort(p27_value)[::-1]

        if category in ("dead_end_stall", "stutter_stall",
                        "passive_map_control"):
            if self.stall_fire_penalty > 0.0:
                fire_mask = np.asarray([
                    action[2] == 1 for action in self.candidates])
                score[fire_mask] -= self.stall_fire_penalty
                self._count("stall_fire_penalty")
            if self.escape_bonus > 0.0:
                escape_index, escape_gain = self._best_escape_move(
                    p27_aux, p27_value, base_value)
                if escape_index is not None:
                    score[escape_index] += self.escape_bonus + max(
                        0.0, escape_gain) * self.assist_weight
                    self._count(f"{category}_escape_bonus")

        if category in ("missed_fire_window", "missed_kill_line"):
            fire_index = self._best_safe_fire(
                p27_score, p27_aux, p27_value, metrics)
            if fire_index is not None and self.opportunity_bonus > 0.0:
                score[fire_index] += self.opportunity_bonus
                self._count("safe_opportunity_fire_bonus")

        for index in order:
            index = int(index)
            action = self.candidates[index]
            fire = action[2] == 1
            movement = index // 2
            fire_delta = float(p27_score[movement * 2 + 1]
                               - p27_score[movement * 2])
            danger = max(float(p27_aux[index, 1]), float(p27_aux[index, 2]))
            gain = float(p27_value[index] - base_value)
            if gain < self.assist_margin:
                break
            if category in ("dead_end_stall", "stutter_stall",
                            "passive_map_control", "unsafe_movement",
                            "movement_value_gap"):
                if fire:
                    continue
            if category in ("missed_fire_window", "missed_kill_line"):
                if fire:
                    if (fire_delta < self.fire_delta_margin
                            or danger > self.risk_threshold
                            or line < self.fire_window_line):
                        continue
                elif line >= self.fire_window_line and risk < 0.30:
                    continue
            if category in ("blind_fire", "waste_or_unsafe_fire",
                            "unsafe_fire_death", "double_death_risk",
                            "fire_into_double_death"):
                if fire or danger > self.risk_threshold:
                    continue
            bonus = min(self.max_bonus, gain * self.assist_weight)
            score[index] += bonus
            self._count(category)
            break

        if category in ("blind_fire", "waste_or_unsafe_fire",
                        "unsafe_fire_death", "double_death_risk",
                        "fire_into_double_death"):
            risky_fire = ((p27_aux[:, 1] > self.risk_threshold)
                          | (p27_aux[:, 2] > self.risk_threshold))
            fire_mask = np.asarray([a[2] == 1 for a in self.candidates])
            score[fire_mask & risky_fire] -= self.max_bonus
            self._count("risk_fire_penalty")

        adjusted["score"] = score
        adjusted["aux"] = outputs["aux"]
        adjusted["fire"] = outputs["fire"]
        return adjusted

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
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
            out = self.base_net(self.torch.as_tensor(stacked).unsqueeze(0))
        outputs = {
            "score": out["score"][0].numpy(),
            "aux": out["aux"][0].numpy(),
            "fire": out["fire"][0].numpy(),
        }
        base_action = select_action(
            outputs, self.candidates, self.fire_margin, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0)
        base_index = self.candidates.index(base_action)
        base_inp = _controls(base_action)
        category = self._detect_category(game, base_inp, metrics)
        if category in ("dead_end_stall", "stutter_stall",
                        "passive_map_control"):
            self.escape_frames_remaining = max(
                self.escape_frames_remaining, self.escape_hold_frames)
            self.active_escape_category = category
        elif self.escape_frames_remaining > 0:
            category = self.active_escape_category
            self.escape_frames_remaining -= 1
        else:
            self.active_escape_category = None
        context = self._update_context(game, metrics)
        p27 = self._p27_value(stacked, context)
        outputs = self._adjust_outputs(
            outputs, category, p27, base_index, metrics)
        throttle, turn, fire = select_action(
            outputs, self.candidates, self.fire_margin, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0)
        if len(game.tanks) > 1 and not game.tanks[1].alive:
            fire = 0
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def _eval_worker(job):
    worker, base_net, value_net, seed, count, args = job
    import torch

    torch.set_num_threads(1)
    policy = P27BRiskValuePolicy(
        base_net=base_net,
        value_net=value_net,
        fire_margin=args.fire_margin,
        assist_margin=args.assist_margin,
        assist_weight=args.assist_weight,
        max_bonus=args.max_bonus,
        kill_weight=args.kill_weight,
        death_weight=args.death_weight,
        double_death_weight=args.double_death_weight,
        survive_weight=args.survive_weight,
        risk_threshold=args.risk_threshold,
        fire_delta_margin=args.fire_delta_margin,
        global_fire_risk_penalty=args.global_fire_risk_penalty,
        global_fire_risk_threshold=args.global_fire_risk_threshold,
        global_fire_dd_threshold=args.global_fire_dd_threshold,
        low_quality_fire_penalty=args.low_quality_fire_penalty,
        low_quality_fire_delta=args.low_quality_fire_delta,
        opportunity_bonus=args.opportunity_bonus,
        opportunity_min_line=args.opportunity_min_line,
        opportunity_max_risk=args.opportunity_max_risk,
        opportunity_max_danger=args.opportunity_max_danger,
        opportunity_min_fire_delta=args.opportunity_min_fire_delta,
        escape_bonus=args.escape_bonus,
        escape_min_gain=args.escape_min_gain,
        escape_max_danger=args.escape_max_danger,
        escape_hold_frames=args.escape_hold_frames,
        stall_fire_penalty=args.stall_fire_penalty,
    )
    return [play_round_dual_engine(policy, seed + index)
            for index in range(count)]


def evaluate(args):
    workers = max(1, min(args.workers, args.n))
    base, remainder = divmod(args.n, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((worker, args.base_net, args.value_net,
                         args.seed + offset, count, args))
            offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rounds = [item for part in pool.map(_eval_worker, jobs)
                  for item in part]
    total = len(rounds)
    count = lambda key: sum(item["true_result"] == key for item in rounds)
    shots = sum(item["shots"] for item in rounds)
    kills = sum(item["kills"] for item in rounds)
    print(f"===== P27b {os.path.basename(args.value_net)} "
          f"{total} games @{args.seed} ({time.time()-started:.0f}s) =====",
          flush=True)
    print(f"  true win {count('win')/total:.1%}  "
          f"loss {count('loss')/total:.1%}  "
          f"double death {count('double_death')/total:.1%}  "
          f"draw {count('draw')/total:.1%}", flush=True)
    print(f"  shots/game {shots/total:.1f}  "
          f"hit rate {kills/max(shots,1):.1%}  "
          f"avg length {sum(r['frames'] for r in rounds)/total/25:.1f}s",
          flush=True)
    return count("win") / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--include-phases", default=",".join(DEFAULT_PHASES))
    parser.add_argument("--phase-weights", default=None)
    parser.add_argument("--category-weights", default=(
        "unsafe_movement=2.0,movement_value_gap=1.6,"
        "missed_fire_window=1.5,missed_kill_line=1.5,"
        "unsafe_fire_death=2.0,double_death_risk=2.4,"
        "fire_into_double_death=2.4,waste_or_unsafe_fire=1.4"))
    parser.add_argument("--out", default=os.path.join(
        MODELS_DIR, "p27b_risk_value_iter00.pt"))
    parser.add_argument("--value-net", default=os.path.join(
        MODELS_DIR, "p27b_risk_value_iter00.pt"))
    parser.add_argument("--base-net", default=os.path.join(
        MODELS_DIR, "p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--aux-weight", type=float, default=0.12)
    parser.add_argument("--rank-weight", type=float, default=0.35)
    parser.add_argument("--rank-margin", type=float, default=0.04)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--assist-margin", type=float, default=0.08)
    parser.add_argument("--assist-weight", type=float, default=0.35)
    parser.add_argument("--max-bonus", type=float, default=0.10)
    parser.add_argument("--kill-weight", type=float, default=0.04)
    parser.add_argument("--death-weight", type=float, default=0.12)
    parser.add_argument("--double-death-weight", type=float, default=0.18)
    parser.add_argument("--survive-weight", type=float, default=0.02)
    parser.add_argument("--risk-threshold", type=float, default=0.55)
    parser.add_argument("--fire-delta-margin", type=float, default=0.14)
    parser.add_argument("--global-fire-risk-penalty", type=float, default=0.0)
    parser.add_argument("--global-fire-risk-threshold", type=float,
                        default=1.10)
    parser.add_argument("--global-fire-dd-threshold", type=float,
                        default=1.10)
    parser.add_argument("--low-quality-fire-penalty", type=float,
                        default=0.0)
    parser.add_argument("--low-quality-fire-delta", type=float, default=0.0)
    parser.add_argument("--opportunity-bonus", type=float, default=0.0)
    parser.add_argument("--opportunity-min-line", type=float, default=0.76)
    parser.add_argument("--opportunity-max-risk", type=float, default=0.25)
    parser.add_argument("--opportunity-max-danger", type=float, default=0.45)
    parser.add_argument("--opportunity-min-fire-delta", type=float,
                        default=0.08)
    parser.add_argument("--escape-bonus", type=float, default=0.0)
    parser.add_argument("--escape-min-gain", type=float, default=-0.02)
    parser.add_argument("--escape-max-danger", type=float, default=0.55)
    parser.add_argument("--escape-hold-frames", type=int, default=0)
    parser.add_argument("--stall-fire-penalty", type=float, default=0.0)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
