"""
P27 macro-score policy.

This wraps the current P26 champion with a small macro-action head. The macro
head is trained on P27 macro probe data and is only allowed to override P26 in
detected hard states such as stalls, dead ends, and missed firing windows.
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.maze import h_open, v_open  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    MODELS_DIR,
    P26Policy,
    build_observation,
    select_action,
    stack_observation,
)

MACRO_NAMES = (
    "hold_fire_reposition",
    "single_fire_policy",
    "fan_left_3",
    "fan_right_3",
    "fan_center_3",
    "escape_back_left",
    "escape_back_right",
    "escape_forward_left",
    "escape_forward_right",
    "escape_forward",
)
MACRO_DIM = len(MACRO_NAMES)


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


def _macro_specs(policy_action):
    throttle, turn, _ = policy_action
    return [
        ("hold_fire_reposition", "hold", [(throttle, turn, 0)],
         (throttle, turn, 0)),
        ("single_fire_policy", "single_fire",
         [(throttle, turn, 1), (throttle, turn, 0)],
         (throttle, turn, 0)),
        ("fan_left_3", "fan_fire", [
            (throttle, 0, 1), (throttle, 0, 0),
            (throttle, 0, 1), (throttle, 0, 0),
            (throttle, 0, 1), (throttle, 1, 0),
        ], (throttle, 1, 0)),
        ("fan_right_3", "fan_fire", [
            (throttle, 2, 1), (throttle, 2, 0),
            (throttle, 2, 1), (throttle, 2, 0),
            (throttle, 2, 1), (throttle, 1, 0),
        ], (throttle, 1, 0)),
        ("fan_center_3", "fan_fire", [
            (throttle, 1, 1), (throttle, 1, 0),
            (throttle, 0, 1), (throttle, 0, 0),
            (throttle, 2, 1), (throttle, 1, 0),
        ], (throttle, 1, 0)),
        ("escape_back_left", "escape", [(0, 0, 0)], (0, 0, 0)),
        ("escape_back_right", "escape", [(0, 2, 0)], (0, 2, 0)),
        ("escape_forward_left", "escape", [(2, 0, 0)], (2, 0, 0)),
        ("escape_forward_right", "escape", [(2, 2, 0)], (2, 2, 0)),
        ("escape_forward", "escape", [(2, 1, 0)], (2, 1, 0)),
    ]


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


def build_macro_net(in_dim, width=512):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, MACRO_DIM),
    )


def load_macro_data(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("no macro data matched")

    xs, ys, categories, best_macro = [], [], [], []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        xs.append(data["X"].astype(np.float32))
        ys.append(data["Y_macro"].astype(np.float32))
        categories.append(data["category"].astype(str))
        best_macro.append(data["best_macro"].astype(str))
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    C = np.concatenate(categories)
    B = np.concatenate(best_macro)
    return X, Y, C, B, paths


def save_macro_model(net, path, in_dim, width):
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "in_dim": int(in_dim),
        "width": int(width),
        "macro_names": MACRO_NAMES,
    }, path)
    print(f"saved {path}", flush=True)


def load_macro_model(path):
    import torch

    payload = torch.load(path, weights_only=False)
    in_dim = int(payload["in_dim"])
    width = int(payload.get("width", 512))
    net = build_macro_net(in_dim, width)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, in_dim


def _weighted_mean(loss, weights):
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _ranking_loss(pred, target, weights, margin):
    import torch

    best = target.argmax(dim=1)
    pred_best = pred.gather(1, best.unsqueeze(1))
    losses = torch.relu(margin - pred_best + pred)
    mask = torch.ones_like(losses)
    mask.scatter_(1, best.unsqueeze(1), 0.0)
    return _weighted_mean(losses * mask, weights)


def train_macro_model(args):
    import torch
    import torch.nn.functional as F

    patterns = args.data or [
        "training/analysis/runs/p27_macro_probe_40_*.npz"
    ]
    X, Y, categories, best_macro, paths = load_macro_data(patterns)
    hold = Y[:, 0]
    best = Y.max(axis=1)
    positive = (best - hold) > args.positive_margin
    weights = np.ones(len(X), dtype=np.float32)
    weights[positive] *= args.positive_weight
    for name, weight in (
            ("dead_end_stall", args.dead_end_weight),
            ("stutter_stall", args.stutter_weight),
            ("passive_map_control", args.passive_weight),
            ("missed_fire_window", args.missed_fire_weight),
            ("blind_fire", args.blind_fire_weight),
    ):
        weights[categories == name] *= weight
    n = len(X)
    n_val = max(1, min(n - 1, int(n * args.val_frac)))
    perm = np.random.default_rng(2700).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xt = torch.as_tensor(X[train_idx])
    Yt = torch.as_tensor(Y[train_idx])
    Wt = torch.as_tensor(weights[train_idx])
    Xv = torch.as_tensor(X[val_idx])
    Yv = torch.as_tensor(Y[val_idx])
    Wv = torch.as_tensor(weights[val_idx])

    net = build_macro_net(X.shape[1], args.width)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def batch_loss(x, y, w):
        pred = net(x)
        mse = _weighted_mean(F.mse_loss(pred, y, reduction="none"), w)
        rank = _ranking_loss(pred, y, w, args.rank_margin)
        return mse + args.rank_weight * rank, mse, rank, pred

    def metrics():
        with torch.no_grad():
            loss, mse, rank, pred = batch_loss(Xv, Yv, Wv)
            top1 = (pred.argmax(1) == Yv.argmax(1)).float().mean().item()
            true_pos = (Yv.max(1).values - Yv[:, 0]) > args.positive_margin
            pred_pos = (pred.max(1).values - pred[:, 0]) > args.positive_margin
            pos_acc = (true_pos == pred_pos).float().mean().item()
            pos_recall = (
                (true_pos & pred_pos).float().sum()
                / true_pos.float().sum().clamp_min(1.0)).item()
        return loss.item(), mse.item(), rank.item(), top1, pos_acc, pos_recall

    print(f"P27 macro data: {len(paths)} files -> {n} samples", flush=True)
    print(f"  positive {positive.mean():.1%}  weights mean {weights.mean():.2f}",
          flush=True)
    t0 = time.time()
    n_train = len(Xt)
    for epoch in range(args.epochs):
        order = torch.randperm(n_train)
        total, batches = 0.0, 0
        for start in range(0, n_train, args.batch):
            idx = order[start:start + args.batch]
            loss, *_ = batch_loss(Xt[idx], Yt[idx], Wt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        vals = metrics()
        print(f"  epoch {epoch+1}/{args.epochs} "
              f"train {total/max(1,batches):.4f} val {vals[0]:.4f} "
              f"mse {vals[1]:.4f} rank {vals[2]:.4f} "
              f"top1 {vals[3]:.1%} pos_acc {vals[4]:.1%} "
              f"pos_recall {vals[5]:.1%} {time.time()-t0:.0f}s",
              flush=True)
    save_macro_model(net, args.out, X.shape[1], args.width)
    return args.out


class P27MacroPolicy:
    name = "p27_macro"

    def __init__(self, base_net, macro_net, fire_margin=0.16,
                 macro_margin=0.08, fan_min_line=0.75, fan_max_risk=0.25,
                 fan_max_bullets=3, single_min_line=0.70,
                 single_max_risk=0.30, stall_window=40,
                 stall_distance=0.22, fire_window_line=0.70,
                 fire_window_frames=8, blind_fire_line=0.35,
                 pressure_radius=0.75, macro_cooldown=20):
        import torch

        self.torch = torch
        self.base = P26Policy(
            net_path=base_net,
            fire_margin=fire_margin,
            fire_threshold=0.0,
            kill_weight=0.0,
            death_weight=0.0,
            double_death_weight=0.0,
            survive_weight=0.0,
            fire_prob_weight=0.0,
        )
        self.macro_net, self.in_dim = load_macro_model(macro_net)
        self.macro_margin = float(macro_margin)
        self.fan_min_line = float(fan_min_line)
        self.fan_max_risk = float(fan_max_risk)
        self.fan_max_bullets = int(fan_max_bullets)
        self.single_min_line = float(single_min_line)
        self.single_max_risk = float(single_max_risk)
        self.stall_window = int(stall_window)
        self.stall_distance = float(stall_distance)
        self.fire_window_line = float(fire_window_line)
        self.fire_window_frames = int(fire_window_frames)
        self.blind_fire_line = float(blind_fire_line)
        self.pressure_radius = float(pressure_radius)
        self.macro_cooldown = int(macro_cooldown)
        self.reset()

    @property
    def history(self):
        return self.base.history

    @property
    def frame_stack(self):
        return self.base.frame_stack

    def reset(self):
        self.base.reset()
        self.active = deque()
        self.pos_window = deque(maxlen=self.stall_window)
        self.input_window = deque(maxlen=self.stall_window)
        self.clear_fire_frames = 0
        self.macro_counts = {}
        self.cooldown_remaining = 0

    def _detect_category(self, game, inp, metrics):
        cmd = _input_tuple(inp)
        me, enemy = game.tanks[0], game.tanks[1]
        line, reach, risk = [float(value) for value in metrics[:3]]
        self.pos_window.append((me.x, me.y))
        self.input_window.append(cmd)

        if cmd[4] and enemy.alive:
            from training.opportunity_distill import _shot_event

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

    def _macro_allowed(self, category, macro_name, macro_kind, game, metrics):
        line, reach, risk = [float(value) for value in metrics[:3]]
        bullets = int(game.tanks[0].bullets_fired)
        if category == "missed_fire_window":
            if macro_kind not in ("single_fire", "fan_fire"):
                return False
        elif category == "blind_fire":
            if macro_kind in ("single_fire", "fan_fire"):
                return False
        elif category in ("dead_end_stall", "passive_map_control",
                          "stutter_stall"):
            if macro_kind != "escape":
                return False
            if line >= self.single_min_line and risk <= self.single_max_risk:
                return False
            if category == "stutter_stall" and reach >= 0.60 and risk < 0.25:
                return False
        if macro_kind == "fan_fire":
            return (line >= self.fan_min_line
                    and risk <= self.fan_max_risk
                    and bullets <= self.fan_max_bullets)
        if macro_kind == "single_fire":
            return (line >= self.single_min_line
                    and risk <= self.single_max_risk
                    and bullets < game.settings_max_bullets)
        if macro_name == "hold_fire_reposition":
            return False
        return True

    def _select_macro(self, category, game, base_action, metrics):
        if not self.base.history:
            return None
        stacked = stack_observation(self.base.history, self.base.frame_stack)
        if len(stacked) != self.in_dim:
            return None
        with self.torch.no_grad():
            scores = self.macro_net(
                self.torch.as_tensor(stacked).unsqueeze(0))[0].numpy()
        order = np.argsort(scores)[::-1]
        specs = _macro_specs(base_action)
        hold_score = float(scores[0])
        for index in order:
            index = int(index)
            name, kind, sequence, _ = specs[index]
            if index == 0:
                return None
            if float(scores[index] - hold_score) < self.macro_margin:
                return None
            if self._macro_allowed(category, name, kind, game, metrics):
                self.macro_counts[name] = self.macro_counts.get(name, 0) + 1
                return deque(sequence)
        return None

    def act(self, game):
        base_inp = self.base.act(game)
        if not game.tanks[0].alive:
            return {}
        if self.active:
            return _controls(self.active.popleft())
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return base_inp
        if self.base.analyzer is None:
            return base_inp
        metrics = self.base.analyzer.metrics(game)
        category = self._detect_category(game, base_inp, metrics)
        if category is None:
            return base_inp
        sequence = self._select_macro(
            category, game, _action_tuple(base_inp), metrics)
        if not sequence:
            return base_inp
        action = sequence.popleft()
        self.active = sequence
        self.cooldown_remaining = self.macro_cooldown
        return _controls(action)


class P27ValueAssistPolicy:
    name = "p27_value_assist"

    def __init__(self, base_net, macro_net, fire_margin=0.16,
                 fire_threshold=0.0, kill_weight=0.0, death_weight=0.0,
                 double_death_weight=0.0, survive_weight=0.0,
                 fire_prob_weight=0.0, value_margin=0.10,
                 fire_value_margin=0.12, escape_bonus_weight=0.05,
                 fire_bonus_weight=0.04, max_escape_bonus=0.06,
                 max_fire_bonus=0.05, fire_line=0.76, fire_max_risk=0.22,
                 suppress_blind_fire_line=0.35, stall_window=40,
                 stall_distance=0.22, fire_window_line=0.70,
                 fire_window_frames=8, blind_fire_line=0.35,
                 pressure_radius=0.75):
        import torch
        from training.mpc_agent import CANDIDATES
        from training.tt_gym_env import TankTroubleGym
        from training.p26_amortized_mpc import load_p26_network

        self.torch = torch
        self.candidates = CANDIDATES
        self.network, self.frame_stack = load_p26_network(base_net)
        self.macro_net, self.in_dim = load_macro_model(macro_net)
        self.fire_margin = float(fire_margin)
        self.fire_threshold = float(fire_threshold)
        self.kill_weight = float(kill_weight)
        self.death_weight = float(death_weight)
        self.double_death_weight = float(double_death_weight)
        self.survive_weight = float(survive_weight)
        self.fire_prob_weight = float(fire_prob_weight)
        self.value_margin = float(value_margin)
        self.fire_value_margin = float(fire_value_margin)
        self.escape_bonus_weight = float(escape_bonus_weight)
        self.fire_bonus_weight = float(fire_bonus_weight)
        self.max_escape_bonus = float(max_escape_bonus)
        self.max_fire_bonus = float(max_fire_bonus)
        self.fire_line = float(fire_line)
        self.fire_max_risk = float(fire_max_risk)
        self.suppress_blind_fire_line = float(suppress_blind_fire_line)
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
        self.assist_counts = {}

    def _detect_category(self, game, inp, metrics):
        cmd = _input_tuple(inp)
        me, enemy = game.tanks[0], game.tanks[1]
        line, reach, risk = [float(value) for value in metrics[:3]]
        self.pos_window.append((me.x, me.y))
        self.input_window.append(cmd)

        if cmd[4] and enemy.alive:
            from training.opportunity_distill import _shot_event

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

    def _macro_scores(self, stacked):
        if len(stacked) != self.in_dim:
            return None
        with self.torch.no_grad():
            return self.macro_net(
                self.torch.as_tensor(stacked).unsqueeze(0))[0].numpy()

    def _add_assist(self, name):
        self.assist_counts[name] = self.assist_counts.get(name, 0) + 1

    def _apply_value_assist(self, outputs, category, macro_scores, metrics,
                            base_action):
        if category is None or macro_scores is None:
            return outputs
        line, reach, risk = [float(value) for value in metrics[:3]]
        hold = float(macro_scores[0])
        score = outputs["score"].copy()
        if category == "missed_fire_window":
            fire_gain = float(macro_scores[1:5].max() - hold)
            if (fire_gain >= self.fire_value_margin
                    and line >= self.fire_line
                    and risk <= self.fire_max_risk):
                bonus = min(self.max_fire_bonus,
                            fire_gain * self.fire_bonus_weight)
                score[1::2] += bonus
                self._add_assist("fire_score_bonus")
        elif category in ("dead_end_stall", "passive_map_control",
                          "stutter_stall"):
            if line >= self.fire_line and risk <= self.fire_max_risk:
                return outputs
            escape_scores = macro_scores[5:]
            best_escape = int(escape_scores.argmax()) + 5
            escape_gain = float(macro_scores[best_escape] - hold)
            if escape_gain >= self.value_margin:
                specs = _macro_specs(base_action)
                _, _, sequence, _ = specs[best_escape]
                first = sequence[0]
                try:
                    action_index = self.candidates.index(first)
                except ValueError:
                    return outputs
                bonus = min(self.max_escape_bonus,
                            escape_gain * self.escape_bonus_weight)
                score[action_index] += bonus
                self._add_assist("escape_score_bonus")
        elif category == "blind_fire":
            score[1::2] -= self.max_fire_bonus
            self._add_assist("blind_fire_penalty")
        adjusted = dict(outputs)
        adjusted["score"] = score
        return adjusted

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
        observation, metrics = build_observation(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        self.history.append(observation)
        stacked = stack_observation(self.history, self.frame_stack)
        with self.torch.no_grad():
            out = self.network(self.torch.as_tensor(stacked).unsqueeze(0))
        outputs = {
            "score": out["score"][0].numpy(),
            "aux": out["aux"][0].numpy(),
            "fire": out["fire"][0].numpy(),
        }
        base_action = select_action(
            outputs, self.candidates, self.fire_margin, self.fire_threshold,
            self.kill_weight, self.death_weight, self.double_death_weight,
            self.survive_weight, self.fire_prob_weight)
        base_inp = _controls(base_action)
        macro_scores = self._macro_scores(stacked)
        category = self._detect_category(game, base_inp, metrics)
        outputs = self._apply_value_assist(
            outputs, category, macro_scores, metrics, base_action)
        throttle, turn, fire = select_action(
            outputs, self.candidates, self.fire_margin, self.fire_threshold,
            self.kill_weight, self.death_weight, self.double_death_weight,
            self.survive_weight, self.fire_prob_weight)
        if len(game.tanks) > 1 and game.tanks[1].alive:
            line, _, risk = [float(value) for value in metrics[:3]]
            movement = throttle * 3 + turn
            paired_score = outputs["score"].reshape(9, 2)
            fire_delta = float(paired_score[movement, 1]
                               - paired_score[movement, 0])
            if (fire == 1 and self.suppress_blind_fire_line > 0.0
                    and line < self.suppress_blind_fire_line):
                fire = 0
            if (fire == 0 and line >= self.fire_line
                    and risk <= self.fire_max_risk
                    and fire_delta >= self.fire_margin):
                fire = 1
        elif len(game.tanks) > 1:
            fire = 0
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def _eval_worker(job):
    (worker, base_net, macro_net, seed, count, args) = job
    import torch
    torch.set_num_threads(1)
    from training.evaluate import play_round_dual_engine

    policy = P27MacroPolicy(
        base_net=base_net,
        macro_net=macro_net,
        fire_margin=args.fire_margin,
        macro_margin=args.macro_margin,
        fan_min_line=args.fan_min_line,
        fan_max_risk=args.fan_max_risk,
        fan_max_bullets=args.fan_max_bullets,
        single_min_line=args.single_min_line,
        single_max_risk=args.single_max_risk,
        macro_cooldown=args.macro_cooldown,
    )
    return [play_round_dual_engine(policy, seed + index)
            for index in range(count)]


def evaluate_macro(args):
    jobs, offset = [], 0
    workers = max(1, min(args.workers, args.n))
    base, remainder = divmod(args.n, workers)
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((worker, args.base_net, args.macro_net,
                         args.seed + offset, count, args))
            offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rounds = [item for part in pool.map(_eval_worker, jobs)
                  for item in part]
    total = len(rounds)
    count = lambda key: sum(result["true_result"] == key
                            for result in rounds)
    shots = sum(result["shots"] for result in rounds)
    kills = sum(result["kills"] for result in rounds)
    print(f"===== P27 macro {os.path.basename(args.macro_net)} "
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
    parser.add_argument("--data", action="append")
    parser.add_argument("--out", default=os.path.join(
        MODELS_DIR, "p27_macro_iter00.pt"))
    parser.add_argument("--macro-net", default=os.path.join(
        MODELS_DIR, "p27_macro_iter00.pt"))
    parser.add_argument("--base-net", default=os.path.join(
        MODELS_DIR, "p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--rank-margin", type=float, default=0.04)
    parser.add_argument("--rank-weight", type=float, default=0.35)
    parser.add_argument("--positive-margin", type=float, default=0.02)
    parser.add_argument("--positive-weight", type=float, default=2.5)
    parser.add_argument("--dead-end-weight", type=float, default=1.5)
    parser.add_argument("--stutter-weight", type=float, default=1.4)
    parser.add_argument("--passive-weight", type=float, default=1.4)
    parser.add_argument("--missed-fire-weight", type=float, default=1.2)
    parser.add_argument("--blind-fire-weight", type=float, default=1.8)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--macro-margin", type=float, default=0.08)
    parser.add_argument("--fan-min-line", type=float, default=0.78)
    parser.add_argument("--fan-max-risk", type=float, default=0.20)
    parser.add_argument("--fan-max-bullets", type=int, default=3)
    parser.add_argument("--single-min-line", type=float, default=0.72)
    parser.add_argument("--single-max-risk", type=float, default=0.25)
    parser.add_argument("--macro-cooldown", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "train":
        train_macro_model(args)
    else:
        evaluate_macro(args)


if __name__ == "__main__":
    main()
