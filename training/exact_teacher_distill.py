"""Distill the frozen Exact-State Safety-Shielded MPC teacher.

The collector deliberately uses all 18 exact root actions.  Labels therefore
describe the teacher that actually acted, rather than a second approximate
rollout objective.  The student remains a single-forward-pass P26 network.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import Game  # noqa: E402
from tank_trouble_original.maze import h_open, v_open  # noqa: E402
from training.evaluate import RoundTracker  # noqa: E402
from training.exact_state_mpc_teacher import (  # noqa: E402
    ExactStatePriorGuidedMPC,
    exact_root_search,
)
from training.mpc_agent import CANDIDATES  # noqa: E402
from training.opportunity_distill import _shot_event  # noqa: E402
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    AUX_DIM,
    AUX_NAMES,
    DATA_DIR,
    SCORE_SCALE,
    build_observation,
    build_p26_net,
    load_p26_network,
    stack_observation,
)
from training.p27_risk_value import P27BRiskValuePolicy  # noqa: E402
from training.tt_gym_env import TRUNCATE_FRAMES, TankTroubleGym  # noqa: E402


FROZEN_TEACHER_NAME = "exact_state_safety_shielded_mpc_v1"
OBJECTIVE_VERSION = "exact_state_safety_shielded_mpc_v1"
DEFAULT_BASE_NET = "training/models/p26_amortized_mpc_iter05.pt"
DEFAULT_VALUE_NET = "training/models/p27b_risk_value_iter00.pt"
DEFAULT_MANIFEST = "training/champions/exact_state_safety_shielded_mpc_v1.json"

FROZEN_TEACHER_CONFIG = {
    "top_k": 12,
    "search_horizon": 72,
    "search_samples": 1,
    "search_death_penalty": 0.18,
    "search_dd_penalty": 0.45,
    "search_kill_bonus": 0.05,
    "search_max_death": 0.0,
    "search_max_dd": 0.0,
    "successor_shield": True,
    "successor_horizon": 72,
    "successor_shield_max_safe_roots": 2,
    "suppress_secured_fire": True,
    "min_unsecured_fire_gain": 2.0,
    "fire_margin": 0.16,
}

LABEL_CATEGORIES = (
    "standard",
    "fire_action",
    "danger_state",
    "safety_widening",
    "successor_shield_checked",
    "successor_shield_override",
    "secured_kill_fire_suppressed",
    "low_gain_fire_suppressed",
    "no_safe_search_action",
)

CATEGORY_WEIGHTS = {
    "standard": 1.0,
    "fire_action": 1.3,
    "danger_state": 2.5,
    "safety_widening": 2.5,
    "successor_shield_checked": 2.5,
    "successor_shield_override": 4.0,
    "secured_kill_fire_suppressed": 2.5,
    "low_gain_fire_suppressed": 2.2,
    "no_safe_search_action": 2.0,
}

GOAL_NAMES = (
    "idle", "shootAfter", "sprayBullets", "detonate", "driveTo",
    "backAway", "dodgeBullet", "dodgeFragbomb", "dodgeLaser",
    "runAway", "goForCrate",
)
ACTION_NAMES = (
    "idle", "fireWeapon", "turnTo", "driveToField", "driveToPos",
    "forward", "forwardAndTurn", "backup", "backupAndTurn",
)
MAX_PRIVILEGED_ACTIONS = 16
MAX_PRIVILEGED_BULLETS = 10
PRIVILEGED_RNG_DRAWS = 128
GOAL_STATE_NAMES = (
    "priority", "period", "id", "update_continuously", "x", "y", "t",
    "max_time", "dist", "max_dist", "dir_x", "dir_y", "closest_x",
    "closest_y", "target_me", "target_enemy", "target_other",
)
GAME_STATE_NAMES = (
    "maze_width", "maze_height", "scale", "frame", "alive_count",
    "end_count", "reset_count", "frozen", "crate_timer", "bullet_depth",
    "hit_immunity_me", "hit_immunity_enemy", "ai_action_depth",
    "ai_action_truncated", "ai_stuck_time", "ai_aggressiveness",
    "ai_goal_id",
)
TANK_STATE_NAMES = (
    "x", "y", "rotation_sin", "rotation_cos", "forward", "backup",
    "turn_left", "turn_right", "fire", "trigger_released",
    "bullets_fired", "alive", "hit_something", "laser_ready",
    "frag_fired", "gatling_ready", "homing_ready", "mines_layed",
    "death_ray_ready", "remote_controlling", "electric_ready",
    "weapon_bullet", "weapon_other",
)
ACTION_STATE_NAMES = (
    "x", "y", "angle_sin", "angle_cos", "delay", "dist",
    "can_reverse", "dir_left", "dir_right",
)
BULLET_STATE_NAMES = (
    "present", "owner_me", "owner_enemy", "x", "y", "x_speed",
    "y_speed", "lifetime", "deadly", "removed",
)
PRIVILEGED_FEATURE_NAMES = (
    *(f"goal_{name}" for name in GOAL_NAMES),
    "goal_unknown",
    *(f"goal_{name}" for name in GOAL_STATE_NAMES),
    *GAME_STATE_NAMES,
    *(f"tank_{tank}_{name}" for tank in ("me", "enemy")
      for name in TANK_STATE_NAMES),
    *(f"action_{slot}_{name}" for slot in range(MAX_PRIVILEGED_ACTIONS)
      for name in (
          "present", *(f"type_{value}" for value in ACTION_NAMES),
          "type_unknown", *ACTION_STATE_NAMES,
      )),
    *(f"bullet_{slot}_{name}" for slot in range(MAX_PRIVILEGED_BULLETS)
      for name in BULLET_STATE_NAMES),
    *(f"rng_next_{index}" for index in range(PRIVILEGED_RNG_DRAWS)),
)
PRIVILEGED_DIM = len(PRIVILEGED_FEATURE_NAMES)


def _controls_action(controls):
    return (
        2 if controls.get("forward") else 0 if controls.get("backup") else 1,
        0 if controls.get("turn_left") else
        2 if controls.get("turn_right") else 1,
        1 if controls.get("fire") else 0,
    )


def _set_controls(game, controls):
    tank = game.tanks[0]
    tank.forward = bool(controls.get("forward", False))
    tank.backup = bool(controls.get("backup", False))
    tank.turn_left = bool(controls.get("turn_left", False))
    tank.turn_right = bool(controls.get("turn_right", False))
    tank.fire = bool(controls.get("fire", False))


def _round_result(winner):
    if winner == 0:
        return "win"
    if winner == 1:
        return "loss"
    return "double_death"


def _normalized_position(value, game):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    maze_extent = max(len(game.maze), len(game.maze[0]), 1)
    if abs(value) <= maze_extent + 1:
        return float(np.clip(value / maze_extent, -1.5, 1.5))
    return float(np.clip(value / (game.scale * maze_extent), -1.5, 1.5))


def _numeric(value, scale=1.0):
    try:
        return float(value) / float(scale)
    except (TypeError, ValueError):
        return 0.0


def privileged_state_features(game):
    """Expose complete local-engine state without advancing its RNG."""
    import math

    me = game.tanks[0]
    enemy = game.tanks[1]
    ai = enemy.ai
    goal = ai.my_goal or {}
    goal_name = str(goal.get("goal", ""))
    target = goal.get("target")
    direction = goal.get("dir")
    direction = direction if isinstance(direction, dict) else {}
    closest = goal.get("closest")
    closest = closest if isinstance(closest, dict) else {}
    width = max(1.0, len(game.maze) * game.scale)
    height = max(1.0, len(game.maze[0]) * game.scale)

    values = [float(goal_name == name) for name in GOAL_NAMES]
    values.extend((
        float(goal_name not in GOAL_NAMES),
        _numeric(goal.get("priority", 0.0)),
        _numeric(goal.get("period", 0.0), 75.0),
        _numeric(goal.get("id", 0.0), 100.0),
        float(bool(goal.get("updateContinuously", False))),
        _normalized_position(goal.get("x", 0.0), game),
        _normalized_position(goal.get("y", 0.0), game),
        _numeric(goal.get("t", 0.0), 100.0),
        _numeric(goal.get("maxTime", 0.0), 100.0),
        _numeric(goal.get("dist", 0.0), max(1.0, 10.0 * game.scale)),
        _numeric(goal.get("maxDist", 0.0), max(1.0, 10.0 * game.scale)),
        _numeric(direction.get("x", 0.0)),
        _numeric(direction.get("y", 0.0)),
        _numeric(closest.get("x", 0.0), width),
        _numeric(closest.get("y", 0.0), height),
        float(getattr(target, "number", -1) == 0),
        float(getattr(target, "number", -1) == 1),
        float(target is not None and getattr(target, "number", -1) not in (0, 1)),
    ))

    values.extend((
        len(game.maze) / 12.0,
        len(game.maze[0]) / 10.0,
        game.scale / 100.0,
        float(game.frame) / 2500.0,
        float(game.alive_count) / 2.0,
        float(game.end_count) / 200.0,
        float(game.reset_count) / 10.0,
        float(game.frozen),
        float(game.crate_timer) / 2500.0,
        float(game._bullet_depth) / 10.0,
        float(game.hit_immunity_remaining[0]) / 100.0,
        float(game.hit_immunity_remaining[1]) / 100.0,
        min(1.5, len(ai.my_actions) / MAX_PRIVILEGED_ACTIONS),
        float(len(ai.my_actions) > MAX_PRIVILEGED_ACTIONS),
        float(ai.stuck_time) / max(1.0, float(ai.MAXSTUCKTIME)),
        float(ai.current_aggresiveness),
        float(ai.goal_id) / 100.0,
    ))

    for tank in (me, enemy):
        rotation = math.radians(float(tank.rotation))
        values.extend((
            float(tank.x) / width,
            float(tank.y) / height,
            math.sin(rotation),
            math.cos(rotation),
            float(tank.forward),
            float(tank.backup),
            float(tank.turn_left),
            float(tank.turn_right),
            float(tank.fire),
            float(tank.trigger_released),
            float(tank.bullets_fired) / max(1.0, game.settings_max_bullets),
            float(tank.alive),
            float(tank.hit_something),
            float(tank.laser_ready),
            float(tank.frag_fired),
            float(tank.gatling_ready),
            float(tank.homing_ready),
            float(tank.mines_layed) / 5.0,
            float(tank.death_ray_ready),
            float(tank.remote_controlling),
            float(tank.electric_ready),
            float(tank.current_weapon == "bullet"),
            float(tank.current_weapon != "bullet"),
        ))

    actions = list(reversed(ai.my_actions[-MAX_PRIVILEGED_ACTIONS:]))
    for slot in range(MAX_PRIVILEGED_ACTIONS):
        action = actions[slot] if slot < len(actions) else {}
        action_name = str(action.get("action", ""))
        angle = math.radians(_numeric(action.get("angle", 0.0)))
        values.append(float(slot < len(actions)))
        values.extend(float(action_name == name) for name in ACTION_NAMES)
        values.extend((
            float(action_name not in ACTION_NAMES and bool(action_name)),
            _numeric(action.get("x", 0.0), width),
            _numeric(action.get("y", 0.0), height),
            math.sin(angle),
            math.cos(angle),
            _numeric(action.get("delay", 0.0), 75.0),
            _numeric(action.get("dist", 0.0), 10.0),
            float(bool(action.get("canReverse", False))),
            float(action.get("dir") == "left"),
            float(action.get("dir") == "right"),
        ))

    for slot in range(MAX_PRIVILEGED_BULLETS):
        bullet = game.bullets[slot] if slot < len(game.bullets) else None
        if bullet is None:
            values.extend((0.0,) * len(BULLET_STATE_NAMES))
            continue
        values.extend((
            1.0,
            float(bullet.owner is me),
            float(bullet.owner is enemy),
            float(bullet.x) / width,
            float(bullet.y) / height,
            float(bullet.x_speed) / max(1.0, game.scale),
            float(bullet.y_speed) / max(1.0, game.scale),
            float(bullet.lifetime) / 250.0,
            float(bullet.deadly) / 10.0,
            float(bullet.removed),
        ))

    rng = random.Random()
    rng.setstate(game.rng.getstate())
    values.extend(rng.random() for _ in range(PRIVILEGED_RNG_DRAWS))
    feature = np.asarray(values, dtype=np.float32)
    if feature.shape != (PRIVILEGED_DIM,):
        raise RuntimeError(
            f"privileged feature mismatch: {feature.shape} != {PRIVILEGED_DIM}")
    return feature


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_manifest(base_net=DEFAULT_BASE_NET, value_net=DEFAULT_VALUE_NET):
    sources = (
        "training/exact_state.py",
        "training/exact_state_mpc_teacher.py",
        base_net,
        value_net,
    )
    return {
        "name": FROZEN_TEACHER_NAME,
        "scope": "privileged local exact-engine teacher",
        "configuration": dict(FROZEN_TEACHER_CONFIG),
        "base_net": base_net,
        "value_net": value_net,
        "sha256": {path: _sha256(path) for path in sources},
        "benchmark": {
            "report": (
                "training/analysis/runs/"
                "exact_state_fire_cost_official_3x40.json"),
            "seeds": ["970000:40", "990000:40", "973000:40"],
            "wins": 120,
            "games": 120,
            "losses": 0,
            "double_deaths": 0,
        },
    }


def write_manifest(path, base_net=DEFAULT_BASE_NET,
                   value_net=DEFAULT_VALUE_NET):
    payload = frozen_manifest(base_net, value_net)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def verify_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("name") != FROZEN_TEACHER_NAME:
        raise RuntimeError(f"unexpected frozen teacher: {manifest.get('name')}")
    if manifest.get("configuration") != FROZEN_TEACHER_CONFIG:
        raise RuntimeError("frozen teacher configuration changed")
    mismatches = []
    for source, expected in manifest.get("sha256", {}).items():
        actual = _sha256(source)
        if actual != expected:
            mismatches.append((source, expected, actual))
    if mismatches:
        lines = [f"{path}: expected {old}, got {new}"
                 for path, old, new in mismatches]
        raise RuntimeError("frozen teacher hash mismatch:\n" + "\n".join(lines))
    return manifest


def make_frozen_teacher(base_net=DEFAULT_BASE_NET,
                        value_net=DEFAULT_VALUE_NET):
    return ExactStatePriorGuidedMPC(
        base_net=base_net,
        value_net=value_net,
        **FROZEN_TEACHER_CONFIG,
    )


def _label_category(decision):
    interventions = set(decision.get("interventions", ()))
    priority = (
        "successor_shield_override",
        "secured_kill_fire_suppressed",
        "low_gain_fire_suppressed",
        "safety_widening",
        "successor_shield_checked",
    )
    for name in priority:
        if name in interventions:
            return name
    if decision.get("executed_index") is None:
        return "no_safe_search_action"
    if int(decision.get("safe_root_count", 18)) <= 2:
        return "danger_state"
    if int(decision["executed_index"]) % 2 == 1:
        return "fire_action"
    return "standard"


def build_consistent_label(decision, policy_margin=0.15,
                           fire_gain_margin=2.0):
    """Convert one complete exact decision into policy-consistent targets."""
    rows = {int(row["index"]): row for row in decision.get("rows", ())}
    if set(rows) != set(range(len(CANDIDATES))):
        raise ValueError("exact distillation requires all 18 root actions")

    value = np.asarray([rows[index]["value"] for index in range(18)],
                       dtype=np.float32) / SCORE_SCALE
    raw_score = np.asarray([rows[index]["score"] for index in range(18)],
                           dtype=np.float32) / SCORE_SCALE
    aux = np.asarray([rows[index]["aux"] for index in range(18)],
                     dtype=np.float32)
    allowed = np.asarray([rows[index]["allowed"] for index in range(18)],
                         dtype=np.bool_)
    selected = decision.get("executed_index")
    action_valid = bool(
        selected is not None and allowed[int(selected)])

    if allowed.any():
        best_safe = float(value[allowed].max())
        policy_score = np.clip(value - best_safe, -2.5, 0.0)
    else:
        policy_score = np.clip(value - float(value.max()), -2.5, 0.0)
    policy_score[~allowed] = np.minimum(policy_score[~allowed], -1.25)
    if action_valid:
        policy_score[int(selected)] = float(policy_margin)

    fire = np.zeros(9, dtype=np.float32)
    fire_mask = np.zeros(9, dtype=np.float32)
    for movement in range(9):
        no_fire = rows[movement * 2]
        yes_fire = rows[movement * 2 + 1]
        no_allowed = bool(no_fire["allowed"])
        yes_allowed = bool(yes_fire["allowed"])
        target = None
        if no_allowed != yes_allowed:
            target = float(yes_allowed)
        elif no_allowed and yes_allowed:
            gain = float(yes_fire["value"] - no_fire["value"])
            secured = (
                yes_fire["kill"] >= 1.0
                and no_fire["kill"] >= yes_fire["kill"]
                and no_fire["value"] >= yes_fire["value"] - 1e-6
            )
            low_gain = yes_fire["kill"] <= 0.0 and gain < fire_gain_margin
            if secured or low_gain:
                target = 0.0
            elif abs(gain) >= fire_gain_margin:
                target = float(gain > 0.0)
        if action_valid and movement == int(selected) // 2:
            target = float(int(selected) % 2)
        if target is not None:
            fire[movement] = target
            fire_mask[movement] = 1.0

    category = _label_category(decision)
    return {
        "Y_score": policy_score.astype(np.float32),
        "Y_value": value,
        "Y_raw_score": raw_score,
        "Y_aux": aux,
        "Y_fire": fire,
        "Y_fire_mask": fire_mask,
        "Y_action": -1 if selected is None else int(selected),
        "Y_prior_action": int(decision.get("prior_index", -1)),
        "action_valid": action_valid,
        "allowed": allowed,
        "category": category,
        "W": float(CATEGORY_WEIGHTS[category]),
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


class BehaviorAudit:
    """Record the same visible issue classes used in interactive reviews."""

    def __init__(self):
        self.issues = Counter()
        self.issue_frames = {}
        self.positions = deque(maxlen=40)
        self.inputs = deque(maxlen=40)
        self.clear_fire_frames = 0

    def _mark(self, name, frame):
        self.issues[name] += 1
        self.issue_frames.setdefault(name, []).append(int(frame))

    def observe(self, game, controls, metrics, frame):
        before = Counter(self.issues)
        me, enemy = game.tanks[0], game.tanks[1]
        line, reach, risk = [float(value) for value in metrics[:3]]
        command = (
            bool(controls.get("forward")), bool(controls.get("backup")),
            bool(controls.get("turn_left")),
            bool(controls.get("turn_right")), bool(controls.get("fire")),
        )
        self.positions.append((float(me.x), float(me.y)))
        self.inputs.append(command)

        if command[4] and not enemy.alive:
            self._mark("post_kill_fire", frame)
        if command[4] and enemy.alive:
            shot = _shot_event(game)
            closest = float("inf") if shot is None else float(
                shot.get("closest", float("inf")))
            result = None if shot is None else shot.get("result")
            if line < 0.35 and result != "HIT" and closest > 0.75 * game.scale:
                self._mark("blind_fire", frame)
        if not command[4] and enemy.alive and line >= 0.70:
            self.clear_fire_frames += 1
            if self.clear_fire_frames >= 8:
                self._mark("missed_fire_window", frame)
                self.clear_fire_frames = 0
        else:
            self.clear_fire_frames = 0

        if len(self.positions) == self.positions.maxlen:
            displacement = math.hypot(
                self.positions[-1][0] - self.positions[0][0],
                self.positions[-1][1] - self.positions[0][1])
            moving = sum(any(item[:4]) for item in self.inputs)
            x, y = _cell(game, me)
            stalled = displacement < 0.22 * game.scale
            if stalled and (_dead_end_penalty(game, x, y) > 0
                            or _open_neighbors(game, x, y) <= 1):
                self._mark("dead_end_stall", frame)
            elif stalled and line < 0.35 and reach < 0.55 and risk < 0.35:
                self._mark("passive_map_control", frame)
            elif stalled and moving >= self.positions.maxlen // 4:
                self._mark("stutter_stall", frame)
            if stalled:
                self.positions.clear()
                self.inputs.clear()
        return tuple(
            name for name, count in self.issues.items()
            if count > before.get(name, 0)
        )

    def result(self):
        return {
            "issues": dict(self.issues),
            "issue_frames": {
                key: values[:20] for key, values in self.issue_frames.items()
            },
        }


def _collect_round(policy, seed, stride, danger_threshold,
                   collect_labels=True):
    game = Game(seed=seed, ai_enabled=True)
    policy.reset()
    policy.set_round_seed(seed)
    tracker = RoundTracker(game)
    observer = BehaviorAudit()
    records = []
    frames = 0
    true_result = None

    while frames < TRUNCATE_FRAMES or tracker.first_destroy is not None:
        analyzer = policy.analyzer or OpportunityAnalyzer360(game)
        metrics = analyzer.metrics(game)
        controls = policy.act(game)
        observer.observe(game, controls, metrics, frames)
        decision = policy.last_search_decision
        if collect_labels and decision is not None:
            interventions = decision.get("interventions", ())
            capture = (
                frames % max(1, stride) == 0
                or bool(interventions)
                or int(decision.get("safe_root_count", 18)) <= danger_threshold
                or bool(controls.get("fire"))
            )
            if capture:
                label_decision = dict(decision)
                label_rows = list(decision.get("rows", ()))
                labelled_indices = {int(row["index"]) for row in label_rows}
                missing = tuple(
                    index for index in range(len(CANDIDATES))
                    if index not in labelled_indices)
                if missing:
                    _, missing_rows = exact_root_search(
                        game,
                        policy.analyzer,
                        metrics,
                        missing,
                        horizon=FROZEN_TEACHER_CONFIG["search_horizon"],
                        death_penalty=FROZEN_TEACHER_CONFIG[
                            "search_death_penalty"],
                        dd_penalty=FROZEN_TEACHER_CONFIG[
                            "search_dd_penalty"],
                        kill_bonus=FROZEN_TEACHER_CONFIG[
                            "search_kill_bonus"],
                        max_death=FROZEN_TEACHER_CONFIG[
                            "search_max_death"],
                        max_dd=FROZEN_TEACHER_CONFIG["search_max_dd"],
                    )
                    label_rows.extend(missing_rows)
                label_decision["rows"] = label_rows
                label_decision["safe_root_count"] = sum(
                    bool(row["allowed"]) for row in label_rows)
                try:
                    label = build_consistent_label(label_decision)
                except ValueError:
                    label = None
                if label is not None:
                    label.update({
                        "X": stack_observation(
                            policy.history, policy.frame_stack).copy(),
                        "round_seed": int(seed),
                        "frame": int(frames),
                        "interventions": ",".join(interventions) or "none",
                        "safe_root_count": int(
                            decision.get("safe_root_count", 0)),
                    })
                    records.append(label)

        _set_controls(game, controls)
        tracker.pre_step()
        events = game.step()
        frames += 1
        tracker.post_step(events, 1)
        for event in events:
            if event[0] == "round_end":
                true_result = _round_result(event[1])
        if true_result:
            break

    audit = observer.result()
    result = {
        "seed": int(seed),
        "result": true_result or "draw",
        "frames": int(frames),
        "shots": int(tracker.shots),
        "kills": int(tracker.kills),
        "hit_rate": tracker.kills / max(1, tracker.shots),
        "move_cells": tracker.move_px / tracker.scale,
        "death_cause": tracker.death_cause,
        "labels": len(records),
        "label_categories": dict(Counter(
            item["category"] for item in records)),
        "interventions": dict(Counter(
            item["interventions"] for item in records
            if item["interventions"] != "none")),
        **audit,
    }
    return result, records


def _save_shard(path, records, manifest):
    if not records:
        raise RuntimeError(f"no exact labels produced for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray([item["X"] for item in records], np.float32),
        Y_score=np.asarray([item["Y_score"] for item in records], np.float32),
        Y_value=np.asarray([item["Y_value"] for item in records], np.float32),
        Y_raw_score=np.asarray(
            [item["Y_raw_score"] for item in records], np.float32),
        Y_aux=np.asarray([item["Y_aux"] for item in records], np.float32),
        Y_fire=np.asarray([item["Y_fire"] for item in records], np.float32),
        Y_fire_mask=np.asarray(
            [item["Y_fire_mask"] for item in records], np.float32),
        Y_action=np.asarray([item["Y_action"] for item in records], np.int64),
        Y_prior_action=np.asarray(
            [item["Y_prior_action"] for item in records], np.int64),
        action_valid=np.asarray(
            [item["action_valid"] for item in records], np.bool_),
        allowed=np.asarray([item["allowed"] for item in records], np.bool_),
        W=np.asarray([item["W"] for item in records], np.float32),
        category=np.asarray([item["category"] for item in records]),
        category_names=np.asarray(LABEL_CATEGORIES),
        interventions=np.asarray(
            [item["interventions"] for item in records]),
        safe_root_count=np.asarray(
            [item["safe_root_count"] for item in records], np.int16),
        round_seed=np.asarray(
            [item["round_seed"] for item in records], np.int64),
        frame=np.asarray([item["frame"] for item in records], np.int32),
        capture_reason=np.asarray([
            item.get("capture_reason", "teacher_on_policy")
            for item in records
        ]),
        champion_action=np.asarray([
            item.get("champion_action", -1) for item in records
        ], np.int16),
        champion_action_allowed=np.asarray([
            item.get("champion_action_allowed", False) for item in records
        ], np.bool_),
        teacher_override=np.asarray([
            item.get("teacher_override", False) for item in records
        ], np.bool_),
        teacher_advantage=np.asarray([
            item.get("teacher_advantage", 0.0) for item in records
        ], np.float32),
        teacher_margin=np.asarray([
            item.get("teacher_margin", 0.0) for item in records
        ], np.float32),
        residual_target=np.asarray([
            item.get("residual_target", False) for item in records
        ], np.bool_),
        P=np.asarray([
            item.get("P", np.zeros(PRIVILEGED_DIM, dtype=np.float32))
            for item in records
        ], np.float32),
        privileged_valid=np.asarray([
            item.get("privileged_valid", False) for item in records
        ], np.bool_),
        privileged_feature_names=np.asarray(PRIVILEGED_FEATURE_NAMES),
        frame_stack=np.asarray([4], np.int16),
        aux_names=np.asarray(AUX_NAMES),
        objective_version=np.asarray([OBJECTIVE_VERSION]),
        teacher_name=np.asarray([FROZEN_TEACHER_NAME]),
        teacher_manifest_sha256=np.asarray([
            hashlib.sha256(json.dumps(
                manifest, sort_keys=True).encode("utf-8")).hexdigest()
        ]),
    )


def build_champion_label(decision, champion_index):
    """Attach bounded-residual targets to one frozen-teacher decision."""
    label = build_consistent_label(decision)
    champion_index = int(champion_index)
    rows = {int(row["index"]): row for row in decision.get("rows", ())}
    selected = int(label["Y_action"]) if label["action_valid"] else -1
    champion_allowed = bool(rows[champion_index]["allowed"])
    advantage = 0.0
    teacher_margin = 0.0
    if selected >= 0:
        advantage = (
            float(rows[selected]["value"])
            - float(rows[champion_index]["value"])
        ) / SCORE_SCALE
        search_indices = {
            int(index) for index in decision.get("search_indices", rows)
        }
        alternatives = [
            float(row["value"]) for index, row in rows.items()
            if index != selected and index in search_indices and row["allowed"]
        ]
        if alternatives:
            teacher_margin = (
                float(rows[selected]["value"]) - max(alternatives)
            ) / SCORE_SCALE
    label.update({
        "champion_action": champion_index,
        "champion_action_allowed": champion_allowed,
        "teacher_override": bool(selected >= 0 and selected != champion_index),
        "teacher_advantage": float(advantage),
        "teacher_margin": float(teacher_margin),
    })
    return label


def _label_champion_state(teacher, champion, game, metrics, controls,
                          capture_reasons, seed, frame):
    """Query the frozen teacher without changing the champion trajectory."""
    champion_action = _controls_action(controls)
    champion_index = CANDIDATES.index(champion_action)
    stacked = stack_observation(
        champion.history, champion.frame_stack).copy()
    p27 = champion._p27_value(stacked, champion.last_context)
    if p27 is None:
        raise RuntimeError("champion value head is unavailable for exact label")

    teacher.analyzer = champion.analyzer
    teacher.pending_successor_audit = None
    indices = teacher._candidate_order(
        {}, p27, champion_index, metrics)
    teacher.last_search_decision = None
    teacher._search(game, metrics, indices)
    decision = dict(teacher.last_search_decision or {})
    selected = decision.get("selected_index")
    decision["prior_index"] = champion_index
    decision["prior_action"] = champion_action
    decision["executed_index"] = selected
    decision["executed_action"] = (
        None if selected is None else CANDIDATES[int(selected)])

    rows = list(decision.get("rows", ()))
    decision["search_indices"] = [int(row["index"]) for row in rows]
    labelled = {int(row["index"]) for row in rows}
    missing = tuple(index for index in range(len(CANDIDATES))
                    if index not in labelled)
    if missing:
        _, missing_rows = exact_root_search(
            game,
            teacher.analyzer,
            metrics,
            missing,
            horizon=FROZEN_TEACHER_CONFIG["search_horizon"],
            death_penalty=FROZEN_TEACHER_CONFIG["search_death_penalty"],
            dd_penalty=FROZEN_TEACHER_CONFIG["search_dd_penalty"],
            kill_bonus=FROZEN_TEACHER_CONFIG["search_kill_bonus"],
            max_death=FROZEN_TEACHER_CONFIG["search_max_death"],
            max_dd=FROZEN_TEACHER_CONFIG["search_max_dd"],
        )
        rows.extend(missing_rows)
    decision["rows"] = rows
    decision["safe_root_count"] = sum(
        bool(row["allowed"]) for row in rows)
    label = build_champion_label(decision, champion_index)
    label.update({
        "X": stacked,
        "P": privileged_state_features(game),
        "privileged_valid": True,
        "round_seed": int(seed),
        "frame": int(frame),
        "interventions": ",".join(
            decision.get("interventions", ())) or "none",
        "safe_root_count": int(decision["safe_root_count"]),
        "capture_reason": ",".join(sorted(set(capture_reasons))),
    })
    return label


def _champion_collect_round(champion, teacher, seed, args):
    game = Game(seed=seed, ai_enabled=True)
    champion.reset()
    teacher.reset()
    teacher.set_round_seed(seed)
    tracker = RoundTracker(game)
    observer = BehaviorAudit()
    records = []
    frames = 0
    last_capture = -10**9
    capture_counts = Counter()
    true_result = None

    while frames < TRUNCATE_FRAMES or tracker.first_destroy is not None:
        controls = champion.act(game)
        analyzer = champion.analyzer or OpportunityAnalyzer360(game)
        metrics = analyzer.metrics(game)
        new_issues = observer.observe(game, controls, metrics, frames)
        stacked = stack_observation(
            champion.history, champion.frame_stack)
        p27 = champion._p27_value(stacked, champion.last_context)
        champion_index = CANDIDATES.index(_controls_action(controls))
        danger = 0.0
        gap = float("inf")
        if p27 is not None:
            _, aux, value = p27
            danger = max(float(aux[champion_index, 1]),
                         float(aux[champion_index, 2]))
            ordered = np.sort(np.asarray(value, dtype=np.float32))
            if len(ordered) >= 2:
                gap = float(ordered[-1] - ordered[-2])

        reasons = list(new_issues)
        if frames % max(1, args.stride) == 0:
            reasons.append("background")
        if controls.get("fire"):
            reasons.append("champion_fire")
        if danger >= args.danger_probability:
            reasons.append("predicted_danger")
        if (gap <= args.uncertainty_gap
                and float(metrics[2]) >= args.uncertainty_min_risk):
            reasons.append("low_confidence_risk")

        reason_limits = {
            "background": args.max_background_labels,
            "champion_fire": args.max_fire_labels,
            "predicted_danger": args.max_risk_labels,
            "low_confidence_risk": args.max_risk_labels,
        }
        eligible_reasons = [
            reason for reason in reasons
            if capture_counts[reason] < reason_limits.get(
                reason, args.max_issue_labels_per_type)
        ]
        urgent = any(reason in new_issues for reason in eligible_reasons)
        spaced = frames - last_capture >= args.min_gap_frames
        if (eligible_reasons and len(records) < args.max_labels_per_round
                and (spaced or urgent)):
            label = _label_champion_state(
                teacher, champion, game, metrics, controls, eligible_reasons,
                seed, frames)
            label["residual_target"] = bool(
                label["teacher_override"]
                and (
                    not label["champion_action_allowed"]
                    or (
                        label["teacher_advantage"]
                        >= args.min_override_advantage
                        and label["teacher_margin"]
                        >= args.min_teacher_margin
                    )
                )
            )
            records.append(label)
            capture_counts.update(eligible_reasons)
            last_capture = frames

        _set_controls(game, controls)
        tracker.pre_step()
        events = game.step()
        frames += 1
        tracker.post_step(events, 1)
        for event in events:
            if event[0] == "round_end":
                true_result = _round_result(event[1])
        if true_result:
            break

    audit = observer.result()
    return {
        "seed": int(seed),
        "result": true_result or "draw",
        "frames": int(frames),
        "shots": int(tracker.shots),
        "kills": int(tracker.kills),
        "hit_rate": tracker.kills / max(1, tracker.shots),
        "move_cells": tracker.move_px / tracker.scale,
        "death_cause": tracker.death_cause,
        "labels": len(records),
        "teacher_overrides": sum(
            bool(item["teacher_override"]) for item in records),
        "residual_targets": sum(
            bool(item["residual_target"]) for item in records),
        "champion_unsafe": sum(
            not bool(item["champion_action_allowed"]) for item in records),
        "capture_reasons": dict(Counter(
            reason for item in records
            for reason in item["capture_reason"].split(",") if reason)),
        "label_categories": dict(Counter(
            item["category"] for item in records)),
        **audit,
    }, records


def _champion_collect_worker(job):
    worker, seed, count, args, manifest = job
    import torch

    torch.set_num_threads(1)
    champion = P27BRiskValuePolicy(
        base_net=args.base_net,
        value_net=args.value_net,
        fire_margin=args.fire_margin,
    )
    teacher = make_frozen_teacher(args.base_net, args.value_net)
    rounds = []
    records = []
    for offset in range(count):
        result, labels = _champion_collect_round(
            champion, teacher, seed + offset, args)
        rounds.append(result)
        records.extend(labels)
        print(
            f"worker={worker} seed={seed + offset} {result['result']} "
            f"labels={len(labels)} overrides={result['teacher_overrides']} "
            f"unsafe={result['champion_unsafe']} issues={result['issues']}",
            flush=True,
        )
    shard = os.path.join(DATA_DIR, args.phase, f"shard_{worker}.npz")
    _save_shard(shard, records, manifest)
    return rounds, shard, len(records)


def champion_collect(args):
    manifest = verify_manifest(args.manifest)
    phase_dir = os.path.join(DATA_DIR, args.phase)
    if glob.glob(os.path.join(phase_dir, "shard_*.npz")):
        raise RuntimeError(
            f"phase already has shards; use a new phase name: {args.phase}")
    os.makedirs(phase_dir, exist_ok=True)
    workers = max(1, min(args.workers, args.rounds))
    base, remainder = divmod(args.rounds, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        jobs.append((worker, args.seed + offset, count, args, manifest))
        offset += count
    started = time.time()
    if workers == 1:
        outputs = [_champion_collect_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            outputs = pool.map(_champion_collect_worker, jobs)

    rounds = [row for part, _, _ in outputs for row in part]
    results = Counter(row["result"] for row in rounds)
    issues = Counter()
    captures = Counter()
    categories = Counter()
    for row in rounds:
        issues.update(row["issues"])
        captures.update(row["capture_reasons"])
        categories.update(row["label_categories"])
    labels = sum(count for _, _, count in outputs)
    overrides = sum(row["teacher_overrides"] for row in rounds)
    residual_targets = sum(row["residual_targets"] for row in rounds)
    unsafe = sum(row["champion_unsafe"] for row in rounds)
    report = {
        "teacher": FROZEN_TEACHER_NAME,
        "rollout_policy": "p27b_champion",
        "phase": args.phase,
        "seed": args.seed,
        "rounds": args.rounds,
        "results": dict(results),
        "win_rate": results["win"] / max(1, len(rounds)),
        "labelled_states": labels,
        "teacher_overrides": overrides,
        "teacher_override_rate": overrides / max(1, labels),
        "residual_targets": residual_targets,
        "residual_target_rate": residual_targets / max(1, labels),
        "champion_unsafe": unsafe,
        "champion_unsafe_rate": unsafe / max(1, labels),
        "capture_reasons": dict(captures),
        "label_categories": dict(categories),
        "behavior_issues": dict(issues),
        "elapsed_seconds": time.time() - started,
        "shards": [path for _, path, _ in outputs],
        "round_details": sorted(rounds, key=lambda row: row["seed"]),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _collect_worker(job):
    worker, seed, count, args, manifest = job
    import torch

    torch.set_num_threads(1)
    policy = make_frozen_teacher(args.base_net, args.value_net)
    rounds = []
    records = []
    for offset in range(count):
        result, labels = _collect_round(
            policy, seed + offset, args.stride, args.danger_threshold)
        rounds.append(result)
        records.extend(labels)
        print(
            f"worker={worker} seed={seed + offset} {result['result']} "
            f"labels={len(labels)} issues={result['issues']}",
            flush=True,
        )
    shard = os.path.join(DATA_DIR, args.phase, f"shard_{worker}.npz")
    _save_shard(shard, records, manifest)
    return rounds, shard, len(records)


def collect(args):
    if not os.path.exists(args.manifest):
        manifest = write_manifest(
            args.manifest, args.base_net, args.value_net)
    else:
        manifest = verify_manifest(args.manifest)
    phase_dir = os.path.join(DATA_DIR, args.phase)
    existing = glob.glob(os.path.join(phase_dir, "shard_*.npz"))
    if existing:
        raise RuntimeError(
            f"phase already has shards; use a new phase name: {args.phase}")
    os.makedirs(phase_dir, exist_ok=True)

    workers = max(1, min(args.workers, args.rounds))
    base, remainder = divmod(args.rounds, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        jobs.append((worker, args.seed + offset, count, args, manifest))
        offset += count
    started = time.time()
    if workers == 1:
        outputs = [_collect_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            outputs = pool.map(_collect_worker, jobs)

    rounds = [row for part, _, _ in outputs for row in part]
    shards = [path for _, path, _ in outputs]
    results = Counter(row["result"] for row in rounds)
    issues = Counter()
    categories = Counter()
    for row in rounds:
        issues.update(row["issues"])
        categories.update(row["label_categories"])
    report = {
        "teacher": FROZEN_TEACHER_NAME,
        "manifest": args.manifest,
        "phase": args.phase,
        "seed": args.seed,
        "rounds": args.rounds,
        "stride": args.stride,
        "elapsed_seconds": time.time() - started,
        "results": dict(results),
        "win_rate": results["win"] / max(1, len(rounds)),
        "labelled_states": sum(count for _, _, count in outputs),
        "label_categories": dict(categories),
        "behavior_issues": dict(issues),
        "shards": shards,
        "round_details": sorted(rounds, key=lambda row: row["seed"]),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _parse_seed_bands(value):
    bands = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            start_text, count_text = item.split(":", 1)
            start, count = int(start_text), int(count_text)
            if count <= 0:
                raise ValueError("seed-band count must be positive")
            seeds = list(range(start, start + count))
        else:
            seeds = [int(item)]
        bands.append({"name": item, "seeds": seeds})
    if not bands:
        raise ValueError("at least one seed band is required")
    return bands


def _teacher_eval_worker(job):
    worker, seeds, args = job
    import torch

    torch.set_num_threads(1)
    policy = make_frozen_teacher(args.base_net, args.value_net)
    rounds = []
    for seed in seeds:
        started = time.time()
        result, _ = _collect_round(
            policy, seed, stride=1, danger_threshold=0,
            collect_labels=False)
        result["elapsed_seconds"] = time.time() - started
        rounds.append(result)
        print(
            f"worker={worker} seed={seed} {result['result']} "
            f"frames={result['frames']} shots={result['shots']} "
            f"issues={result['issues']} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return rounds


def teacher_evaluate(args):
    verify_manifest(args.manifest)
    bands = _parse_seed_bands(args.seed_bands)
    all_seeds = [seed for band in bands for seed in band["seeds"]]
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("teacher evaluation seed bands overlap")
    workers = max(1, min(args.workers, len(all_seeds)))
    assignments = [all_seeds[index::workers] for index in range(workers)]
    jobs = [(worker, seeds, args)
            for worker, seeds in enumerate(assignments) if seeds]
    started = time.time()
    if len(jobs) == 1:
        outputs = [_teacher_eval_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            outputs = pool.map(_teacher_eval_worker, jobs)
    rounds = sorted(
        [row for part in outputs for row in part], key=lambda row: row["seed"])
    by_seed = {row["seed"]: row for row in rounds}
    results = Counter(row["result"] for row in rounds)
    issues = Counter()
    for row in rounds:
        issues.update(row["issues"])
    band_reports = []
    for band in bands:
        items = [by_seed[seed] for seed in band["seeds"]]
        counts = Counter(row["result"] for row in items)
        band_issues = Counter()
        for row in items:
            band_issues.update(row["issues"])
        band_reports.append({
            "name": band["name"],
            "games": len(items),
            "results": dict(counts),
            "win_rate": counts["win"] / max(1, len(items)),
            "behavior_issues": dict(band_issues),
            "avg_frames": sum(row["frames"] for row in items)
            / max(1, len(items)),
        })
    report = {
        "teacher": FROZEN_TEACHER_NAME,
        "manifest": args.manifest,
        "seed_bands": args.seed_bands,
        "games": len(rounds),
        "workers": workers,
        "results": dict(results),
        "win_rate": results["win"] / max(1, len(rounds)),
        "losses": results["loss"],
        "double_deaths": results["double_death"],
        "draws": results["draw"],
        "passed": (
            results["win"] >= args.min_wins
            and results["double_death"] <= args.max_double_deaths
        ),
        "gate": {
            "min_wins": args.min_wins,
            "max_double_deaths": args.max_double_deaths,
        },
        "elapsed_seconds": time.time() - started,
        "avg_round_elapsed_seconds": sum(
            row["elapsed_seconds"] for row in rounds) / max(1, len(rounds)),
        "avg_frames": sum(row["frames"] for row in rounds)
        / max(1, len(rounds)),
        "shots_per_game": sum(row["shots"] for row in rounds)
        / max(1, len(rounds)),
        "hit_rate": sum(row["kills"] for row in rounds)
        / max(1, sum(row["shots"] for row in rounds)),
        "move_cells_per_game": sum(row["move_cells"] for row in rounds)
        / max(1, len(rounds)),
        "behavior_issues": dict(issues),
        "bands": band_reports,
        "round_details": rounds,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        key: report[key] for key in (
            "teacher", "games", "results", "win_rate", "passed",
            "behavior_issues", "bands", "elapsed_seconds", "report")
        if key in report
    }, indent=2, sort_keys=True), flush=True)
    return report


def _load_exact_phase(phase):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, phase, "shard_*.npz")))
    if not paths:
        raise RuntimeError(f"no shards for exact phase {phase}")
    keys = (
        "X", "Y_score", "Y_aux", "Y_fire", "Y_fire_mask", "Y_action",
        "action_valid", "allowed", "W", "category", "round_seed", "frame",
    )
    rows = []
    for path in paths:
        data = np.load(path)
        objective = str(np.asarray(data["objective_version"]).reshape(-1)[0])
        if objective != OBJECTIVE_VERSION:
            raise RuntimeError(f"wrong objective in {path}: {objective}")
        rows.append({key: data[key] for key in keys})
    joined = {key: np.concatenate([row[key] for row in rows]) for key in keys}
    return joined, paths


def verify_phase(phase):
    data, paths = _load_exact_phase(phase)
    valid = data["action_valid"].astype(bool)
    action = data["Y_action"].astype(np.int64)
    rows = np.arange(len(action))[valid]
    action_argmax = data["Y_score"].argmax(axis=1)
    policy_mismatches = int((action_argmax[valid] != action[valid]).sum())
    selected_unsafe = int((~data["allowed"][rows, action[valid]]).sum())
    fire_rows = rows
    movements = action[valid] // 2
    fire_mask = data["Y_fire_mask"][fire_rows, movements] > 0.5
    fire_target = data["Y_fire"][fire_rows, movements] > 0.5
    fire_truth = action[valid] % 2 == 1
    fire_mismatches = int((fire_target[fire_mask] != fire_truth[fire_mask]).sum())
    report = {
        "phase": phase,
        "shards": paths,
        "states": int(len(action)),
        "valid_actions": int(valid.sum()),
        "action_valid_rate": float(valid.mean()),
        "policy_target_mismatches": policy_mismatches,
        "selected_unsafe": selected_unsafe,
        "selected_fire_label_mismatches": fire_mismatches,
        "finite": bool(
            np.isfinite(data["X"]).all()
            and np.isfinite(data["Y_score"]).all()
            and np.isfinite(data["Y_aux"]).all()),
        "categories": dict(Counter(data["category"].astype(str))),
    }
    report["passed"] = bool(
        report["finite"]
        and policy_mismatches == 0
        and selected_unsafe == 0
        and fire_mismatches == 0
        and valid.any())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["passed"]:
        raise RuntimeError("exact distillation phase consistency check failed")
    return report


def _group_split(seeds, fraction, split_seed):
    unique = np.unique(seeds)
    if len(unique) < 2:
        raise RuntimeError("training needs at least two complete round seeds")
    shuffled = np.random.default_rng(split_seed).permutation(unique)
    count = max(1, min(len(unique) - 1, int(round(len(unique) * fraction))))
    validation = set(int(value) for value in shuffled[:count])
    mask = np.asarray([int(seed) in validation for seed in seeds])
    return np.flatnonzero(~mask), np.flatnonzero(mask), sorted(validation)


def train(args):
    import torch
    import torch.nn.functional as functional

    verify_phase(args.phase)
    data, paths = _load_exact_phase(args.phase)
    train_index, val_index, val_seeds = _group_split(
        data["round_seed"], args.val_fraction, args.split_seed)
    torch.manual_seed(args.split_seed)
    payload = torch.load(args.init, weights_only=False)
    in_dim = int(payload["in_dim"])
    width = int(payload.get("width", 1024))
    if data["X"].shape[1] != in_dim:
        raise RuntimeError("exact labels do not match initial network input")
    network = build_p26_net(in_dim, width)
    network.load_state_dict(payload["state_dict"])
    if args.freeze_trunk:
        for parameter in network.trunk.parameters():
            parameter.requires_grad_(False)
    initial = {name: value.detach().clone()
               for name, value in network.named_parameters()}
    optimizer = torch.optim.AdamW(
        (parameter for parameter in network.parameters()
         if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)

    tensors = {
        key: torch.as_tensor(value)
        for key, value in data.items()
        if value.dtype.kind not in "USO"
    }
    tensors["critical"] = torch.as_tensor(
        data["category"].astype(str) != "standard")
    aux_positive_rate = tensors["Y_aux"][train_index].float().mean(
        dim=(0, 1)).clamp_min(1e-4)
    aux_positive_weight = ((1.0 - aux_positive_rate) / aux_positive_rate).clamp(
        min=1.0, max=args.aux_positive_weight_cap)
    with torch.no_grad():
        initial_output = network(tensors["X"])
        initial_score = initial_output["score"].detach().clone()

    def weighted(values, weights, mask=None):
        if mask is not None:
            values = values[mask]
            weights = weights[mask]
        if len(values) == 0:
            return values.sum() * 0.0
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def loss_for(index):
        output = network(tensors["X"][index])
        weights = tensors["W"][index]
        valid = tensors["action_valid"][index].bool()
        score = weighted(functional.smooth_l1_loss(
            output["score"], tensors["Y_score"][index], reduction="none",
            beta=args.huber_beta).mean(dim=1), weights)
        score_anchor = weighted(functional.smooth_l1_loss(
            output["score"], initial_score[index], reduction="none",
            beta=args.huber_beta).mean(dim=1), weights)
        hard_mask = valid & tensors["critical"][index]
        hard_policy = weighted(functional.cross_entropy(
            output["score"], tensors["Y_action"][index].clamp_min(0),
            reduction="none"), weights, hard_mask)
        target_policy = torch.softmax(
            tensors["Y_score"][index] / args.policy_temperature, dim=1)
        soft_policy_per_sample = -(
            target_policy * torch.log_softmax(
                output["score"] / args.policy_temperature, dim=1)
        ).sum(dim=1)
        soft_policy = weighted(soft_policy_per_sample, weights, valid)
        aux = weighted(functional.binary_cross_entropy_with_logits(
            output["aux"], tensors["Y_aux"][index],
            reduction="none", pos_weight=aux_positive_weight).mean(
                dim=(1, 2)), weights)
        fire_element = functional.binary_cross_entropy_with_logits(
            output["fire"], tensors["Y_fire"][index], reduction="none")
        fire_mask = tensors["Y_fire_mask"][index]
        fire_per_sample = (
            (fire_element * fire_mask).sum(dim=1)
            / fire_mask.sum(dim=1).clamp_min(1.0))
        fire = weighted(fire_per_sample, weights, fire_mask.sum(dim=1) > 0)
        anchor = sum((parameter - initial[name]).square().mean()
                     for name, parameter in network.named_parameters())
        total = (
            args.hard_policy_weight * hard_policy
            + args.soft_policy_weight * soft_policy
            + args.score_weight * score
            + args.score_anchor_weight * score_anchor
            + args.aux_weight * aux
            + args.fire_weight * fire
            + args.anchor_weight * anchor
        )
        return total, (
            hard_policy, soft_policy, score, score_anchor, aux, fire, anchor
        ), output

    def metrics():
        network.eval()
        index = torch.as_tensor(val_index)
        with torch.no_grad():
            total, parts, output = loss_for(index)
            valid = tensors["action_valid"][index].bool()
            action_acc = (output["score"].argmax(1)[valid]
                          == tensors["Y_action"][index][valid]).float().mean()
            target_danger = (
                (tensors["Y_aux"][index, :, 1] > 0.5)
                | (tensors["Y_aux"][index, :, 2] > 0.5))
            predicted_danger = (
                (torch.sigmoid(output["aux"][:, :, 1]) > 0.5)
                | (torch.sigmoid(output["aux"][:, :, 2]) > 0.5))
            danger_recall = (
                (predicted_danger & target_danger).sum()
                / target_danger.sum().clamp_min(1))
            fire_mask = tensors["Y_fire_mask"][index] > 0.5
            fire_acc = (
                ((torch.sigmoid(output["fire"]) > 0.5)
                 == (tensors["Y_fire"][index] > 0.5))[fire_mask]
                .float().mean()) if fire_mask.any() else torch.tensor(0.0)
        network.train()
        return {
            "total": float(total),
            "hard_policy": float(parts[0]),
            "soft_policy": float(parts[1]),
            "score": float(parts[2]),
            "score_anchor": float(parts[3]),
            "aux": float(parts[4]),
            "fire": float(parts[5]),
            "anchor": float(parts[6]),
            "action_accuracy": float(action_acc),
            "danger_recall": float(danger_recall),
            "fire_accuracy": float(fire_acc),
        }

    baseline_metrics = metrics()
    best_state = {name: value.detach().clone()
                  for name, value in network.state_dict().items()}
    best_metrics = baseline_metrics
    best_epoch = 0
    best_total = baseline_metrics["total"]
    stale = 0
    train_tensor = torch.as_tensor(train_index)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = train_tensor[torch.randperm(len(train_tensor))]
        running = 0.0
        batches = 0
        network.train()
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            loss = loss_for(index)[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        values = metrics()
        passes_regression_gate = (
            values["action_accuracy"]
            >= baseline_metrics["action_accuracy"]
            - args.max_action_accuracy_drop
            and values["danger_recall"]
            >= baseline_metrics["danger_recall"]
            - args.max_danger_recall_drop
        )
        if (passes_regression_gate
                and values["total"] < best_total - args.min_delta):
            best_total = values["total"]
            best_epoch = epoch
            best_metrics = values
            best_state = {name: value.detach().clone()
                          for name, value in network.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train={running/max(1, batches):.4f} "
            f"val={values['total']:.4f} action={values['action_accuracy']:.1%} "
            f"danger_recall={values['danger_recall']:.1%} "
            f"fire={values['fire_accuracy']:.1%} stale={stale} "
            f"gate={'pass' if passes_regression_gate else 'reject'} "
            f"elapsed={time.time()-started:.0f}s",
            flush=True,
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("exact distillation produced no checkpoint")
    network.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output_payload = {
        "state_dict": network.state_dict(),
        "in_dim": in_dim,
        "frame_stack": int(payload.get("frame_stack", 4)),
        "width": width,
        "aux_names": AUX_NAMES,
        "teacher": FROZEN_TEACHER_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "phase": args.phase,
        "init": args.init,
        "best_epoch": best_epoch,
        "best_val": best_total,
    }
    torch.save(output_payload, args.out)
    report = {
        "model": args.out,
        "teacher": FROZEN_TEACHER_NAME,
        "phase": args.phase,
        "init": args.init,
        "states": int(len(data["X"])),
        "rounds": int(len(np.unique(data["round_seed"]))),
        "train_states": int(len(train_index)),
        "val_states": int(len(val_index)),
        "val_seeds": val_seeds,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "baseline_metrics": baseline_metrics,
        "categories": dict(Counter(data["category"].astype(str))),
        "aux_positive_rate": aux_positive_rate.tolist(),
        "aux_positive_weight": aux_positive_weight.tolist(),
        "freeze_trunk": args.freeze_trunk,
        "shards": paths,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


class ExactDistilledPolicy:
    """Pure-network runtime with learned safety and fire gates."""

    name = "exact_teacher_distilled_network"

    def __init__(self, net, death_threshold=0.50, dd_threshold=0.50,
                 fire_threshold=0.50, fire_margin=0.0):
        import torch

        self.torch = torch
        self.network, self.frame_stack = load_p26_network(net)
        self.death_threshold = float(death_threshold)
        self.dd_threshold = float(dd_threshold)
        self.fire_threshold = float(fire_threshold)
        self.fire_margin = float(fire_margin)
        self.env = TankTroubleGym(seed=0, reward_version=1,
                                  terminal_mode="score", obs_traj=True,
                                  obs_nav=True)
        self.reset()

    def reset(self):
        self.game = None
        self.analyzer = None
        self.frames = 0
        self.history = []

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer360(game)
            self.frames = 0
            self.history = []
        observation, _ = build_observation(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        self.history.append(observation)
        stacked = stack_observation(self.history, self.frame_stack)
        with self.torch.no_grad():
            output = self.network(
                self.torch.as_tensor(stacked).unsqueeze(0))
        score = output["score"][0].numpy().copy()
        aux = 1.0 / (1.0 + np.exp(-output["aux"][0].numpy()))
        fire_probability = 1.0 / (
            1.0 + np.exp(-output["fire"][0].numpy()))
        danger = np.maximum(aux[:, 1], aux[:, 2])
        unsafe = ((aux[:, 1] >= self.death_threshold)
                  | (aux[:, 2] >= self.dd_threshold))
        paired = score.reshape(9, 2)
        for movement in range(9):
            if (fire_probability[movement] < self.fire_threshold
                    or paired[movement, 1] - paired[movement, 0]
                    < self.fire_margin):
                score[movement * 2 + 1] = -1e9
        safe_score = score.copy()
        safe_score[unsafe] = -1e9
        if np.max(safe_score) <= -1e8:
            index = int(np.lexsort((-score, danger))[0])
        else:
            index = int(safe_score.argmax())
        throttle, turn, fire = CANDIDATES[index]
        if len(game.tanks) > 1 and not game.tanks[1].alive:
            fire = 0
        return {
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": fire == 1,
        }


def _eval_round(policy, seed):
    game = Game(seed=seed, ai_enabled=True)
    policy.reset()
    tracker = RoundTracker(game)
    observer = BehaviorAudit()
    frames = 0
    result = None
    while frames < TRUNCATE_FRAMES or tracker.first_destroy is not None:
        analyzer = policy.analyzer or OpportunityAnalyzer360(game)
        metrics = analyzer.metrics(game)
        controls = policy.act(game)
        observer.observe(game, controls, metrics, frames)
        _set_controls(game, controls)
        tracker.pre_step()
        events = game.step()
        frames += 1
        tracker.post_step(events, 1)
        for event in events:
            if event[0] == "round_end":
                result = _round_result(event[1])
        if result:
            break
    return {
        "seed": seed,
        "result": result or "draw",
        "frames": frames,
        "shots": tracker.shots,
        "kills": tracker.kills,
        "hit_rate": tracker.kills / max(1, tracker.shots),
        "move_cells": tracker.move_px / tracker.scale,
        "death_cause": tracker.death_cause,
        **observer.result(),
    }


def evaluate(args):
    if args.runtime == "p27b":
        policy = P27BRiskValuePolicy(
            base_net=args.net,
            value_net=args.value_net,
            fire_margin=args.fire_margin,
        )
    else:
        policy = ExactDistilledPolicy(
            args.net, args.death_threshold, args.dd_threshold,
            args.fire_threshold, args.fire_margin)
    started = time.time()
    rounds = [_eval_round(policy, args.seed + offset)
              for offset in range(args.n)]
    results = Counter(row["result"] for row in rounds)
    issues = Counter()
    for row in rounds:
        issues.update(row["issues"])
    report = {
        "policy": policy.name,
        "runtime": args.runtime,
        "net": args.net,
        "seed": args.seed,
        "n": args.n,
        "results": dict(results),
        "win_rate": results["win"] / max(1, len(rounds)),
        "shots_per_game": sum(row["shots"] for row in rounds)
        / max(1, len(rounds)),
        "hit_rate": sum(row["kills"] for row in rounds)
        / max(1, sum(row["shots"] for row in rounds)),
        "move_cells_per_game": sum(row["move_cells"] for row in rounds)
        / max(1, len(rounds)),
        "behavior_issues": dict(issues),
        "elapsed_seconds": time.time() - started,
        "round_details": rounds,
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    freeze_parser.add_argument("--base-net", default=DEFAULT_BASE_NET)
    freeze_parser.add_argument("--value-net", default=DEFAULT_VALUE_NET)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--phase", required=True)
    collect_parser.add_argument("--rounds", type=int, default=2)
    collect_parser.add_argument("--seed", type=int, default=974000)
    collect_parser.add_argument("--workers", type=int, default=2)
    collect_parser.add_argument("--stride", type=int, default=4)
    collect_parser.add_argument("--danger-threshold", type=int, default=3)
    collect_parser.add_argument("--base-net", default=DEFAULT_BASE_NET)
    collect_parser.add_argument("--value-net", default=DEFAULT_VALUE_NET)
    collect_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    collect_parser.add_argument("--report", default=(
        "training/analysis/runs/exact_teacher_distill_collect.json"))

    champion_collect_parser = subparsers.add_parser("champion-collect")
    champion_collect_parser.add_argument("--phase", required=True)
    champion_collect_parser.add_argument("--rounds", type=int, default=2)
    champion_collect_parser.add_argument("--seed", type=int, default=982000)
    champion_collect_parser.add_argument("--workers", type=int, default=2)
    champion_collect_parser.add_argument("--stride", type=int, default=48)
    champion_collect_parser.add_argument("--min-gap-frames", type=int,
                                         default=12)
    champion_collect_parser.add_argument("--max-labels-per-round", type=int,
                                         default=12)
    champion_collect_parser.add_argument("--max-background-labels", type=int,
                                         default=2)
    champion_collect_parser.add_argument("--max-fire-labels", type=int,
                                         default=4)
    champion_collect_parser.add_argument("--max-risk-labels", type=int,
                                         default=4)
    champion_collect_parser.add_argument("--max-issue-labels-per-type",
                                         type=int, default=3)
    champion_collect_parser.add_argument("--min-override-advantage",
                                         type=float, default=0.03)
    champion_collect_parser.add_argument("--min-teacher-margin", type=float,
                                         default=0.01)
    champion_collect_parser.add_argument("--danger-probability", type=float,
                                         default=0.45)
    champion_collect_parser.add_argument("--uncertainty-gap", type=float,
                                         default=0.04)
    champion_collect_parser.add_argument("--uncertainty-min-risk", type=float,
                                         default=0.35)
    champion_collect_parser.add_argument("--fire-margin", type=float,
                                         default=0.16)
    champion_collect_parser.add_argument("--base-net",
                                         default=DEFAULT_BASE_NET)
    champion_collect_parser.add_argument("--value-net",
                                         default=DEFAULT_VALUE_NET)
    champion_collect_parser.add_argument("--manifest",
                                         default=DEFAULT_MANIFEST)
    champion_collect_parser.add_argument("--report", required=True)

    teacher_eval_parser = subparsers.add_parser("teacher-eval")
    teacher_eval_parser.add_argument("--seed-bands", required=True)
    teacher_eval_parser.add_argument("--workers", type=int, default=12)
    teacher_eval_parser.add_argument("--min-wins", type=int, default=297)
    teacher_eval_parser.add_argument("--max-double-deaths", type=int,
                                     default=1)
    teacher_eval_parser.add_argument("--base-net", default=DEFAULT_BASE_NET)
    teacher_eval_parser.add_argument("--value-net", default=DEFAULT_VALUE_NET)
    teacher_eval_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    teacher_eval_parser.add_argument("--report", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--phase", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--phase", required=True)
    train_parser.add_argument("--init", default=DEFAULT_BASE_NET)
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--report", required=True)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--min-epochs", type=int, default=15)
    train_parser.add_argument("--patience", type=int, default=12)
    train_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_parser.add_argument("--batch", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=8e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--grad-clip", type=float, default=2.0)
    train_parser.add_argument("--val-fraction", type=float, default=0.20)
    train_parser.add_argument("--split-seed", type=int, default=31001)
    train_parser.add_argument("--huber-beta", type=float, default=0.35)
    train_parser.add_argument("--hard-policy-weight", type=float, default=0.35)
    train_parser.add_argument("--soft-policy-weight", type=float, default=0.65)
    train_parser.add_argument("--policy-temperature", type=float, default=0.35)
    train_parser.add_argument("--score-weight", type=float, default=0.0)
    train_parser.add_argument("--score-anchor-weight", type=float, default=1.0)
    train_parser.add_argument("--aux-weight", type=float, default=0.35)
    train_parser.add_argument("--aux-positive-weight-cap", type=float,
                              default=12.0)
    train_parser.add_argument("--fire-weight", type=float, default=0.35)
    train_parser.add_argument("--anchor-weight", type=float, default=0.02)
    train_parser.add_argument("--freeze-trunk",
                              action=argparse.BooleanOptionalAction,
                              default=True)
    train_parser.add_argument("--max-action-accuracy-drop", type=float,
                              default=0.02)
    train_parser.add_argument("--max-danger-recall-drop", type=float,
                              default=0.02)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--net", required=True)
    eval_parser.add_argument("--runtime", choices=("p27b", "hard_gate"),
                             default="p27b")
    eval_parser.add_argument("--value-net", default=DEFAULT_VALUE_NET)
    eval_parser.add_argument("--n", type=int, default=20)
    eval_parser.add_argument("--seed", type=int, default=970000)
    eval_parser.add_argument("--death-threshold", type=float, default=0.50)
    eval_parser.add_argument("--dd-threshold", type=float, default=0.50)
    eval_parser.add_argument("--fire-threshold", type=float, default=0.50)
    eval_parser.add_argument("--fire-margin", type=float, default=0.16)
    eval_parser.add_argument("--report", default=None)

    args = parser.parse_args()
    if args.mode == "freeze":
        print(json.dumps(write_manifest(
            args.manifest, args.base_net, args.value_net),
            indent=2, sort_keys=True))
    elif args.mode == "collect":
        collect(args)
    elif args.mode == "champion-collect":
        champion_collect(args)
    elif args.mode == "teacher-eval":
        teacher_evaluate(args)
    elif args.mode == "verify":
        verify_phase(args.phase)
    elif args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
