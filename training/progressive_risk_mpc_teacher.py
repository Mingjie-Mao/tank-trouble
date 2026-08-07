"""Progressive, closed-loop, risk-constrained MPC search teacher.

The current P28/P30 teacher restarts every h48/h72/h96 rollout and keeps one
root action active for almost the whole horizon. This method reuses branches
between horizons, shares Laika RNG seeds across candidate actions, and switches
to a cheap navigation/aiming macro controller after a short root commitment.

This remains a teacher/search policy. It is intentionally separate from the
deployment champion and does not replace P27b or P30 checkpoints.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import Game  # noqa: E402
from training.evaluate import play_round_dual_engine  # noqa: E402
from training.mpc_agent import CANDIDATES, make_sandbox  # noqa: E402
from training.opportunity_distill import (  # noqa: E402
    GOOD_FIRE_BONUS,
    PRESSURE_BONUS,
    SUICIDE_FIRE_PENALTY,
)
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    AUX_HORIZONS,
    SCORE_SCALE,
    build_observation,
    select_action,
    stack_observation,
)
from training.p27_risk_value import P27BRiskValuePolicy, _controls  # noqa: E402


def _csv_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _set_controls(tank, action):
    throttle, turn, fire = action
    tank.forward = throttle == 2
    tank.backup = throttle == 0
    tank.turn_left = turn == 0
    tank.turn_right = turn == 2
    tank.fire = fire == 1


def _movement_group(action_index):
    """Return the throttle/turn group for an adjacent no-fire/fire pair."""
    return int(action_index) // 2


def _movement_pair(group_index):
    no_fire = int(group_index) * 2
    return no_fire, no_fire + 1


def _action_shot_event(game, action):
    """Evaluate the shot after the selected one-frame turn is applied."""
    if action[2] != 1:
        return None
    me, enemy = game.tanks[0], game.tanks[1]
    if not (me.alive and enemy.alive and me.trigger_released
            and game.weapon_ready(me)):
        return None

    from tank_trouble_original.laika import LaikaAI

    sandbox = make_sandbox(game, "L1", rng_seed=0)
    _set_controls(sandbox.tanks[0], action)
    events = sandbox.step()
    if not any(event[0] == "fire" and event[1] == 0 for event in events):
        return None
    shooter = sandbox.tanks[0]
    return LaikaAI(sandbox, shooter).check_bullet_path(shooter.rotation)


def _credible_root_fire(game, action, metrics, min_line=0.35,
                        max_alignment=0.30, pressure_radius=0.75):
    """Hard gate for root fire; movement remains available as a paired action."""
    shot = _action_shot_event(game, action)
    if shot is None:
        return False, shot, "unavailable"
    if shot.get("result") == "HIT":
        return True, shot, "predicted_hit"
    if shot.get("closest", float("inf")) <= pressure_radius * game.scale:
        return True, shot, "pressure"

    line = float(metrics[0])
    direction_x, direction_y = [float(value) for value in metrics[3:5]]
    alignment = abs(math.atan2(direction_y, direction_x)) if line > 0.0 else math.pi
    if line >= min_line and alignment <= max_alignment:
        return True, shot, "aligned_line"
    return False, shot, "blind"


class _ProgressiveBranch:
    def __init__(self, game, action_index, rng_seed, root_metrics,
                 commit_frames, replan_interval, macro_fire_line,
                 macro_fire_max_risk, macro_turn_deadzone,
                 macro_move_alignment, root_fire_min_line,
                 root_fire_max_alignment, root_fire_pressure_radius,
                 root_fire_check=None):
        self.game = make_sandbox(game, "L2", rng_seed=rng_seed)
        self.action_index = int(action_index)
        self.root_action = CANDIDATES[self.action_index]
        self.analyzer = OpportunityAnalyzer360(self.game)
        self.root_metrics = np.asarray(root_metrics, dtype=np.float32)
        self.root_potential = self.analyzer.potential(self.root_metrics)
        self.commit_frames = int(commit_frames)
        self.replan_interval = int(replan_interval)
        self.macro_fire_line = float(macro_fire_line)
        self.macro_fire_max_risk = float(macro_fire_max_risk)
        self.macro_turn_deadzone = float(macro_turn_deadzone)
        self.macro_move_alignment = float(macro_move_alignment)
        self.frame = 0
        self.simulated_frames = 0
        self.me_dead_frame = None
        self.enemy_dead_frame = None
        self.true_result = None
        self.last_fire_frame = None
        if self.root_action[2] == 1:
            if root_fire_check is None:
                root_fire_check = _credible_root_fire(
                    game, self.root_action, self.root_metrics,
                    min_line=root_fire_min_line,
                    max_alignment=root_fire_max_alignment,
                    pressure_radius=root_fire_pressure_radius,
                )
            (self.root_fire_allowed, self.shot,
             self.root_fire_reason) = root_fire_check
        else:
            self.root_fire_allowed = False
            self.root_fire_reason = "no_fire"
            self.shot = None
        self.fired = False
        self.stage_survival = {}
        _set_controls(self.game.tanks[0], self.root_action)
        if self.root_action[2] == 1:
            self.last_fire_frame = 0

    @property
    def finished(self):
        return self.true_result is not None

    def _macro_action(self):
        me, enemy = self.game.tanks[0], self.game.tanks[1]
        if not me.alive:
            return (1, 1, 0)
        metrics = self.analyzer.metrics(self.game)
        line, _, risk, direction_x, direction_y = [
            float(value) for value in metrics]
        angle_error = math.atan2(direction_y, direction_x)
        if angle_error > self.macro_turn_deadzone:
            turn = 2
        elif angle_error < -self.macro_turn_deadzone:
            turn = 0
        else:
            turn = 1

        alignment = abs(angle_error)
        if alignment <= self.macro_move_alignment:
            throttle = 2
        elif alignment >= math.pi - 0.45:
            throttle = 0
        else:
            throttle = 1
        fire = int(
            enemy.alive
            and line >= self.macro_fire_line
            and risk <= self.macro_fire_max_risk
            and alignment <= max(0.16, self.macro_turn_deadzone * 1.5)
        )
        return throttle, turn, fire

    def advance(self, target_frame):
        me = self.game.tanks[0]
        enemy = self.game.tanks[1]
        while self.frame < target_frame and not self.finished:
            if (self.last_fire_frame is not None
                    and self.frame == self.last_fire_frame + 1):
                me.fire = False
            if (self.frame >= self.commit_frames
                    and (self.frame - self.commit_frames)
                    % max(1, self.replan_interval) == 0):
                action = self._macro_action()
                _set_controls(me, action)
                if action[2] == 1:
                    self.last_fire_frame = self.frame

            events = self.game.step()
            self.simulated_frames += 1
            self.fired = self.fired or any(
                event[0] == "fire" and event[1] == 0 for event in events)
            if self.me_dead_frame is None and not me.alive:
                self.me_dead_frame = self.frame
            if self.enemy_dead_frame is None and not enemy.alive:
                self.enemy_dead_frame = self.frame
            self.frame += 1
            if self.frame in AUX_HORIZONS:
                self.stage_survival[self.frame] = float(me.alive)
            for event in events:
                if event[0] == "round_end":
                    winner = event[1]
                    self.true_result = (
                        "win" if winner == 0 else
                        "loss" if winner == 1 else "double_death")

    def score(self):
        me, enemy = self.game.tanks[0], self.game.tanks[1]
        enemy_frame = (self.enemy_dead_frame if self.enemy_dead_frame is not None
                       else self.frame)
        me_frame = (self.me_dead_frame if self.me_dead_frame is not None
                    else self.frame)
        if self.true_result == "win":
            return 1000.0 - float(enemy_frame)
        if self.true_result == "loss":
            return -1000.0 + float(me_frame)
        if self.true_result == "double_death":
            return -900.0 + float(me_frame)
        if self.me_dead_frame is not None:
            return -1000.0 + float(self.me_dead_frame)
        if self.enemy_dead_frame is not None and me.alive:
            return 1000.0 - float(self.enemy_dead_frame)

        metrics = self.analyzer.metrics(self.game)
        score = self.analyzer.potential(metrics) - self.root_potential
        if self.fired and self.shot is not None:
            result = self.shot["result"]
            if result == "HIT" and self.root_metrics[0] >= 0.60:
                score += GOOD_FIRE_BONUS
            elif result == "SUICIDE":
                score -= SUICIDE_FIRE_PENALTY
            elif self.shot.get("closest", float("inf")) <= (
                    0.75 * self.game.scale):
                score += PRESSURE_BONUS
        return float(score)

    def aux(self):
        if self.true_result == "win":
            kill, death, double_death = 1.0, 0.0, 0.0
        elif self.true_result == "loss":
            kill, death, double_death = 0.0, 1.0, 0.0
        elif self.true_result == "double_death":
            kill, death, double_death = 1.0, 1.0, 1.0
        else:
            kill = float(self.enemy_dead_frame is not None)
            death = float(self.me_dead_frame is not None)
            double_death = float(bool(kill and death))
        survival = []
        for horizon in AUX_HORIZONS:
            if horizon in self.stage_survival:
                survival.append(self.stage_survival[horizon])
            elif death:
                survival.append(float(
                    self.me_dead_frame is not None
                    and self.me_dead_frame >= horizon))
            else:
                survival.append(1.0)
        return np.asarray(
            [kill, death, double_death, *survival], dtype=np.float32)


class ProgressiveSearchEngine:
    def __init__(self, horizons=(24, 48, 72, 96), widths=(6, 3, 2, 2),
                 final_samples=4, commit_frames=24, replan_interval=16,
                 death_penalty=0.55, dd_penalty=1.00, kill_bonus=0.04,
                 tail_penalty=0.15, max_death=0.0, max_dd=0.0,
                 prior_bonus=0.004, macro_fire_line=0.76,
                 macro_fire_max_risk=0.30, macro_turn_deadzone=0.16,
                 macro_move_alignment=0.72, fire_min_gain=0.015,
                 fire_max_extra_death=0.0, fire_max_extra_dd=0.0,
                 root_fire_min_line=0.35,
                 root_fire_max_alignment=0.30,
                 root_fire_pressure_radius=0.75):
        self.horizons = tuple(int(value) for value in horizons)
        self.widths = tuple(int(value) for value in widths)
        if len(self.horizons) != len(self.widths):
            raise ValueError(
                "progressive MPC horizons and widths must have equal length")
        if sorted(self.horizons) != list(self.horizons):
            raise ValueError("progressive MPC horizons must be increasing")
        self.final_samples = max(1, int(final_samples))
        self.commit_frames = int(commit_frames)
        self.replan_interval = int(replan_interval)
        self.death_penalty = float(death_penalty)
        self.dd_penalty = float(dd_penalty)
        self.kill_bonus = float(kill_bonus)
        self.tail_penalty = float(tail_penalty)
        self.max_death = float(max_death)
        self.max_dd = float(max_dd)
        self.prior_bonus = float(prior_bonus)
        self.fire_min_gain = float(fire_min_gain)
        self.fire_max_extra_death = float(fire_max_extra_death)
        self.fire_max_extra_dd = float(fire_max_extra_dd)
        self.branch_kwargs = {
            "commit_frames": commit_frames,
            "replan_interval": replan_interval,
            "macro_fire_line": macro_fire_line,
            "macro_fire_max_risk": macro_fire_max_risk,
            "macro_turn_deadzone": macro_turn_deadzone,
            "macro_move_alignment": macro_move_alignment,
            "root_fire_min_line": root_fire_min_line,
            "root_fire_max_alignment": root_fire_max_alignment,
            "root_fire_pressure_radius": root_fire_pressure_radius,
        }

    def _new_branch(self, game, action_index, seed, metrics,
                    root_fire_checks=None):
        root_fire_check = None
        if root_fire_checks is not None:
            root_fire_check = root_fire_checks.get(int(action_index))
        return _ProgressiveBranch(
            game, action_index, seed, metrics,
            root_fire_check=root_fire_check, **self.branch_kwargs)

    def _value(self, score, aux, score_std=0.0):
        return (
            float(score) / SCORE_SCALE
            - self.death_penalty * float(aux[1])
            - self.dd_penalty * float(aux[2])
            + self.kill_bonus * float(aux[0])
            - self.tail_penalty * float(score_std) / SCORE_SCALE
        )

    @staticmethod
    def _keep_diverse_groups(order, width, base_group):
        selected = []
        must_include = [base_group]
        if width >= 3:
            represented = {base_group // 3}
            for throttle in (0, 1, 2):
                if throttle in represented:
                    continue
                must_include.append(next(
                    (group for group in order if group // 3 == throttle), None))
        for group in must_include + list(order):
            if group is None or group in selected:
                continue
            selected.append(int(group))
            if len(selected) >= width:
                break
        return selected

    def _safe(self, row):
        return (row["aux"][1] <= self.max_death
                and row["aux"][2] <= self.max_dd)

    def _select_pair(self, group, final, root_branches):
        no_fire, fire = _movement_pair(group)
        no_row = final[no_fire]
        fire_row = final[fire]
        fire_allowed = root_branches[fire].root_fire_allowed
        gain = float(fire_row["value"] - no_row["value"])
        extra_death = float(fire_row["aux"][1] - no_row["aux"][1])
        extra_dd = float(fire_row["aux"][2] - no_row["aux"][2])
        no_safe = self._safe(no_row)
        fire_safe = self._safe(fire_row)

        reason = root_branches[fire].root_fire_reason
        selected = no_fire
        if fire_allowed and fire_safe and not no_safe:
            selected = fire
            reason = "fire_only_safe"
        elif (fire_allowed and fire_safe and no_safe
              and gain >= self.fire_min_gain
              and extra_death <= self.fire_max_extra_death
              and extra_dd <= self.fire_max_extra_dd):
            selected = fire
            reason = "positive_paired_gain"
        elif fire_allowed:
            reason = "insufficient_paired_gain"

        return selected, {
            "group": int(group),
            "no_fire": int(no_fire),
            "fire": int(fire),
            "selected": int(selected),
            "fire_allowed": bool(fire_allowed),
            "fire_reason": reason,
            "fire_gain": gain,
            "fire_extra_death": extra_death,
            "fire_extra_dd": extra_dd,
            "no_fire_safe": bool(no_safe),
            "fire_safe": bool(fire_safe),
        }

    def search(self, game, analyzer, base_index, root_seed):
        metrics = analyzer.metrics(game)
        common_seed = int(root_seed % (1 << 30))
        root_fire_checks = {}
        for group in range(len(CANDIDATES) // 2):
            _, fire = _movement_pair(group)
            action = CANDIDATES[fire]
            root_fire_checks[fire] = _credible_root_fire(
                game, action, metrics,
                min_line=self.branch_kwargs["root_fire_min_line"],
                max_alignment=self.branch_kwargs[
                    "root_fire_max_alignment"],
                pressure_radius=self.branch_kwargs[
                    "root_fire_pressure_radius"],
            )
        branches = {
            index: self._new_branch(
                game, index, common_seed, metrics, root_fire_checks)
            for index in range(len(CANDIDATES))
        }
        base_group = _movement_group(base_index)
        active_groups = list(range(len(CANDIDATES) // 2))
        stage_rows = []
        simulated_frames = 0

        for horizon, width in zip(self.horizons, self.widths):
            active = [
                index for group in active_groups
                for index in _movement_pair(group)
            ]
            for index in active:
                before = branches[index].simulated_frames
                branches[index].advance(horizon)
                simulated_frames += branches[index].simulated_frames - before
            values = {}
            for index in active:
                branch = branches[index]
                values[index] = self._value(branch.score(), branch.aux())
                if index == base_index:
                    values[index] += self.prior_bonus
            group_best = {}
            group_action = {}
            for group in active_groups:
                no_fire, fire = _movement_pair(group)
                eligible = [no_fire]
                if branches[fire].root_fire_allowed:
                    eligible.append(fire)
                best_action = max(eligible, key=lambda index: values[index])
                group_best[group] = values[best_action]
                group_action[group] = best_action
            order = sorted(active_groups, key=lambda group: group_best[group],
                           reverse=True)
            keep = min(max(1, width), len(order))
            active_groups = self._keep_diverse_groups(
                order, keep, base_group)
            stage_rows.append({
                "horizon": int(horizon),
                "evaluated_groups": int(len(order)),
                "kept_groups": list(active_groups),
                "best_group": int(order[0]),
                "best_action": int(group_action[order[0]]),
                "best_value": float(group_best[order[0]]),
            })

        final_horizon = self.horizons[-1]
        active = [
            index for group in active_groups
            for index in _movement_pair(group)
        ]
        samples = {
            index: [(branches[index].score(), branches[index].aux())]
            for index in active
        }
        for sample in range(1, self.final_samples):
            sample_seed = int((root_seed + sample * 17_071) % (1 << 30))
            for index in active:
                branch = self._new_branch(
                    game, index, sample_seed, metrics, root_fire_checks)
                branch.advance(final_horizon)
                simulated_frames += branch.simulated_frames
                samples[index].append((branch.score(), branch.aux()))

        final = {}
        for index in active:
            scores = np.asarray(
                [item[0] for item in samples[index]], dtype=np.float32)
            aux = np.mean(
                [item[1] for item in samples[index]], axis=0).astype(
                    np.float32)
            final[index] = {
                "score": float(scores.mean()),
                "score_std": float(scores.std()),
                "aux": aux,
                "value": self._value(scores.mean(), aux, scores.std()),
            }

        pair_rows = {}
        group_choices = []
        for group in active_groups:
            selected, row = self._select_pair(group, final, branches)
            pair_rows[str(group)] = row
            group_choices.append(selected)

        safe = [index for index in group_choices if self._safe(final[index])]
        if safe:
            best = max(safe, key=lambda index: final[index]["value"])
            selection = (
                "safe_value_fire" if CANDIDATES[best][2] == 1
                else "safe_value_no_fire")
        else:
            best = min(
                group_choices,
                key=lambda index: (
                    float(final[index]["aux"][1]),
                    float(final[index]["aux"][2]),
                    -float(final[index]["value"]),
                ),
            )
            selection = "minimum_risk_fallback"

        diagnostics = {
            "base_index": int(base_index),
            "best_index": int(best),
            "selection": selection,
            "simulated_frames": int(simulated_frames),
            "stages": stage_rows,
            "active_groups": list(active_groups),
            "fire_pairs": pair_rows,
            "final": {
                str(index): {
                    "action": list(CANDIDATES[index]),
                    "score": row["score"],
                    "score_std": row["score_std"],
                    "value": row["value"],
                    "aux": row["aux"].tolist(),
                }
                for index, row in final.items()
            },
        }
        return CANDIDATES[best], diagnostics


class ProgressiveRiskMPCPolicy(P27BRiskValuePolicy):
    name = "progressive_risk_mpc_teacher"

    def __init__(self, base_net, value_net, seed=0,
                 deterministic_search_seeds=True, **kwargs):
        search_keys = {
            "horizons", "widths", "final_samples", "commit_frames",
            "replan_interval", "death_penalty", "dd_penalty", "kill_bonus",
            "tail_penalty", "max_death", "max_dd", "prior_bonus",
            "macro_fire_line", "macro_fire_max_risk",
            "macro_turn_deadzone", "macro_move_alignment", "fire_min_gain",
            "fire_max_extra_death", "fire_max_extra_dd",
            "root_fire_min_line", "root_fire_max_alignment",
            "root_fire_pressure_radius",
        }
        search_args = {
            key: kwargs.pop(key) for key in list(kwargs) if key in search_keys
        }
        super().__init__(base_net=base_net, value_net=value_net, **kwargs)
        self.search = ProgressiveSearchEngine(**search_args)
        self.rng = random.Random(seed)
        self.deterministic_search_seeds = bool(deterministic_search_seeds)
        self.round_seed = 0
        self.search_counts = {}
        self.search_frames = 0

    def set_round_seed(self, seed):
        self.round_seed = int(seed)

    def reset(self):
        super().reset()
        self.search_counts = {}
        self.search_frames = 0

    def _root_seed(self):
        if not self.deterministic_search_seeds:
            return self.rng.randrange(1 << 30)
        return int((
            self.round_seed * 1_000_003
            + self.frames * 9_176
            + 31_001
        ) % (1 << 30))

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
            output = self.base_net(
                self.torch.as_tensor(stacked).unsqueeze(0))
        outputs = {
            "score": output["score"][0].numpy(),
            "aux": output["aux"][0].numpy(),
            "fire": output["fire"][0].numpy(),
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

        action, diagnostics = self.search.search(
            game, self.analyzer, p27_index, self._root_seed())
        self.search_frames += diagnostics["simulated_frames"]
        key = diagnostics["selection"]
        self.search_counts[key] = self.search_counts.get(key, 0) + 1
        if diagnostics["best_index"] != p27_index:
            self.search_counts["override"] = (
                self.search_counts.get("override", 0) + 1)
        selected_group = str(_movement_group(diagnostics["best_index"]))
        pair = diagnostics["fire_pairs"][selected_group]
        fire_key = (
            "fire_selected" if CANDIDATES[diagnostics["best_index"]][2] == 1
            else f"fire_suppressed_{pair['fire_reason']}"
        )
        self.search_counts[fire_key] = self.search_counts.get(fire_key, 0) + 1
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


def _policy(args, worker=0):
    return ProgressiveRiskMPCPolicy(
        base_net=args.base_net,
        value_net=args.value_net,
        fire_margin=args.fire_margin,
        seed=args.seed + worker * 10007,
        deterministic_search_seeds=args.deterministic_search_seeds,
        horizons=_csv_ints(args.horizons),
        widths=_csv_ints(args.widths),
        final_samples=args.final_samples,
        commit_frames=args.commit_frames,
        replan_interval=args.replan_interval,
        death_penalty=args.death_penalty,
        dd_penalty=args.dd_penalty,
        kill_bonus=args.kill_bonus,
        tail_penalty=args.tail_penalty,
        max_death=args.max_death,
        max_dd=args.max_dd,
        prior_bonus=args.prior_bonus,
        macro_fire_line=args.macro_fire_line,
        macro_fire_max_risk=args.macro_fire_max_risk,
        macro_turn_deadzone=args.macro_turn_deadzone,
        macro_move_alignment=args.macro_move_alignment,
        fire_min_gain=args.fire_min_gain,
        fire_max_extra_death=args.fire_max_extra_death,
        fire_max_extra_dd=args.fire_max_extra_dd,
        root_fire_min_line=args.root_fire_min_line,
        root_fire_max_alignment=args.root_fire_max_alignment,
        root_fire_pressure_radius=args.root_fire_pressure_radius,
    )


def probe(args):
    game = Game(seed=args.seed, ai_enabled=True)
    policy = _policy(args)
    policy.set_round_seed(args.seed)
    policy.game = game
    policy.analyzer = OpportunityAnalyzer360(game)
    observation, metrics = build_observation(
        policy.env, game, policy.analyzer, 0)
    policy.history.append(observation)
    stacked = stack_observation(policy.history, policy.frame_stack)
    with policy.torch.no_grad():
        output = policy.base_net(
            policy.torch.as_tensor(stacked).unsqueeze(0))
    outputs = {
        "score": output["score"][0].numpy(),
        "aux": output["aux"][0].numpy(),
        "fire": output["fire"][0].numpy(),
    }
    base_action = select_action(
        outputs, CANDIDATES, policy.fire_margin,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    base_index = CANDIDATES.index(base_action)
    started = time.time()
    action, diagnostics = policy.search.search(
        game, policy.analyzer, base_index, policy._root_seed())
    diagnostics["seed"] = args.seed
    diagnostics["base_action"] = list(base_action)
    diagnostics["selected_action"] = list(action)
    diagnostics["wall_seconds"] = time.time() - started
    print(json.dumps(diagnostics, indent=2, sort_keys=True), flush=True)
    if args.summary:
        os.makedirs(os.path.dirname(args.summary), exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2, sort_keys=True)
    return diagnostics


def _eval_worker(job):
    worker, seed, count, args = job
    import torch

    torch.set_num_threads(1)
    policy = _policy(args, worker)
    rounds = []
    counts = Counter()
    simulated_frames = 0
    for offset in range(count):
        round_seed = seed + offset
        policy.set_round_seed(round_seed)
        result = play_round_dual_engine(policy, round_seed)
        result["seed"] = round_seed
        result["search_counts"] = dict(policy.search_counts)
        result["simulated_frames"] = int(policy.search_frames)
        rounds.append(result)
        counts.update(policy.search_counts)
        simulated_frames += policy.search_frames
    return rounds, counts, simulated_frames


def evaluate(args):
    workers = max(1, min(args.workers, args.n))
    base, remainder = divmod(args.n, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count:
            jobs.append((worker, args.seed + offset, count, args))
            offset += count
    started = time.time()
    if workers == 1:
        outputs = [_eval_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            outputs = pool.map(_eval_worker, jobs)
    rounds = [row for part, _, _ in outputs for row in part]
    counts = Counter()
    simulated_frames = 0
    for _, worker_counts, worker_frames in outputs:
        counts.update(worker_counts)
        simulated_frames += worker_frames
    total = max(1, len(rounds))
    result_count = Counter(row["true_result"] for row in rounds)
    shots = sum(row["shots"] for row in rounds)
    kills = sum(row["kills"] for row in rounds)
    elapsed = time.time() - started
    summary = {
        "n": len(rounds),
        "seed": args.seed,
        "results": dict(result_count),
        "win_rate": result_count["win"] / total,
        "loss_rate": result_count["loss"] / total,
        "double_death_rate": result_count["double_death"] / total,
        "shots_per_game": shots / total,
        "hit_rate": kills / max(1, shots),
        "avg_seconds": sum(row["frames"] for row in rounds) / total / 25.0,
        "wall_seconds": elapsed,
        "wall_seconds_per_game": elapsed / total,
        "simulated_frames": simulated_frames,
        "search_counts": dict(counts),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in sorted(rounds, key=lambda item: item["seed"]):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    if args.summary:
        os.makedirs(os.path.dirname(args.summary), exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "eval"])
    parser.add_argument("--base-net", default=(
        "training/models/p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--value-net", default=(
        "training/models/p27b_risk_value_iter00.pt"))
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=973034)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--horizons", default="24,48,72,96")
    parser.add_argument("--widths", default="6,3,2,2")
    parser.add_argument("--final-samples", type=int, default=4)
    parser.add_argument("--commit-frames", type=int, default=24)
    parser.add_argument("--replan-interval", type=int, default=16)
    parser.add_argument("--death-penalty", type=float, default=0.55)
    parser.add_argument("--dd-penalty", type=float, default=1.0)
    parser.add_argument("--kill-bonus", type=float, default=0.04)
    parser.add_argument("--tail-penalty", type=float, default=0.15)
    parser.add_argument("--max-death", type=float, default=0.0)
    parser.add_argument("--max-dd", type=float, default=0.0)
    parser.add_argument("--prior-bonus", type=float, default=0.004)
    parser.add_argument("--macro-fire-line", type=float, default=0.76)
    parser.add_argument("--macro-fire-max-risk", type=float, default=0.30)
    parser.add_argument("--macro-turn-deadzone", type=float, default=0.16)
    parser.add_argument("--macro-move-alignment", type=float, default=0.72)
    parser.add_argument("--fire-min-gain", type=float, default=0.015)
    parser.add_argument("--fire-max-extra-death", type=float, default=0.0)
    parser.add_argument("--fire-max-extra-dd", type=float, default=0.0)
    parser.add_argument("--root-fire-min-line", type=float, default=0.35)
    parser.add_argument("--root-fire-max-alignment", type=float, default=0.30)
    parser.add_argument("--root-fire-pressure-radius", type=float, default=0.75)
    parser.add_argument("--deterministic-search-seeds", action="store_true")
    parser.add_argument("--out", default=(
        "training/analysis/runs/progressive_risk_mpc_rounds.jsonl"))
    parser.add_argument("--summary", default=(
        "training/analysis/runs/progressive_risk_mpc_summary.json"))
    args = parser.parse_args()
    if args.mode == "probe":
        probe(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
