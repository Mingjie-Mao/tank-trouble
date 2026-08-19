"""Sparse exact-state safety policy.

The P27b champion supplies normal actions. A one-candidate exact rollout audits
the proposed action; full Exact-State Safety-Shielded MPC is invoked only when
that action is unsafe, when a behavior failure is detected, or at a low-rate
proactive planning interval.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluate import play_round_dual_engine  # noqa: E402
from training.behavior_events import BehaviorEventTracker  # noqa: E402
from training.dagger_correction_recorder import (  # noqa: E402
    build_correction_record,
)
from training.battle_supervision import diagnose_battle  # noqa: E402
from training.exact_state_mpc_teacher import (  # noqa: E402
    ExactStatePriorGuidedMPC,
    choose_unsafe_fallback,
    exact_root_search,
)
from training.mpc_agent import CANDIDATES  # noqa: E402
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    build_observation,
    select_action,
    stack_observation,
)
from training.p27_risk_value import _controls  # noqa: E402
from training.map_topology_planner import (  # noqa: E402
    MapTopologyPlanner,
    MovementHysteresis,
    cardinal_neighbors,
    dead_end_depth,
    drive_action_to_cell,
    shortest_cardinal_path,
    tank_cell,
)
from training.temporal_sequence_teacher import (  # noqa: E402
    rollout_exact_sequence,
)
from training.temporal_intent_model import (  # noqa: E402
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_STATE_FEATURE_DIM,
    TemporalIntentRuntime,
    build_temporal_features,
)


BEHAVIOR_SEARCH_CATEGORIES = frozenset((
    "blind_fire",
    "missed_fire_window",
    "dead_end_stall",
    "stutter_stall",
    "passive_map_control",
))


def should_arm_narrow_replan(safe_roots, threshold, reason):
    return bool(
        reason != "narrow"
        and int(threshold) > 0
        and 0 < int(safe_roots) <= int(threshold)
    )


def exceeded_nonwin_gate(rounds, max_nonwins):
    if int(max_nonwins) < 0:
        return False
    return sum(row["true_result"] != "win" for row in rounds) > int(
        max_nonwins)


def movement_only_action(action):
    throttle, turn, _ = action
    return throttle, turn, 0


def should_trigger_key_search(action, incoming_risk, risk_threshold,
                              search_on_fire=True):
    return bool(
        (search_on_fire and int(action[2]) == 1)
        or (float(risk_threshold) > 0.0
            and float(incoming_risk) >= float(risk_threshold))
    )


class SparseExactSafetyPolicy(ExactStatePriorGuidedMPC):
    name = "sparse_exact_state_safety"

    def __init__(self, *args, audit_interval=1, proactive_interval=24,
                 behavior_full_search=True, search_hold_frames=6,
                 search_on_fire=True, risk_search_threshold=0.18,
                 long_tail_fire_horizon=375, topology_assist=False,
                 topology_max_risk=0.18, topology_max_line=0.65,
                 topology_intent_max_frames=75,
                 topology_cooldown_frames=25,
                 topology_pursuit_delay_frames=20,
                 topology_pursuit_max_reach=0.55,
                 topology_pursuit_max_line=0.35,
                 network_move_hold_frames=0, temporal_intent_net=None,
                 temporal_confidence=0.55,
                 learned_proactive_threshold=0.0,
                 proactive_max_interval=0,
                 temporal_record_state=False,
                 proactive_top_k=0, behavior_top_k=0, **kwargs):
        critical_safe_roots = kwargs.pop("critical_safe_roots", 0)
        critical_hold_frames = kwargs.pop("critical_hold_frames", 0)
        narrow_replan_safe_roots = kwargs.pop(
            "narrow_replan_safe_roots", 0)
        super().__init__(*args, **kwargs)
        self.audit_interval = max(1, int(audit_interval))
        self.proactive_interval = max(0, int(proactive_interval))
        self.behavior_full_search = bool(behavior_full_search)
        self.search_hold_frames = max(0, int(search_hold_frames))
        self.search_on_fire = bool(search_on_fire)
        self.risk_search_threshold = float(risk_search_threshold)
        self.long_tail_fire_horizon = max(
            self.search_horizon, int(long_tail_fire_horizon))
        self.topology_assist = bool(topology_assist)
        self.topology_max_risk = float(topology_max_risk)
        self.topology_max_line = float(topology_max_line)
        self.topology_intent_max_frames = max(
            1, int(topology_intent_max_frames))
        self.topology_cooldown_frames = max(
            0, int(topology_cooldown_frames))
        self.topology_pursuit_delay_frames = max(
            1, int(topology_pursuit_delay_frames))
        self.topology_pursuit_max_reach = float(topology_pursuit_max_reach)
        self.topology_pursuit_max_line = float(topology_pursuit_max_line)
        self.network_move_hold_frames = max(0, int(network_move_hold_frames))
        self.temporal_intent_net = temporal_intent_net
        self.temporal_confidence = float(temporal_confidence)
        self.learned_proactive_threshold = float(
            learned_proactive_threshold)
        self.proactive_max_interval = max(0, int(proactive_max_interval))
        self.temporal_record_state = bool(temporal_record_state)
        self.proactive_top_k = max(0, int(proactive_top_k))
        self.behavior_top_k = max(0, int(behavior_top_k))
        self.temporal_runtime = (
            TemporalIntentRuntime(temporal_intent_net)
            if temporal_intent_net else None)
        # Fail fast: temporal_record_state appends STATE_FEATURE_DIM extra
        # features, so pairing it with a checkpoint trained on the compact
        # feature set produces a matmul shape error ~1 frame into a rollout,
        # inside a worker process, with no useful message.
        if (self.temporal_record_state and self.temporal_runtime is not None
                and self.temporal_runtime.feature_dim <= TEMPORAL_FEATURE_DIM):
            raise ValueError(
                "temporal_record_state=True appends state features "
                f"({TEMPORAL_STATE_FEATURE_DIM}-dim) but "
                f"{temporal_intent_net} expects "
                f"{self.temporal_runtime.feature_dim}-dim input. "
                "Leave temporal_record_state=False when a compact checkpoint "
                "drives the rollout; the feature set then follows the loaded "
                "model automatically.")
        self.movement_hysteresis = MovementHysteresis(
            self.network_move_hold_frames)
        self.topology_planner = MapTopologyPlanner()
        self.critical_safe_roots = max(0, int(critical_safe_roots))
        self.critical_hold_frames = max(0, int(critical_hold_frames))
        self.narrow_replan_safe_roots = max(
            0, int(narrow_replan_safe_roots))

    def reset(self):
        super().reset()
        self.audit_frames = 0
        self.audit_candidates = 0
        self.unsafe_audits = 0
        self.skipped_audits = 0
        self.proactive_searches = 0
        self.behavior_searches = 0
        self.followup_searches = 0
        self.unsafe_searches = 0
        self.policy_frames = 0
        self.behavior_categories = Counter()
        self.committed_action = None
        self.commit_remaining = 0
        self.committed_frames = 0
        self.critical_mode_remaining = 0
        self.critical_searches = 0
        self.narrow_replan_pending = False
        self.narrow_replans = 0
        self.last_incoming_risk = 0.0
        self.max_incoming_risk = 0.0
        self.risk_trace = deque(maxlen=256)
        self.no_safe_events = deque(maxlen=64)
        self.last_proactive_search_frame = 0
        self.fire_searches = 0
        self.risk_searches = 0
        self.long_tail_fire_checks = 0
        self.long_tail_fire_rejections = 0
        self.long_tail_simulated_frames = 0
        self.event_tracker = BehaviorEventTracker()
        self.topology_requests = 0
        self.topology_accepts = 0
        self.topology_rejections = 0
        self.topology_goal_kinds = Counter()
        self.active_topology_target = None
        self.active_topology_kind = None
        self.active_topology_frames = 0
        self.topology_cooldown = 0
        self.topology_route_frames = 0
        self.topology_completions = 0
        self.topology_aborts = 0
        self.low_map_control_frames = 0
        self.movement_hysteresis = MovementHysteresis(
            getattr(self, "network_move_hold_frames", 0))
        temporal_runtime = getattr(self, "temporal_runtime", None)
        if temporal_runtime is not None:
            temporal_runtime.reset()
        self.temporal_last_movement = None
        self.temporal_frames_since_change = 0
        self.temporal_overrides = 0
        self.unsafe_fallback_corrections = 0
        self.last_temporal_sample = None
        self.learned_proactive_searches = 0
        self.forced_proactive_searches = 0

    def _clear_topology_intent(self, cooldown=True):
        self.active_topology_target = None
        self.active_topology_kind = None
        self.active_topology_frames = 0
        if cooldown:
            self.topology_cooldown = self.topology_cooldown_frames

    def _topology_action(self, game, metrics, category, network_action):
        if self.topology_cooldown > 0:
            self.topology_cooldown -= 1
        line, reach, risk = [float(value) for value in metrics[:3]]
        low_map_control = (
            line < self.topology_pursuit_max_line
            and reach < self.topology_pursuit_max_reach
            and risk <= self.topology_max_risk)
        if low_map_control:
            self.low_map_control_frames += 1
        else:
            self.low_map_control_frames = 0
        if self.active_topology_target is not None:
            if network_action[2] == 1 or risk > self.topology_max_risk:
                return None, False
            if (self.active_topology_kind == "seek_firing_position"
                    and line >= self.topology_max_line):
                self.topology_completions += 1
                self._clear_topology_intent()
                return None, False
            if (self.active_topology_frames
                    >= self.topology_intent_max_frames):
                self.topology_aborts += 1
                self._clear_topology_intent()
                return None, False
            current = tank_cell(game, game.tanks[0])
            if current == self.active_topology_target:
                self.topology_completions += 1
                self._clear_topology_intent()
                return None, False
            path = shortest_cardinal_path(
                game.maze, current, self.active_topology_target)
            if not path:
                self.topology_aborts += 1
                self._clear_topology_intent()
                return None, False
            self.active_topology_frames += 1
            self.topology_route_frames += 1
            return drive_action_to_cell(
                game, game.tanks[0], path[0], can_reverse=True), False

        current = tank_cell(game, game.tanks[0])
        immediate_escape = (
            dead_end_depth(game, current) > 0.0
            or len(cardinal_neighbors(game.maze, current)) <= 1)
        behavior_trigger = category in (
            "dead_end_stall", "passive_map_control", "stutter_stall")
        pursuit_trigger = (
            self.low_map_control_frames >= self.topology_pursuit_delay_frames)
        if (self.topology_cooldown > 0
                or not (immediate_escape or behavior_trigger or pursuit_trigger)
                or network_action[2] == 1
                or risk > self.topology_max_risk
                or (not immediate_escape and line >= self.topology_max_line)):
            return None, False
        goal = self.topology_planner.choose_goal(game, self.analyzer)
        if goal.kind == "hold_position" or not goal.path:
            return None, False
        self.active_topology_target = goal.target
        self.active_topology_kind = goal.kind
        self.active_topology_frames = 1
        self.topology_requests += 1
        self.topology_route_frames += 1
        self.topology_goal_kinds[goal.kind] += 1
        self.committed_action = None
        self.commit_remaining = 0
        return drive_action_to_cell(
            game, game.tanks[0], goal.next_cell, can_reverse=True), True

    def _audit_action(self, game, metrics, index):
        _, rows = exact_root_search(
            game,
            self.analyzer,
            metrics,
            (int(index),),
            horizon=self.search_horizon,
            death_penalty=self.search_death_penalty,
            dd_penalty=self.search_dd_penalty,
            kill_bonus=self.search_kill_bonus,
            max_death=self.search_max_death,
            max_dd=self.search_max_dd,
        )
        self.audit_frames += 1
        self.audit_candidates += len(rows)
        allowed = bool(rows and rows[0]["allowed"])
        if not allowed:
            self.unsafe_audits += 1
        return allowed

    def _full_search(self, game, metrics, outputs, p27, p27_index, reason):
        indices = self._candidate_order(
            outputs, p27, p27_index, metrics)
        budget = (self.proactive_top_k if reason == "proactive" else
                  self.behavior_top_k if reason == "behavior" else 0)
        if budget > 0:
            indices = indices[:budget]
        self._fb_count(f"sparse_{reason}")
        if reason == "proactive":
            self.proactive_searches += 1
            self.last_proactive_search_frame = self.frames
        elif reason == "behavior":
            self.behavior_searches += 1
        elif reason == "followup":
            self.followup_searches += 1
        elif reason == "unsafe":
            self.unsafe_searches += 1
        elif reason == "fire":
            self.fire_searches += 1
        elif reason == "risk":
            self.risk_searches += 1
        elif reason == "critical":
            self.critical_searches += 1
        elif reason == "narrow":
            self.narrow_replans += 1
            self.narrow_replan_pending = False
        action = self._search(game, metrics, indices)
        if action is not None and action[2] == 1:
            self.long_tail_fire_checks += 1
            action_index = self.candidates.index(action)
            movement_index = self.candidates.index(
                movement_only_action(action))
            tail = rollout_exact_sequence(
                game,
                self.analyzer,
                metrics,
                action_index,
                movement_index,
                chunk_frames=1,
                score_horizon=self.search_horizon,
                fire_tail_horizon=self.long_tail_fire_horizon,
            )
            self.long_tail_simulated_frames += int(tail["simulated_frames"])
            if not tail["allowed"]:
                self.long_tail_fire_rejections += 1
                self._fb_count("sparse_long_tail_fire_rejected")
                nonfire = tuple(
                    index for index in indices
                    if self.candidates[int(index)][2] == 0)
                action = self._search(game, metrics, nonfire)
        safe_roots = int((self.last_search_decision or {}).get(
            "safe_root_count", len(CANDIDATES)))
        if reason != "narrow":
            self.narrow_replan_pending = should_arm_narrow_replan(
                safe_roots, self.narrow_replan_safe_roots, reason)
            if self.narrow_replan_pending:
                self._fb_count("sparse_narrow_replan_armed")
        if action is None:
            self.no_safe_events.append({
                "frame": int(game.frame),
                "incoming_risk": float(self.last_incoming_risk),
                "reason": reason,
            })
        if (self.critical_safe_roots > 0
                and 0 < safe_roots <= self.critical_safe_roots):
            self.critical_mode_remaining = max(
                self.critical_mode_remaining, self.critical_hold_frames)
            self._fb_count("sparse_critical_enter")
        return action

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
        self.last_incoming_risk = float(metrics[2])
        self.max_incoming_risk = max(
            self.max_incoming_risk, self.last_incoming_risk)
        if self.last_incoming_risk > 0.0:
            self.risk_trace.append({
                "frame": int(game.frame),
                "risk": self.last_incoming_risk,
            })
        self.frames += 1
        self.policy_frames += 1
        self.history.append(observation)
        stacked = stack_observation(self.history, self.frame_stack)
        with self.torch.no_grad():
            out = self.base_net(self.torch.as_tensor(stacked).unsqueeze(0))
        outputs = {
            "score": out["score"][0].numpy(),
            "aux": out["aux"][0].numpy(),
            "fire": out["fire"][0].numpy(),
        }
        topology_goal = self.topology_planner.choose_goal(game, self.analyzer)
        topology_features = self.topology_planner.features(game, topology_goal)
        include_state = (
            self.temporal_record_state
            or (self.temporal_runtime is not None
                and self.temporal_runtime.feature_dim > TEMPORAL_FEATURE_DIM))
        temporal_features = build_temporal_features(
            outputs["score"], outputs["aux"], outputs["fire"],
            self.temporal_last_movement,
            self.temporal_frames_since_change,
            topology_features=topology_features,
            state_features=observation if include_state else None,
        )
        default_action = select_action(
            outputs, self.candidates, self.fire_margin,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        default_index = self.candidates.index(default_action)
        category = self._detect_category(
            game, _controls(default_action), metrics)
        if category is not None:
            self.behavior_categories[category] += 1
        context = self._update_context(game, metrics)
        p27 = self._p27_value(stacked, context)
        outputs = self._adjust_outputs(
            outputs, category, p27, default_index, metrics)
        p27_action = select_action(
            outputs, self.candidates, self.fire_margin,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        p27_index = self.candidates.index(p27_action)

        network_action = p27_action
        network_movement = self.candidates.index(
            movement_only_action(network_action)) // 2
        temporal_movement = None
        temporal_prediction = None
        if self.temporal_runtime is not None:
            temporal_prediction = self.temporal_runtime.predict(
                temporal_features)
            movement_prob = temporal_prediction["movement_prob"]
            movement = int(np.argmax(movement_prob))
            if (p27_action[2] == 0
                    and self.last_incoming_risk < self.risk_search_threshold
                    and float(movement_prob[movement])
                    >= self.temporal_confidence):
                candidate = self.candidates[movement * 2]
                if candidate != p27_action:
                    self.temporal_overrides += 1
                p27_action = candidate
                p27_index = movement * 2
                temporal_movement = movement
        topology_action = None
        topology_started = False
        if self.topology_assist:
            topology_action, topology_started = self._topology_action(
                game, metrics, category, network_action)
            if topology_action is not None:
                p27_action = topology_action
                p27_index = self.candidates.index(topology_action)
        if topology_action is None:
            p27_action = self.movement_hysteresis.choose(
                p27_action,
                interrupt=(
                    float(metrics[2]) >= self.risk_search_threshold
                    or bool(getattr(game.tanks[0], "hit_something", False))
                    or self.active_topology_target is not None),
            )
            p27_index = self.candidates.index(p27_action)
        else:
            self.movement_hysteresis.reset()

        topology_movement = (
            None if topology_action is None
            else self.candidates.index(
                movement_only_action(topology_action)) // 2)

        proposed_action = p27_action
        proposed_index = p27_index
        if self.committed_action is not None and self.commit_remaining > 0:
            proposed_action = self.committed_action
            proposed_index = self.candidates.index(proposed_action)

        reason = None
        if self.pending_successor_audit is not None:
            reason = "followup"
        elif self.critical_mode_remaining > 0:
            reason = "critical"
            self.critical_mode_remaining -= 1
        elif (self.narrow_replan_pending
              and not (self.committed_action is not None
                       and self.commit_remaining > 0)):
            reason = "narrow"
        elif should_trigger_key_search(
                p27_action, self.last_incoming_risk,
                self.risk_search_threshold, self.search_on_fire):
            reason = "fire" if p27_action[2] == 1 else "risk"
        elif topology_action is not None:
            reason = "topology"
        elif self.proactive_interval > 0:
            since_proactive = self.frames - self.last_proactive_search_frame
            learned_due = (
                self.learned_proactive_threshold > 0.0
                and temporal_prediction is not None
                and since_proactive >= self.proactive_interval
                and temporal_prediction["search_needed_prob"]
                >= self.learned_proactive_threshold)
            forced_due = (
                self.proactive_max_interval > 0
                and since_proactive >= self.proactive_max_interval)
            fixed_due = (
                self.learned_proactive_threshold <= 0.0
                and since_proactive >= self.proactive_interval)
            if learned_due or forced_due or fixed_due:
                reason = "proactive"
                if learned_due:
                    self.learned_proactive_searches += 1
                elif forced_due:
                    self.forced_proactive_searches += 1
        elif self.committed_action is not None and self.commit_remaining > 0:
            reason = "committed"
        elif (self.behavior_full_search
              and category in BEHAVIOR_SEARCH_CATEGORIES):
            reason = "behavior"

        action = None
        full_search_ran = False
        audit_failed = False
        topology_aborted = False
        long_tail_rejections_before = self.long_tail_fire_rejections
        search_needed_mask = False
        search_needed_target = False
        if reason in (
                "followup", "critical", "narrow", "behavior", "proactive",
                "fire", "risk"):
            action = self._full_search(
                game, metrics, outputs, p27, p27_index, reason)
            full_search_ran = True
            if reason == "proactive":
                decision = self.last_search_decision or {}
                selected = decision.get("selected_action")
                search_needed_mask = True
                search_needed_target = bool(
                    selected is None
                    or tuple(selected) != tuple(proposed_action)
                    or int(decision.get("safe_root_count", 18)) <= 2
                    or decision.get("interventions"))
        elif reason == "committed":
            audit_now = (
                self.frames % self.audit_interval == 0
                or (self.risk_search_threshold > 0.0
                    and self.last_incoming_risk
                    >= self.risk_search_threshold))
            if not audit_now:
                action = proposed_action
                self.commit_remaining -= 1
                self.committed_frames += 1
                if self.commit_remaining <= 0:
                    self.committed_action = None
                self.skipped_audits += 1
            elif self._audit_action(game, metrics, proposed_index):
                action = proposed_action
                search_needed_mask = True
                search_needed_target = False
                self.commit_remaining -= 1
                self.committed_frames += 1
                if self.commit_remaining <= 0:
                    self.committed_action = None
            else:
                audit_failed = True
                action = self._full_search(
                    game, metrics, outputs, p27, p27_index, "unsafe")
                full_search_ran = True
                search_needed_mask = True
                search_needed_target = True
        elif reason == "topology":
            audit_now = topology_started or self.frames % self.audit_interval == 0
            if not audit_now:
                action = p27_action
                self.skipped_audits += 1
            elif self._audit_action(game, metrics, p27_index):
                action = p27_action
                search_needed_mask = True
                search_needed_target = False
                if topology_started:
                    self.topology_accepts += 1
            else:
                audit_failed = True
                topology_aborted = True
                if topology_started:
                    self.topology_rejections += 1
                else:
                    self.topology_aborts += 1
                self._clear_topology_intent()
                action = self._full_search(
                    game, metrics, outputs, p27, p27_index, "unsafe")
                full_search_ran = True
                search_needed_mask = True
                search_needed_target = True
        elif self.frames % self.audit_interval == 0:
            if not self._audit_action(game, metrics, p27_index):
                audit_failed = True
                action = self._full_search(
                    game, metrics, outputs, p27, p27_index, "unsafe")
                full_search_ran = True
                search_needed_mask = True
                search_needed_target = True
            else:
                search_needed_mask = True
                search_needed_target = False
        else:
            self.skipped_audits += 1
        if action is not None and full_search_ran:
            self.committed_action = movement_only_action(action)
            self.commit_remaining = self.search_hold_frames
        elif full_search_ran:
            self.committed_action = None
            self.commit_remaining = 0
        if action is None and full_search_ran:
            # The exact search found no safe root.  Falling straight through to
            # proposed_action executes an action nothing validated, and skips
            # the long-tail fire check entirely -- see choose_unsafe_fallback.
            fallback = choose_unsafe_fallback(
                (self.last_search_decision or {}).get("rows") or (),
                proposed_action, mode=self.unsafe_fallback_mode)
            if fallback != proposed_action:
                self.unsafe_fallback_corrections += 1
                self._fb_count("unsafe_fallback_corrected")
            action = fallback
        action = action or proposed_action

        throttle, turn, fire = action
        if len(game.tanks) > 1 and not game.tanks[1].alive:
            fire = 0
        self.event_tracker.update_action((throttle, turn))
        event_frame = self.frames - 1
        self.event_tracker.update_episode(
            "stutter_stall", category == "stutter_stall", event_frame)
        self.event_tracker.update_episode(
            "dead_end_stall", category == "dead_end_stall", event_frame)
        self.event_tracker.update_episode(
            "passive_map_control",
            category == "passive_map_control",
            event_frame,
        )
        self.event_tracker.update_fire_window(
            event_frame,
            clear_line=(float(metrics[0]) >= self.fire_window_line
                        and float(metrics[2]) <= 0.35),
            fired=fire == 1,
            enemy_alive=len(game.tanks) > 1 and game.tanks[1].alive,
        )
        movement = self.candidates.index(
            movement_only_action((throttle, turn, fire))) // 2
        if movement == self.temporal_last_movement:
            self.temporal_frames_since_change += 1
        else:
            self.temporal_last_movement = movement
            self.temporal_frames_since_change = 1
        # Ground truth for the exact teacher's continuity preference: what was
        # actually executed, which can differ from what _search returned once
        # the long-tail fire check has replaced a fire action.
        self.continuity_movement = movement
        target_x = (topology_goal.target[0] + 0.5) * game.scale
        target_y = (topology_goal.target[1] + 0.5) * game.scale
        self.last_temporal_sample = {
            "features": temporal_features,
            "movement": int(movement),
            "risk": float(self.last_incoming_risk),
            "category": category or "standard",
            "reason": reason or "network",
            "full_search": bool(full_search_ran),
            "topology_active": bool(topology_action is not None),
            "topology_kind": topology_goal.kind,
            "target": tuple(int(value) for value in topology_goal.target),
            "target_distance_before": float(np.hypot(
                game.tanks[0].x - target_x,
                game.tanks[0].y - target_y) / game.scale),
            "temporal_confidence": (
                0.0 if temporal_prediction is None else float(np.max(
                    temporal_prediction["movement_prob"]))),
            "search_needed_mask": bool(search_needed_mask),
            "search_needed_target": bool(search_needed_target),
            "search_needed_probability": (
                0.0 if temporal_prediction is None else float(
                    temporal_prediction["search_needed_prob"])),
        }
        # --- DAgger correction trace -----------------------------------
        # Everything the distiller needs to know *which* stage proposed the
        # move and whether the exact teacher had to overrule it.  Cheap to
        # collect (no extra rollouts) and inert when nobody reads it.
        decision = self.last_search_decision or {}
        self.last_temporal_sample.update(build_correction_record(
            frame=event_frame,
            features=temporal_features,
            executed_movement=movement,
            # action[2], not the local `fire`: `fire` is forced to 0 once the
            # enemy is already dead, which would look like a fire correction.
            executed_fire=action[2] == 1,
            network_movement=network_movement,
            temporal_movement=temporal_movement,
            temporal_confidence=self.last_temporal_sample[
                "temporal_confidence"],
            topology_movement=topology_movement,
            proposed_movement=self.candidates.index(
                movement_only_action(proposed_action)) // 2,
            proposed_fire=proposed_action[2] == 1,
            reason=reason,
            category=category,
            full_search=full_search_ran,
            audit_failed=audit_failed,
            safe_root_count=int(
                decision.get("safe_root_count", len(self.candidates))
                if full_search_ran else len(self.candidates)),
            successor_shield_triggered=bool(
                full_search_ran
                and decision.get("successor_shield_triggered", False)),
            interventions=(
                decision.get("interventions", ()) if full_search_ran else ()),
            long_tail_fire_rejected=bool(
                self.long_tail_fire_rejections > long_tail_rejections_before),
            risk=float(self.last_incoming_risk),
            topology_active=topology_action is not None,
            topology_kind=topology_goal.kind,
            topology_started=topology_started,
            topology_aborted=topology_aborted,
        ))
        return {
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": fire == 1,
        }


def parse_seeds(value):
    seeds = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            start, count = (int(part) for part in item.split(":", 1))
            seeds.extend(range(start, start + count))
        else:
            seeds.append(int(item))
    return tuple(seeds)


def _make_policy(args):
    return SparseExactSafetyPolicy(
        base_net=args.base_net,
        value_net=args.value_net,
        fire_margin=args.fire_margin,
        top_k=args.top_k,
        search_horizon=args.search_horizon,
        search_death_penalty=args.search_death_penalty,
        search_dd_penalty=args.search_dd_penalty,
        search_kill_bonus=args.search_kill_bonus,
        search_max_death=0.0,
        search_max_dd=0.0,
        successor_shield=args.successor_shield,
        successor_horizon=args.successor_horizon,
        successor_shield_max_safe_roots=args.successor_shield_max_safe_roots,
        suppress_secured_fire=True,
        min_unsecured_fire_gain=2.0,
        audit_interval=args.audit_interval,
        proactive_interval=args.proactive_interval,
        behavior_full_search=args.behavior_full_search,
        search_hold_frames=args.search_hold_frames,
        search_on_fire=args.search_on_fire,
        risk_search_threshold=args.risk_search_threshold,
        long_tail_fire_horizon=args.long_tail_fire_horizon,
        topology_assist=args.topology_assist,
        topology_max_risk=args.topology_max_risk,
        topology_max_line=args.topology_max_line,
        topology_intent_max_frames=args.topology_intent_max_frames,
        topology_cooldown_frames=args.topology_cooldown_frames,
        topology_pursuit_delay_frames=args.topology_pursuit_delay_frames,
        topology_pursuit_max_reach=args.topology_pursuit_max_reach,
        topology_pursuit_max_line=args.topology_pursuit_max_line,
        network_move_hold_frames=args.network_move_hold_frames,
        temporal_intent_net=args.temporal_intent_net,
        temporal_confidence=args.temporal_confidence,
        learned_proactive_threshold=args.learned_proactive_threshold,
        proactive_max_interval=args.proactive_max_interval,
        proactive_top_k=args.proactive_top_k,
        behavior_top_k=args.behavior_top_k,
        critical_safe_roots=args.critical_safe_roots,
        critical_hold_frames=args.critical_hold_frames,
        narrow_replan_safe_roots=args.narrow_replan_safe_roots,
        movement_continuity_epsilon=args.movement_continuity_epsilon,
        unsafe_fallback_mode=args.unsafe_fallback_mode,
        deterministic_search_seeds=True,
    )


def _run_seed(job):
    seed, args = job
    import torch

    torch.set_num_threads(1)
    policy = _make_policy(args)
    policy.set_round_seed(seed)
    started = time.time()
    result = play_round_dual_engine(policy, seed)
    policy.event_tracker.finish(result["frames"])
    full_candidates = int(policy.exact_candidates + policy.successor_candidates)
    simulated_frames = int(
        (policy.audit_candidates + full_candidates) * policy.search_horizon
        + policy.long_tail_simulated_frames)
    result.update({
        "seed": int(seed),
        "elapsed_seconds": time.time() - started,
        "policy_frames": int(policy.policy_frames),
        "audit_frames": int(policy.audit_frames),
        "audit_candidates": int(policy.audit_candidates),
        "unsafe_audits": int(policy.unsafe_audits),
        "skipped_audits": int(policy.skipped_audits),
        "full_searches": int(policy.exact_searches),
        "root_candidates": int(policy.exact_candidates),
        "successor_candidates": int(policy.successor_candidates),
        "simulated_frames": simulated_frames,
        "proactive_searches": int(policy.proactive_searches),
        "behavior_searches": int(policy.behavior_searches),
        "followup_searches": int(policy.followup_searches),
        "unsafe_searches": int(policy.unsafe_searches),
        "search_frame_rate": (
            policy.exact_searches / max(1, policy.policy_frames)),
        "audit_frame_rate": (
            policy.audit_frames / max(1, policy.policy_frames)),
        "behavior_categories": dict(policy.behavior_categories),
        "committed_frames": int(policy.committed_frames),
        "fire_searches": int(policy.fire_searches),
        "risk_searches": int(policy.risk_searches),
        "long_tail_fire_checks": int(policy.long_tail_fire_checks),
        "long_tail_fire_rejections": int(
            policy.long_tail_fire_rejections),
        "long_tail_simulated_frames": int(
            policy.long_tail_simulated_frames),
        "critical_searches": int(policy.critical_searches),
        "narrow_replans": int(policy.narrow_replans),
        "max_incoming_risk": float(policy.max_incoming_risk),
        "fallback_counts": dict(policy.fallback_counts),
        "event_metrics": policy.event_tracker.summary(),
        "topology_requests": int(policy.topology_requests),
        "topology_accepts": int(policy.topology_accepts),
        "topology_rejections": int(policy.topology_rejections),
        "topology_goal_kinds": dict(policy.topology_goal_kinds),
        "topology_route_frames": int(policy.topology_route_frames),
        "topology_completions": int(policy.topology_completions),
        "topology_aborts": int(policy.topology_aborts),
        "movement_hysteresis_suppressions": int(
            policy.movement_hysteresis.suppressions),
        "temporal_overrides": int(policy.temporal_overrides),
        "movement_continuity_holds": int(policy.movement_continuity_holds),
        "unsafe_fallback_corrections": int(
            policy.unsafe_fallback_corrections),
        # Counted for every round.  The full trace below is only kept for
        # non-wins, which silently made winning rounds look like they had
        # zero no-safe frames when diffing two runs.
        "no_safe_event_count": len(policy.no_safe_events),
        "learned_proactive_searches": int(
            policy.learned_proactive_searches),
        "forced_proactive_searches": int(policy.forced_proactive_searches),
    })
    if result["true_result"] != "win":
        result["search_trace"] = list(policy.search_trace)
        result["incoming_risk_trace"] = list(policy.risk_trace)
        result["no_safe_events"] = list(policy.no_safe_events)
    result["diagnosis"] = diagnose_battle(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-net", default=(
        "training/models/p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--value-net", default=(
        "training/models/p27b_risk_value_iter00.pt"))
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--search-horizon", type=int, default=72)
    parser.add_argument("--search-death-penalty", type=float, default=0.18)
    parser.add_argument("--search-dd-penalty", type=float, default=0.45)
    parser.add_argument("--search-kill-bonus", type=float, default=0.05)
    parser.add_argument("--successor-shield",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--successor-horizon", type=int, default=72)
    parser.add_argument("--successor-shield-max-safe-roots", type=int,
                        default=2)
    parser.add_argument("--audit-interval", type=int, default=1)
    parser.add_argument("--proactive-interval", type=int, default=24)
    parser.add_argument("--behavior-full-search",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-hold-frames", type=int, default=6)
    parser.add_argument("--search-on-fire",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--risk-search-threshold", type=float, default=0.18)
    parser.add_argument("--long-tail-fire-horizon", type=int, default=375)
    parser.add_argument("--topology-assist",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--topology-max-risk", type=float, default=0.18)
    parser.add_argument("--topology-max-line", type=float, default=0.65)
    parser.add_argument("--topology-intent-max-frames", type=int, default=75)
    parser.add_argument("--topology-cooldown-frames", type=int, default=25)
    parser.add_argument("--topology-pursuit-delay-frames", type=int,
                        default=20)
    parser.add_argument("--topology-pursuit-max-reach", type=float,
                        default=0.55)
    parser.add_argument("--topology-pursuit-max-line", type=float,
                        default=0.35)
    parser.add_argument("--network-move-hold-frames", type=int, default=0)
    parser.add_argument("--temporal-intent-net")
    parser.add_argument("--temporal-confidence", type=float, default=0.55)
    parser.add_argument("--learned-proactive-threshold", type=float,
                        default=0.0)
    parser.add_argument("--proactive-max-interval", type=int, default=0)
    parser.add_argument("--proactive-top-k", type=int, default=0)
    parser.add_argument("--behavior-top-k", type=int, default=0)
    parser.add_argument("--critical-safe-roots", type=int, default=0)
    parser.add_argument("--critical-hold-frames", type=int, default=0)
    parser.add_argument("--narrow-replan-safe-roots", type=int, default=0)
    parser.add_argument(
        "--movement-continuity-epsilon", type=float, default=0.0,
        help="keep the current movement when an equally-safe action is "
             "within this value gap; 0.0 disables (default)")
    parser.add_argument(
        "--unsafe-fallback-mode",
        choices=("strip_fire", "least_bad", "proposed"),
        default="strip_fire",
        help="搜索找不到安全动作时: strip_fire=保留网络的移动但去掉开火位(最小修复), "
             "least_bad=再把移动也换成已评估的最优行, "
             "proposed=旧行为(直接执行未验证的网络提议)")
    parser.add_argument("--max-nonwins", type=int, default=-1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress-jsonl")
    parser.add_argument("--print-report", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seed_list)
    started = time.time()
    worker_count = max(1, min(args.workers, len(seeds)))
    progress_path = args.progress_jsonl or f"{args.out}.partial.jsonl"
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    progress_handle = open(progress_path, "w", encoding="utf-8")
    stopped_early = False
    if worker_count == 1:
        iterator = (_run_seed((seed, args)) for seed in seeds)
    else:
        pool = mp.get_context("spawn").Pool(worker_count)
        iterator = pool.imap_unordered(
            _run_seed, ((seed, args) for seed in seeds))
    rounds = []
    try:
        for row in iterator:
            rounds.append(row)
            progress_handle.write(json.dumps(row, sort_keys=True) + "\n")
            progress_handle.flush()
            print(
                f"progress={len(rounds)}/{len(seeds)} seed={row['seed']} "
                f"result={row['true_result']} "
                f"search_rate={row['search_frame_rate']:.1%}",
                flush=True,
            )
            if exceeded_nonwin_gate(rounds, args.max_nonwins):
                nonwins = sum(
                    item["true_result"] != "win" for item in rounds)
                stopped_early = True
                print(
                    f"early_stop nonwins={nonwins} "
                    f"max_nonwins={args.max_nonwins}",
                    flush=True,
                )
                break
    finally:
        progress_handle.close()
        if worker_count != 1:
            if stopped_early:
                pool.terminate()
            else:
                pool.close()
            pool.join()
    rounds.sort(key=lambda row: row["seed"])
    results = Counter(row["true_result"] for row in rounds)
    behavior = Counter()
    behavior_events = Counter()
    behavior_event_durations = Counter()
    for row in rounds:
        behavior.update(row["behavior_categories"])
        behavior_events.update(row["event_metrics"]["events"])
        behavior_event_durations.update(
            row["event_metrics"]["durations"])
    report = {
        "method": "sparse_exact_state_safety",
        "configuration": {
            "base_net": args.base_net,
            "value_net": args.value_net,
            "fire_margin": args.fire_margin,
            "top_k": args.top_k,
            "search_horizon": args.search_horizon,
            "search_death_penalty": args.search_death_penalty,
            "search_dd_penalty": args.search_dd_penalty,
            "search_kill_bonus": args.search_kill_bonus,
            "successor_shield": args.successor_shield,
            "successor_horizon": args.successor_horizon,
            "successor_shield_max_safe_roots": (
                args.successor_shield_max_safe_roots),
            "audit_interval": args.audit_interval,
            "proactive_interval": args.proactive_interval,
            "behavior_full_search": args.behavior_full_search,
            "search_hold_frames": args.search_hold_frames,
            "search_on_fire": args.search_on_fire,
            "risk_search_threshold": args.risk_search_threshold,
            "long_tail_fire_horizon": args.long_tail_fire_horizon,
            "topology_assist": args.topology_assist,
            "topology_max_risk": args.topology_max_risk,
            "topology_max_line": args.topology_max_line,
            "topology_intent_max_frames": args.topology_intent_max_frames,
            "topology_cooldown_frames": args.topology_cooldown_frames,
            "topology_pursuit_delay_frames": (
                args.topology_pursuit_delay_frames),
            "topology_pursuit_max_reach": args.topology_pursuit_max_reach,
            "topology_pursuit_max_line": args.topology_pursuit_max_line,
        "network_move_hold_frames": args.network_move_hold_frames,
        "temporal_intent_net": args.temporal_intent_net,
        "temporal_confidence": args.temporal_confidence,
        "learned_proactive_threshold": args.learned_proactive_threshold,
        "proactive_max_interval": args.proactive_max_interval,
        "proactive_top_k": args.proactive_top_k,
        "behavior_top_k": args.behavior_top_k,
            "critical_safe_roots": args.critical_safe_roots,
            "critical_hold_frames": args.critical_hold_frames,
            "narrow_replan_safe_roots": args.narrow_replan_safe_roots,
        },
        "audit_interval": args.audit_interval,
        "proactive_interval": args.proactive_interval,
        "behavior_full_search": args.behavior_full_search,
        "expected_games": len(seeds),
        "completed_games": len(rounds),
        "stopped_early": stopped_early,
        "seeds": seeds,
        "results": dict(results),
        "win_rate": results["win"] / max(1, len(rounds)),
        "double_death_rate": results["double_death"] / max(1, len(rounds)),
        "total_policy_frames": int(sum(
            row["policy_frames"] for row in rounds)),
        "total_audit_frames": int(sum(
            row["audit_frames"] for row in rounds)),
        "total_full_searches": int(sum(
            row["full_searches"] for row in rounds)),
        "total_simulated_frames": int(sum(
            row["simulated_frames"] for row in rounds)),
        "mean_search_frame_rate": float(np.mean([
            row["search_frame_rate"] for row in rounds])),
        "mean_elapsed_seconds": float(np.mean([
            row["elapsed_seconds"] for row in rounds])),
        "behavior_categories": dict(behavior),
        "behavior_events": dict(behavior_events),
        "behavior_event_durations": dict(behavior_event_durations),
        "elapsed_seconds": time.time() - started,
        "rounds": rounds,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for row in rounds:
        print(
            f"seed={row['seed']} result={row['true_result']} "
            f"death={row['death_cause']} frames={row['frames']} "
            f"audit={row['audit_frames']} unsafe={row['unsafe_audits']} "
            f"search={row['full_searches']} "
            f"rate={row['search_frame_rate']:.1%} "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    summary = report if args.print_report else {
        key: report[key] for key in (
            "method", "configuration", "expected_games",
            "completed_games", "stopped_early", "results", "win_rate",
            "double_death_rate",
            "total_policy_frames", "total_audit_frames",
            "total_full_searches", "total_simulated_frames",
            "mean_search_frame_rate", "mean_elapsed_seconds",
            "behavior_categories", "elapsed_seconds",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
