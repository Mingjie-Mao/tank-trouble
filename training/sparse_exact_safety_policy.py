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
from training.exact_state_mpc_teacher import (  # noqa: E402
    ExactStatePriorGuidedMPC,
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


class SparseExactSafetyPolicy(ExactStatePriorGuidedMPC):
    name = "sparse_exact_state_safety"

    def __init__(self, *args, audit_interval=1, proactive_interval=24,
                 behavior_full_search=True, search_hold_frames=6, **kwargs):
        critical_safe_roots = kwargs.pop("critical_safe_roots", 0)
        critical_hold_frames = kwargs.pop("critical_hold_frames", 0)
        narrow_replan_safe_roots = kwargs.pop(
            "narrow_replan_safe_roots", 0)
        super().__init__(*args, **kwargs)
        self.audit_interval = max(1, int(audit_interval))
        self.proactive_interval = max(0, int(proactive_interval))
        self.behavior_full_search = bool(behavior_full_search)
        self.search_hold_frames = max(0, int(search_hold_frames))
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
        elif reason == "critical":
            self.critical_searches += 1
        elif reason == "narrow":
            self.narrow_replans += 1
            self.narrow_replan_pending = False
        action = self._search(game, metrics, indices)
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
        elif (self.proactive_interval > 0
              and self.frames - self.last_proactive_search_frame
              >= self.proactive_interval):
            reason = "proactive"
        elif self.committed_action is not None and self.commit_remaining > 0:
            reason = "committed"
        elif (self.behavior_full_search
              and category in BEHAVIOR_SEARCH_CATEGORIES):
            reason = "behavior"

        action = None
        full_search_ran = False
        if reason in (
                "followup", "critical", "narrow", "behavior", "proactive"):
            action = self._full_search(
                game, metrics, outputs, p27, p27_index, reason)
            full_search_ran = True
        elif reason == "committed":
            if self._audit_action(game, metrics, proposed_index):
                action = proposed_action
                self.commit_remaining -= 1
                self.committed_frames += 1
                if self.commit_remaining <= 0:
                    self.committed_action = None
            else:
                action = self._full_search(
                    game, metrics, outputs, p27, p27_index, "unsafe")
                full_search_ran = True
        elif self.frames % self.audit_interval == 0:
            if not self._audit_action(game, metrics, p27_index):
                action = self._full_search(
                    game, metrics, outputs, p27, p27_index, "unsafe")
                full_search_ran = True
        else:
            self.skipped_audits += 1
        if action is not None and full_search_ran:
            self.committed_action = action
            self.commit_remaining = self.search_hold_frames
        elif full_search_ran:
            self.committed_action = None
            self.commit_remaining = 0
        action = action or proposed_action

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
        critical_safe_roots=args.critical_safe_roots,
        critical_hold_frames=args.critical_hold_frames,
        narrow_replan_safe_roots=args.narrow_replan_safe_roots,
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
    full_candidates = int(policy.exact_candidates + policy.successor_candidates)
    simulated_frames = int(
        (policy.audit_candidates + full_candidates) * policy.search_horizon)
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
        "critical_searches": int(policy.critical_searches),
        "narrow_replans": int(policy.narrow_replans),
        "max_incoming_risk": float(policy.max_incoming_risk),
        "fallback_counts": dict(policy.fallback_counts),
    })
    if result["true_result"] != "win":
        result["search_trace"] = list(policy.search_trace)
        result["incoming_risk_trace"] = list(policy.risk_trace)
        result["no_safe_events"] = list(policy.no_safe_events)
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
    parser.add_argument("--critical-safe-roots", type=int, default=0)
    parser.add_argument("--critical-hold-frames", type=int, default=0)
    parser.add_argument("--narrow-replan-safe-roots", type=int, default=0)
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
    for row in rounds:
        behavior.update(row["behavior_categories"])
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
