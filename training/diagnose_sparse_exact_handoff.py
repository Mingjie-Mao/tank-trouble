"""Diagnose when the full exact teacher must take over a sparse trajectory.

Both policies observe every frame of the same trajectory. The sparse policy is
executed before ``handoff_frame`` and the full teacher is executed afterward.
This preserves the teacher's temporal context and turns a guessed trigger
threshold into a causal handoff experiment.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluate import play_round_dual_engine  # noqa: E402
from training.exact_state_mpc_teacher import (  # noqa: E402
    ExactStatePriorGuidedMPC,
)
from training.mpc_agent import CANDIDATES  # noqa: E402
from training.sparse_exact_safety_policy import (  # noqa: E402
    SparseExactSafetyPolicy,
)


def controls_to_action(controls):
    return (
        2 if controls.get("forward") else
        0 if controls.get("backup") else 1,
        0 if controls.get("turn_left") else
        2 if controls.get("turn_right") else 1,
        1 if controls.get("fire") else 0,
    )


def action_index(controls):
    action = controls_to_action(controls)
    return CANDIDATES.index(action) if action in CANDIDATES else None


def policy_kwargs(base_net, value_net):
    return {
        "base_net": base_net,
        "value_net": value_net,
        "fire_margin": 0.16,
        "top_k": 12,
        "search_horizon": 72,
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
    }


class SparseTeacherHandoffPolicy:
    name = "sparse_exact_teacher_handoff"

    def __init__(self, handoff_frame, base_net, value_net, trace_start=0):
        kwargs = policy_kwargs(base_net, value_net)
        self.handoff_frame = int(handoff_frame)
        self.trace_start = max(0, int(trace_start))
        self.sparse = SparseExactSafetyPolicy(
            **kwargs,
            audit_interval=1,
            proactive_interval=48,
            behavior_full_search=True,
            search_hold_frames=12,
            critical_safe_roots=0,
            critical_hold_frames=0,
            narrow_replan_safe_roots=0,
            deterministic_search_seeds=True,
        )
        self.teacher = ExactStatePriorGuidedMPC(
            **kwargs,
            search_samples=1,
        )
        self.trace = []

    def set_round_seed(self, seed):
        self.sparse.set_round_seed(seed)
        self.teacher.set_round_seed(seed)

    def reset(self):
        self.sparse.reset()
        self.teacher.reset()
        self.trace = []

    def _search_reason(self, before, sparse_search_ran):
        for name in ("followup", "critical", "proactive", "behavior", "unsafe"):
            if getattr(self.sparse, f"{name}_searches") > before[name]:
                return name
        if sparse_search_ran:
            return "search"
        if self.sparse.committed_frames > before["committed"]:
            return "committed"
        return "audit"

    def act(self, game):
        before = {
            "exact": self.sparse.exact_searches,
            "followup": self.sparse.followup_searches,
            "critical": self.sparse.critical_searches,
            "proactive": self.sparse.proactive_searches,
            "behavior": self.sparse.behavior_searches,
            "unsafe": self.sparse.unsafe_searches,
            "committed": self.sparse.committed_frames,
        }
        sparse_controls = self.sparse.act(game)
        teacher_controls = self.teacher.act(game)
        sparse_search_ran = self.sparse.exact_searches > before["exact"]
        sparse_decision = (
            self.sparse.last_search_decision if sparse_search_ran else None)
        teacher_decision = self.teacher.last_search_decision or {}
        source = "teacher" if game.frame >= self.handoff_frame else "sparse"
        selected = teacher_controls if source == "teacher" else sparse_controls

        if game.frame >= self.trace_start:
            sparse_index = action_index(sparse_controls)
            teacher_index = action_index(teacher_controls)
            self.trace.append({
                "frame": int(game.frame),
                "source": source,
                "sparse_index": sparse_index,
                "teacher_index": teacher_index,
                "actions_differ": sparse_index != teacher_index,
                "sparse_reason": self._search_reason(
                    before, sparse_search_ran),
                "sparse_search_ran": sparse_search_ran,
                "sparse_safe_roots": (
                    None if sparse_decision is None else
                    int(sparse_decision.get("safe_root_count", 0))),
                "sparse_search_selected": (
                    None if sparse_decision is None else
                    sparse_decision.get("selected_index")),
                "teacher_safe_roots": int(
                    teacher_decision.get("safe_root_count", 0)),
                "teacher_search_selected": teacher_decision.get(
                    "selected_index"),
                "incoming_risk": float(self.sparse.last_incoming_risk),
                "me": {
                    "x": float(game.tanks[0].x),
                    "y": float(game.tanks[0].y),
                    "rotation": float(game.tanks[0].rotation),
                },
                "enemy": {
                    "x": float(game.tanks[1].x),
                    "y": float(game.tanks[1].y),
                    "rotation": float(game.tanks[1].rotation),
                },
                "bullets": len(game.bullets),
            })
        return selected


def _run_handoff(job):
    handoff_frame, args = job
    import torch

    torch.set_num_threads(1)
    policy = SparseTeacherHandoffPolicy(
        handoff_frame,
        args.base_net,
        args.value_net,
        trace_start=args.trace_start,
    )
    policy.set_round_seed(args.seed)
    started = time.time()
    result = play_round_dual_engine(policy, args.seed)
    first_no_safe = next((
        row["frame"] for row in policy.trace
        if row["sparse_search_ran"]
        and row["sparse_search_selected"] is None
    ), None)
    result.update({
        "seed": args.seed,
        "handoff_frame": handoff_frame,
        "elapsed_seconds": time.time() - started,
        "first_no_safe_frame": first_no_safe,
        "action_disagreements": sum(
            row["actions_differ"] for row in policy.trace),
        "sparse_full_searches": policy.sparse.exact_searches,
        "teacher_full_searches": policy.teacher.exact_searches,
        "trace": policy.trace,
    })
    return result


def parse_handoffs(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=990008)
    parser.add_argument(
        "--handoff-list", default="144,168,180,191,196,202")
    parser.add_argument("--trace-start", type=int, default=120)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-net", default=(
        "training/models/p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--value-net", default=(
        "training/models/p27b_risk_value_iter00.pt"))
    parser.add_argument(
        "--out",
        default="training/analysis/runs/sparse_exact_handoff_990008.json",
    )
    args = parser.parse_args()
    handoffs = parse_handoffs(args.handoff_list)
    started = time.time()
    worker_count = max(1, min(args.workers, len(handoffs)))
    if worker_count == 1:
        rows = [_run_handoff((frame, args)) for frame in handoffs]
    else:
        with mp.get_context("spawn").Pool(worker_count) as pool:
            rows = list(pool.imap_unordered(
                _run_handoff, ((frame, args) for frame in handoffs)))
    rows.sort(key=lambda row: row["handoff_frame"])
    report = {
        "method": "sparse_exact_teacher_handoff",
        "seed": args.seed,
        "handoffs": handoffs,
        "elapsed_seconds": time.time() - started,
        "results": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for row in rows:
        print(
            f"handoff={row['handoff_frame']} result={row['true_result']} "
            f"frames={row['frames']} first_no_safe={row['first_no_safe_frame']} "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
