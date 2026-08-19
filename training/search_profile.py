"""Measure where the hybrid policy actually spends wall time.

Why
---
The DAgger round-1 collection falsified the "GRU proposes unsafe moves" theory:
the temporal net is overruled on only 7.05% of the frames it drives, and
GRU-driven frames search *five times less* than non-GRU frames (4.76% vs
23.23%).  Search load is architectural, not learned:

    proactive 32.2% | fire 21.9% | risk 17.9% | followup 15.4% |
    audit-fail 7.9% | topology 4.7%

Back-of-envelope from that run put one ``full_search`` at ~1.2 s, but that
number came from dividing total CPU seconds across 6 worker processes on a
laptop, so efficiency-core scheduling could inflate it badly.  Optimising
against a number derived that way is guessing.

This module measures directly: single process, single thread, real timers
around each stage.  It changes nothing in the hot path -- the wrappers are
installed at runtime and removed afterwards.

Timings are **inclusive**: ``_full_search`` contains ``_search`` contains
``exact_root_search``.  Read the output as a tree, not as a sum.  The
``exclusive`` column is inclusive time minus time attributed to the direct
children listed under ``CHILDREN``.

Usage
-----
    python3 training/search_profile.py \
        --temporal-intent-net training/models/temporal_intent_topology_v1.pt \
        --seed-list 970000:5 \
        --out training/analysis/dagger/search_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Game/policy imports live inside run() on purpose: they pull in tkinter,
# gymnasium and torch, and keeping them out of module scope means the
# Profiler itself stays unit-testable in a bare environment.


# label -> direct children, used to derive exclusive time.
CHILDREN = {
    "act": ["build_observation", "p27_value", "temporal_predict",
            "topology_plan", "audit_action", "full_search"],
    "full_search": ["search", "long_tail_fire"],
    "search": ["exact_root_search", "successor_viability"],
    "audit_action": ["audit_root_search"],
}

# Primitives shared by every rollout path.  exact_root_search,
# successor_viability, long_tail_fire and audit_root_search all call them, so
# they belong to no single parent and must stay out of CHILDREN -- otherwise
# their time would be subtracted from one caller and double-counted against
# the others.  They are reported standalone, for attribution only.
PRIMITIVES = ("clone_exact_game", "rollout_targets")


class Profiler:
    """Runtime wrappers with per-label call counts and inclusive seconds."""

    def __init__(self):
        self.stats = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
        self._undo = []

    def wrap(self, module_or_obj, attribute, label):
        original = getattr(module_or_obj, attribute)
        stats = self.stats[label]

        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                stats["calls"] += 1
                stats["seconds"] += time.perf_counter() - started

        setattr(module_or_obj, attribute, wrapped)
        self._undo.append((module_or_obj, attribute, original))
        return original

    def restore(self):
        for module_or_obj, attribute, original in reversed(self._undo):
            setattr(module_or_obj, attribute, original)
        self._undo.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        return False

    def report(self, total_frames):
        rows = {}
        for label, values in self.stats.items():
            calls = values["calls"]
            seconds = values["seconds"]
            children = sum(
                self.stats[child]["seconds"] for child in CHILDREN.get(label, [])
                if child in self.stats)
            rows[label] = {
                "calls": calls,
                "inclusive_seconds": seconds,
                "exclusive_seconds": max(0.0, seconds - children),
                "ms_per_call": 1000.0 * seconds / calls if calls else 0.0,
                "calls_per_frame": calls / max(1, total_frames),
                "ms_per_frame": 1000.0 * seconds / max(1, total_frames),
            }
        return rows


def install(profiler):
    """Wrap every stage we care about, at the site where it is looked up."""
    import training.exact_state_mpc_teacher as teacher_module
    import training.sparse_exact_safety_policy as policy_module
    import training.temporal_intent_model as temporal_module
    from training.exact_state_mpc_teacher import ExactStatePriorGuidedMPC
    from training.map_topology_planner import MapTopologyPlanner
    from training.sparse_exact_safety_policy import SparseExactSafetyPolicy

    # Module-level functions must be patched where they are *imported to*,
    # because both modules did `from ... import name` at import time.
    profiler.wrap(policy_module, "exact_root_search", "audit_root_search")
    profiler.wrap(teacher_module, "exact_root_search", "exact_root_search")
    profiler.wrap(teacher_module, "exact_successor_viability",
                  "successor_viability")
    profiler.wrap(policy_module, "rollout_exact_sequence", "long_tail_fire")

    # Rollout primitives, to separate "cloning the world" from "simulating
    # it".  If clone dominates, a hand-written clone is a pure win: it changes
    # no decision at all.  If simulation dominates, only horizon/top-k move
    # the needle and those do trade against safety.
    profiler.wrap(teacher_module, "clone_exact_game", "clone_exact_game")
    profiler.wrap(teacher_module, "rollout_targets", "rollout_targets")

    # Per-frame work that runs whether or not a search happens.
    profiler.wrap(policy_module, "build_observation", "build_observation")
    profiler.wrap(SparseExactSafetyPolicy, "_p27_value", "p27_value")

    profiler.wrap(SparseExactSafetyPolicy, "act", "act")
    profiler.wrap(SparseExactSafetyPolicy, "_full_search", "full_search")
    profiler.wrap(SparseExactSafetyPolicy, "_audit_action", "audit_action")
    profiler.wrap(ExactStatePriorGuidedMPC, "_search", "search")
    profiler.wrap(temporal_module.TemporalIntentRuntime, "predict",
                  "temporal_predict")
    profiler.wrap(MapTopologyPlanner, "choose_goal", "topology_plan")


def run(args):
    import torch

    from play_tank_trouble import Game
    from training.dagger_distill import make_dagger_policy
    from training.evaluate import RoundTracker, _round_stats
    from training.sparse_exact_safety_policy import parse_seeds

    torch.set_num_threads(1)
    seeds = parse_seeds(args.seed_list)
    profiler = Profiler()
    install(profiler)

    # The base-net forward is inside act() and has no separate seam, so it is
    # measured as act()'s exclusive time rather than wrapped directly.
    rounds = []
    total_frames = 0
    started = time.perf_counter()
    try:
        for seed in seeds:
            game = Game(seed=seed, ai_enabled=True)
            policy = make_dagger_policy(
                args.base_net, args.value_net, args.temporal_intent_net,
                temporal_confidence=args.temporal_confidence,
                top_k=args.top_k, search_horizon=args.search_horizon)
            policy.set_round_seed(seed)
            tracker = RoundTracker(game)
            frames = 0
            true_result = None
            round_started = time.perf_counter()
            while frames < args.max_frames:
                controls = policy.act(game)
                tank = game.tanks[0]
                tank.forward = bool(controls.get("forward", False))
                tank.backup = bool(controls.get("backup", False))
                tank.turn_left = bool(controls.get("turn_left", False))
                tank.turn_right = bool(controls.get("turn_right", False))
                tank.fire = bool(controls.get("fire", False))
                tracker.pre_step()
                events = game.step()
                frames += 1
                tracker.post_step(events, 1)
                for event in events:
                    if event[0] == "round_end":
                        winner = event[1]
                        true_result = ("win" if winner == 0 else
                                       "loss" if winner == 1 else
                                       "double_death")
                if true_result:
                    break
            elapsed = time.perf_counter() - round_started
            total_frames += frames
            row = {
                "seed": int(seed),
                "result": true_result or "draw",
                "frames": frames,
                "elapsed_seconds": elapsed,
                "ms_per_frame": 1000.0 * elapsed / max(1, frames),
                "full_searches": int(policy.exact_searches),
                "search_frame_rate": float(
                    policy.exact_searches / max(1, frames)),
            }
            row.update(_round_stats(tracker, row["result"], frames))
            rounds.append(row)
            print(
                f"seed={seed} result={row['result']} frames={frames} "
                f"searches={row['full_searches']} "
                f"rate={row['search_frame_rate']:.1%} "
                f"{row['ms_per_frame']:.1f} ms/frame", flush=True)
    finally:
        profiler.restore()

    wall = time.perf_counter() - started
    stages = profiler.report(total_frames)
    searches = sum(row["full_searches"] for row in rounds)
    full_search = stages.get("full_search", {})
    payload = {
        "seeds": list(seeds),
        "rounds": rounds,
        "total_frames": total_frames,
        "wall_seconds": wall,
        "ms_per_frame": 1000.0 * wall / max(1, total_frames),
        "frame_budget_ms_at_60fps": 1000.0 / 60.0,
        "total_full_searches": searches,
        "seconds_per_full_search": (
            full_search.get("inclusive_seconds", 0.0) / searches
            if searches else 0.0),
        "mean_search_frame_rate": float(np.mean(
            [row["search_frame_rate"] for row in rounds])),
        "stages": stages,
        "children": CHILDREN,
        "note": "stage seconds are inclusive; see CHILDREN for nesting",
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(format_stages(payload), flush=True)
    return payload


def format_stages(payload):
    stages = payload["stages"]
    header = (f"{'stage':<22}{'calls':>8}{'incl s':>10}{'excl s':>10}"
              f"{'ms/call':>10}{'ms/frame':>10}")
    lines = ["", header, "-" * len(header)]
    for label in sorted(stages, key=lambda key: -stages[key]["inclusive_seconds"]):
        row = stages[label]
        lines.append(
            f"{label:<22}{row['calls']:>8}{row['inclusive_seconds']:>10.2f}"
            f"{row['exclusive_seconds']:>10.2f}{row['ms_per_call']:>10.2f}"
            f"{row['ms_per_frame']:>10.2f}")
    lines.append("")
    lines.append(
        f"frames={payload['total_frames']}  wall={payload['wall_seconds']:.1f}s"
        f"  {payload['ms_per_frame']:.1f} ms/frame"
        f"  (60fps budget {payload['frame_budget_ms_at_60fps']:.1f} ms)")
    lines.append(
        f"full_searches={payload['total_full_searches']}  "
        f"{payload['seconds_per_full_search']:.3f} s per full_search  "
        f"search_rate={payload['mean_search_frame_rate']:.1%}")
    clone = stages.get("clone_exact_game", {}).get("inclusive_seconds", 0.0)
    sim = stages.get("rollout_targets", {}).get("inclusive_seconds", 0.0)
    if clone or sim:
        lines.append(
            f"rollout primitives: clone {clone:.1f}s vs simulate {sim:.1f}s"
            f"  -> clone is {100 * clone / max(1e-9, clone + sim):.0f}% "
            "of rollout cost")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-net",
                    default="training/models/p26_amortized_mpc_iter05.pt")
    ap.add_argument("--value-net",
                    default="training/models/p27b_risk_value_iter00.pt")
    ap.add_argument("--temporal-intent-net", default=None)
    ap.add_argument("--temporal-confidence", type=float, default=0.60)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--search-horizon", type=int, default=72)
    ap.add_argument("--seed-list", default="970000:5")
    ap.add_argument("--max-frames", type=int, default=1800)
    ap.add_argument("--out",
                    default="training/analysis/dagger/search_profile.json")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
