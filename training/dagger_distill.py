"""DAgger round for the temporal movement policy.

Why this exists
---------------
``temporal_intent_pipeline.collect`` builds its dataset with a teacher that has
**no** temporal net loaded, and it throws away every non-winning round
(``rows if true_result == "win"``).  The deployed hybrid runs *with* the GRU in
the loop, so the training states and the deployment states come from different
distributions.  The observable symptom is that the exact safety layer keeps
overruling the GRU: on the live log the long-tail fire check rejects 275 of 476
proposals and search rates reach 40-54% on the hard seeds.

This module closes that loop:

1. roll out **with the current temporal net driving movement** (on-policy),
2. keep the exact teacher as the label source (executed action = label),
3. tag each state by which correction channel fired,
4. keep the pre-death window of losing rounds instead of discarding it,
5. emit an npz that ``temporal_intent_pipeline.train`` can consume unchanged.

What it does *not* claim: retraining on these states is not guaranteed to raise
win rate.  The mechanism it targets is narrow and falsifiable — if the next
run's ``temporal_correction_rate`` and ``search_frame_rate`` do not drop while
win rate holds, the hypothesis is wrong and the checkpoint must be rejected.

Usage
-----
    python3 training/dagger_distill.py collect \
        --temporal-intent-net training/models/temporal_intent_topology_v1.pt \
        --seed-list 970000:120 --workers 6 \
        --out training/temporal_intent_data/dagger_round1.npz \
        --report training/analysis/dagger/collect_round1.json

    python3 training/temporal_intent_pipeline.py train \
        --data training/temporal_intent_data/dagger_round1.npz \
        --out training/models/temporal_intent_dagger_r1.pt
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from play_tank_trouble import Game  # noqa: E402
from training.battle_supervision import diagnose_battle  # noqa: E402
from training.dagger_correction_recorder import (  # noqa: E402
    DISAGREEMENT_TAGS,
    TERMINAL_WINDOW_FRAMES,
    build_dataset,
)
from training.evaluate import RoundTracker, _round_stats  # noqa: E402
from training.sparse_exact_safety_policy import (  # noqa: E402
    SparseExactSafetyPolicy,
    parse_seeds,
)


# Must match temporal_intent_pipeline / watch.py exactly.  A mismatch here
# silently changes the state distribution and invalidates the whole round.
DEFAULT_BASE_NET = "training/models/p26_amortized_mpc_iter05.pt"
DEFAULT_VALUE_NET = "training/models/p27b_risk_value_iter00.pt"


def make_dagger_policy(base_net, value_net, temporal_intent_net,
                       temporal_confidence=0.60, top_k=12,
                       search_horizon=72):
    """Deployment-shaped policy with the temporal net actually driving.

    Kept in one place so collection and evaluation cannot silently drift
    apart; a mismatch here reintroduces exactly the off-policy bug this
    module exists to fix.
    """
    return SparseExactSafetyPolicy(
        base_net=base_net,
        value_net=value_net,
        fire_margin=0.16,
        top_k=top_k,
        search_horizon=search_horizon,
        search_death_penalty=0.18,
        search_dd_penalty=0.45,
        search_kill_bonus=0.05,
        search_max_death=0.0,
        search_max_dd=0.0,
        successor_shield=True,
        successor_horizon=72,
        successor_shield_max_safe_roots=2,
        suppress_secured_fire=True,
        min_unsecured_fire_gain=2.0,
        audit_interval=6,
        proactive_interval=24,
        behavior_full_search=True,
        search_hold_frames=6,
        search_on_fire=True,
        risk_search_threshold=0.18,
        long_tail_fire_horizon=375,
        topology_assist=True,
        topology_intent_max_frames=75,
        topology_cooldown_frames=25,
        topology_pursuit_delay_frames=20,
        network_move_hold_frames=4,
        # NOT temporal_record_state=True.  temporal_intent_pipeline sets it
        # because it loads no net, so nothing consumes the features at
        # runtime.  Here the net IS driving, and the recorded feature vector
        # must be exactly the one it was trained on.  Left False, the feature
        # set follows the loaded checkpoint's feature_dim automatically, so
        # this works for both compact (157) and state (597) checkpoints.
        #
        # It also keeps the experiment honest: this round changes the state
        # distribution only.  Changing the feature set at the same time would
        # confound the two, and the state-feature variant already failed once
        # (temporal_intent_state_searchgate_v3).
        temporal_intent_net=temporal_intent_net,
        temporal_confidence=temporal_confidence,
        deterministic_search_seeds=True,
    )


def _collect_seed(job):
    (seed, base_net, value_net, temporal_intent_net, temporal_confidence,
     top_k, search_horizon, max_frames) = job
    import torch

    torch.set_num_threads(1)
    game = Game(seed=seed, ai_enabled=True)
    policy = make_dagger_policy(
        base_net, value_net, temporal_intent_net,
        temporal_confidence=temporal_confidence, top_k=top_k,
        search_horizon=search_horizon)
    policy.set_round_seed(seed)
    tracker = RoundTracker(game)
    rows = []
    true_result = None
    frames = 0
    started = time.time()
    while frames < max_frames or tracker.first_destroy is not None:
        controls = policy.act(game)
        sample = dict(policy.last_temporal_sample)
        tank = game.tanks[0]
        target = sample["target"]
        target_x = (target[0] + 0.5) * game.scale
        target_y = (target[1] + 0.5) * game.scale

        tank.forward = bool(controls.get("forward", False))
        tank.backup = bool(controls.get("backup", False))
        tank.turn_left = bool(controls.get("turn_left", False))
        tank.turn_right = bool(controls.get("turn_right", False))
        tank.fire = bool(controls.get("fire", False))
        tracker.pre_step()
        events = game.step()
        frames += 1
        tracker.post_step(events, 1)
        distance_after = float(np.hypot(
            tank.x - target_x, tank.y - target_y) / game.scale)
        sample["progress"] = float(np.clip(
            sample["target_distance_before"] - distance_after, -1.0, 1.0))
        sample["frame"] = frames - 1
        rows.append(sample)
        for event in events:
            if event[0] == "round_end":
                winner = event[1]
                true_result = ("win" if winner == 0 else
                               "loss" if winner == 1 else "double_death")
        if true_result:
            break

    result_label = true_result or "draw"
    stats = _round_stats(tracker, result_label, frames)
    result = {
        "seed": int(seed),
        "result": result_label,
        "frames": int(frames),
        "elapsed_seconds": time.time() - started,
        # NOTE: unlike temporal_intent_pipeline, losing rounds keep their rows.
        "rows": rows,
        "search_rate": float(policy.exact_searches / max(1, frames)),
        "temporal_overrides": int(policy.temporal_overrides),
        "long_tail_fire_checks": int(policy.long_tail_fire_checks),
        "long_tail_fire_rejections": int(policy.long_tail_fire_rejections),
    }
    result.update(stats)
    result["event_metrics"] = policy.event_tracker.summary()
    result["diagnosis"] = diagnose_battle(result)
    return result


def collect(args):
    seeds = parse_seeds(args.seed_list)
    jobs = [(seed, args.base_net, args.value_net, args.temporal_intent_net,
             args.temporal_confidence, args.top_k, args.search_horizon,
             args.max_frames) for seed in seeds]
    workers = max(1, min(args.workers, len(jobs)))
    if workers == 1:
        iterator = map(_collect_seed, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers)
        iterator = pool.imap_unordered(_collect_seed, jobs)
    rounds = []
    started = time.time()
    try:
        for row in iterator:
            rounds.append(row)
            print(
                f"collect={len(rounds)}/{len(jobs)} seed={row['seed']} "
                f"result={row['result']} frames={row['frames']} "
                f"search={row['search_rate']:.1%}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    rounds.sort(key=lambda item: item["seed"])

    payload, round_rows = build_dataset(
        rounds,
        terminal_window=args.terminal_window,
        keep_accepted_fraction=args.keep_accepted_fraction,
        rng_seed=args.split_seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **payload)

    tag_counts = Counter(payload["tag"].tolist())
    temporal_used = int(payload["temporal_used"].sum())
    report = {
        "out": args.out,
        "temporal_intent_net": args.temporal_intent_net,
        "requested_rounds": len(seeds),
        "collected_rounds": len(rounds),
        "states": int(len(payload["movement"])),
        "results": dict(Counter(row["result"] for row in rounds)),
        "tags": dict(tag_counts),
        "disagreement_states": int(sum(
            count for tag, count in tag_counts.items()
            if tag in DISAGREEMENT_TAGS)),
        "temporal_frames": temporal_used,
        "temporal_correction_rate": float(
            payload["temporal_corrected"].sum() / max(1, temporal_used)),
        "movement_correction_rate": float(
            payload["movement_corrected"].mean()),
        "mean_search_rate": float(np.mean(
            [row["search_rate"] for row in rounds])),
        "max_search_rate": float(max(
            row["search_rate"] for row in rounds)),
        "mean_frames": float(np.mean([row["frames"] for row in rounds])),
        "max_frames_seen": int(max(row["frames"] for row in rounds)),
        "active_kill_rounds": int(sum(
            1 for row in rounds if (row.get("kills") or 0) > 0)),
        "elapsed_seconds": time.time() - started,
        # Supervision requirement: every round is logged, not just failures.
        "rounds": round_rows,
        "failed_rounds": [
            {key: row[key] for key in (
                "seed", "result", "frames", "death_cause", "diagnosis")}
            for row in rounds if row["result"] != "win"
        ],
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    summary = {key: value for key, value in report.items()
               if key not in ("rounds", "failed_rounds")}
    summary["failed_seeds"] = [row["seed"] for row in rounds
                               if row["result"] != "win"]
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    collect_ap = sub.add_parser(
        "collect", help="roll out with the GRU in the loop and tag corrections")
    collect_ap.add_argument("--base-net", default=DEFAULT_BASE_NET)
    collect_ap.add_argument("--value-net", default=DEFAULT_VALUE_NET)
    collect_ap.add_argument("--temporal-intent-net", required=True)
    collect_ap.add_argument("--temporal-confidence", type=float, default=0.60)
    collect_ap.add_argument("--top-k", type=int, default=12)
    collect_ap.add_argument("--search-horizon", type=int, default=72)
    collect_ap.add_argument("--seed-list", default="970000:120")
    collect_ap.add_argument("--max-frames", type=int, default=1800)
    collect_ap.add_argument("--workers", type=int, default=6)
    collect_ap.add_argument("--terminal-window", type=int,
                            default=TERMINAL_WINDOW_FRAMES)
    collect_ap.add_argument(
        "--keep-accepted-fraction", type=float, default=1.0,
        help="subsample uncorrected states; 1.0 keeps all of them. "
             "Below 1.0 it breaks GRU sequence continuity — ablations only.")
    collect_ap.add_argument("--split-seed", type=int, default=2701)
    collect_ap.add_argument(
        "--out", default="training/temporal_intent_data/dagger_round1.npz")
    collect_ap.add_argument(
        "--report", default="training/analysis/dagger/collect_round1.json")
    collect_ap.set_defaults(func=collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
