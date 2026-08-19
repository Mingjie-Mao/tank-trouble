"""Staged gate for a candidate temporal checkpoint.

The project rule is that a checkpoint is never accepted on win rate alone.  A
DAgger round can trade win rate for speed, or buy a lower search rate by
refusing to propose *any* aggressive move — both look fine on win rate and are
regressions.  This module makes the full comparison mechanical.

It consumes the JSON that ``training/sparse_exact_safety_policy.py --out``
already writes, so no new evaluation path is introduced and baseline and
candidate are always measured by the same code.

Gate order (each level must pass before the next runs):

    permanent -> paired12 -> unseen100 -> broad300 -> official

Metric classes
--------------
``hard``      regression fails the gate outright (win rate, non-win count).
``objective`` the round is pointless unless at least one improves
              (search rate, wall time).
``watch``     regression is reported and must be explained, not auto-failed;
              these are the behaviour metrics where noise is large at small n.

Usage
-----
    python3 training/dagger_gate.py compare \
        --baseline training/analysis/dagger/base_permanent.json \
        --candidate training/analysis/dagger/cand_permanent.json \
        --level permanent \
        --out training/analysis/dagger/gate_permanent.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


# Seeds that have ever produced a loss or double death and must pass forever.
# 996004/998002/979000 come from the earlier top-k experiments; the 97000xx
# seeds come from the 268-round live GUI baseline.
PERMANENT_REGRESSION_SEEDS = (
    979000, 996004, 998002,
    970105, 970128, 970163, 970170, 970197, 970243, 970252,
)

GATE_LEVELS = ("permanent", "paired12", "unseen100", "broad300", "official")

HARD = "hard"
OBJECTIVE = "objective"
WATCH = "watch"

# name -> (direction, class, tolerance)
# direction  +1: higher is better, -1: lower is better
METRICS = {
    "win_rate": (+1, HARD, 0.0),
    "nonwin_rate": (-1, HARD, 0.0),
    "double_death_rate": (-1, HARD, 0.0),
    "mean_search_frame_rate": (-1, OBJECTIVE, 0.0),
    "mean_elapsed_seconds": (-1, OBJECTIVE, 0.0),
    "high_search_round_rate": (-1, OBJECTIVE, 0.0),
    "mean_frames": (-1, WATCH, 0.05),
    "max_frames": (-1, WATCH, 0.10),
    "reversal_per_1000": (-1, WATCH, 0.05),
    "stutter_per_1000": (-1, WATCH, 0.10),
    "dead_end_per_1000": (-1, WATCH, 0.10),
    "missed_fire_per_1000": (-1, WATCH, 0.10),
    "fire_capture_rate": (+1, WATCH, 0.05),
    "active_kill_rate": (+1, WATCH, 0.05),
    "long_tail_rejection_rate": (-1, WATCH, 0.05),
    "topology_abort_rate": (-1, WATCH, 0.05),
}

HIGH_SEARCH_THRESHOLD = 0.25


def _events(row):
    metrics = row.get("event_metrics") or {}
    return metrics.get("events") or {}


def _sum(rows, key, default=0):
    return sum(int(row.get(key) or default) for row in rows)


def summarize_report(report):
    """Reduce one evaluation report to the comparable metric vector."""
    rows = report.get("rounds") or []
    if not rows:
        raise ValueError("report contains no rounds")
    count = len(rows)
    results = [row.get("true_result") for row in rows]
    wins = sum(1 for value in results if value == "win")
    double = sum(1 for value in results if value == "double_death")
    frames = [int(row.get("frames") or 0) for row in rows]
    action_frames = sum(
        int((row.get("event_metrics") or {}).get("action_frames") or 0)
        for row in rows)
    action_frames = max(1, action_frames)
    search_rates = [float(row.get("search_frame_rate") or 0.0)
                    for row in rows]
    elapsed = [float(row.get("elapsed_seconds") or 0.0) for row in rows]

    windows = sum(_events(row).get("fire_window", 0) for row in rows)
    captured = sum(_events(row).get("captured_fire_window", 0) for row in rows)
    reversals = sum(
        _events(row).get("throttle_reversal", 0)
        + _events(row).get("turn_reversal", 0) for row in rows)
    checks = _sum(rows, "long_tail_fire_checks")
    rejections = _sum(rows, "long_tail_fire_rejections")
    requests = _sum(rows, "topology_requests")
    aborts = _sum(rows, "topology_aborts")

    return {
        "rounds": count,
        "win_rate": wins / count,
        "nonwin_rate": (count - wins) / count,
        "double_death_rate": double / count,
        "mean_search_frame_rate": float(np.mean(search_rates)),
        "max_search_frame_rate": float(np.max(search_rates)),
        "high_search_round_rate": sum(
            1 for value in search_rates
            if value > HIGH_SEARCH_THRESHOLD) / count,
        "mean_elapsed_seconds": float(np.mean(elapsed)),
        "mean_frames": float(np.mean(frames)),
        "max_frames": float(np.max(frames)),
        "reversal_per_1000": 1000.0 * reversals / action_frames,
        "stutter_per_1000": 1000.0 * sum(
            _events(row).get("stutter_stall", 0) for row in rows
        ) / action_frames,
        "dead_end_per_1000": 1000.0 * sum(
            _events(row).get("dead_end_stall", 0) for row in rows
        ) / action_frames,
        "missed_fire_per_1000": 1000.0 * sum(
            _events(row).get("missed_fire_window", 0) for row in rows
        ) / action_frames,
        "fire_capture_rate": captured / windows if windows else 0.0,
        "active_kill_rate": sum(
            1 for row in rows if int(row.get("kills") or 0) > 0) / count,
        "long_tail_rejection_rate": rejections / checks if checks else 0.0,
        "topology_abort_rate": aborts / requests if requests else 0.0,
        "nonwin_seeds": sorted(
            int(row["seed"]) for row in rows
            if row.get("true_result") != "win"),
    }


def compare_metric(name, baseline, candidate):
    """Signed improvement for one metric, normalised so + is always better."""
    direction, metric_class, tolerance = METRICS[name]
    delta = (candidate - baseline) * direction
    scale = max(abs(baseline), 1e-9)
    relative = delta / scale
    return {
        "metric": name,
        "class": metric_class,
        "baseline": float(baseline),
        "candidate": float(candidate),
        "delta": float(candidate - baseline),
        "improved": bool(delta > 0),
        "regressed": bool(relative < -tolerance),
        "within_tolerance": bool(-tolerance <= relative <= 0),
        "tolerance": tolerance,
    }


def compare_summaries(baseline, candidate):
    return [compare_metric(name, baseline[name], candidate[name])
            for name in METRICS]


def gate_verdict(rows, level, permanent_seeds=PERMANENT_REGRESSION_SEEDS,
                 candidate_nonwin_seeds=()):
    """Decide pass/fail from the comparison rows.

    Rules, in order:

    1. On the ``permanent`` level any non-win at all is a failure -- these are
       seeds that a correct policy has to survive, so a rate comparison is not
       enough.
    2. Any ``hard`` regression fails.
    3. At least one ``objective`` metric must improve, otherwise the round
       bought nothing and the checkpoint is not worth the risk.
    """
    hard_regressions = [row["metric"] for row in rows
                        if row["class"] == HARD and row["regressed"]]
    objective_gains = [row["metric"] for row in rows
                       if row["class"] == OBJECTIVE and row["improved"]]
    watch_regressions = [row["metric"] for row in rows
                         if row["class"] == WATCH and row["regressed"]]

    failures = []
    if level == "permanent":
        blocked = sorted(set(candidate_nonwin_seeds) & set(permanent_seeds))
        # A brand new failure on a permanent seed is still a failure.
        blocked = sorted(set(blocked) | set(candidate_nonwin_seeds))
        if blocked:
            failures.append(
                f"permanent regression seeds not won: {blocked}")
    if hard_regressions:
        failures.append(f"hard metric regression: {hard_regressions}")
    if not objective_gains:
        failures.append(
            "no objective metric improved (search rate / wall time / "
            "high-search rounds)")

    return {
        "level": level,
        "passed": not failures,
        "failures": failures,
        "hard_regressions": hard_regressions,
        "objective_gains": objective_gains,
        "watch_regressions": watch_regressions,
        "requires_explanation": watch_regressions,
        "next_level": _next_level(level) if not failures else None,
    }


def _next_level(level):
    index = GATE_LEVELS.index(level)
    return GATE_LEVELS[index + 1] if index + 1 < len(GATE_LEVELS) else None


def compare(args):
    with open(args.baseline, encoding="utf-8") as handle:
        baseline_report = json.load(handle)
    with open(args.candidate, encoding="utf-8") as handle:
        candidate_report = json.load(handle)

    baseline = summarize_report(baseline_report)
    candidate = summarize_report(candidate_report)
    rows = compare_summaries(baseline, candidate)
    verdict = gate_verdict(
        rows, args.level,
        candidate_nonwin_seeds=candidate["nonwin_seeds"])

    payload = {
        "level": args.level,
        "baseline_file": args.baseline,
        "candidate_file": args.candidate,
        "baseline": baseline,
        "candidate": candidate,
        "metrics": rows,
        "verdict": verdict,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(format_table(rows), flush=True)
    print(json.dumps(verdict, indent=2, sort_keys=True), flush=True)
    return payload


def format_table(rows):
    header = f"{'metric':<26}{'baseline':>12}{'candidate':>12}  flag"
    lines = [header, "-" * len(header)]
    for row in rows:
        flag = ("REGRESSED" if row["regressed"] else
                "improved" if row["improved"] else "flat/tol")
        lines.append(
            f"{row['metric']:<26}{row['baseline']:>12.4f}"
            f"{row['candidate']:>12.4f}  {flag} [{row['class']}]")
    return "\n".join(lines)


def seeds(args):
    print(",".join(str(seed) for seed in PERMANENT_REGRESSION_SEEDS))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    compare_ap = sub.add_parser("compare", help="compare two eval reports")
    compare_ap.add_argument("--baseline", required=True)
    compare_ap.add_argument("--candidate", required=True)
    compare_ap.add_argument("--level", choices=GATE_LEVELS,
                            default="permanent")
    compare_ap.add_argument("--out", default=None)
    compare_ap.set_defaults(func=compare)

    seeds_ap = sub.add_parser(
        "seeds", help="print the permanent regression seed list")
    seeds_ap.set_defaults(func=seeds)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
