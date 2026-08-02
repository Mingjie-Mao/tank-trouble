"""Tiered policy calibration for the rejected P29 distillation head.

The sweep changes only inference-time score adjustments. It screens several
small, interpretable configurations on identical seed bands, confirms only
qualified candidates, and runs medium validation only after a strict safety
gate. Existing summaries are reused when their parameters match.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "training", "analysis", "runs")


BASE = {
    "assist_margin": 0.08,
    "assist_weight": 0.35,
    "max_bonus": 0.10,
    "kill_weight": 0.04,
    "death_weight": 0.12,
    "double_death_weight": 0.18,
    "survive_weight": 0.02,
    "risk_threshold": 0.55,
    "fire_delta_margin": 0.14,
    "global_fire_risk_penalty": 0.0,
    "global_fire_risk_threshold": 1.10,
    "global_fire_dd_threshold": 1.10,
    "low_quality_fire_penalty": 0.0,
    "low_quality_fire_delta": 0.0,
    "opportunity_bonus": 0.0,
    "opportunity_min_line": 0.76,
    "opportunity_max_risk": 0.25,
    "opportunity_max_danger": 0.45,
    "opportunity_min_fire_delta": 0.08,
    "escape_bonus": 0.0,
    "escape_min_gain": -0.02,
    "escape_max_danger": 0.55,
    "escape_hold_frames": 0,
    "stall_fire_penalty": 0.0,
}


def _config(name, **overrides):
    params = dict(BASE)
    params.update(overrides)
    return {"name": name, "params": params}


CONFIGS = [
    _config("baseline"),
    _config(
        "fire_safe_mild",
        death_weight=0.14,
        double_death_weight=0.24,
        global_fire_risk_penalty=0.035,
        global_fire_risk_threshold=0.58,
        global_fire_dd_threshold=0.50,
        low_quality_fire_penalty=0.02,
        low_quality_fire_delta=-0.02,
        opportunity_bonus=0.04,
        opportunity_min_fire_delta=0.04,
        opportunity_max_danger=0.40,
    ),
    _config(
        "balanced_mild",
        death_weight=0.14,
        double_death_weight=0.26,
        global_fire_risk_penalty=0.04,
        global_fire_risk_threshold=0.55,
        global_fire_dd_threshold=0.45,
        low_quality_fire_penalty=0.02,
        low_quality_fire_delta=-0.02,
        opportunity_bonus=0.05,
        opportunity_min_line=0.74,
        opportunity_min_fire_delta=0.03,
        opportunity_max_danger=0.40,
        escape_bonus=0.04,
        escape_min_gain=-0.01,
        escape_max_danger=0.50,
        escape_hold_frames=12,
        stall_fire_penalty=0.03,
    ),
    _config(
        "balanced_medium",
        death_weight=0.16,
        double_death_weight=0.30,
        risk_threshold=0.50,
        fire_delta_margin=0.10,
        global_fire_risk_penalty=0.055,
        global_fire_risk_threshold=0.52,
        global_fire_dd_threshold=0.42,
        low_quality_fire_penalty=0.03,
        low_quality_fire_delta=-0.01,
        opportunity_bonus=0.065,
        opportunity_min_line=0.72,
        opportunity_max_risk=0.24,
        opportunity_max_danger=0.38,
        opportunity_min_fire_delta=0.01,
        escape_bonus=0.055,
        escape_max_danger=0.48,
        escape_hold_frames=16,
        stall_fire_penalty=0.04,
    ),
    _config(
        "mobility_safe",
        death_weight=0.15,
        double_death_weight=0.30,
        risk_threshold=0.50,
        fire_delta_margin=0.11,
        global_fire_risk_penalty=0.06,
        global_fire_risk_threshold=0.52,
        global_fire_dd_threshold=0.42,
        low_quality_fire_penalty=0.025,
        low_quality_fire_delta=-0.01,
        opportunity_bonus=0.04,
        opportunity_min_line=0.74,
        opportunity_min_fire_delta=0.03,
        opportunity_max_danger=0.38,
        escape_bonus=0.075,
        escape_min_gain=-0.03,
        escape_max_danger=0.45,
        escape_hold_frames=20,
        stall_fire_penalty=0.06,
    ),
    _config(
        "finish_safe",
        death_weight=0.17,
        double_death_weight=0.34,
        risk_threshold=0.48,
        fire_delta_margin=0.09,
        global_fire_risk_penalty=0.075,
        global_fire_risk_threshold=0.50,
        global_fire_dd_threshold=0.38,
        low_quality_fire_penalty=0.04,
        low_quality_fire_delta=0.0,
        opportunity_bonus=0.08,
        opportunity_min_line=0.72,
        opportunity_max_risk=0.22,
        opportunity_max_danger=0.35,
        opportunity_min_fire_delta=0.0,
        escape_bonus=0.045,
        escape_max_danger=0.45,
        escape_hold_frames=12,
        stall_fire_penalty=0.035,
    ),
    _config(
        "conservative_finish",
        death_weight=0.18,
        double_death_weight=0.38,
        risk_threshold=0.46,
        fire_delta_margin=0.10,
        global_fire_risk_penalty=0.08,
        global_fire_risk_threshold=0.48,
        global_fire_dd_threshold=0.36,
        low_quality_fire_penalty=0.05,
        low_quality_fire_delta=0.0,
        opportunity_bonus=0.06,
        opportunity_max_risk=0.20,
        opportunity_max_danger=0.32,
        opportunity_min_fire_delta=0.02,
        escape_bonus=0.05,
        escape_max_danger=0.42,
        escape_hold_frames=16,
        stall_fire_penalty=0.05,
    ),
]


CLI_NAMES = {
    "assist_margin": "p27b-assist-margin",
    "assist_weight": "p27b-assist-weight",
    "max_bonus": "p27b-max-bonus",
    "kill_weight": "p27b-kill-weight",
    "death_weight": "p27b-death-weight",
    "double_death_weight": "p27b-double-death-weight",
    "survive_weight": "p27b-survive-weight",
    "risk_threshold": "p27b-risk-threshold",
    "fire_delta_margin": "p27b-fire-delta-margin",
    "global_fire_risk_penalty": "p27b-global-fire-risk-penalty",
    "global_fire_risk_threshold": "p27b-global-fire-risk-threshold",
    "global_fire_dd_threshold": "p27b-global-fire-dd-threshold",
    "low_quality_fire_penalty": "p27b-low-quality-fire-penalty",
    "low_quality_fire_delta": "p27b-low-quality-fire-delta",
    "opportunity_bonus": "p27b-opportunity-bonus",
    "opportunity_min_line": "p27b-opportunity-min-line",
    "opportunity_max_risk": "p27b-opportunity-max-risk",
    "opportunity_max_danger": "p27b-opportunity-max-danger",
    "opportunity_min_fire_delta": "p27b-opportunity-min-fire-delta",
    "escape_bonus": "p27b-escape-bonus",
    "escape_min_gain": "p27b-escape-min-gain",
    "escape_max_danger": "p27b-escape-max-danger",
    "escape_hold_frames": "p27b-escape-hold-frames",
    "stall_fire_penalty": "p27b-stall-fire-penalty",
}


def _close(a, b):
    return abs(float(a) - float(b)) < 1e-9


def _summary_matches(summary, config, n, seed, value_net):
    if summary.get("n") != n or summary.get("seed") != seed:
        return False
    if summary.get("p27b_net") != value_net:
        return False
    for key, expected in config["params"].items():
        actual = summary.get("p27b_" + key)
        if actual is None or not _close(actual, expected):
            return False
    return True


def _paths(config_name, phase, n, seed):
    stem = f"p29b_{phase}_{config_name}_{n}_{seed}"
    return (
        os.path.join(RUN_DIR, stem + ".jsonl"),
        os.path.join(RUN_DIR, stem + "_summary.json"),
    )


def _run_eval(args, config, phase, n, seed):
    round_path, summary_path = _paths(config["name"], phase, n, seed)
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        if _summary_matches(summary, config, n, seed, args.value_net):
            print(f"reuse {phase} {config['name']} {n}@{seed}", flush=True)
            return summary

    command = [
        sys.executable,
        "training/p26_behavior_observer.py",
        "--net", args.base_net,
        "--p27b-net", args.value_net,
        "--n", str(n),
        "--seed", str(seed),
        "--workers", str(args.workers),
        "--fire-margin", str(args.fire_margin),
        "--out", round_path,
        "--summary", summary_path,
    ]
    for key, value in config["params"].items():
        command.extend(["--" + CLI_NAMES[key], str(value)])
    print(f"start {phase} {config['name']} {n}@{seed}", flush=True)
    started = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    print(
        f"done {phase} {config['name']} {n}@{seed}: "
        f"win={summary['win_rate']:.1%} loss={summary['loss_rate']:.1%} "
        f"dd={summary['double_death_rate']:.1%} "
        f"elapsed={time.time() - started:.0f}s",
        flush=True,
    )
    return summary


def _aggregate(config, summaries):
    games = sum(item["n"] for item in summaries)
    results = Counter()
    issues = Counter()
    assists = Counter()
    shots = kills = frames = 0.0
    for item in summaries:
        results.update(item.get("results", {}))
        issues.update(item.get("issues", {}))
        assists.update(item.get("assist_counts", {}))
        shots += item.get("shots_per_game", 0.0) * item["n"]
        kills += (item.get("hit_rate", 0.0)
                  * item.get("shots_per_game", 0.0) * item["n"])
        frames += item.get("avg_seconds", 0.0) * item["n"]
    win = results["win"] / games
    loss = results["loss"] / games
    dd = results["double_death"] / games
    draw = results["draw"] / games
    issue_pressure = (
        issues["missed_fire_window"] + issues["stutter_stall"]
        + 8 * issues["dead_end_stall"] + 4 * issues["blind_fire"]
    ) / games
    score = win - 0.80 * dd - 0.35 * draw - 0.00015 * issue_pressure
    return {
        "name": config["name"],
        "params": config["params"],
        "games": games,
        "results": dict(results),
        "win_rate": win,
        "loss_rate": loss,
        "double_death_rate": dd,
        "draw_rate": draw,
        "min_seed_win_rate": min(item["win_rate"] for item in summaries),
        "max_seed_double_death_rate": max(
            item["double_death_rate"] for item in summaries),
        "shots_per_game": shots / games,
        "hit_rate": kills / max(1.0, shots),
        "avg_seconds": frames / games,
        "issues": dict(issues),
        "issues_per_game": {key: value / games for key, value in issues.items()},
        "assist_counts": dict(assists),
        "selection_score": score,
        "seed_results": [
            {
                "seed": item["seed"],
                "win_rate": item["win_rate"],
                "loss_rate": item["loss_rate"],
                "double_death_rate": item["double_death_rate"],
                "draw_rate": item.get("draw_rate", 0.0),
            }
            for item in summaries
        ],
    }


def _screen_qualified(row):
    return (
        row["win_rate"] >= 0.90
        and row["min_seed_win_rate"] >= 0.875
        and row["double_death_rate"] <= 0.034
        and row["max_seed_double_death_rate"] <= 0.05
    )


def _confirm_qualified(row):
    return (
        row["win_rate"] >= 0.92
        and row["min_seed_win_rate"] >= 0.90
        and row["double_death_rate"] <= 0.025
        and row["max_seed_double_death_rate"] <= 0.038
    )


def _medium_qualified(row):
    return (
        row["win_rate"] >= 0.93
        and row["min_seed_win_rate"] >= 0.92
        and row["double_death_rate"] <= 0.02
        and row["max_seed_double_death_rate"] <= 0.025
    )


def _save_report(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _print_table(label, rows):
    print(f"===== {label} =====", flush=True)
    for row in sorted(rows, key=lambda item: item["selection_score"], reverse=True):
        print(
            f"{row['name']:20s} win={row['win_rate']:.1%} "
            f"min={row['min_seed_win_rate']:.1%} "
            f"loss={row['loss_rate']:.1%} dd={row['double_death_rate']:.1%} "
            f"max_dd={row['max_seed_double_death_rate']:.1%} "
            f"shots={row['shots_per_game']:.2f} hit={row['hit_rate']:.1%} "
            f"score={row['selection_score']:.4f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-net", default=(
        "training/models/p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--value-net", default=(
        "training/models/p29_p28_distill_iter00.pt"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--seeds", default="970000,990000,973000")
    parser.add_argument("--screen-n", type=int, default=40)
    parser.add_argument("--confirm-n", type=int, default=80)
    parser.add_argument("--medium-n", type=int, default=300)
    parser.add_argument("--confirm-top", type=int, default=2)
    parser.add_argument("--skip-medium", action="store_true")
    parser.add_argument("--report", default=(
        "training/analysis/runs/p29b_policy_sweep_report.json"))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",")]
    os.makedirs(RUN_DIR, exist_ok=True)

    report = {
        "base_net": args.base_net,
        "value_net": args.value_net,
        "seeds": seeds,
        "screen_n": args.screen_n,
        "confirm_n": args.confirm_n,
        "medium_n": args.medium_n,
        "started_at": time.time(),
        "screen": [],
        "confirm": [],
        "medium": [],
        "decision": "running",
    }

    for config in CONFIGS:
        summaries = [
            _run_eval(args, config, "screen", args.screen_n, seed)
            for seed in seeds
        ]
        report["screen"].append(_aggregate(config, summaries))
        _save_report(args.report, report)
    _print_table("P29b screen", report["screen"])

    qualified = [row for row in report["screen"] if _screen_qualified(row)]
    qualified.sort(key=lambda item: item["selection_score"], reverse=True)
    selected_names = [row["name"] for row in qualified[:args.confirm_top]]
    if not selected_names:
        report["decision"] = "reject_calibration_no_screen_candidate"
        report["finished_at"] = time.time()
        _save_report(args.report, report)
        print("No configuration passed the screen gate; stop before confirmation.", flush=True)
        return 0

    by_name = {config["name"]: config for config in CONFIGS}
    for name in selected_names:
        config = by_name[name]
        summaries = [
            _run_eval(args, config, "confirm", args.confirm_n, seed)
            for seed in seeds
        ]
        report["confirm"].append(_aggregate(config, summaries))
        _save_report(args.report, report)
    _print_table("P29b confirm", report["confirm"])

    confirmed = [row for row in report["confirm"] if _confirm_qualified(row)]
    confirmed.sort(key=lambda item: item["selection_score"], reverse=True)
    if not confirmed:
        report["decision"] = "reject_calibration_no_confirm_candidate"
        report["finished_at"] = time.time()
        _save_report(args.report, report)
        print("No configuration passed confirmation; stop before medium validation.", flush=True)
        return 0

    if args.skip_medium:
        report["decision"] = "confirm_passed_medium_skipped"
        report["recommended_config"] = confirmed[0]["name"]
        report["finished_at"] = time.time()
        _save_report(args.report, report)
        return 0

    best = by_name[confirmed[0]["name"]]
    summaries = [
        _run_eval(args, best, "medium", args.medium_n, seed)
        for seed in seeds
    ]
    medium = _aggregate(best, summaries)
    report["medium"].append(medium)
    report["recommended_config"] = best["name"]
    report["decision"] = (
        "medium_passed_ready_for_official"
        if _medium_qualified(medium)
        else "medium_failed_retrain_with_consistent_p28_labels"
    )
    report["finished_at"] = time.time()
    _save_report(args.report, report)
    _print_table("P29b medium", report["medium"])
    print(f"Final decision: {report['decision']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
