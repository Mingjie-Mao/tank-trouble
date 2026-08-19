"""Standing monitor over the live GUI supervision log.

``training/watch.py`` appends one row per round to
``training/analysis/live/watch_supervision.jsonl``.  This module turns that
stream into an attribution report: what failed, why, and which *wins* were
still played badly.

It is deliberately dependency-light (json + statistics only) so it can run
anywhere, including while a GUI session is still appending.

Rounds are de-duplicated by seed, keeping the newest row: re-running the same
seed range in a second GUI session would otherwise double-count it, which is
exactly what happened to the first 23 seeds of the 2026-08-03 session.

Usage
-----
    python3 training/watch_monitor.py
    python3 training/watch_monitor.py --since 2026-08-04 --top 15
    python3 training/watch_monitor.py --out training/analysis/live/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter

DEFAULT_LOG = "training/analysis/live/watch_supervision.jsonl"

# A win can still be a bad round.  These are the tells, in priority order.
BAD_WIN_RULES = (
    ("no_shot_win", "整局没开过枪，赢是因为 Laika 自己撞死",
     lambda row: not (row.get("shots") or 0)),
    ("excessive_search", "搜索率过高，画面会明显卡",
     lambda row: (row.get("search_frame_rate") or 0.0) > 0.25),
    ("very_long_round", "局太长，说明没能主动结束战斗",
     lambda row: (row.get("frames") or 0) > 600),
    ("no_active_kill", "赢了但没有主动击杀",
     lambda row: not (row.get("kills") or 0)),
)


CONFIG_KEYS = (
    "cfg_policy", "cfg_movement_continuity_epsilon",
    "cfg_temporal_intent_net", "cfg_top_k", "cfg_search_horizon",
    "cfg_temporal_confidence",
)


def run_key(row):
    """Identify which configuration produced a round.

    De-duplicating on seed alone is wrong once an A/B is running: the second
    session replays the same seeds and would overwrite the baseline it is
    supposed to be compared against.  Rounds logged before configs were
    stamped fall back to the policy name.
    """
    if any(key in row for key in CONFIG_KEYS):
        return tuple(str(row.get(key)) for key in CONFIG_KEYS)
    return ("legacy", str(row.get("policy")))


def config_label(key):
    if key[0] == "legacy":
        return f"legacy/{key[1]}"
    policy, epsilon = key[0], key[1]
    return f"{policy} eps={epsilon}"


def load_rounds(path=DEFAULT_LOG, since=None, config=None):
    """Read the log, split by configuration, de-duplicate seeds within each.

    Returns ``(raw_rows, rounds, runs)`` where ``runs`` maps a config key to
    its de-duplicated rounds.  ``rounds`` is the newest configuration unless
    ``config`` selects one explicitly.
    """
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # The GUI may be mid-write on the last line; skip it.
                continue
            if since and str(row.get("timestamp") or "") < str(since):
                continue
            rows.append(row)

    latest = {}
    newest_time = {}
    for row in rows:
        key = run_key(row)
        latest.setdefault(key, {})[row.get("seed")] = row
        stamp = str(row.get("timestamp") or "")
        if stamp > newest_time.get(key, ""):
            newest_time[key] = stamp

    runs = {
        key: sorted(by_seed.values(),
                    key=lambda row: (row.get("seed") is None,
                                     row.get("seed") or 0))
        for key, by_seed in latest.items()
    }
    if not runs:
        return rows, [], {}
    if config is not None:
        selected = next(
            (key for key in runs if config in config_label(key)), None)
    else:
        selected = max(runs, key=lambda key: newest_time.get(key, ""))
    return rows, runs.get(selected, []), runs


def events(row):
    return ((row.get("event_metrics") or {}).get("events") or {})


def action_frames(rows):
    return max(1, sum(
        int((row.get("event_metrics") or {}).get("action_frames") or 0)
        for row in rows))


def group_stats(rows):
    """Aggregate one group of rounds into comparable rates."""
    if not rows:
        return {"rounds": 0}
    frames = action_frames(rows)
    windows = sum(events(row).get("fire_window", 0) for row in rows)
    captured = sum(events(row).get("captured_fire_window", 0) for row in rows)

    def per_1000(key):
        return round(1000.0 * sum(
            events(row).get(key, 0) for row in rows) / frames, 1)

    return {
        "rounds": len(rows),
        "mean_frames": round(statistics.mean(
            [row.get("frames") or 0 for row in rows]), 1),
        "max_frames": max(row.get("frames") or 0 for row in rows),
        "mean_search_rate": round(statistics.mean(
            [row.get("search_frame_rate") or 0.0 for row in rows]), 4),
        "mean_shots": round(statistics.mean(
            [row.get("shots") or 0 for row in rows]), 2),
        "zero_shot_rounds": sum(
            1 for row in rows if not (row.get("shots") or 0)),
        "active_kill_rounds": sum(
            1 for row in rows if (row.get("kills") or 0) > 0),
        "fire_capture_rate": round(captured / windows, 3) if windows else 0.0,
        "missed_fire_per_1000": per_1000("missed_fire_window"),
        "dead_end_per_1000": per_1000("dead_end_stall"),
        "stutter_per_1000": per_1000("stutter_stall"),
        "reversal_per_1000": round(1000.0 * sum(
            events(row).get("throttle_reversal", 0)
            + events(row).get("turn_reversal", 0)
            for row in rows) / frames, 1),
    }


def classify_bad_win(row):
    """Return every bad-round tell that applies to a winning round."""
    return [name for name, _, predicate in BAD_WIN_RULES if predicate(row)]


def failure_rows(rounds):
    return [row for row in rounds if row.get("true_result") != "win"]


def bad_wins(rounds):
    out = []
    for row in rounds:
        if row.get("true_result") != "win":
            continue
        tells = classify_bad_win(row)
        if tells:
            out.append({
                "seed": row.get("seed"),
                "frames": row.get("frames"),
                "search_frame_rate": row.get("search_frame_rate"),
                "shots": row.get("shots") or 0,
                "kills": row.get("kills") or 0,
                "tells": tells,
                "issues": (row.get("diagnosis") or {}).get(
                    "issue_categories", []),
            })
    return out


def compare_groups(rounds):
    """Win vs non-win on every behaviour rate, with the ratio."""
    wins = [row for row in rounds if row.get("true_result") == "win"]
    losses = failure_rows(rounds)
    win_stats = group_stats(wins)
    loss_stats = group_stats(losses)
    ratios = {}
    for key, value in loss_stats.items():
        base = win_stats.get(key)
        if isinstance(value, (int, float)) and isinstance(base, (int, float)):
            if base:
                ratios[key] = round(value / base, 2)
    return {"win": win_stats, "nonwin": loss_stats, "nonwin_over_win": ratios}


def analyze(rounds):
    results = Counter(row.get("true_result") for row in rounds)
    failures = failure_rows(rounds)
    total = max(1, len(rounds))
    return {
        "rounds": len(rounds),
        "results": dict(results),
        "win_rate": round(results.get("win", 0) / total, 4),
        "death_causes": dict(Counter(
            f"{row.get('true_result')}/{row.get('death_cause')}"
            for row in failures)),
        "self_kill_failures": sum(
            1 for row in failures if row.get("death_cause") == "self"),
        # A truncated round is a stalemate the policy could not resolve.  It
        # used to hang the GUI forever and silently stop this log, so it is
        # tracked separately rather than lumped in with deaths.
        "truncated_rounds": sum(
            1 for row in rounds if row.get("true_result") == "truncated"),
        "zero_shot_failures": sum(
            1 for row in failures if not (row.get("shots") or 0)),
        "issue_rounds": dict(Counter(
            issue for row in rounds
            for issue in (row.get("diagnosis") or {}).get(
                "issue_categories", []))),
        "comparison": compare_groups(rounds),
        "failures": [{
            "seed": row.get("seed"),
            "result": row.get("true_result"),
            "death_cause": row.get("death_cause"),
            "frames": row.get("frames"),
            "search_frame_rate": row.get("search_frame_rate"),
            "shots": row.get("shots") or 0,
            "kills": row.get("kills") or 0,
            "issues": (row.get("diagnosis") or {}).get(
                "issue_categories", []),
        } for row in failures],
        "bad_wins": bad_wins(rounds),
    }


def format_report(report, top=12):
    lines = []
    lines.append(
        f"局数 {report['rounds']}  结果 {report['results']}  "
        f"胜率 {report['win_rate']:.1%}")
    lines.append("")
    lines.append(f"死因: {report['death_causes']}")
    lines.append(
        f"自杀失败 {report['self_kill_failures']} 局  |  "
        f"0 射击失败 {report['zero_shot_failures']} 局  |  "
        f"僵持截断 {report.get('truncated_rounds', 0)} 局")
    lines.append("")
    comparison = report["comparison"]
    header = f"{'指标':<24}{'胜局':>12}{'非胜局':>12}{'倍数':>8}"
    lines.append(header)
    lines.append("-" * 58)
    for key in ("mean_frames", "mean_search_rate", "mean_shots",
                "fire_capture_rate", "missed_fire_per_1000",
                "dead_end_per_1000", "stutter_per_1000",
                "reversal_per_1000"):
        win = comparison["win"].get(key, 0)
        non = comparison["nonwin"].get(key, 0)
        ratio = comparison["nonwin_over_win"].get(key, 0)
        lines.append(f"{key:<24}{win:>12}{non:>12}{ratio:>8}")
    lines.append("")
    lines.append(f"=== 失败局 (前 {top}) ===")
    for row in sorted(report["failures"],
                      key=lambda item: -(item["search_frame_rate"] or 0))[:top]:
        lines.append(
            f"  {row['seed']} {row['result']:<13} "
            f"death={str(row['death_cause']):<14} "
            f"frames={row['frames']:<5} "
            f"search={(row['search_frame_rate'] or 0):.1%} "
            f"shots={row['shots']} kills={row['kills']}")
    lines.append("")
    tells = Counter(tell for row in report["bad_wins"]
                    for tell in row["tells"])
    lines.append(f"=== 赢了但打得不好: {len(report['bad_wins'])} 局 ===")
    for name, description, _ in BAD_WIN_RULES:
        if tells.get(name):
            lines.append(f"  {name:<18}{tells[name]:>5} 局   {description}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--since", default=None,
                    help="只统计 timestamp >= 该前缀的行, 例如 2026-08-04")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--config", default=None,
                    help="只看匹配该子串的配置, 默认取最新的一次运行")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw, rounds, runs = load_rounds(args.log, args.since, args.config)
    if not rounds:
        print("没有可分析的回合")
        return
    if len(runs) > 1:
        print("日志里有多个配置 (--config 可指定):")
        for key, group in sorted(
                runs.items(), key=lambda item: -len(item[1])):
            wins = sum(1 for row in group
                       if row.get("true_result") == "win")
            print(f"  {config_label(key):<48}{len(group):>5} 局  "
                  f"胜率 {wins / max(1, len(group)):.1%}")
        print()
    report = analyze(rounds)
    report["raw_rows"] = len(raw)
    report["log"] = args.log
    report["configs"] = {config_label(key): len(group)
                         for key, group in runs.items()}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False,
                      sort_keys=True)
            handle.write("\n")
    print(format_report(report, args.top))


if __name__ == "__main__":
    main()
