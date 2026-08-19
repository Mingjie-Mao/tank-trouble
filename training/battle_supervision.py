"""Uniform per-round supervision and actionable behavior diagnosis."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def _event_rate(events, name, frames):
    return 1000.0 * float(events.get(name, 0)) / max(1, int(frames))


def diagnose_battle(row):
    """Return evidence-backed issues and recommendations for one round."""
    frames = int(row.get("frames", row.get("policy_frames", 0)))
    result = row.get("true_result", row.get("result", "unknown"))
    event_metrics = row.get("event_metrics") or {}
    events = event_metrics.get("events") or row.get("behavior_events") or {}
    issues = []

    def add(category, severity, evidence, recommendation):
        issues.append({
            "category": category,
            "severity": severity,
            "evidence": evidence,
            "recommendation": recommendation,
        })

    if result == "double_death":
        add(
            "double_death_risk", "critical",
            {"result": result, "death_cause": row.get("death_cause")},
            "Replay this seed with long-tail fire counterfactuals and add the "
            "unsafe branch to the safety regression set.")
    elif result not in ("win", "unknown"):
        add(
            "terminal_loss", "critical",
            {"result": result, "death_cause": row.get("death_cause")},
            "Preserve the full pre-death window and relabel it with the frozen "
            "exact teacher before retraining.")

    checks = (
        ("missed_fire_window", 4.0, "fire_opportunity_gap", "high",
         "Train a calibrated fire-opportunity head; keep exact long-tail "
         "self-hit rejection as the final gate."),
        ("stutter_stall", 5.0, "movement_stutter", "medium",
         "Add this sequence to temporal intent training and penalize rapid "
         "turn/throttle reversals only when the teacher intent is unchanged."),
        ("dead_end_stall", 3.0, "dead_end_navigation", "high",
         "Upweight escape-dead-end sequences and retain the topology exit goal "
         "until completion or an explicit danger interrupt."),
        ("passive_map_control", 1.0, "passive_map_control", "medium",
         "Relabel with an active firing-position goal and measure progress "
         "toward Laika or a viable ricochet lane."),
    )
    for event, threshold, category, severity, recommendation in checks:
        rate = _event_rate(events, event, frames)
        if rate > threshold:
            add(category, severity, {
                "events": int(events.get(event, 0)),
                "per_1000_frames": rate,
                "threshold": threshold,
            }, recommendation)

    search_rate = float(row.get("search_frame_rate", 0.0))
    if search_rate > 0.20:
        add(
            "excessive_search_handoff", "medium",
            {"search_frame_rate": search_rate},
            "Inspect rejected temporal moves and distill only exact-safe "
            "corrections; do not lower safety thresholds to gain speed.")
    return {
        "result": result,
        "frames": frames,
        "issues": issues,
        "issue_categories": [issue["category"] for issue in issues],
        "needs_failure_replay": result != "win",
    }


def append_supervision(path, row):
    payload = dict(row)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    payload["diagnosis"] = diagnose_battle(payload)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload
