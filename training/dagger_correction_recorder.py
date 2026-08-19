"""DAgger correction bookkeeping for the temporal (GRU) movement policy.

Motivation
----------
``training/temporal_intent_pipeline.py`` collects states by running
``_make_teacher()``, which does **not** load a temporal intent net.  Every
state in ``topology_temporal_*.npz`` therefore comes from the P27b + topology
distribution, while the deployed hybrid runs with the GRU in the loop.  The
GRU is trained on states it never visits, proposes moves the exact safety
teacher has to reject, and the safety layer compensates with searches.  That
is the mechanism behind the 23-40% search rates, not the top-k budget.

This module holds the *pure* part of the fix so it can be tested without
torch: given the decision trace of one frame, decide

* whether the executed action differs from what the learned stack proposed,
* which correction channel is responsible,
* how the state should be weighted when it is distilled back.

The policy fills :func:`build_correction_record` from ``act()``; the dataset
builder in ``training/dagger_distill.py`` consumes it.

Labels are always the **executed** (exact-teacher approved) movement.  The
temporal net never owns firing; fire stays behind the exact long-tail check.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from training.temporal_intent_model import movement_run_targets


# --- correction channels ------------------------------------------------
# Ordered by priority: the first matching tag wins, so a frame that is both a
# shield override and a hard-case event is booked as the shield override.
TAG_TERMINAL_WINDOW = "terminal_window"
TAG_SUCCESSOR_SHIELD_OVERRIDE = "successor_shield_override"
TAG_FIRE_REJECTED = "fire_rejected"
TAG_UNSAFE_TEMPORAL = "unsafe_temporal_movement"
TAG_SEARCH_OVERRIDE = "search_override"
TAG_TOPOLOGY_ABORT = "topology_abort"
TAG_HARDCASE = "hardcase_event"
TAG_ACCEPTED = "accepted"

CORRECTION_TAGS = (
    TAG_TERMINAL_WINDOW,
    TAG_SUCCESSOR_SHIELD_OVERRIDE,
    TAG_FIRE_REJECTED,
    TAG_UNSAFE_TEMPORAL,
    TAG_SEARCH_OVERRIDE,
    TAG_TOPOLOGY_ABORT,
    TAG_HARDCASE,
    TAG_ACCEPTED,
)

# Tags that mean "the learned stack proposed something the exact teacher did
# not accept".  These are the states the distribution mismatch lives in.
DISAGREEMENT_TAGS = frozenset((
    TAG_TERMINAL_WINDOW,
    TAG_SUCCESSOR_SHIELD_OVERRIDE,
    TAG_FIRE_REJECTED,
    TAG_UNSAFE_TEMPORAL,
    TAG_SEARCH_OVERRIDE,
    TAG_TOPOLOGY_ABORT,
))

HARDCASE_CATEGORIES = frozenset((
    "blind_fire",
    "missed_fire_window",
    "dead_end_stall",
    "stutter_stall",
    "passive_map_control",
))

# Reasons that mean a full exact search replaced the proposal.
SEARCH_REASONS = frozenset((
    "unsafe", "followup", "critical", "narrow", "behavior", "fire", "risk",
))

# Weights are deliberately modest.  Aggressive upweighting of corrections
# collapses the movement head onto the safety teacher's conservative moves and
# reproduces the stutter we are trying to remove, so accepted states stay in
# the mix at weight 1.0 and act as the continuity prior.
DEFAULT_TAG_WEIGHTS = {
    TAG_TERMINAL_WINDOW: 4.0,
    TAG_SUCCESSOR_SHIELD_OVERRIDE: 3.0,
    TAG_FIRE_REJECTED: 2.0,
    TAG_UNSAFE_TEMPORAL: 3.0,
    TAG_SEARCH_OVERRIDE: 2.0,
    TAG_TOPOLOGY_ABORT: 1.5,
    TAG_HARDCASE: 1.5,
    TAG_ACCEPTED: 1.0,
}

# Frames before a loss / double death that are preserved as a failure window.
TERMINAL_WINDOW_FRAMES = 90


def build_correction_record(*, frame, features, executed_movement,
                            executed_fire, network_movement,
                            temporal_movement, temporal_confidence,
                            topology_movement, proposed_movement,
                            proposed_fire, reason, category, full_search,
                            audit_failed, safe_root_count,
                            successor_shield_triggered, interventions,
                            long_tail_fire_rejected, risk,
                            topology_active, topology_kind, topology_started,
                            topology_aborted):
    """Normalise one frame of the hybrid decision trace.

    ``*_movement`` values are movement indices in ``[0, 9)`` or ``None`` when
    that stage did not run.  ``executed_movement`` is the ground-truth label.
    """
    record = {
        "frame": int(frame),
        "features": np.asarray(features, dtype=np.float32),
        "movement": int(executed_movement),
        "executed_fire": bool(executed_fire),
        "network_movement": _opt_int(network_movement),
        "temporal_movement": _opt_int(temporal_movement),
        "temporal_confidence": float(temporal_confidence or 0.0),
        "topology_movement": _opt_int(topology_movement),
        "proposed_movement": _opt_int(proposed_movement),
        "proposed_fire": bool(proposed_fire),
        "reason": str(reason or "network"),
        "category": str(category or "standard"),
        "full_search": bool(full_search),
        "audit_failed": bool(audit_failed),
        "safe_root_count": int(safe_root_count),
        "successor_shield_triggered": bool(successor_shield_triggered),
        "interventions": tuple(str(item) for item in (interventions or ())),
        "long_tail_fire_rejected": bool(long_tail_fire_rejected),
        "risk": float(risk),
        "topology_active": bool(topology_active),
        "topology_kind": str(topology_kind or "none"),
        "topology_started": bool(topology_started),
        "topology_aborted": bool(topology_aborted),
    }
    record["temporal_used"] = record["temporal_movement"] is not None
    record["movement_corrected"] = bool(
        record["proposed_movement"] is not None
        and record["proposed_movement"] != record["movement"])
    record["fire_corrected"] = bool(
        record["proposed_fire"] != record["executed_fire"])
    # Did the exact layer specifically reject what the GRU asked for?
    record["temporal_corrected"] = bool(
        record["temporal_used"]
        and record["temporal_movement"] != record["movement"])
    return record


def _opt_int(value):
    return None if value is None else int(value)


def classify_correction(record, terminal_window=False):
    """Return the correction channel this state belongs to."""
    if terminal_window:
        return TAG_TERMINAL_WINDOW
    if TAG_SUCCESSOR_SHIELD_OVERRIDE in record["interventions"]:
        return TAG_SUCCESSOR_SHIELD_OVERRIDE
    if record["long_tail_fire_rejected"] or record["fire_corrected"]:
        return TAG_FIRE_REJECTED
    if record["temporal_corrected"]:
        # The GRU proposed a move the exact teacher would not execute.  This
        # is the channel the DAgger round is aimed at.
        return TAG_UNSAFE_TEMPORAL
    if record["movement_corrected"] and (
            record["full_search"] or record["reason"] in SEARCH_REASONS):
        return TAG_SEARCH_OVERRIDE
    if record["topology_aborted"]:
        return TAG_TOPOLOGY_ABORT
    if record["category"] in HARDCASE_CATEGORIES:
        return TAG_HARDCASE
    return TAG_ACCEPTED


def correction_weight(tag, record, tag_weights=None):
    """Weight for one state, combining channel and local difficulty."""
    weights = DEFAULT_TAG_WEIGHTS if tag_weights is None else tag_weights
    weight = float(weights.get(tag, 1.0))
    # Narrow safe-root situations are where a wrong move actually kills.
    safe_roots = record["safe_root_count"]
    if 0 < safe_roots <= 2:
        weight *= 1.5
    elif 0 < safe_roots <= 4:
        weight *= 1.2
    if record["topology_active"]:
        weight *= 1.2
    return float(weight)


def tag_round(records, result, terminal_window=TERMINAL_WINDOW_FRAMES):
    """Tag every frame of one round; returns ``(tags, weights)``.

    Non-win rounds keep their full pre-death window at ``terminal_window``
    priority so failures are never silently dropped, which is what the current
    pipeline does when it stores ``rows if true_result == "win"``.
    """
    records = list(records)
    count = len(records)
    terminal_start = count
    if result != "win" and count:
        terminal_start = max(0, count - int(terminal_window))
    tags = []
    weights = []
    for index, record in enumerate(records):
        tag = classify_correction(record, terminal_window=index >= terminal_start)
        tags.append(tag)
        weights.append(correction_weight(tag, record))
    return tags, weights


def disagreement_rate(tags):
    """Fraction of states where the learned stack was overruled."""
    tags = list(tags)
    if not tags:
        return 0.0
    hits = sum(1 for tag in tags if tag in DISAGREEMENT_TAGS)
    return hits / len(tags)


def summarize_tags(tags):
    counts = Counter(tags)
    total = max(1, len(list(tags)) if not isinstance(tags, list) else len(tags))
    return {
        "counts": dict(counts),
        "total": total,
        "disagreement_rate": disagreement_rate(tags),
    }


def sequence_targets(movements):
    """Hold/interrupt targets for one round (mirrors the v1 pipeline)."""
    movements = np.asarray(movements, dtype=np.int64)
    remaining, hold = movement_run_targets(movements)
    interrupt = np.zeros(len(movements), dtype=np.float32)
    if len(movements) > 1:
        interrupt[:-1] = movements[:-1] != movements[1:]
    return remaining, hold, interrupt


DATASET_KEYS = (
    "features", "movement", "hold", "interrupt", "progress", "weight",
    "round_seed", "frame", "risk", "category", "reason", "tag",
    "topology_active", "topology_kind", "full_search",
    "search_needed", "search_needed_mask",
    "movement_corrected", "temporal_corrected", "temporal_used",
    "safe_root_count", "round_result_win",
)

_DATASET_DTYPES = {
    "features": np.float32,
    "movement": np.int64,
    "hold": np.int64,
    "interrupt": np.float32,
    "progress": np.float32,
    "weight": np.float32,
    "round_seed": np.int64,
    "frame": np.int32,
    "risk": np.float32,
    "category": "U32",
    "reason": "U16",
    "tag": "U32",
    "topology_active": np.bool_,
    "topology_kind": "U32",
    "full_search": np.bool_,
    "search_needed": np.float32,
    "search_needed_mask": np.bool_,
    "movement_corrected": np.bool_,
    "temporal_corrected": np.bool_,
    "temporal_used": np.bool_,
    "safe_root_count": np.int32,
    "round_result_win": np.bool_,
}


def build_dataset(rounds, terminal_window=TERMINAL_WINDOW_FRAMES,
                  keep_accepted_fraction=1.0, rng_seed=2701):
    """Flatten tagged rounds into the npz payload the trainer consumes.

    The payload is a superset of ``temporal_intent_pipeline``'s format, so
    ``temporal_intent_pipeline.train`` reads it unchanged; the extra columns
    exist for slicing and for the post-run comparison.

    WARNING: ``keep_accepted_fraction < 1.0`` punches holes in the frame
    sequence of a round.  The trainer groups states by ``round_seed`` and feeds
    them to a GRU as one contiguous sequence, so subsampling silently teaches
    the recurrent state to skip time.  Leave it at 1.0 unless the run is purely
    a feed-forward ablation.
    """
    rng = np.random.default_rng(rng_seed)
    arrays = {key: [] for key in DATASET_KEYS}
    round_rows = []
    for round_row in rounds:
        records = round_row["rows"]
        if not records:
            continue
        tags, weights = tag_round(
            records, round_row["result"], terminal_window=terminal_window)
        round_rows.append(round_report(
            round_row["seed"], round_row["result"], round_row["frames"],
            records, tags, weights))
        movement = np.asarray([row["movement"] for row in records],
                              dtype=np.int64)
        _, hold, interrupt = sequence_targets(movement)
        is_win = round_row["result"] == "win"
        for index, row in enumerate(records):
            tag = tags[index]
            if (keep_accepted_fraction < 1.0
                    and tag not in DISAGREEMENT_TAGS
                    and rng.random() > keep_accepted_fraction):
                continue
            arrays["features"].append(row["features"])
            arrays["movement"].append(row["movement"])
            arrays["hold"].append(hold[index])
            arrays["interrupt"].append(interrupt[index])
            arrays["progress"].append(row.get("progress", 0.0))
            arrays["weight"].append(weights[index])
            arrays["round_seed"].append(round_row["seed"])
            arrays["frame"].append(row["frame"])
            arrays["risk"].append(row["risk"])
            arrays["category"].append(row["category"])
            arrays["reason"].append(row["reason"])
            arrays["tag"].append(tag)
            arrays["topology_active"].append(row["topology_active"])
            arrays["topology_kind"].append(row["topology_kind"])
            arrays["full_search"].append(row["full_search"])
            arrays["search_needed"].append(
                row.get("search_needed_target", False))
            arrays["search_needed_mask"].append(
                row.get("search_needed_mask", False))
            arrays["movement_corrected"].append(row["movement_corrected"])
            arrays["temporal_corrected"].append(row["temporal_corrected"])
            arrays["temporal_used"].append(row["temporal_used"])
            arrays["safe_root_count"].append(row["safe_root_count"])
            arrays["round_result_win"].append(is_win)

    if not arrays["features"]:
        raise RuntimeError("dagger collection produced no states")
    payload = {key: np.asarray(arrays[key], dtype=_DATASET_DTYPES[key])
               for key in DATASET_KEYS}
    return payload, round_rows


def round_report(seed, result, frames, records, tags, weights):
    """Per-round supervision row for the DAgger collection log."""
    reasons = Counter(record["reason"] for record in records)
    categories = Counter(record["category"] for record in records)
    corrected = sum(1 for record in records if record["movement_corrected"])
    temporal_used = sum(1 for record in records if record["temporal_used"])
    temporal_corrected = sum(
        1 for record in records if record["temporal_corrected"])
    return {
        "seed": int(seed),
        "result": str(result),
        "frames": int(frames),
        "states": len(records),
        "tags": dict(Counter(tags)),
        "disagreement_rate": disagreement_rate(tags),
        "movement_correction_rate": corrected / max(1, len(records)),
        "temporal_frames": temporal_used,
        "temporal_correction_rate": (
            temporal_corrected / max(1, temporal_used)),
        "full_search_frames": sum(
            1 for record in records if record["full_search"]),
        "mean_weight": float(np.mean(weights)) if weights else 0.0,
        "reasons": dict(reasons),
        "categories": dict(categories),
    }
