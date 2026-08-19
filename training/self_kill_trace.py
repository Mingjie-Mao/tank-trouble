"""Reconstruct how a round killed itself with its own bullet.

The hypothesis, from 805 live GUI rounds (765W/31L/9DD):

* 13 of 40 non-win rounds died to their own bullet (``death_cause == "self"``),
  and every one of them had fired at least once.
* Failure rounds search 2.15x more than wins and stall in dead ends 1.79x more.

The suspected chain:

1. Fire.  ``rollout_exact_sequence`` validates the shot over ``fire_tail_horizon``
   (375) frames, but ``_sequence_action`` holds a *single* movement for that
   whole tail.  It answers "is this shot safe if I keep driving this way",
   which is almost never what happens -- the measured median action run is 4
   frames.
2. ``incoming_risk`` projects every bullet only ``RISK_HORIZON = 30`` frames
   ahead, so our own bullet is invisible for most of its flight.
3. It reappears ~25 frames before impact (risk crosses the 0.18 search
   threshold), a full search runs, and if we are in a corridor or dead end
   there is no allowed action left.

This module holds the *pure* reconstruction so the chain can be confirmed or
refuted from a replay instead of argued about.  Each link produces a separate
number, so a wrong link fails loudly:

``continuation_deviation_frames``
    how long the actual movement matched the continuation the long-tail check
    validated.  Near 0 means link 1 holds.
``blind_frames`` / ``blind_fraction``
    frames with our own bullet in flight but zero measured risk.  Large means
    link 2 holds.
``reaction_frames``
    frames between risk first crossing the threshold and death.  Small, or
    accompanied by a ``no_safe_event``, means link 3 holds.
"""

from __future__ import annotations

RISK_SEARCH_THRESHOLD = 0.18


def continuation_deviation(frames, fire_frame, validated_movement):
    """Frames the actual movement held the continuation that was validated.

    Returns ``None`` when the fire frame is not in the trace.  A value of 0
    means the very next frame already departed from the validated plan, so the
    long-tail guarantee never applied to the trajectory actually taken.
    """
    after = [row for row in frames if row["frame"] > fire_frame]
    if not after:
        return None
    for offset, row in enumerate(after):
        if row["movement"] != validated_movement:
            return offset
    return len(after)


def blind_window(frames):
    """Frames where our own bullet was airborne but measured risk was zero.

    ``incoming_risk`` truncates every bullet at RISK_HORIZON frames, so a
    long-lived own bullet contributes nothing until it is nearly on top of us.
    """
    airborne = [row for row in frames if row.get("own_bullets", 0) > 0]
    if not airborne:
        return {"airborne_frames": 0, "blind_frames": 0,
                "blind_fraction": 0.0}
    blind = sum(1 for row in airborne if not row.get("risk"))
    return {
        "airborne_frames": len(airborne),
        "blind_frames": blind,
        "blind_fraction": blind / len(airborne),
    }


def reaction_window(frames, death_frame, threshold=RISK_SEARCH_THRESHOLD):
    """Frames between risk first crossing ``threshold`` and death.

    Only the final unbroken run of elevated risk counts: an earlier scare that
    was successfully escaped is not the warning that mattered.
    """
    if death_frame is None:
        return None
    before = [row for row in frames if row["frame"] <= death_frame]
    if not before:
        return None
    start = None
    for row in reversed(before):
        if (row.get("risk") or 0.0) >= threshold:
            start = row["frame"]
        elif start is not None:
            break
    if start is None:
        return None
    return int(death_frame - start)


def fire_events(frames):
    """Every frame where we actually fired, with what was validated there."""
    return [{
        "frame": row["frame"],
        "validated_movement": row.get("validated_movement"),
        "long_tail_checked": bool(row.get("long_tail_checked")),
        "long_tail_rejected": bool(row.get("long_tail_rejected")),
    } for row in frames if row.get("fired")]


def summarize(frames, death_frame=None, death_cause=None,
              no_safe_events=(), threshold=RISK_SEARCH_THRESHOLD):
    """One round's self-kill reconstruction."""
    fires = fire_events(frames)
    deviations = []
    for event in fires:
        validated = event["validated_movement"]
        if validated is None:
            continue
        held = continuation_deviation(frames, event["frame"], validated)
        if held is not None:
            deviations.append({"fire_frame": event["frame"],
                               "held_frames": held})
    blind = blind_window(frames)
    late = [event for event in no_safe_events
            if death_frame is not None
            and death_frame - int(event.get("frame", -10 ** 9)) <= 60]
    return {
        "frames": len(frames),
        "death_frame": death_frame,
        "death_cause": death_cause,
        "fires": fires,
        "shots": len(fires),
        "continuation_deviation": deviations,
        "min_held_frames": (min(item["held_frames"] for item in deviations)
                            if deviations else None),
        "blind_window": blind,
        "reaction_frames": reaction_window(frames, death_frame, threshold),
        "no_safe_events_near_death": list(late),
        "max_risk": max((row.get("risk") or 0.0 for row in frames),
                        default=0.0),
    }


def verdict(summary, held_frames_limit=30, blind_fraction_limit=0.5,
            reaction_frames_limit=40):
    """Which links of the hypothesised chain this round actually supports."""
    links = {}
    held = summary.get("min_held_frames")
    links["continuation_broken"] = bool(
        held is not None and held < held_frames_limit)
    blind = summary.get("blind_window") or {}
    links["bullet_was_invisible"] = bool(
        blind.get("blind_fraction", 0.0) >= blind_fraction_limit)
    reaction = summary.get("reaction_frames")
    links["no_time_to_escape"] = bool(
        (reaction is not None and reaction <= reaction_frames_limit)
        or summary.get("no_safe_events_near_death"))
    links["all_links_hold"] = all(
        links[key] for key in (
            "continuation_broken", "bullet_was_invisible",
            "no_time_to_escape"))
    return links
