"""Counterfactual labels for safe ricochet pressure and route denial."""

from __future__ import annotations

from training.mpc_agent import CANDIDATES
from training.temporal_sequence_teacher import rollout_exact_sequence


def space_control_score(no_fire, fire):
    if not fire["allowed"] or not fire["fired"]:
        return 0.0
    if fire.get("agent_kill", False):
        return 1.0
    cell_reduction = max(
        0, int(no_fire["enemy_unique_cells"])
        - int(fire["enemy_unique_cells"]))
    exit_reduction = max(
        0, int(no_fire["enemy_min_exits"])
        - int(fire["enemy_min_exits"]))
    dead_end_gain = max(
        0.0, float(fire["enemy_end_dead_end_depth"])
        - float(no_fire["enemy_end_dead_end_depth"]))
    return min(
        1.0,
        0.20 * cell_reduction
        + 0.12 * exit_reduction
        + 0.10 * dead_end_gain,
    )


def evaluate_space_control_pair(
        game, analyzer, metrics, movement_index, *,
        score_horizon=96, fire_tail_horizon=375,
        positive_threshold=0.20):
    movement_id = int(movement_index)
    if not 0 <= movement_id < 9:
        raise ValueError(f"movement_index must be in [0, 8], got {movement_id}")
    movement_index = movement_id * 2
    fire_index = movement_index + 1
    if CANDIDATES[fire_index][2] != 1:
        raise ValueError("paired fire action is missing")
    no_fire = rollout_exact_sequence(
        game,
        analyzer,
        metrics,
        movement_index,
        movement_index,
        chunk_frames=1,
        score_horizon=score_horizon,
        fire_tail_horizon=fire_tail_horizon,
    )
    fire = rollout_exact_sequence(
        game,
        analyzer,
        metrics,
        fire_index,
        movement_index,
        chunk_frames=1,
        score_horizon=score_horizon,
        fire_tail_horizon=fire_tail_horizon,
    )
    score = space_control_score(no_fire, fire)
    return {
        "movement_index": movement_id,
        "no_fire": no_fire,
        "fire": fire,
        "space_control_score": float(score),
        "positive": bool(
            fire["allowed"] and fire["fired"]
            and score >= float(positive_threshold)),
        "cell_reduction": max(
            0, no_fire["enemy_unique_cells"] - fire["enemy_unique_cells"]),
        "exit_reduction": max(
            0, no_fire["enemy_min_exits"] - fire["enemy_min_exits"]),
        "dead_end_gain": max(
            0.0,
            fire["enemy_end_dead_end_depth"]
            - no_fire["enemy_end_dead_end_depth"],
        ),
        "self_tail_safe": bool(fire["allowed"]),
    }
