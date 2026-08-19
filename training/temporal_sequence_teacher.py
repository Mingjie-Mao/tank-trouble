"""Exact two-stage sequence teacher for smooth, safety-constrained intent.

The current exact teacher replans every frame and evaluates one movement held
for most of the horizon.  This experiment evaluates a short first movement
segment followed by a second segment, commits only while that intent remains
exactly safe, and performs a long-tail simulation for every firing sequence.

Safety is a hard constraint.  Attack value and movement smoothness only rank
sequences that remain alive and avoid double death.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluate import play_round_dual_engine  # noqa: E402
from training.exact_state import clone_exact_game, state_fingerprint  # noqa: E402
from training.exact_state_mpc_teacher import (  # noqa: E402
    ALL_CANDIDATE_INDICES,
    ExactStatePriorGuidedMPC,
    _set_candidate_controls,
    exact_root_search,
)
from training.mpc_agent import CANDIDATES  # noqa: E402
from training.map_topology_planner import (  # noqa: E402
    cardinal_neighbors,
    dead_end_depth,
    tank_cell,
)
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import build_observation  # noqa: E402
from training.p27_risk_value import _controls, _input_tuple  # noqa: E402


def _movement_only(index):
    return (int(index) // 2) * 2


def _result_from_winner(winner):
    if winner == 0:
        return "win"
    if winner == 1:
        return "loss"
    return "double_death"


def _sequence_action(first_index, second_index, frame, chunk_frames,
                     score_horizon):
    index = int(first_index if frame < chunk_frames else second_index)
    throttle, turn, fire = CANDIDATES[index]
    segment_start = frame == 0 or frame == chunk_frames
    if frame >= score_horizon or not segment_start:
        fire = 0
    return throttle, turn, fire


def rollout_exact_sequence(
        game,
        analyzer,
        start_metrics,
        first_index,
        second_index,
        *,
        chunk_frames=8,
        score_horizon=72,
        fire_tail_horizon=375):
    """Evaluate a two-stage action sequence without mutating the live game."""
    chunk_frames = max(1, min(int(chunk_frames), int(score_horizon) - 1))
    score_horizon = max(chunk_frames + 1, int(score_horizon))
    first_index, second_index = int(first_index), int(second_index)
    contains_fire = bool(
        CANDIDATES[first_index][2] or CANDIDATES[second_index][2])
    max_frames = max(
        score_horizon,
        int(fire_tail_horizon) if contains_fire else score_horizon,
    )

    sandbox = clone_exact_game(game)
    before = state_fingerprint(game)
    root_potential = analyzer.potential(start_metrics)
    me, enemy = sandbox.tanks[0], sandbox.tanks[1]
    me_dead_frame = None
    enemy_dead_frame = None
    true_result = None
    fired = False
    agent_hit = False
    score_at_horizon = None
    simulated_frames = 0
    previous_position = (me.x, me.y)
    movement_distance = 0.0
    enemy_previous_position = (enemy.x, enemy.y)
    enemy_movement_distance = 0.0
    enemy_cells = {tank_cell(sandbox, enemy)}
    enemy_min_exits = len(cardinal_neighbors(
        sandbox.maze, tank_cell(sandbox, enemy)))

    for frame in range(max_frames):
        action = _sequence_action(
            first_index, second_index, frame, chunk_frames, score_horizon)
        if not enemy.alive:
            action = (action[0], action[1], 0)
        _set_candidate_controls(sandbox, action)
        events = sandbox.step()
        simulated_frames += 1
        movement_distance += math.hypot(
            me.x - previous_position[0], me.y - previous_position[1])
        previous_position = (me.x, me.y)
        enemy_movement_distance += math.hypot(
            enemy.x - enemy_previous_position[0],
            enemy.y - enemy_previous_position[1],
        )
        enemy_previous_position = (enemy.x, enemy.y)
        enemy_cell = tank_cell(sandbox, enemy)
        enemy_cells.add(enemy_cell)
        enemy_min_exits = min(
            enemy_min_exits,
            len(cardinal_neighbors(sandbox.maze, enemy_cell)),
        )
        fired = fired or any(
            event[0] == "fire" and event[1] == 0 for event in events)
        agent_hit = agent_hit or any(
            event[0] == "hit" and event[1] == 0 and event[2] == 1
            for event in events)
        if me_dead_frame is None and not me.alive:
            me_dead_frame = frame
        if enemy_dead_frame is None and not enemy.alive:
            enemy_dead_frame = frame
        for event in events:
            if event[0] == "round_end":
                true_result = _result_from_winner(event[1])
        if frame + 1 == score_horizon and true_result is None:
            score_at_horizon = (
                analyzer.potential(analyzer.metrics(sandbox)) - root_potential)
        if true_result is not None:
            break

    if true_result == "win":
        score = 1000.0 - float(enemy_dead_frame or 0)
    elif true_result == "loss":
        score = -1000.0 + float(me_dead_frame or 0)
    elif true_result == "double_death":
        score = -900.0 + float(me_dead_frame or 0)
    elif me_dead_frame is not None:
        score = -1000.0 + float(me_dead_frame)
    elif enemy_dead_frame is not None:
        score = 1000.0 - float(enemy_dead_frame)
    else:
        score = float(score_at_horizon or 0.0)

    kill = enemy_dead_frame is not None
    death = me_dead_frame is not None
    double_death = bool(kill and death)
    movement_switches = int(
        CANDIDATES[first_index][:2] != CANDIDATES[second_index][:2])
    return {
        "first_index": first_index,
        "second_index": second_index,
        "first_action": CANDIDATES[first_index],
        "second_action": CANDIDATES[second_index],
        "score": float(score),
        "value": float(score),
        "kill": bool(kill),
        "agent_kill": bool(kill and agent_hit),
        "unassisted_enemy_death": bool(kill and not agent_hit),
        "death": bool(death),
        "double_death": bool(double_death),
        "allowed": bool(not death and not double_death),
        "true_result": true_result,
        "kill_frame": enemy_dead_frame,
        "death_frame": me_dead_frame,
        "fired": bool(fired),
        "tail_checked": bool(contains_fire),
        "simulated_frames": int(simulated_frames),
        "movement_switches": movement_switches,
        "movement_cells": float(movement_distance / max(1.0, game.scale)),
        "enemy_movement_cells": float(
            enemy_movement_distance / max(1.0, game.scale)),
        "enemy_unique_cells": int(len(enemy_cells)),
        "enemy_end_dead_end_depth": float(
            dead_end_depth(sandbox, tank_cell(sandbox, enemy))),
        "enemy_min_exits": int(enemy_min_exits),
        "live_fingerprint_unchanged": before == state_fingerprint(game),
    }


def choose_smooth_safe_sequence(rows, value_tolerance=1.0):
    """Apply lexicographic safety/value/attack/smoothness selection."""
    allowed = [row for row in rows if row["allowed"]]
    if not allowed:
        return None
    best_value = max(float(row["value"]) for row in allowed)
    near_best = [
        row for row in allowed
        if float(row["value"]) >= best_value - float(value_tolerance)
    ]
    return max(
        near_best,
        key=lambda row: (
            int(bool(row.get("agent_kill", False))),
            int(bool(row["kill"])),
            -int(row["kill_frame"] if row["kill_frame"] is not None else 10**9),
            float(row.get("movement_cells", 0.0)),
            -int(row["movement_switches"]),
            float(row["value"]),
        ),
    )


def _advance_root(game, action_index, chunk_frames):
    successor = clone_exact_game(game)
    for frame in range(int(chunk_frames)):
        throttle, turn, fire = CANDIDATES[int(action_index)]
        _set_candidate_controls(
            successor, (throttle, turn, fire if frame == 0 else 0))
        successor.step()
        if not successor.tanks[0].alive:
            break
    return successor


def exact_two_stage_search(
        game,
        analyzer,
        metrics,
        root_indices,
        *,
        chunk_frames=8,
        score_horizon=72,
        root_beam=4,
        continuation_beam=3,
        fire_tail_horizon=375,
        value_tolerance=1.0):
    """Build a small exact beam, then validate complete sequence candidates."""
    _, root_rows = exact_root_search(
        game, analyzer, metrics, root_indices,
        horizon=score_horizon, max_death=0.0, max_dd=0.0)
    safe_roots = sorted(
        (row for row in root_rows if row["allowed"]),
        key=lambda row: row["value"], reverse=True)[:max(1, int(root_beam))]
    rows = []
    for root in safe_roots:
        first_index = int(root["index"])
        successor = _advance_root(game, first_index, chunk_frames)
        if not successor.tanks[0].alive:
            continue
        successor_metrics = analyzer.metrics(successor)
        _, continuation_rows = exact_root_search(
            successor,
            analyzer,
            successor_metrics,
            ALL_CANDIDATE_INDICES,
            horizon=max(8, int(score_horizon) - int(chunk_frames)),
            max_death=0.0,
            max_dd=0.0,
        )
        safe_continuations = sorted(
            (row for row in continuation_rows if row["allowed"]),
            key=lambda row: row["value"], reverse=True,
        )[:max(1, int(continuation_beam))]
        if not safe_continuations:
            safe_continuations = [{"index": _movement_only(first_index)}]
        for continuation in safe_continuations:
            rows.append(rollout_exact_sequence(
                game,
                analyzer,
                metrics,
                first_index,
                int(continuation["index"]),
                chunk_frames=chunk_frames,
                score_horizon=score_horizon,
                fire_tail_horizon=fire_tail_horizon,
            ))
    return choose_smooth_safe_sequence(rows, value_tolerance), rows


class ExactTemporalSequencePolicy(ExactStatePriorGuidedMPC):
    name = "exact_temporal_sequence_teacher"

    def __init__(self, *args, intent_chunk_frames=8, root_beam=4,
                 continuation_beam=3, fire_tail_horizon=375,
                 value_tolerance=1.0, commit_safety_horizon=48, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent_chunk_frames = int(intent_chunk_frames)
        self.root_beam = int(root_beam)
        self.continuation_beam = int(continuation_beam)
        self.fire_tail_horizon = int(fire_tail_horizon)
        self.value_tolerance = float(value_tolerance)
        self.commit_safety_horizon = int(commit_safety_horizon)

    def reset(self):
        super().reset()
        self.committed_index = None
        self.commit_remaining = 0
        self.sequence_searches = 0
        self.sequence_candidates = 0
        self.intent_frames = 0
        self.intent_interrupts = 0
        self.last_sequence_decision = None

    def _search(self, game, metrics, indices):
        selected, rows = exact_two_stage_search(
            game,
            self.analyzer,
            metrics,
            indices,
            chunk_frames=self.intent_chunk_frames,
            score_horizon=self.search_horizon,
            root_beam=self.root_beam,
            continuation_beam=self.continuation_beam,
            fire_tail_horizon=self.fire_tail_horizon,
            value_tolerance=self.value_tolerance,
        )
        self.sequence_searches += 1
        self.sequence_candidates += len(rows)
        if selected is None:
            return super()._search(game, metrics, indices)
        first_index = int(selected["first_index"])
        self.committed_index = _movement_only(first_index)
        self.commit_remaining = max(0, self.intent_chunk_frames - 1)
        self.last_sequence_decision = dict(selected)
        return CANDIDATES[first_index]

    def _committed_action(self, game):
        if self.committed_index is None or self.commit_remaining <= 0:
            return None
        metrics = self.analyzer.metrics(game)
        _, rows = exact_root_search(
            game,
            self.analyzer,
            metrics,
            (self.committed_index,),
            horizon=self.commit_safety_horizon,
            max_death=0.0,
            max_dd=0.0,
        )
        if not rows or not rows[0]["allowed"]:
            self.intent_interrupts += 1
            self.committed_index = None
            self.commit_remaining = 0
            return None

        action = CANDIDATES[self.committed_index]
        observation, _ = build_observation(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        self.history.append(observation)
        self.pos_window.append((game.tanks[0].x, game.tanks[0].y))
        self.input_window.append(_input_tuple(_controls(action)))
        self._update_context(game, metrics)
        self.commit_remaining -= 1
        self.intent_frames += 1
        if self.commit_remaining <= 0:
            self.committed_index = None
        throttle, turn, _ = action
        return {
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": False,
        }

    def act(self, game):
        if game is self.game and game.tanks[0].alive:
            committed = self._committed_action(game)
            if committed is not None:
                return committed
        return super().act(game)


def _make_policy(args):
    return ExactTemporalSequencePolicy(
        base_net=args.base_net,
        value_net=args.value_net,
        fire_margin=args.fire_margin,
        top_k=args.top_k,
        search_horizon=args.search_horizon,
        search_samples=1,
        search_death_penalty=0.18,
        search_dd_penalty=0.45,
        search_kill_bonus=0.05,
        search_max_death=0.0,
        search_max_dd=0.0,
        successor_shield=True,
        successor_horizon=args.search_horizon,
        successor_shield_max_safe_roots=2,
        suppress_secured_fire=True,
        min_unsecured_fire_gain=2.0,
        intent_chunk_frames=args.intent_chunk_frames,
        root_beam=args.root_beam,
        continuation_beam=args.continuation_beam,
        fire_tail_horizon=args.fire_tail_horizon,
        value_tolerance=args.value_tolerance,
        commit_safety_horizon=args.commit_safety_horizon,
    )


def _run_seed(job):
    seed, args = job
    import torch

    torch.set_num_threads(1)
    policy = _make_policy(args)
    policy.set_round_seed(seed)
    started = time.time()
    result = play_round_dual_engine(policy, seed)
    result.update({
        "seed": seed,
        "elapsed_seconds": time.time() - started,
        "sequence_searches": policy.sequence_searches,
        "sequence_candidates": policy.sequence_candidates,
        "intent_frames": policy.intent_frames,
        "intent_interrupts": policy.intent_interrupts,
    })
    return result


def _parse_seed_list(value):
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, count = (int(part) for part in item.split(":", 1))
            seeds.extend(range(start, start + count))
        else:
            seeds.append(int(item))
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-net", default=(
        "training/models/p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--value-net", default=(
        "training/models/p27b_risk_value_iter00.pt"))
    parser.add_argument("--seed-list", default=(
        "975062,981086,991011,983043,983064,993007,993042"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--search-horizon", type=int, default=72)
    parser.add_argument("--intent-chunk-frames", type=int, default=8)
    parser.add_argument("--root-beam", type=int, default=4)
    parser.add_argument("--continuation-beam", type=int, default=3)
    parser.add_argument("--fire-tail-horizon", type=int, default=375)
    parser.add_argument("--value-tolerance", type=float, default=1.0)
    parser.add_argument("--commit-safety-horizon", type=int, default=48)
    parser.add_argument("--out", default=(
        "training/analysis/runs/temporal_sequence_teacher_smoke.json"))
    args = parser.parse_args()

    seeds = _parse_seed_list(args.seed_list)
    started = time.time()
    if max(1, min(args.workers, len(seeds))) == 1:
        rounds = [_run_seed((seed, args)) for seed in seeds]
    else:
        with mp.get_context("spawn").Pool(
                min(args.workers, len(seeds))) as pool:
            rounds = pool.map(_run_seed, [(seed, args) for seed in seeds])
    rounds.sort(key=lambda row: row["seed"])
    results = Counter(row["true_result"] for row in rounds)
    report = {
        "method": "exact_temporal_sequence_teacher",
        "configuration": vars(args),
        "games": len(rounds),
        "results": dict(results),
        "win_rate": results["win"] / max(1, len(rounds)),
        "elapsed_seconds": time.time() - started,
        "sequence_searches": sum(row["sequence_searches"] for row in rounds),
        "sequence_candidates": sum(
            row["sequence_candidates"] for row in rounds),
        "intent_frames": sum(row["intent_frames"] for row in rounds),
        "intent_interrupts": sum(row["intent_interrupts"] for row in rounds),
        "rounds": rounds,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
