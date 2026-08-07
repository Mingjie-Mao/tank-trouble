"""
P27 macro-action probe.

This is an offline diagnostic for the next P26/P27 direction. It runs the
current deployed P26 policy against Laika, captures human-observed hard states,
and evaluates short macro actions such as fan-fire bursts and escape moves in
the sandbox teacher. It does not change the deployment policy.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import Game  # noqa: E402
from training.evaluate import RoundTracker  # noqa: E402
from training.mpc_agent import CANDIDATES, make_sandbox  # noqa: E402
from training.opportunity_distill import SCORE_SCALE, _shot_event  # noqa: E402
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    P26Policy,
    label_actions,
    stack_observation,
)
from training.p26_behavior_observer import (  # noqa: E402
    _cell,
    _dead_end_penalty,
    _input_tuple,
    _open_neighbors,
    _round_result,
)
from training.tt_gym_env import TRUNCATE_FRAMES  # noqa: E402

MACRO_NAMES = (
    "hold_fire_reposition",
    "single_fire_policy",
    "fan_left_3",
    "fan_right_3",
    "fan_center_3",
    "escape_back_left",
    "escape_back_right",
    "escape_forward_left",
    "escape_forward_right",
    "escape_forward",
)


def _action_tuple(inp):
    return (
        2 if inp.get("forward") else 0 if inp.get("backup") else 1,
        0 if inp.get("turn_left") else 2 if inp.get("turn_right") else 1,
        1 if inp.get("fire") else 0,
    )


def _controls(action):
    throttle, turn, fire = action
    return {
        "forward": throttle == 2,
        "backup": throttle == 0,
        "turn_left": turn == 0,
        "turn_right": turn == 2,
        "fire": fire == 1,
    }


def _macro_specs(policy_action):
    throttle, turn, _ = policy_action
    return [
        {
            "name": "hold_fire_reposition",
            "kind": "hold",
            "sequence": [(throttle, turn, 0)],
            "tail": (throttle, turn, 0),
        },
        {
            "name": "single_fire_policy",
            "kind": "single_fire",
            "sequence": [(throttle, turn, 1), (throttle, turn, 0)],
            "tail": (throttle, turn, 0),
        },
        {
            "name": "fan_left_3",
            "kind": "fan_fire",
            "sequence": [
                (throttle, 0, 1), (throttle, 0, 0),
                (throttle, 0, 1), (throttle, 0, 0),
                (throttle, 0, 1), (throttle, 1, 0),
            ],
            "tail": (throttle, 1, 0),
        },
        {
            "name": "fan_right_3",
            "kind": "fan_fire",
            "sequence": [
                (throttle, 2, 1), (throttle, 2, 0),
                (throttle, 2, 1), (throttle, 2, 0),
                (throttle, 2, 1), (throttle, 1, 0),
            ],
            "tail": (throttle, 1, 0),
        },
        {
            "name": "fan_center_3",
            "kind": "fan_fire",
            "sequence": [
                (throttle, 1, 1), (throttle, 1, 0),
                (throttle, 0, 1), (throttle, 0, 0),
                (throttle, 2, 1), (throttle, 1, 0),
            ],
            "tail": (throttle, 1, 0),
        },
        {
            "name": "escape_back_left",
            "kind": "escape",
            "sequence": [(0, 0, 0)],
            "tail": (0, 0, 0),
        },
        {
            "name": "escape_back_right",
            "kind": "escape",
            "sequence": [(0, 2, 0)],
            "tail": (0, 2, 0),
        },
        {
            "name": "escape_forward_left",
            "kind": "escape",
            "sequence": [(2, 0, 0)],
            "tail": (2, 0, 0),
        },
        {
            "name": "escape_forward_right",
            "kind": "escape",
            "sequence": [(2, 2, 0)],
            "tail": (2, 2, 0),
        },
        {
            "name": "escape_forward",
            "kind": "escape",
            "sequence": [(2, 1, 0)],
            "tail": (2, 1, 0),
        },
    ]


def _apply_action(tank, action):
    controls = _controls(action)
    tank.forward = controls["forward"]
    tank.backup = controls["backup"]
    tank.turn_left = controls["turn_left"]
    tank.turn_right = controls["turn_right"]
    tank.fire = controls["fire"]


def macro_rollout(sandbox, macro, analyzer, start_metrics, horizon, hold):
    me, enemy = sandbox.tanks[0], sandbox.tanks[1]
    shots = 0
    kill_frame = None
    death_frame = None
    round_result = None
    sequence = macro["sequence"]
    tail = macro["tail"]

    for frame in range(horizon):
        if frame < len(sequence):
            action = sequence[frame]
        elif frame < hold:
            action = tail
        else:
            action = (tail[0], tail[1], 0)
        _apply_action(me, action)
        events = sandbox.step()
        for event in events:
            if event[0] == "fire" and event[1] == 0:
                shots += 1
            elif event[0] == "destroy":
                if event[1] == 0 and death_frame is None:
                    death_frame = frame
                elif event[1] == 1 and kill_frame is None:
                    kill_frame = frame
            elif event[0] == "round_end":
                round_result = _round_result(event[1])

        if not me.alive:
            score = -1000.0 + frame
            if kill_frame is not None:
                score -= 100.0
            return score, {
                "kill": kill_frame is not None,
                "death": True,
                "double_death": kill_frame is not None,
                "shots": shots,
                "frames": frame + 1,
                "round_result": round_result,
            }
        if not enemy.alive and frame >= hold:
            return 1000.0 - frame, {
                "kill": True,
                "death": False,
                "double_death": False,
                "shots": shots,
                "frames": frame + 1,
                "round_result": round_result,
            }

    end_metrics = analyzer.metrics(sandbox)
    score = float(analyzer.potential(end_metrics)
                  - analyzer.potential(start_metrics))
    if kill_frame is not None:
        score += 800.0
    if death_frame is not None:
        score -= 1000.0
    score -= 1.5 * shots
    return score, {
        "kill": kill_frame is not None or not enemy.alive,
        "death": death_frame is not None or not me.alive,
        "double_death": ((kill_frame is not None or not enemy.alive)
                         and (death_frame is not None or not me.alive)),
        "shots": shots,
        "frames": horizon,
        "round_result": round_result,
    }


def _should_capture(category, frame, records, last_frames, args):
    if len(records) >= args.max_states_per_round:
        return False
    if frame - last_frames.get(category, -10**9) < args.min_gap_frames:
        return False
    return True


def _probe_state(game, policy, action, category, frame, seed, rng, args):
    analyzer = OpportunityAnalyzer360(game)
    metrics = analyzer.metrics(game)
    atomic_scores, _ = label_actions(
        game, analyzer, metrics, rng.randrange(1 << 30), args.horizon,
        score_samples=args.samples)
    chosen = CANDIDATES.index(action)
    atomic_best = int(np.argmax(atomic_scores))

    macro_scores = []
    macro_facts = []
    specs = _macro_specs(action)
    for macro in specs:
        total = 0.0
        facts_acc = Counter()
        shots = 0
        for _ in range(max(1, args.samples)):
            sandbox = make_sandbox(
                game, "L2", rng_seed=rng.randrange(1 << 30))
            score, facts = macro_rollout(
                sandbox, macro, analyzer, metrics, args.horizon, args.hold)
            total += score
            shots += facts["shots"]
            for key in ("kill", "death", "double_death"):
                facts_acc[key] += int(bool(facts[key]))
        denom = max(1, args.samples)
        macro_scores.append(total / denom)
        macro_facts.append({
            "kill_rate": facts_acc["kill"] / denom,
            "death_rate": facts_acc["death"] / denom,
            "double_death_rate": facts_acc["double_death"] / denom,
            "shots": shots / denom,
        })

    best_macro = int(np.argmax(macro_scores))
    best_macro_name = specs[best_macro]["name"]
    hold_score = float(macro_scores[0])
    return {
        "seed": seed,
        "frame": frame,
        "category": category,
        "metrics": {
            "line": float(metrics[0]),
            "reach": float(metrics[1]),
            "risk": float(metrics[2]),
        },
        "bullets_fired": int(game.tanks[0].bullets_fired),
        "chosen_action": list(action),
        "chosen_score": float(atomic_scores[chosen]),
        "atomic_best": int(atomic_best),
        "atomic_best_action": list(CANDIDATES[atomic_best]),
        "atomic_best_score": float(atomic_scores[atomic_best]),
        "macro_names": [macro["name"] for macro in specs],
        "macro_kinds": [macro["kind"] for macro in specs],
        "macro_scores": [float(value) for value in macro_scores],
        "macro_facts": macro_facts,
        "best_macro": best_macro_name,
        "best_macro_kind": specs[best_macro]["kind"],
        "best_macro_score": float(macro_scores[best_macro]),
        "hold_macro_score": hold_score,
        "best_macro_advantage_vs_hold": float(
            macro_scores[best_macro] - hold_score),
        "best_macro_advantage_vs_chosen": float(
            macro_scores[best_macro] - atomic_scores[chosen]),
        "best_macro_advantage_vs_atomic": float(
            macro_scores[best_macro] - atomic_scores[atomic_best]),
        "X": stack_observation(policy.history, policy.frame_stack).tolist(),
    }


def probe_round(policy, seed, args):
    game = Game(seed=seed, ai_enabled=True)
    policy.reset()
    analyzer = OpportunityAnalyzer360(game)
    tracker = RoundTracker(game)
    rng = random.Random(seed + 27001)
    records = []
    last_frames = {}
    pos_window = deque(maxlen=args.stall_window)
    input_window = deque(maxlen=args.stall_window)
    clear_fire_frames = 0
    true_result = None
    frames = 0

    while frames < TRUNCATE_FRAMES or tracker.first_destroy is not None:
        t0, t1 = game.tanks[0], game.tanks[1]
        metrics = analyzer.metrics(game)
        line, reach, risk = [float(value) for value in metrics[:3]]
        inp = policy.act(game)
        cmd = _input_tuple(inp)
        action = _action_tuple(inp)
        pos_window.append((t0.x, t0.y))
        input_window.append(cmd)

        category = None
        if cmd[4] and t1.alive:
            shot = _shot_event(game)
            closest = float("inf") if shot is None else shot.get(
                "closest", float("inf"))
            result = None if shot is None else shot.get("result")
            if (line < args.blind_fire_line and result != "HIT"
                    and closest > args.pressure_radius * game.scale):
                category = "blind_fire"
        if category is None and (not cmd[4]) and t1.alive:
            if line >= args.fire_window_line:
                clear_fire_frames += 1
                if clear_fire_frames >= args.fire_window_frames:
                    category = "missed_fire_window"
                    clear_fire_frames = 0
            else:
                clear_fire_frames = 0
        elif category is None:
            clear_fire_frames = 0

        if category is None and len(pos_window) == args.stall_window:
            dx = pos_window[-1][0] - pos_window[0][0]
            dy = pos_window[-1][1] - pos_window[0][1]
            displacement = math.hypot(dx, dy)
            moving_cmds = sum(any(command[:4]) for command in input_window)
            x, y = _cell(game, t0)
            dead_end = _dead_end_penalty(game, x, y)
            exits = _open_neighbors(game, x, y)
            stalled = displacement < args.stall_distance * game.scale
            if stalled and (dead_end > 0 or exits <= 1):
                category = "dead_end_stall"
            elif stalled and line < 0.35 and reach < 0.55 and risk < 0.35:
                category = "passive_map_control"
            elif stalled and moving_cmds >= args.stall_window // 4:
                category = "stutter_stall"
            if stalled:
                pos_window.clear()
                input_window.clear()

        if (category is not None and t0.alive
                and _should_capture(category, frames, records,
                                    last_frames, args)):
            last_frames[category] = frames
            records.append(_probe_state(
                game, policy, action, category, frames, seed, rng, args))

        t0.forward, t0.backup = cmd[0], cmd[1]
        t0.turn_left, t0.turn_right = cmd[2], cmd[3]
        t0.fire = cmd[4]
        tracker.pre_step()
        events = game.step()
        frames += 1
        tracker.post_step(events, 1)
        for event in events:
            if event[0] == "round_end":
                true_result = _round_result(event[1])
        if true_result:
            break

    return {
        "seed": seed,
        "result": true_result or "draw",
        "frames": frames,
        "shots": tracker.shots,
        "kills": tracker.kills,
    }, records


def _make_policy(args):
    return P26Policy(
        net_path=args.net,
        fire_margin=args.fire_margin,
        fire_threshold=0.0,
        kill_weight=0.0,
        death_weight=0.0,
        double_death_weight=0.0,
        survive_weight=0.0,
        fire_prob_weight=0.0,
    )


def _worker(job):
    worker, seed, count, args = job
    import torch

    torch.set_num_threads(1)
    policy = _make_policy(args)
    rounds = []
    records = []
    for index in range(count):
        payload, round_records = probe_round(policy, seed + index, args)
        payload["worker"] = worker
        rounds.append(payload)
        records.extend(round_records)
    return rounds, records


def _summarize(rounds, records, args, elapsed):
    results = Counter(item["result"] for item in rounds)
    category_counts = Counter(item["category"] for item in records)
    best_macro_counts = Counter(item["best_macro"] for item in records)
    best_kind_counts = Counter(item["best_macro_kind"] for item in records)
    fan_records = [
        item for item in records if item["best_macro_kind"] == "fan_fire"
    ]
    escape_records = [
        item for item in records if item["best_macro_kind"] == "escape"
    ]
    positive_macro = [
        item for item in records
        if item["best_macro_advantage_vs_hold"] > args.min_advantage
    ]
    fan_positive = [
        item for item in records
        if (item["best_macro_kind"] == "fan_fire"
            and item["best_macro_advantage_vs_hold"] > args.min_advantage)
    ]
    total = max(1, len(rounds))
    record_total = max(1, len(records))
    return {
        "net": args.net,
        "n": args.n,
        "seed": args.seed,
        "workers": args.workers,
        "horizon": args.horizon,
        "hold": args.hold,
        "samples": args.samples,
        "elapsed_seconds": elapsed,
        "rounds": len(rounds),
        "results": dict(results),
        "win_rate": results["win"] / total,
        "loss_rate": results["loss"] / total,
        "double_death_rate": results["double_death"] / total,
        "records": len(records),
        "category_counts": dict(category_counts),
        "best_macro_counts": dict(best_macro_counts),
        "best_kind_counts": dict(best_kind_counts),
        "fan_best_rate": len(fan_records) / record_total,
        "escape_best_rate": len(escape_records) / record_total,
        "positive_macro_rate": len(positive_macro) / record_total,
        "fan_positive_rate": len(fan_positive) / record_total,
        "avg_best_advantage_vs_hold": (
            sum(item["best_macro_advantage_vs_hold"] for item in records)
            / record_total),
        "avg_best_advantage_vs_chosen": (
            sum(item["best_macro_advantage_vs_chosen"] for item in records)
            / record_total),
        "avg_best_advantage_vs_atomic": (
            sum(item["best_macro_advantage_vs_atomic"] for item in records)
            / record_total),
        "out": args.out,
        "summary": args.summary,
    }


def run(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    workers = max(1, min(args.workers, args.n))
    base, remainder = divmod(args.n, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count <= 0:
            continue
        jobs.append((worker, args.seed + offset, count, args))
        offset += count

    started = time.time()
    if workers == 1:
        outputs = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            outputs = pool.map(_worker, jobs)
    rounds = [item for part, _ in outputs for item in part]
    records = [item for _, part in outputs for item in part]
    rounds.sort(key=lambda item: item["seed"])
    records.sort(key=lambda item: (item["seed"], item["frame"],
                                   item["category"]))

    with open(args.out, "w", encoding="utf-8") as handle:
        for item in records:
            slim = {key: value for key, value in item.items() if key != "X"}
            handle.write(json.dumps(slim, sort_keys=True) + "\n")
    if args.macro_data:
        os.makedirs(os.path.dirname(args.macro_data), exist_ok=True)
        x = np.asarray([item["X"] for item in records], dtype=np.float32)
        scores = np.asarray([item["macro_scores"] for item in records],
                            dtype=np.float32)
        np.savez_compressed(
            args.macro_data,
            X=x,
            Y_macro=scores / SCORE_SCALE,
            macro_names=np.asarray(MACRO_NAMES),
            category=np.asarray([item["category"] for item in records]),
            best_macro=np.asarray([item["best_macro"] for item in records]),
            seed=np.asarray([item["seed"] for item in records], np.int32),
            frame=np.asarray([item["frame"] for item in records], np.int32),
        )

    summary = _summarize(rounds, records, args, time.time() - started)
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print("===== P27 macro probe =====", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="training/models/p26_amortized_mpc_iter05.pt")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--hold", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-states-per-round", type=int, default=5)
    parser.add_argument("--min-gap-frames", type=int, default=24)
    parser.add_argument("--min-advantage", type=float, default=20.0)
    parser.add_argument("--stall-window", type=int, default=40)
    parser.add_argument("--stall-distance", type=float, default=0.22)
    parser.add_argument("--fire-window-line", type=float, default=0.70)
    parser.add_argument("--fire-window-frames", type=int, default=8)
    parser.add_argument("--blind-fire-line", type=float, default=0.35)
    parser.add_argument("--pressure-radius", type=float, default=0.75)
    parser.add_argument("--out", default="training/analysis/runs/p27_macro_probe.jsonl")
    parser.add_argument("--summary", default="training/analysis/runs/p27_macro_probe_summary.json")
    parser.add_argument("--macro-data", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
