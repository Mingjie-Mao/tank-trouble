"""
Human-style behavior observer for P26 policies.

Runs real P26-vs-Laika rounds in the original engine and records per-round
behavior issues that are visible in watch mode but not captured by win rate
alone.
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
from training.mpc_agent import CANDIDATES, make_sandbox  # noqa: E402
from tank_trouble_original.maze import h_open, v_open  # noqa: E402
from training.evaluate import RoundTracker  # noqa: E402
from training.opportunity_distill import _shot_event  # noqa: E402
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    AUX_NAMES,
    DATA_DIR,
    SCORE_SCALE,
    P26Policy,
    fire_targets,
    label_actions,
    sample_weight,
    stack_observation,
)
from training.tt_gym_env import TRUNCATE_FRAMES  # noqa: E402

OBSERVER_CATEGORY_NAMES = (
    "missed_fire_window",
    "blind_fire",
    "stutter_stall",
    "dead_end_stall",
    "passive_map_control",
    "post_kill_fire",
)


def _input_tuple(inp):
    return (
        bool(inp.get("forward", False)),
        bool(inp.get("backup", False)),
        bool(inp.get("turn_left", False)),
        bool(inp.get("turn_right", False)),
        bool(inp.get("fire", False)),
    )


def _action_tuple(inp):
    return (
        2 if inp.get("forward") else 0 if inp.get("backup") else 1,
        0 if inp.get("turn_left") else 2 if inp.get("turn_right") else 1,
        1 if inp.get("fire") else 0,
    )


def _cell(game, tank):
    x = max(0, min(len(game.maze) - 1, int(tank.x // game.scale)))
    y = max(0, min(len(game.maze[0]) - 1, int(tank.y // game.scale)))
    return x, y


def _open_neighbors(game, x, y):
    w, h = len(game.maze), len(game.maze[0])
    count = 0
    if x > 0 and v_open(game.maze, x, y):
        count += 1
    if x < w - 1 and v_open(game.maze, x + 1, y):
        count += 1
    if y > 0 and h_open(game.maze, x, y - 1):
        count += 1
    if y < h - 1 and h_open(game.maze, x, y):
        count += 1
    return count


def _dead_end_penalty(game, x, y):
    if not getattr(game, "dead_ends", None):
        return 0.0
    value = game.dead_ends[x][y]
    return 0.0 if value is None else float(value)


def _round_result(winner):
    if winner == 0:
        return "win"
    if winner == 1:
        return "loss"
    return "double_death"


def _summarize_notes(issues, strengths):
    notes = []
    if issues["post_kill_fire"]:
        notes.append("fires after Laika is already dead")
    if issues["blind_fire"]:
        notes.append("fires without a good direct/rebound line")
    if issues["missed_fire_window"]:
        notes.append("declines clear firing windows")
    if issues["passive_map_control"]:
        notes.append("stalls while map-control value is low")
    if issues["stutter_stall"]:
        notes.append("jitters or stalls with low displacement")
    if issues["dead_end_stall"]:
        notes.append("stalls in a dead-end/corner-like cell")
    if strengths["bullet_dodge_good"]:
        notes.append("dodges bullets well under incoming risk")
    return notes


def _capture_issue(records, last_frames, category, frame, policy, game, action,
                   rng, args):
    if not args.hard_phase:
        return
    if len(records) >= args.hard_max_per_round:
        return
    if frame - last_frames.get(category, -10**9) < args.hard_min_gap_frames:
        return
    if not game.tanks[0].alive:
        return
    if not policy.history:
        return
    last_frames[category] = frame
    records.append({
        "category": category,
        "frame": frame,
        "action": action,
        "stacked": stack_observation(policy.history, policy.frame_stack).copy(),
        "game": make_sandbox(game, "L2", rng_seed=rng.randrange(1 << 30)),
    })


def _label_issue_record(record, rng, args):
    game = record["game"]
    analyzer = OpportunityAnalyzer360(game)
    metrics = analyzer.metrics(game)
    scores, aux = label_actions(
        game, analyzer, metrics, rng.randrange(1 << 30), args.score_horizon,
        score_samples=args.score_samples)
    chosen = CANDIDATES.index(record["action"])
    best = int(scores.argmax())
    y_score = scores / SCORE_SCALE
    return {
        "X": record["stacked"],
        "Y_score": y_score.astype("float32"),
        "Y_aux": aux.astype("float32"),
        "Y_fire": fire_targets(y_score, args.fire_target_margin),
        "W": float(sample_weight(scores, chosen, aux)),
        "category": record["category"],
        "regret": float((scores[best] - scores[chosen]) / SCORE_SCALE),
        "chosen": chosen,
        "best": best,
        "frame": record["frame"],
    }


def observe_round(policy, seed, args):
    game = Game(seed=seed, ai_enabled=True)
    policy.reset()
    if hasattr(policy, "set_round_seed"):
        policy.set_round_seed(seed)
    analyzer = OpportunityAnalyzer360(game)
    tracker = RoundTracker(game)
    rng = random.Random(seed + 26001)
    issues = Counter()
    strengths = Counter()
    issue_frames = {name: [] for name in (
        "post_kill_fire",
        "blind_fire",
        "missed_fire_window",
        "passive_map_control",
        "stutter_stall",
        "dead_end_stall",
    )}
    pos_window = deque(maxlen=args.stall_window)
    input_window = deque(maxlen=args.stall_window)
    high_risk_frames = 0
    clear_fire_frames = 0
    true_result = None
    frames = 0
    hard_records = []
    hard_last_frames = {}

    while frames < TRUNCATE_FRAMES or tracker.first_destroy is not None:
        t0 = game.tanks[0]
        t1 = game.tanks[1]
        metrics = analyzer.metrics(game)
        line, reach, risk = [float(value) for value in metrics[:3]]
        inp = policy.act(game)
        cmd = _input_tuple(inp)
        action = _action_tuple(inp)
        pos_window.append((t0.x, t0.y))
        input_window.append(cmd)

        if risk >= args.good_dodge_risk:
            high_risk_frames += 1

        if cmd[4] and not t1.alive:
            issues["post_kill_fire"] += 1
            issue_frames["post_kill_fire"].append(frames)
            _capture_issue(
                hard_records, hard_last_frames, "post_kill_fire", frames,
                policy, game, action, rng, args)

        if cmd[4] and t1.alive:
            shot = _shot_event(game)
            closest = float("inf") if shot is None else shot.get("closest", float("inf"))
            result = None if shot is None else shot.get("result")
            weak_line = line < args.blind_fire_line
            weak_shot = result != "HIT" and closest > args.pressure_radius * game.scale
            if weak_line and weak_shot:
                issues["blind_fire"] += 1
                issue_frames["blind_fire"].append(frames)
                _capture_issue(
                    hard_records, hard_last_frames, "blind_fire", frames,
                    policy, game, action, rng, args)

        if (not cmd[4]) and t1.alive and line >= args.fire_window_line:
            clear_fire_frames += 1
            if clear_fire_frames >= args.fire_window_frames:
                issues["missed_fire_window"] += 1
                issue_frames["missed_fire_window"].append(frames)
                _capture_issue(
                    hard_records, hard_last_frames, "missed_fire_window",
                    frames, policy, game, action, rng, args)
                clear_fire_frames = 0
        else:
            clear_fire_frames = 0

        if len(pos_window) == args.stall_window:
            dx = pos_window[-1][0] - pos_window[0][0]
            dy = pos_window[-1][1] - pos_window[0][1]
            displacement = math.hypot(dx, dy)
            moving_cmds = sum(any(command[:4]) for command in input_window)
            x, y = _cell(game, t0)
            dead_end = _dead_end_penalty(game, x, y)
            exits = _open_neighbors(game, x, y)
            stalled = displacement < args.stall_distance * game.scale
            if stalled and moving_cmds >= args.stall_window // 4:
                issues["stutter_stall"] += 1
                issue_frames["stutter_stall"].append(frames)
                _capture_issue(
                    hard_records, hard_last_frames, "stutter_stall", frames,
                    policy, game, action, rng, args)
                pos_window.clear()
                input_window.clear()
            elif stalled and line < 0.35 and reach < 0.55 and risk < 0.35:
                issues["passive_map_control"] += 1
                issue_frames["passive_map_control"].append(frames)
                _capture_issue(
                    hard_records, hard_last_frames, "passive_map_control",
                    frames, policy, game, action, rng, args)
                pos_window.clear()
                input_window.clear()
            elif stalled and (dead_end > 0 or exits <= 1):
                issues["dead_end_stall"] += 1
                issue_frames["dead_end_stall"].append(frames)
                _capture_issue(
                    hard_records, hard_last_frames, "dead_end_stall", frames,
                    policy, game, action, rng, args)
                pos_window.clear()
                input_window.clear()

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

    if high_risk_frames >= args.good_dodge_frames and true_result != "loss":
        strengths["bullet_dodge_good"] += 1

    payload = {
        "seed": seed,
        "result": true_result or "draw",
        "frames": frames,
        "seconds": frames / 25.0,
        "shots": tracker.shots,
        "kills": tracker.kills,
        "hit_rate": tracker.kills / max(1, tracker.shots),
        "move_px": tracker.move_px,
        "death_cause": tracker.death_cause,
        "kill_type": tracker.kill_type,
        "issues": dict(issues),
        "issue_frames": {
            key: value[:args.max_issue_frames]
            for key, value in issue_frames.items() if value
        },
        "strengths": dict(strengths),
        "macro_counts": dict(getattr(policy, "macro_counts", {})),
        "assist_counts": dict(getattr(policy, "assist_counts", {})),
        "correction_counts": dict(getattr(
            policy, "correction_counts", {})),
        "search_counts": dict(getattr(policy, "search_counts", {})),
        "search_frames": int(getattr(policy, "search_frames", 0)),
    }
    payload["notes"] = _summarize_notes(issues, strengths)
    labelled = []
    for record in hard_records:
        labelled.append(_label_issue_record(record, rng, args))
    return payload, labelled


def _make_policy(args):
    if args.progressive_risk_mpc:
        from training.progressive_risk_mpc_teacher import (
            ProgressiveRiskMPCPolicy,
        )
        return ProgressiveRiskMPCPolicy(
            base_net=args.net,
            value_net=args.p27b_net,
            fire_margin=args.fire_margin,
            seed=args.seed,
            deterministic_search_seeds=True,
            horizons=[int(value) for value in args.risk_mpc_horizons.split(",")],
            widths=[int(value) for value in args.risk_mpc_widths.split(",")],
            final_samples=args.risk_mpc_final_samples,
            commit_frames=args.risk_mpc_commit_frames,
            replan_interval=args.risk_mpc_replan_interval,
            death_penalty=args.risk_mpc_death_penalty,
            dd_penalty=args.risk_mpc_dd_penalty,
            kill_bonus=args.risk_mpc_kill_bonus,
            tail_penalty=args.risk_mpc_tail_penalty,
            max_death=args.risk_mpc_max_death,
            max_dd=args.risk_mpc_max_dd,
            fire_min_gain=args.risk_mpc_fire_min_gain,
            fire_max_extra_death=args.risk_mpc_fire_max_extra_death,
            fire_max_extra_dd=args.risk_mpc_fire_max_extra_dd,
            root_fire_min_line=args.risk_mpc_root_fire_min_line,
            root_fire_max_alignment=(
                args.risk_mpc_root_fire_max_alignment),
            root_fire_pressure_radius=(
                args.risk_mpc_root_fire_pressure_radius),
        )
    if args.p30_net:
        from training.p30_consensus_correction import P30CorrectionPolicy
        return P30CorrectionPolicy(
            base_net=args.net,
            value_net=args.p27b_net,
            correction_net=args.p30_net,
            fire_margin=args.fire_margin,
            override_threshold=args.p30_override_threshold,
            background_override_threshold=(
                args.p30_background_override_threshold),
            min_predicted_gain=args.p30_min_predicted_gain,
            max_override_death=args.p30_max_override_death,
            max_override_dd=args.p30_max_override_dd,
            assist_margin=args.p27b_assist_margin,
            assist_weight=args.p27b_assist_weight,
            max_bonus=args.p27b_max_bonus,
            kill_weight=args.p27b_kill_weight,
            death_weight=args.p27b_death_weight,
            double_death_weight=args.p27b_double_death_weight,
            survive_weight=args.p27b_survive_weight,
            risk_threshold=args.p27b_risk_threshold,
            fire_delta_margin=args.p27b_fire_delta_margin,
        )
    if args.p27b_net:
        from training.p27_risk_value import P27BRiskValuePolicy
        return P27BRiskValuePolicy(
            base_net=args.net,
            value_net=args.p27b_net,
            fire_margin=args.fire_margin,
            assist_margin=args.p27b_assist_margin,
            assist_weight=args.p27b_assist_weight,
            max_bonus=args.p27b_max_bonus,
            kill_weight=args.p27b_kill_weight,
            death_weight=args.p27b_death_weight,
            double_death_weight=args.p27b_double_death_weight,
            survive_weight=args.p27b_survive_weight,
            risk_threshold=args.p27b_risk_threshold,
            fire_delta_margin=args.p27b_fire_delta_margin,
            global_fire_risk_penalty=args.p27b_global_fire_risk_penalty,
            global_fire_risk_threshold=args.p27b_global_fire_risk_threshold,
            global_fire_dd_threshold=args.p27b_global_fire_dd_threshold,
            low_quality_fire_penalty=args.p27b_low_quality_fire_penalty,
            low_quality_fire_delta=args.p27b_low_quality_fire_delta,
            opportunity_bonus=args.p27b_opportunity_bonus,
            opportunity_min_line=args.p27b_opportunity_min_line,
            opportunity_max_risk=args.p27b_opportunity_max_risk,
            opportunity_max_danger=args.p27b_opportunity_max_danger,
            opportunity_min_fire_delta=args.p27b_opportunity_min_fire_delta,
            escape_bonus=args.p27b_escape_bonus,
            escape_min_gain=args.p27b_escape_min_gain,
            escape_max_danger=args.p27b_escape_max_danger,
            escape_hold_frames=args.p27b_escape_hold_frames,
            stall_fire_penalty=args.p27b_stall_fire_penalty,
        )
    if args.macro_net:
        from training.p27_macro_policy import P27MacroPolicy, P27ValueAssistPolicy
        if args.macro_mode == "value":
            return P27ValueAssistPolicy(
                base_net=args.net,
                macro_net=args.macro_net,
                fire_margin=args.fire_margin,
                value_margin=args.value_margin,
                fire_value_margin=args.fire_value_margin,
                escape_bonus_weight=args.escape_bonus_weight,
                fire_bonus_weight=args.fire_bonus_weight,
                max_escape_bonus=args.max_escape_bonus,
                max_fire_bonus=args.max_fire_bonus,
                fire_line=args.value_fire_line,
                fire_max_risk=args.value_fire_max_risk,
                suppress_blind_fire_line=args.suppress_blind_fire_line,
            )
        return P27MacroPolicy(
            base_net=args.net,
            macro_net=args.macro_net,
            fire_margin=args.fire_margin,
            macro_margin=args.macro_margin,
            fan_min_line=args.fan_min_line,
            fan_max_risk=args.fan_max_risk,
            fan_max_bullets=args.fan_max_bullets,
            single_min_line=args.single_min_line,
            single_max_risk=args.single_max_risk,
            macro_cooldown=args.macro_cooldown,
        )
    return P26Policy(
        net_path=args.net,
        fire_margin=args.fire_margin,
        fire_threshold=0.0,
        kill_weight=0.0,
        death_weight=0.0,
        double_death_weight=0.0,
        survive_weight=0.0,
        fire_prob_weight=0.0,
        fire_assist_line=args.fire_assist_line,
        fire_assist_max_risk=args.fire_assist_max_risk,
        fire_assist_min_delta=args.fire_assist_min_delta,
        suppress_blind_fire_line=args.suppress_blind_fire_line,
    )


def _observe_worker(job):
    worker, seed, count, args = job
    import torch

    torch.set_num_threads(1)
    policy = _make_policy(args)
    rounds = []
    hard_rows = []
    for index in range(count):
        result, labelled = observe_round(policy, seed + index, args)
        result["worker"] = worker
        rounds.append(result)
        hard_rows.extend(labelled)
    return rounds, hard_rows


def run(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    started = time.time()
    workers = max(1, min(args.workers, args.n))
    base, remainder = divmod(args.n, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count <= 0:
            continue
        jobs.append((worker, args.seed + offset, count, args))
        offset += count

    if workers == 1:
        rounds, hard_rows = _observe_worker(jobs[0])
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            outputs = pool.map(_observe_worker, jobs)
        rounds = [item for part, _ in outputs for item in part]
        hard_rows = [item for _, part in outputs for item in part]
    rounds.sort(key=lambda item: item["seed"])

    with open(args.out, "w", encoding="utf-8") as handle:
        for index, result in enumerate(rounds):
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            notes = "; ".join(result["notes"]) if result["notes"] else "clean"
            print(
                f"{index + 1:03d} seed={result['seed']} "
                f"{result['result']} shots={result['shots']} "
                f"hit={result['hit_rate']:.1%} issues={result['issues']} "
                f"{notes}",
                flush=True,
            )

    results = Counter(item["result"] for item in rounds)
    issues = Counter()
    strengths = Counter()
    macro_counts = Counter()
    assist_counts = Counter()
    correction_counts = Counter()
    search_counts = Counter()
    search_frames = 0
    shots = 0
    kills = 0
    frames = 0
    for item in rounds:
        issues.update(item["issues"])
        strengths.update(item["strengths"])
        macro_counts.update(item.get("macro_counts", {}))
        assist_counts.update(item.get("assist_counts", {}))
        correction_counts.update(item.get("correction_counts", {}))
        search_counts.update(item.get("search_counts", {}))
        search_frames += item.get("search_frames", 0)
        shots += item["shots"]
        kills += item["kills"]
        frames += item["frames"]
    total = max(1, len(rounds))
    summary = {
        "net": args.net,
        "p27b_net": args.p27b_net,
        "p30_net": args.p30_net,
        "macro_net": args.macro_net,
        "macro_mode": args.macro_mode,
        "n": args.n,
        "seed": args.seed,
        "workers": workers,
        "fire_margin": args.fire_margin,
        "p27b_assist_margin": args.p27b_assist_margin,
        "p27b_assist_weight": args.p27b_assist_weight,
        "p27b_max_bonus": args.p27b_max_bonus,
        "p27b_kill_weight": args.p27b_kill_weight,
        "p27b_death_weight": args.p27b_death_weight,
        "p27b_double_death_weight": args.p27b_double_death_weight,
        "p27b_survive_weight": args.p27b_survive_weight,
        "p27b_risk_threshold": args.p27b_risk_threshold,
        "p27b_fire_delta_margin": args.p27b_fire_delta_margin,
        "p27b_global_fire_risk_penalty": args.p27b_global_fire_risk_penalty,
        "p27b_global_fire_risk_threshold": args.p27b_global_fire_risk_threshold,
        "p27b_global_fire_dd_threshold": args.p27b_global_fire_dd_threshold,
        "p27b_low_quality_fire_penalty": args.p27b_low_quality_fire_penalty,
        "p27b_low_quality_fire_delta": args.p27b_low_quality_fire_delta,
        "p27b_opportunity_bonus": args.p27b_opportunity_bonus,
        "p27b_opportunity_min_line": args.p27b_opportunity_min_line,
        "p27b_opportunity_max_risk": args.p27b_opportunity_max_risk,
        "p27b_opportunity_max_danger": args.p27b_opportunity_max_danger,
        "p27b_opportunity_min_fire_delta": (
            args.p27b_opportunity_min_fire_delta),
        "p27b_escape_bonus": args.p27b_escape_bonus,
        "p27b_escape_min_gain": args.p27b_escape_min_gain,
        "p27b_escape_max_danger": args.p27b_escape_max_danger,
        "p27b_escape_hold_frames": args.p27b_escape_hold_frames,
        "p27b_stall_fire_penalty": args.p27b_stall_fire_penalty,
        "macro_margin": args.macro_margin,
        "fan_min_line": args.fan_min_line,
        "fan_max_risk": args.fan_max_risk,
        "fan_max_bullets": args.fan_max_bullets,
        "single_min_line": args.single_min_line,
        "single_max_risk": args.single_max_risk,
        "macro_cooldown": args.macro_cooldown,
        "value_margin": args.value_margin,
        "fire_value_margin": args.fire_value_margin,
        "escape_bonus_weight": args.escape_bonus_weight,
        "fire_bonus_weight": args.fire_bonus_weight,
        "max_escape_bonus": args.max_escape_bonus,
        "max_fire_bonus": args.max_fire_bonus,
        "value_fire_line": args.value_fire_line,
        "value_fire_max_risk": args.value_fire_max_risk,
        "fire_assist_line": args.fire_assist_line,
        "fire_assist_max_risk": args.fire_assist_max_risk,
        "fire_assist_min_delta": args.fire_assist_min_delta,
        "suppress_blind_fire_line": args.suppress_blind_fire_line,
        "elapsed_seconds": time.time() - started,
        "results": dict(results),
        "win_rate": results["win"] / total,
        "loss_rate": results["loss"] / total,
        "double_death_rate": results["double_death"] / total,
        "shots_per_game": shots / total,
        "hit_rate": kills / max(1, shots),
        "avg_seconds": frames / total / 25.0,
        "issues": dict(issues),
        "strengths": dict(strengths),
        "macro_counts": dict(macro_counts),
        "assist_counts": dict(assist_counts),
        "correction_counts": dict(correction_counts),
        "search_counts": dict(search_counts),
        "search_frames": int(search_frames),
        "round_log": args.out,
    }
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    hard_path = None
    if args.hard_phase and hard_rows:
        phase_dir = os.path.join(DATA_DIR, args.hard_phase)
        os.makedirs(phase_dir, exist_ok=True)
        hard_path = os.path.join(phase_dir, "shard_0.npz")
        np.savez_compressed(
            hard_path,
            X=np.asarray([item["X"] for item in hard_rows], np.float32),
            Y_score=np.asarray([item["Y_score"] for item in hard_rows],
                               np.float32),
            Y_aux=np.asarray([item["Y_aux"] for item in hard_rows],
                             np.float32),
            Y_fire=np.asarray([item["Y_fire"] for item in hard_rows],
                              np.float32),
            W=np.asarray([item["W"] for item in hard_rows], np.float32),
            category=np.asarray([item["category"] for item in hard_rows]),
            category_names=np.asarray(OBSERVER_CATEGORY_NAMES),
            regret=np.asarray([item["regret"] for item in hard_rows],
                              np.float32),
            chosen=np.asarray([item["chosen"] for item in hard_rows],
                              np.int32),
            best=np.asarray([item["best"] for item in hard_rows], np.int32),
            frame=np.asarray([item["frame"] for item in hard_rows],
                             np.int32),
            frame_stack=np.asarray([4], np.int32),
            aux_names=np.asarray(AUX_NAMES),
        )
        summary["hard_data_path"] = hard_path
        summary["hard_rows"] = len(hard_rows)
        if args.summary:
            with open(args.summary, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
    print("===== behavior summary =====", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if hard_path:
        print(f"saved hard data {hard_path}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="training/models/p26_amortized_mpc_iter05.pt")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--fire-assist-line", type=float, default=0.0)
    parser.add_argument("--fire-assist-max-risk", type=float, default=0.35)
    parser.add_argument("--fire-assist-min-delta", type=float, default=-0.03)
    parser.add_argument("--suppress-blind-fire-line", type=float, default=0.0)
    parser.add_argument("--p27b-net", default=None)
    parser.add_argument("--p30-net", default=None)
    parser.add_argument("--p30-override-threshold", type=float, default=0.72)
    parser.add_argument("--p30-background-override-threshold", type=float,
                        default=0.84)
    parser.add_argument("--p30-min-predicted-gain", type=float, default=0.03)
    parser.add_argument("--p30-max-override-death", type=float, default=0.55)
    parser.add_argument("--p30-max-override-dd", type=float, default=0.50)
    parser.add_argument("--progressive-risk-mpc", action="store_true")
    parser.add_argument("--risk-mpc-horizons", default="24,48,72,96")
    parser.add_argument("--risk-mpc-widths", default="6,3,2,2")
    parser.add_argument("--risk-mpc-final-samples", type=int, default=4)
    parser.add_argument("--risk-mpc-commit-frames", type=int, default=24)
    parser.add_argument("--risk-mpc-replan-interval", type=int, default=16)
    parser.add_argument("--risk-mpc-death-penalty", type=float, default=0.55)
    parser.add_argument("--risk-mpc-dd-penalty", type=float, default=1.0)
    parser.add_argument("--risk-mpc-kill-bonus", type=float, default=0.04)
    parser.add_argument("--risk-mpc-tail-penalty", type=float, default=0.15)
    parser.add_argument("--risk-mpc-max-death", type=float, default=0.0)
    parser.add_argument("--risk-mpc-max-dd", type=float, default=0.0)
    parser.add_argument("--risk-mpc-fire-min-gain", type=float, default=0.015)
    parser.add_argument("--risk-mpc-fire-max-extra-death", type=float,
                        default=0.0)
    parser.add_argument("--risk-mpc-fire-max-extra-dd", type=float,
                        default=0.0)
    parser.add_argument("--risk-mpc-root-fire-min-line", type=float,
                        default=0.35)
    parser.add_argument("--risk-mpc-root-fire-max-alignment", type=float,
                        default=0.30)
    parser.add_argument("--risk-mpc-root-fire-pressure-radius", type=float,
                        default=0.75)
    parser.add_argument("--p27b-assist-margin", type=float, default=0.08)
    parser.add_argument("--p27b-assist-weight", type=float, default=0.35)
    parser.add_argument("--p27b-max-bonus", type=float, default=0.10)
    parser.add_argument("--p27b-kill-weight", type=float, default=0.04)
    parser.add_argument("--p27b-death-weight", type=float, default=0.12)
    parser.add_argument("--p27b-double-death-weight", type=float, default=0.18)
    parser.add_argument("--p27b-survive-weight", type=float, default=0.02)
    parser.add_argument("--p27b-risk-threshold", type=float, default=0.55)
    parser.add_argument("--p27b-fire-delta-margin", type=float, default=0.14)
    parser.add_argument("--p27b-global-fire-risk-penalty", type=float,
                        default=0.0)
    parser.add_argument("--p27b-global-fire-risk-threshold", type=float,
                        default=1.10)
    parser.add_argument("--p27b-global-fire-dd-threshold", type=float,
                        default=1.10)
    parser.add_argument("--p27b-low-quality-fire-penalty", type=float,
                        default=0.0)
    parser.add_argument("--p27b-low-quality-fire-delta", type=float,
                        default=0.0)
    parser.add_argument("--p27b-opportunity-bonus", type=float, default=0.0)
    parser.add_argument("--p27b-opportunity-min-line", type=float,
                        default=0.76)
    parser.add_argument("--p27b-opportunity-max-risk", type=float,
                        default=0.25)
    parser.add_argument("--p27b-opportunity-max-danger", type=float,
                        default=0.45)
    parser.add_argument("--p27b-opportunity-min-fire-delta", type=float,
                        default=0.08)
    parser.add_argument("--p27b-escape-bonus", type=float, default=0.0)
    parser.add_argument("--p27b-escape-min-gain", type=float, default=-0.02)
    parser.add_argument("--p27b-escape-max-danger", type=float, default=0.55)
    parser.add_argument("--p27b-escape-hold-frames", type=int, default=0)
    parser.add_argument("--p27b-stall-fire-penalty", type=float, default=0.0)
    parser.add_argument("--macro-net", default=None)
    parser.add_argument("--macro-mode", choices=["sequence", "value"],
                        default="sequence")
    parser.add_argument("--macro-margin", type=float, default=0.08)
    parser.add_argument("--fan-min-line", type=float, default=0.78)
    parser.add_argument("--fan-max-risk", type=float, default=0.20)
    parser.add_argument("--fan-max-bullets", type=int, default=3)
    parser.add_argument("--single-min-line", type=float, default=0.72)
    parser.add_argument("--single-max-risk", type=float, default=0.25)
    parser.add_argument("--macro-cooldown", type=int, default=20)
    parser.add_argument("--value-margin", type=float, default=0.10)
    parser.add_argument("--fire-value-margin", type=float, default=0.12)
    parser.add_argument("--escape-bonus-weight", type=float, default=0.05)
    parser.add_argument("--fire-bonus-weight", type=float, default=0.04)
    parser.add_argument("--max-escape-bonus", type=float, default=0.06)
    parser.add_argument("--max-fire-bonus", type=float, default=0.05)
    parser.add_argument("--value-fire-line", type=float, default=0.76)
    parser.add_argument("--value-fire-max-risk", type=float, default=0.22)
    parser.add_argument("--hard-phase", default=None)
    parser.add_argument("--hard-max-per-round", type=int, default=8)
    parser.add_argument("--hard-min-gap-frames", type=int, default=24)
    parser.add_argument("--score-horizon", type=int, default=48)
    parser.add_argument("--score-samples", type=int, default=1)
    parser.add_argument("--fire-target-margin", type=float, default=0.16)
    parser.add_argument("--out", default="training/analysis/runs/p26_behavior_observer.jsonl")
    parser.add_argument("--summary", default="training/analysis/runs/p26_behavior_observer_summary.json")
    parser.add_argument("--stall-window", type=int, default=40)
    parser.add_argument("--stall-distance", type=float, default=0.22)
    parser.add_argument("--fire-window-line", type=float, default=0.70)
    parser.add_argument("--fire-window-frames", type=int, default=8)
    parser.add_argument("--blind-fire-line", type=float, default=0.35)
    parser.add_argument("--pressure-radius", type=float, default=0.75)
    parser.add_argument("--good-dodge-risk", type=float, default=0.55)
    parser.add_argument("--good-dodge-frames", type=int, default=20)
    parser.add_argument("--max-issue-frames", type=int, default=12)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
