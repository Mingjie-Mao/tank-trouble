"""
P29 P28 hard-frame distillation.

This collector records only strategically important frames from real
P27b/P28-vs-Laika rounds, then labels those states with the current strongest
P28 teacher objective. The output shard uses the same NPZ schema as the P27b
risk/value head, so training can reuse training/p27_risk_value.py.
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
from tank_trouble_original.maze import h_open, v_open  # noqa: E402
from training.evaluate import RoundTracker  # noqa: E402
from training.mpc_agent import CANDIDATES, make_sandbox  # noqa: E402
from training.opportunity_distill import _shot_event  # noqa: E402
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    AUX_NAMES,
    DATA_DIR,
    SCORE_SCALE,
    fire_targets,
    label_actions,
    sample_weight,
    stack_observation,
)
from training.p27_risk_value import (  # noqa: E402
    P27BRiskValuePolicy,
    P29C_CONTEXT_DIM,
    P29C_CONTEXT_NAMES,
)
from training.p28_hybrid_fallback import P28PriorSearchPolicy  # noqa: E402
from training.tt_gym_env import TRUNCATE_FRAMES  # noqa: E402


P29_CATEGORIES = (
    "direct_shot_loss",
    "bounce_shot_loss",
    "self_shot_loss",
    "finish_window",
    "missed_fire_window",
    "blind_fire",
    "post_kill_fire",
    "stutter_stall",
    "dead_end_stall",
    "passive_map_control",
    "active_pursuit_gap",
    "long_game",
    "background_state",
)

CATEGORY_WEIGHT = {
    "direct_shot_loss": 4.0,
    "bounce_shot_loss": 3.2,
    "self_shot_loss": 3.6,
    "finish_window": 2.6,
    "missed_fire_window": 2.4,
    "blind_fire": 1.8,
    "post_kill_fire": 1.4,
    "stutter_stall": 2.2,
    "dead_end_stall": 2.8,
    "passive_map_control": 2.2,
    "active_pursuit_gap": 2.4,
    "long_game": 1.8,
    "background_state": 0.65,
}


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
    width, height = len(game.maze), len(game.maze[0])
    count = 0
    if x > 0 and v_open(game.maze, x, y):
        count += 1
    if x < width - 1 and v_open(game.maze, x + 1, y):
        count += 1
    if y > 0 and h_open(game.maze, x, y - 1):
        count += 1
    if y < height - 1 and h_open(game.maze, x, y):
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


def _safe_div(a, b):
    return float(a) / max(1.0, float(b))


def _make_policy(args, worker):
    seed = args.seed + worker * 10007
    if args.rollout_policy == "p28":
        return P28PriorSearchPolicy(
            base_net=args.base_net,
            value_net=args.value_net,
            fire_margin=args.fire_margin,
            top_k=args.top_k,
            search_horizon=args.search_horizon,
            search_samples=args.search_samples,
            search_death_penalty=args.search_death_penalty,
            search_dd_penalty=args.search_dd_penalty,
            search_kill_bonus=args.search_kill_bonus,
            deterministic_search_seeds=args.deterministic_search_seeds,
            seed=seed,
        )
    return P27BRiskValuePolicy(
        base_net=args.base_net,
        value_net=args.value_net,
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
    )


def _capture(records, last_frames, category, frame, policy, game, action,
             rng, args, frames_to_end=None):
    if len(records) >= args.max_records_per_round:
        return
    if frame - last_frames.get(category, -10**9) < args.min_gap_frames:
        return
    if not game.tanks[0].alive or not policy.history:
        return
    last_frames[category] = frame
    records.append({
        "category": category,
        "frame": int(frame),
        "frames_to_end": -1 if frames_to_end is None else int(frames_to_end),
        "action": action,
        "stacked": stack_observation(policy.history, policy.frame_stack).copy(),
        "context": np.asarray(getattr(
            policy, "last_context",
            np.zeros(P29C_CONTEXT_DIM, dtype=np.float32))).copy(),
        "game": make_sandbox(game, "L2", rng_seed=rng.randrange(1 << 30)),
    })


def _record_snapshot(policy, game, action, frame, rng):
    if not game.tanks[0].alive or not policy.history:
        return None
    return {
        "category": "terminal_window",
        "frame": int(frame),
        "frames_to_end": -1,
        "action": action,
        "stacked": stack_observation(policy.history, policy.frame_stack).copy(),
        "context": np.asarray(getattr(
            policy, "last_context",
            np.zeros(P29C_CONTEXT_DIM, dtype=np.float32))).copy(),
        "game": make_sandbox(game, "L2", rng_seed=rng.randrange(1 << 30)),
    }


def _teacher_member(game, analyzer, metrics, rng, horizon, samples, args):
    scores, aux = label_actions(
        game,
        analyzer,
        metrics,
        rng.randrange(1 << 30),
        horizon,
        score_samples=samples,
    )
    raw = scores.astype(np.float32) / SCORE_SCALE
    value = (
        raw
        - args.search_death_penalty * aux[:, 1]
        - args.search_dd_penalty * aux[:, 2]
        + args.search_kill_bonus * aux[:, 0]
    ).astype(np.float32)
    return raw, value, aux.astype(np.float32)


def _top_gap(value):
    ordered = np.sort(np.asarray(value, dtype=np.float32))
    if len(ordered) < 2:
        return float("inf")
    return float(ordered[-1] - ordered[-2])


def _consensus_teacher_label(game, analyzer, metrics, rng, args):
    primary = _teacher_member(
        game, analyzer, metrics, rng, args.teacher_horizon,
        args.teacher_samples, args)
    secondary = _teacher_member(
        game, analyzer, metrics, rng, args.teacher_secondary_horizon,
        args.teacher_secondary_samples, args)
    primary_best = int(primary[1].argmax())
    secondary_best = int(secondary[1].argmax())
    primary_gap = _top_gap(primary[1])
    secondary_gap = _top_gap(secondary[1])
    short_consensus = (
        primary_best == secondary_best
        and primary_gap >= args.teacher_consensus_gap
        and secondary_gap >= args.teacher_consensus_gap
    )
    vote_slots = max(2, args.teacher_review_votes)
    votes = np.full(vote_slots, -1, dtype=np.int16)
    votes[:2] = (primary_best, secondary_best)
    if short_consensus:
        return {
            "raw": secondary[0],
            "value": secondary[1],
            "aux": secondary[2],
            "best": secondary_best,
            "action_valid": True,
            "teacher_source": "short_consensus",
            "review_reason": "short_teachers_agree",
            "votes": votes,
            "vote_count": 2,
            "winner_votes": 2,
            "confidence": min(primary_gap, secondary_gap),
            "disagreement": False,
        }

    review_members = []
    review_votes = []
    for _ in range(args.teacher_review_votes):
        member = _teacher_member(
            game, analyzer, metrics, rng, args.teacher_review_horizon,
            args.teacher_review_samples, args)
        review_members.append(member)
        review_votes.append(int(member[1].argmax()))
    if review_votes:
        votes[:len(review_votes)] = review_votes
        counts = Counter(review_votes)
        winner, winner_votes = counts.most_common(1)[0]
        raw = np.mean([item[0] for item in review_members], axis=0).astype(
            np.float32)
        value = np.mean([item[1] for item in review_members], axis=0).astype(
            np.float32)
        aux = np.mean([item[2] for item in review_members], axis=0).astype(
            np.float32)
        review_gap = float(value[winner] - np.max(np.delete(value, winner)))
        action_valid = (
            winner_votes >= args.teacher_review_min_votes
            and review_gap >= args.teacher_review_action_gap
        )
        reason = ("review_consensus" if action_valid
                  else "review_still_ambiguous")
        return {
            "raw": raw,
            "value": value,
            "aux": aux,
            "best": int(winner),
            "action_valid": bool(action_valid),
            "teacher_source": "h96_vote" if action_valid else "risk_only",
            "review_reason": reason,
            "votes": votes,
            "vote_count": len(review_votes),
            "winner_votes": int(winner_votes),
            "confidence": max(0.0, review_gap) * (
                winner_votes / max(1, len(review_votes))),
            "disagreement": True,
        }

    # A state without a decisive teacher action is still useful for learning
    # the risk of keeping the current P27b action.
    return {
        "raw": secondary[0],
        "value": secondary[1],
        "aux": secondary[2],
        "best": secondary_best,
        "action_valid": False,
        "teacher_source": "risk_only",
        "review_reason": "no_review_votes",
        "votes": votes,
        "vote_count": 0,
        "winner_votes": 0,
        "confidence": 0.0,
        "disagreement": True,
    }


def _teacher_label(record, rng, args):
    game = record["game"]
    analyzer = OpportunityAnalyzer360(game)
    metrics = analyzer.metrics(game)
    if args.teacher_label_mode == "consensus":
        teacher = _consensus_teacher_label(
            game, analyzer, metrics, rng, args)
        raw = teacher["raw"]
        value = teacher["value"]
        aux = teacher["aux"]
        best = int(teacher["best"])
        disagreement = bool(teacher["disagreement"])
    else:
        teacher = None
        members = [_teacher_member(
            game, analyzer, metrics, rng, args.teacher_horizon,
            args.teacher_samples, args)]
        disagreement = False
        if args.teacher_secondary_horizon > 0:
            members.append(_teacher_member(
                game, analyzer, metrics, rng,
                args.teacher_secondary_horizon,
                args.teacher_secondary_samples, args))
            first, second = members[0][1], members[1][1]
            disagreement = int(first.argmax()) != int(second.argmax())
            if args.teacher_review_gap > 0.0:
                disagreement = disagreement or min(
                    _top_gap(first), _top_gap(second)) < args.teacher_review_gap
        if disagreement and args.teacher_review_horizon > 0:
            review = _teacher_member(
                game, analyzer, metrics, rng, args.teacher_review_horizon,
                args.teacher_review_samples, args)
            members.extend([review, review])

        raw = np.mean([item[0] for item in members], axis=0).astype(
            np.float32)
        value = np.mean([item[1] for item in members], axis=0).astype(
            np.float32)
        aux = np.mean([item[2] for item in members], axis=0).astype(
            np.float32)
        best = int(value.argmax())
    target_score = value
    if args.advantage_target:
        scale = max(float(np.std(value)), args.advantage_min_scale)
        target_score = np.clip(
            (value - float(np.max(value))) / scale,
            -args.advantage_clip,
            0.0,
        ).astype(np.float32)
    model_input = record["stacked"]
    if args.include_context:
        model_input = np.concatenate((
            np.asarray(model_input, dtype=np.float32),
            np.asarray(record["context"], dtype=np.float32),
        ))
    chosen = CANDIDATES.index(record["action"])
    regret = float(value[best] - value[chosen])
    weight = sample_weight(value * SCORE_SCALE, chosen, aux)
    weight *= CATEGORY_WEIGHT.get(record["category"], 1.0)
    if record["category"] in (
            "direct_shot_loss", "bounce_shot_loss", "self_shot_loss"):
        frames_to_end = max(0, int(record.get("frames_to_end", -1)))
        if 0 <= frames_to_end <= 48:
            weight *= 2.0
        elif frames_to_end <= 96:
            weight *= 1.5
    if regret > 0.03:
        weight *= 1.0 + min(3.0, regret * 5.0)
    return {
        "X": model_input,
        "Y_score": target_score,
        "Y_value": value,
        "Y_aux": aux.astype(np.float32),
        "Y_fire": fire_targets(target_score, args.fire_target_margin),
        "W": float(weight),
        "category": record["category"],
        "regret": regret,
        "chosen": int(chosen),
        "best": int(best),
        "action_valid": bool(
            teacher["action_valid"] if teacher is not None else True),
        "override_target": bool(
            (teacher["action_valid"] if teacher is not None else True)
            and best != chosen and regret >= args.override_min_gain),
        "teacher_source": (
            teacher["teacher_source"] if teacher is not None else "average"),
        "review_reason": (
            teacher["review_reason"] if teacher is not None else
            ("teacher_disagreement" if disagreement else "not_reviewed")),
        "teacher_votes": (
            teacher["votes"] if teacher is not None else
            np.asarray([best, -1], dtype=np.int16)),
        "teacher_vote_count": int(
            teacher["vote_count"] if teacher is not None else 1),
        "teacher_winner_votes": int(
            teacher["winner_votes"] if teacher is not None else 1),
        "teacher_confidence": float(
            teacher["confidence"] if teacher is not None else _top_gap(value)),
        "frame": int(record["frame"]),
        "round_seed": int(record["round_seed"]),
        "frames_to_end": int(record.get("frames_to_end", -1)),
        "raw_score": raw,
        "teacher_disagreement": bool(disagreement),
    }


def _observe_round(policy, seed, args, worker):
    game = Game(seed=seed, ai_enabled=True)
    policy.reset()
    if hasattr(policy, "set_round_seed"):
        policy.set_round_seed(seed)
    analyzer = OpportunityAnalyzer360(game)
    tracker = RoundTracker(game)
    rng = random.Random(seed + worker * 104729 + 29001)
    issues = Counter()
    pos_window = deque(maxlen=args.stall_window)
    input_window = deque(maxlen=args.stall_window)
    pursuit_pos = deque(maxlen=args.pursuit_window)
    clear_fire_frames = 0
    true_result = None
    frames = 0
    records = []
    last_frames = {}
    terminal_ring = deque(maxlen=max(1, args.terminal_window // max(1, args.terminal_stride) + 2))

    while frames < TRUNCATE_FRAMES or tracker.first_destroy is not None:
        me = game.tanks[0]
        enemy = game.tanks[1]
        metrics = analyzer.metrics(game)
        line, reach, risk = [float(value) for value in metrics[:3]]
        inp = policy.act(game)
        cmd = _input_tuple(inp)
        action = _action_tuple(inp)
        pos_window.append((me.x, me.y))
        pursuit_pos.append((me.x, me.y))
        input_window.append(cmd)

        if frames % max(1, args.terminal_stride) == 0:
            snapshot = _record_snapshot(policy, game, action, frames, rng)
            if snapshot:
                terminal_ring.append(snapshot)

        if cmd[4] and not enemy.alive:
            issues["post_kill_fire"] += 1
            _capture(records, last_frames, "post_kill_fire", frames, policy,
                     game, action, rng, args)

        if cmd[4] and enemy.alive:
            shot = _shot_event(game)
            closest = float("inf") if shot is None else shot.get("closest", float("inf"))
            result = None if shot is None else shot.get("result")
            if (line < args.blind_fire_line and result != "HIT"
                    and closest > args.pressure_radius * game.scale):
                issues["blind_fire"] += 1
                _capture(records, last_frames, "blind_fire", frames, policy,
                         game, action, rng, args)

        if (not cmd[4]) and enemy.alive and line >= args.fire_window_line:
            clear_fire_frames += 1
            if clear_fire_frames >= args.fire_window_frames:
                category = "finish_window" if (
                    line >= args.finish_line and risk <= args.finish_max_risk
                ) else "missed_fire_window"
                issues[category] += 1
                _capture(records, last_frames, category, frames, policy,
                         game, action, rng, args)
                clear_fire_frames = 0
        else:
            clear_fire_frames = 0

        if len(pos_window) == args.stall_window:
            dx = pos_window[-1][0] - pos_window[0][0]
            dy = pos_window[-1][1] - pos_window[0][1]
            displacement = math.hypot(dx, dy)
            moving_cmds = sum(any(command[:4]) for command in input_window)
            x, y = _cell(game, me)
            dead_end = _dead_end_penalty(game, x, y)
            exits = _open_neighbors(game, x, y)
            stalled = displacement < args.stall_distance * game.scale
            category = None
            if stalled and (dead_end > 0 or exits <= 1):
                category = "dead_end_stall"
            elif stalled and line < 0.35 and reach < 0.55 and risk < 0.35:
                category = "passive_map_control"
            elif stalled and moving_cmds >= args.stall_window // 4:
                category = "stutter_stall"
            if category:
                issues[category] += 1
                _capture(records, last_frames, category, frames, policy, game,
                         action, rng, args)
                pos_window.clear()
                input_window.clear()

        if len(pursuit_pos) == args.pursuit_window and enemy.alive:
            dx = pursuit_pos[-1][0] - pursuit_pos[0][0]
            dy = pursuit_pos[-1][1] - pursuit_pos[0][1]
            displacement = math.hypot(dx, dy)
            if (line <= args.pursuit_max_line
                    and reach <= args.pursuit_max_reach
                    and risk <= args.pursuit_max_risk
                    and displacement < args.pursuit_distance * game.scale):
                issues["active_pursuit_gap"] += 1
                _capture(records, last_frames, "active_pursuit_gap", frames,
                         policy, game, action, rng, args)
                pursuit_pos.clear()

        if frames >= args.long_frame_start and frames % args.long_stride == 0:
            issues["long_game"] += 1
            _capture(records, last_frames, "long_game", frames, policy, game,
                     action, rng, args)

        if (args.background_stride > 0 and frames > 0
                and frames % args.background_stride == 0):
            _capture(records, last_frames, "background_state", frames,
                     policy, game, action, rng, args)

        me.forward, me.backup = cmd[0], cmd[1]
        me.turn_left, me.turn_right = cmd[2], cmd[3]
        me.fire = cmd[4]
        tracker.pre_step()
        events = game.step()
        frames += 1
        tracker.post_step(events, 1)
        for event in events:
            if event[0] == "round_end":
                true_result = _round_result(event[1])
        if true_result:
            break

    true_result = true_result or "draw"
    terminal_category = None
    if true_result == "loss" and terminal_ring:
        if tracker.death_cause == "laika_direct":
            terminal_category = "direct_shot_loss"
        elif tracker.death_cause == "laika_bounce":
            terminal_category = "bounce_shot_loss"
        elif tracker.death_cause == "self":
            terminal_category = "self_shot_loss"

    if terminal_category:
        for item in terminal_ring:
            copy = dict(item)
            copy["category"] = terminal_category
            copy["frames_to_end"] = frames - int(copy["frame"])
            records.append(copy)
        issues[terminal_category] += len(terminal_ring)
    elif true_result == "draw" and terminal_ring:
        for item in list(terminal_ring)[-args.draw_terminal_records:]:
            copy = dict(item)
            copy["category"] = "long_game"
            copy["frames_to_end"] = frames - int(copy["frame"])
            records.append(copy)
        issues["long_game"] += min(args.draw_terminal_records, len(terminal_ring))

    for record in records:
        record["round_seed"] = int(seed)
    labelled = [_teacher_label(record, rng, args) for record in records]
    result = {
        "seed": int(seed),
        "result": true_result,
        "frames": int(frames),
        "seconds": frames / 25.0,
        "shots": int(tracker.shots),
        "kills": int(tracker.kills),
        "hit_rate": _safe_div(tracker.kills, tracker.shots),
        "move_px": float(tracker.move_px),
        "death_cause": tracker.death_cause,
        "kill_type": tracker.kill_type,
        "issues": dict(issues),
        "labelled": len(labelled),
    }
    return result, labelled


def _worker(job):
    worker, seed, count, args = job
    import torch

    torch.set_num_threads(1)
    policy = _make_policy(args, worker)
    rounds = []
    labels = []
    for offset in range(count):
        result, labelled = _observe_round(policy, seed + offset, args, worker)
        result["worker"] = worker
        rounds.append(result)
        labels.extend(labelled)
    return rounds, labels


def run(args):
    os.makedirs(os.path.join(DATA_DIR, args.phase), exist_ok=True)
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
    print(f"===== P29 collect {args.phase}: {args.n} games @{args.seed} "
          f"policy={args.rollout_policy} teacher=h{args.teacher_horizon}"
          f"s{args.teacher_samples} workers={workers} =====", flush=True)
    if workers == 1:
        outputs = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            outputs = pool.map(_worker, jobs)

    rounds = [item for part, _ in outputs for item in part]
    labels = [item for _, part in outputs for item in part]
    rounds.sort(key=lambda item: item["seed"])
    with open(args.out, "w", encoding="utf-8") as handle:
        for item in rounds:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            print(
                f"  seed={item['seed']} {item['result']} "
                f"shots={item['shots']} hit={item['hit_rate']:.1%} "
                f"labels={item['labelled']} issues={item['issues']}",
                flush=True,
            )

    summary = _write_outputs(args, rounds, labels, started)
    print("===== P29 summary =====", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _write_outputs(args, rounds, labels, started):
    results = Counter(item["result"] for item in rounds)
    issues = Counter()
    shots = kills = frames = 0
    for item in rounds:
        issues.update(item["issues"])
        shots += item["shots"]
        kills += item["kills"]
        frames += item["frames"]

    shard_path = None
    categories = Counter()
    regrets = []
    if labels:
        shard_path = os.path.join(DATA_DIR, args.phase, "shard_0.npz")
        categories.update(item["category"] for item in labels)
        regrets = [item["regret"] for item in labels]
        np.savez_compressed(
            shard_path,
            X=np.asarray([item["X"] for item in labels], np.float32),
            Y_score=np.asarray([item["Y_score"] for item in labels],
                               np.float32),
            Y_value=np.asarray([item["Y_value"] for item in labels],
                               np.float32),
            Y_aux=np.asarray([item["Y_aux"] for item in labels], np.float32),
            Y_fire=np.asarray([item["Y_fire"] for item in labels],
                              np.float32),
            W=np.asarray([item["W"] for item in labels], np.float32),
            category=np.asarray([item["category"] for item in labels]),
            category_names=np.asarray(P29_CATEGORIES),
            regret=np.asarray([item["regret"] for item in labels],
                              np.float32),
            chosen=np.asarray([item["chosen"] for item in labels], np.int32),
            best=np.asarray([item["best"] for item in labels], np.int32),
            frame=np.asarray([item["frame"] for item in labels], np.int32),
            round_seed=np.asarray(
                [item["round_seed"] for item in labels], np.int64),
            frames_to_end=np.asarray(
                [item["frames_to_end"] for item in labels], np.int32),
            raw_score=np.asarray([item["raw_score"] for item in labels],
                                 np.float32),
            teacher_disagreement=np.asarray(
                [item["teacher_disagreement"] for item in labels], np.bool_),
            action_valid=np.asarray(
                [item["action_valid"] for item in labels], np.bool_),
            override_target=np.asarray(
                [item["override_target"] for item in labels], np.bool_),
            teacher_source=np.asarray(
                [item["teacher_source"] for item in labels]),
            review_reason=np.asarray(
                [item["review_reason"] for item in labels]),
            teacher_votes=np.asarray(
                [item["teacher_votes"] for item in labels], np.int16),
            teacher_vote_count=np.asarray(
                [item["teacher_vote_count"] for item in labels], np.int16),
            teacher_winner_votes=np.asarray(
                [item["teacher_winner_votes"] for item in labels], np.int16),
            teacher_confidence=np.asarray(
                [item["teacher_confidence"] for item in labels], np.float32),
            frame_stack=np.asarray([4], np.int32),
            aux_names=np.asarray(AUX_NAMES),
            context_dim=np.asarray([
                P29C_CONTEXT_DIM if args.include_context else 0], np.int32),
            context_names=np.asarray(
                P29C_CONTEXT_NAMES if args.include_context else ()),
            objective_version=np.asarray([args.objective_version]),
        )

    total = max(1, len(rounds))
    summary = {
        "phase": args.phase,
        "rollout_policy": args.rollout_policy,
        "base_net": args.base_net,
        "value_net": args.value_net,
        "n": args.n,
        "seed": args.seed,
        "workers": max(1, min(args.workers, args.n)),
        "teacher_horizon": args.teacher_horizon,
        "teacher_samples": args.teacher_samples,
        "teacher_secondary_horizon": args.teacher_secondary_horizon,
        "teacher_secondary_samples": args.teacher_secondary_samples,
        "teacher_review_horizon": args.teacher_review_horizon,
        "teacher_review_samples": args.teacher_review_samples,
        "teacher_review_gap": args.teacher_review_gap,
        "teacher_label_mode": args.teacher_label_mode,
        "teacher_consensus_gap": args.teacher_consensus_gap,
        "teacher_review_votes": args.teacher_review_votes,
        "teacher_review_min_votes": args.teacher_review_min_votes,
        "teacher_review_action_gap": args.teacher_review_action_gap,
        "advantage_target": args.advantage_target,
        "include_context": args.include_context,
        "objective_version": args.objective_version,
        "search_death_penalty": args.search_death_penalty,
        "search_dd_penalty": args.search_dd_penalty,
        "search_kill_bonus": args.search_kill_bonus,
        "elapsed_seconds": time.time() - started,
        "results": dict(results),
        "win_rate": results["win"] / total,
        "loss_rate": results["loss"] / total,
        "double_death_rate": results["double_death"] / total,
        "draw_rate": results["draw"] / total,
        "shots_per_game": shots / total,
        "hit_rate": kills / max(1, shots),
        "avg_seconds": frames / total / 25.0,
        "issues": dict(issues),
        "labelled_states": len(labels),
        "label_categories": dict(categories),
        "avg_regret": float(np.mean(regrets)) if regrets else 0.0,
        "max_regret": float(np.max(regrets)) if regrets else 0.0,
        "teacher_disagreement_rate": (
            sum(item["teacher_disagreement"] for item in labels)
            / max(1, len(labels))),
        "action_valid_rate": (
            sum(item["action_valid"] for item in labels)
            / max(1, len(labels))),
        "override_target_rate": (
            sum(item["override_target"] for item in labels)
            / max(1, len(labels))),
        "teacher_sources": dict(Counter(
            item["teacher_source"] for item in labels)),
        "round_log": args.out,
        "hard_data_path": shard_path,
    }
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="p29_p28_distill")
    parser.add_argument("--rollout-policy", choices=["p27b", "p28"],
                        default="p27b")
    parser.add_argument("--base-net", default=(
        "training/models/p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--value-net", default=(
        "training/models/p27b_risk_value_iter00.pt"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=973000)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--teacher-horizon", type=int, default=72)
    parser.add_argument("--teacher-samples", type=int, default=2)
    parser.add_argument("--teacher-secondary-horizon", type=int, default=0)
    parser.add_argument("--teacher-secondary-samples", type=int, default=1)
    parser.add_argument("--teacher-review-horizon", type=int, default=0)
    parser.add_argument("--teacher-review-samples", type=int, default=2)
    parser.add_argument("--teacher-review-gap", type=float, default=0.0)
    parser.add_argument("--teacher-label-mode",
                        choices=["average", "consensus"], default="average")
    parser.add_argument("--teacher-consensus-gap", type=float, default=0.08)
    parser.add_argument("--teacher-review-votes", type=int, default=4)
    parser.add_argument("--teacher-review-min-votes", type=int, default=3)
    parser.add_argument("--teacher-review-action-gap", type=float,
                        default=0.05)
    parser.add_argument("--override-min-gain", type=float, default=0.03)
    parser.add_argument("--advantage-target", action="store_true")
    parser.add_argument("--advantage-min-scale", type=float, default=0.05)
    parser.add_argument("--advantage-clip", type=float, default=6.0)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--objective-version", default="p28_value_v1")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--search-horizon", type=int, default=72)
    parser.add_argument("--search-samples", type=int, default=2)
    parser.add_argument("--search-death-penalty", type=float, default=0.35)
    parser.add_argument("--search-dd-penalty", type=float, default=0.75)
    parser.add_argument("--search-kill-bonus", type=float, default=0.04)
    parser.add_argument("--deterministic-search-seeds", action="store_true")
    parser.add_argument("--p27b-assist-margin", type=float, default=0.08)
    parser.add_argument("--p27b-assist-weight", type=float, default=0.35)
    parser.add_argument("--p27b-max-bonus", type=float, default=0.10)
    parser.add_argument("--p27b-kill-weight", type=float, default=0.04)
    parser.add_argument("--p27b-death-weight", type=float, default=0.12)
    parser.add_argument("--p27b-double-death-weight", type=float, default=0.18)
    parser.add_argument("--p27b-survive-weight", type=float, default=0.02)
    parser.add_argument("--p27b-risk-threshold", type=float, default=0.55)
    parser.add_argument("--p27b-fire-delta-margin", type=float, default=0.14)
    parser.add_argument("--max-records-per-round", type=int, default=10)
    parser.add_argument("--min-gap-frames", type=int, default=24)
    parser.add_argument("--stall-window", type=int, default=40)
    parser.add_argument("--stall-distance", type=float, default=0.22)
    parser.add_argument("--fire-window-line", type=float, default=0.70)
    parser.add_argument("--fire-window-frames", type=int, default=8)
    parser.add_argument("--finish-line", type=float, default=0.82)
    parser.add_argument("--finish-max-risk", type=float, default=0.25)
    parser.add_argument("--blind-fire-line", type=float, default=0.35)
    parser.add_argument("--pressure-radius", type=float, default=0.75)
    parser.add_argument("--pursuit-window", type=int, default=60)
    parser.add_argument("--pursuit-distance", type=float, default=0.40)
    parser.add_argument("--pursuit-max-line", type=float, default=0.45)
    parser.add_argument("--pursuit-max-reach", type=float, default=0.55)
    parser.add_argument("--pursuit-max-risk", type=float, default=0.30)
    parser.add_argument("--long-frame-start", type=int, default=625)
    parser.add_argument("--long-stride", type=int, default=125)
    parser.add_argument("--background-stride", type=int, default=0)
    parser.add_argument("--terminal-window", type=int, default=160)
    parser.add_argument("--terminal-stride", type=int, default=4)
    parser.add_argument("--draw-terminal-records", type=int, default=12)
    parser.add_argument("--fire-target-margin", type=float, default=0.16)
    parser.add_argument("--out", default=(
        "training/analysis/runs/p29_p28_distill.jsonl"))
    parser.add_argument("--summary", default=(
        "training/analysis/runs/p29_p28_distill_summary.json"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
