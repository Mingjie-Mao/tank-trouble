"""
Failure attribution for P26 policies.

Runs the deployed network policy in the real scoring engine. For loss and
double-death rounds, it labels only the final few seconds with the MPC teacher,
then summarizes where the policy diverged from teacher actions.
"""

import argparse
import json
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
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    AUX_NAMES,
    SCORE_SCALE,
    P26Policy,
    fire_targets,
    label_actions,
    sample_weight,
    stack_observation,
)
from training.tt_gym_env import TRUNCATE_FRAMES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "p26_amortized_data")
ANALYSIS_DIR = os.path.join(HERE, "analysis", "runs")
CATEGORY_NAMES = (
    "teacher_close",
    "fire_into_double_death",
    "unsafe_fire_death",
    "double_death_risk",
    "unsafe_movement",
    "missed_fire_window",
    "waste_or_unsafe_fire",
    "missed_kill_line",
    "movement_value_gap",
)


def _action_tuple(inp):
    return (
        2 if inp.get("forward") else 0 if inp.get("backup") else 1,
        0 if inp.get("turn_left") else 2 if inp.get("turn_right") else 1,
        1 if inp.get("fire") else 0,
    )


def _classify(chosen, best, scores, aux, regret):
    chosen_fire = chosen % 2 == 1
    best_fire = best % 2 == 1
    chosen_aux = aux[chosen]
    best_aux = aux[best]

    if regret < 0.03:
        return "teacher_close"
    if chosen_fire and chosen_aux[2] > 0.5:
        return "fire_into_double_death"
    if chosen_fire and chosen_aux[1] > 0.5 and best_aux[1] < 0.5:
        return "unsafe_fire_death"
    if chosen_aux[2] > 0.5 and best_aux[2] < 0.5:
        return "double_death_risk"
    if chosen_aux[1] > 0.5 and best_aux[1] < 0.5:
        return "unsafe_movement"
    if (not chosen_fire) and best_fire:
        return "missed_fire_window"
    if chosen_fire and not best_fire:
        return "waste_or_unsafe_fire"
    if chosen_aux[0] < 0.5 and best_aux[0] > 0.5:
        return "missed_kill_line"
    return "movement_value_gap"


def _label_record(record, score_horizon, fire_target_margin, score_samples,
                  rng):
    game = record["game"]
    analyzer = OpportunityAnalyzer360(game)
    metrics = analyzer.metrics(game)
    scores, aux = label_actions(
        game, analyzer, metrics, rng.randrange(1 << 30), score_horizon,
        score_samples=score_samples)
    chosen = CANDIDATES.index(record["action"])
    best = int(scores.argmax())
    regret_raw = float(scores[best] - scores[chosen])
    regret = regret_raw / SCORE_SCALE
    category = _classify(chosen, best, scores, aux, regret)
    y_score = scores / SCORE_SCALE
    y_fire = fire_targets(y_score, fire_target_margin)
    weight = sample_weight(scores, chosen, aux)
    if category != "teacher_close":
        weight *= 2.0 + min(4.0, max(0.0, regret) * 8.0)
    return {
        "frame": record["frame"],
        "frames_to_end": record["frames_to_end"],
        "chosen": chosen,
        "best": best,
        "chosen_action": CANDIDATES[chosen],
        "best_action": CANDIDATES[best],
        "regret": regret,
        "category": category,
        "chosen_aux": chosen_aux_dict(aux[chosen]),
        "best_aux": chosen_aux_dict(aux[best]),
        "chosen_score": float(y_score[chosen]),
        "best_score": float(y_score[best]),
        "chosen_fire": bool(chosen % 2),
        "best_fire": bool(best % 2),
        "X": record["stacked"],
        "Y_score": y_score.astype(np.float32),
        "Y_aux": aux.astype(np.float32),
        "Y_fire": y_fire.astype(np.float32),
        "W": float(weight),
    }


def chosen_aux_dict(values):
    return {name: float(value) for name, value in zip(AUX_NAMES, values)}


def _empty_stats():
    return {
        "rounds": 0,
        "results": Counter(),
        "death_causes": Counter(),
        "kill_types": Counter(),
        "states": 0,
        "state_categories": Counter(),
        "root_categories": Counter(),
        "chosen_fire": 0,
        "best_fire": 0,
        "fire_disagreement": 0,
        "regret_sum": 0.0,
        "regret_max": 0.0,
        "shots": 0,
        "kills": 0,
        "frames": 0,
    }


def _merge_stats(target, source):
    for key in ("rounds", "states", "chosen_fire", "best_fire",
                "fire_disagreement", "shots", "kills", "frames"):
        target[key] += source[key]
    target["regret_sum"] += source["regret_sum"]
    target["regret_max"] = max(target["regret_max"], source["regret_max"])
    for key in ("results", "death_causes", "kill_types", "state_categories",
                "root_categories"):
        target[key].update(source[key])


def _worker(job):
    (worker, net_path, p27b_net, seed0, count, fire_margin, window_frames,
     stride, score_horizon, fire_target_margin, score_samples, max_cases,
     hard_phase) = job
    import torch
    torch.set_num_threads(1)

    rng = random.Random(seed0 + worker * 104729)
    if p27b_net:
        from training.p27_risk_value import P27BRiskValuePolicy
        policy = P27BRiskValuePolicy(
            base_net=net_path,
            value_net=p27b_net,
            fire_margin=fire_margin,
        )
    else:
        policy = P26Policy(
            net_path=net_path,
            fire_margin=fire_margin,
            fire_threshold=0.0,
            kill_weight=0.0,
            death_weight=0.0,
            double_death_weight=0.0,
            survive_weight=0.0,
            fire_prob_weight=0.0,
        )
    stats = _empty_stats()
    cases = []
    hard_x, hard_score, hard_aux, hard_fire, hard_w = [], [], [], [], []
    hard_category, hard_regret, hard_chosen, hard_best = [], [], [], []
    hard_result, hard_frame, hard_frames_to_end = [], [], []

    max_records = max(1, window_frames // max(1, stride) + 2)
    for offset in range(count):
        seed = seed0 + offset
        game = Game(seed=seed, ai_enabled=True)
        policy.reset()
        tracker = RoundTracker(game)
        ring = deque(maxlen=max_records)
        true_result = None
        frames = 0

        while frames < TRUNCATE_FRAMES or tracker.first_destroy is not None:
            inp = policy.act(game)
            action = _action_tuple(inp)
            if frames % stride == 0 and game.tanks[0].alive:
                ring.append({
                    "frame": frames,
                    "action": action,
                    "stacked": stack_observation(
                        policy.history, policy.frame_stack).copy(),
                    "game": make_sandbox(
                        game, "L2", rng_seed=rng.randrange(1 << 30)),
                })

            t0 = game.tanks[0]
            t0.forward = bool(inp.get("forward", False))
            t0.backup = bool(inp.get("backup", False))
            t0.turn_left = bool(inp.get("turn_left", False))
            t0.turn_right = bool(inp.get("turn_right", False))
            t0.fire = bool(inp.get("fire", False))
            tracker.pre_step()
            events = game.step()
            frames += 1
            tracker.post_step(events, 1)
            for event in events:
                if event[0] == "round_end":
                    winner = event[1]
                    true_result = ("win" if winner == 0 else
                                   "loss" if winner == 1 else "double_death")
            if true_result:
                break

        true_result = true_result or "draw"
        stats["rounds"] += 1
        stats["results"][true_result] += 1
        stats["shots"] += tracker.shots
        stats["kills"] += tracker.kills
        stats["frames"] += frames
        if tracker.death_cause:
            stats["death_causes"][tracker.death_cause] += 1
        if tracker.kill_type:
            stats["kill_types"][tracker.kill_type] += 1

        if true_result not in ("loss", "double_death"):
            continue

        labelled = []
        for record in ring:
            record["frames_to_end"] = frames - record["frame"]
            labelled.append(_label_record(
                record, score_horizon, fire_target_margin, score_samples,
                rng))
        if not labelled:
            continue

        root = max(labelled, key=lambda item: item["regret"])
        stats["root_categories"][root["category"]] += 1
        for item in labelled:
            stats["states"] += 1
            stats["state_categories"][item["category"]] += 1
            stats["regret_sum"] += item["regret"]
            stats["regret_max"] = max(stats["regret_max"], item["regret"])
            stats["chosen_fire"] += int(item["chosen_fire"])
            stats["best_fire"] += int(item["best_fire"])
            stats["fire_disagreement"] += int(
                item["chosen_fire"] != item["best_fire"])
            if item["regret"] >= 0.03:
                hard_x.append(item["X"])
                hard_score.append(item["Y_score"])
                hard_aux.append(item["Y_aux"])
                hard_fire.append(item["Y_fire"])
                hard_w.append(item["W"])
                hard_category.append(item["category"])
                hard_regret.append(item["regret"])
                hard_chosen.append(item["chosen"])
                hard_best.append(item["best"])
                hard_result.append(true_result)
                hard_frame.append(item["frame"])
                hard_frames_to_end.append(item["frames_to_end"])

        cases.append({
            "seed": seed,
            "result": true_result,
            "frames": frames,
            "death_cause": tracker.death_cause,
            "shots": tracker.shots,
            "kills": tracker.kills,
            "root_category": root["category"],
            "root_regret": root["regret"],
            "root_frame": root["frame"],
            "root_frames_to_end": root["frames_to_end"],
            "chosen_action": root["chosen_action"],
            "best_action": root["best_action"],
            "chosen_score": root["chosen_score"],
            "best_score": root["best_score"],
            "chosen_aux": root["chosen_aux"],
            "best_aux": root["best_aux"],
        })
        cases = sorted(cases, key=lambda item: item["root_regret"],
                       reverse=True)[:max_cases]

    hard_path = None
    if hard_phase and hard_x:
        phase_dir = os.path.join(DATA_DIR, hard_phase)
        os.makedirs(phase_dir, exist_ok=True)
        hard_path = os.path.join(phase_dir, f"shard_{worker}.npz")
        np.savez_compressed(
            hard_path,
            X=np.asarray(hard_x, np.float32),
            Y_score=np.asarray(hard_score, np.float32),
            Y_aux=np.asarray(hard_aux, np.float32),
            Y_fire=np.asarray(hard_fire, np.float32),
            W=np.asarray(hard_w, np.float32),
            category=np.asarray(hard_category),
            category_names=np.asarray(CATEGORY_NAMES),
            regret=np.asarray(hard_regret, np.float32),
            chosen=np.asarray(hard_chosen, np.int32),
            best=np.asarray(hard_best, np.int32),
            result=np.asarray(hard_result),
            frame=np.asarray(hard_frame, np.int32),
            frames_to_end=np.asarray(hard_frames_to_end, np.int32),
            frame_stack=np.asarray([policy.frame_stack], np.int32),
            aux_names=np.asarray(AUX_NAMES),
        )

    return stats, cases, hard_path


def _jsonable_stats(stats):
    total = max(1, stats["rounds"])
    states = max(1, stats["states"])
    return {
        "rounds": stats["rounds"],
        "results": dict(stats["results"]),
        "win_rate": stats["results"]["win"] / total,
        "loss_rate": stats["results"]["loss"] / total,
        "double_death_rate": stats["results"]["double_death"] / total,
        "draw_rate": stats["results"]["draw"] / total,
        "shots_per_game": stats["shots"] / total,
        "hit_rate": stats["kills"] / max(1, stats["shots"]),
        "avg_seconds": stats["frames"] / total / 25.0,
        "death_causes": dict(stats["death_causes"]),
        "kill_types": dict(stats["kill_types"]),
        "labelled_failure_states": stats["states"],
        "avg_regret": stats["regret_sum"] / states,
        "max_regret": stats["regret_max"],
        "chosen_fire_rate": stats["chosen_fire"] / states,
        "teacher_fire_rate": stats["best_fire"] / states,
        "fire_disagreement_rate": stats["fire_disagreement"] / states,
        "state_categories": dict(stats["state_categories"]),
        "root_categories": dict(stats["root_categories"]),
    }


def run(args):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    base, remainder = divmod(args.n, args.workers)
    jobs, offset = [], 0
    for worker in range(args.workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((worker, args.net, args.p27b_net,
                         args.seed + offset, count, args.fire_margin,
                         args.window_frames, args.stride, args.score_horizon,
                         args.fire_target_margin, args.score_samples,
                         args.max_cases,
                         args.hard_phase))
            offset += count

    started = time.time()
    print(f"===== P26 failure attribution {args.n} games @{args.seed} "
          f"margin={args.fire_margin:g} =====", flush=True)
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        outputs = pool.map(_worker, jobs)

    stats = _empty_stats()
    cases, hard_paths = [], []
    for worker_stats, worker_cases, hard_path in outputs:
        _merge_stats(stats, worker_stats)
        cases.extend(worker_cases)
        if hard_path:
            hard_paths.append(hard_path)
    cases = sorted(cases, key=lambda item: item["root_regret"],
                   reverse=True)[:args.max_cases]

    payload = {
        "net": args.net,
        "p27b_net": args.p27b_net,
        "seed": args.seed,
        "n": args.n,
        "fire_margin": args.fire_margin,
        "window_frames": args.window_frames,
        "stride": args.stride,
        "score_horizon": args.score_horizon,
        "score_samples": args.score_samples,
        "elapsed_seconds": time.time() - started,
        "stats": _jsonable_stats(stats),
        "top_cases": cases,
        "hard_data_paths": hard_paths,
    }
    out_path = args.out
    if not out_path:
        name = (f"p26_failure_attribution_{os.path.basename(args.net)}_"
                f"{args.n}_{args.seed}.json").replace(".pt", "")
        out_path = os.path.join(ANALYSIS_DIR, name)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    s = payload["stats"]
    print(f"  true win {s['win_rate']:.1%}  loss {s['loss_rate']:.1%}  "
          f"double death {s['double_death_rate']:.1%}  draw {s['draw_rate']:.1%}",
          flush=True)
    print(f"  shots/game {s['shots_per_game']:.1f}  hit {s['hit_rate']:.1%}  "
          f"avg length {s['avg_seconds']:.1f}s", flush=True)
    print(f"  labelled failure states {s['labelled_failure_states']}  "
          f"avg regret {s['avg_regret']:.3f}  max regret {s['max_regret']:.3f}",
          flush=True)
    print(f"  root categories {s['root_categories']}", flush=True)
    print(f"  state categories {s['state_categories']}", flush=True)
    print(f"  hard shards {len(hard_paths)}", flush=True)
    print(f"saved {out_path}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="training/models/p26_amortized_mpc_iter05.pt")
    parser.add_argument("--p27b-net", default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--window-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--score-horizon", type=int, default=48)
    parser.add_argument("--score-samples", type=int, default=1)
    parser.add_argument("--fire-target-margin", type=float, default=0.16)
    parser.add_argument("--hard-phase", default=None)
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
