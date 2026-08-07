"""P36 MPC teacher for the exponential chain-coin combat course.

The teacher uses one continuous objective: real coin gain, signed route
progress, combat opportunity, and survival risk are all active at every score.
There is no score-gated firing phase.  A kill is accepted only after a
one-second self-survival window.

This is an upper-bound probe for the rule design, not a deployable policy.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.coin_path_rl import SAFE_SETTLE_FRAMES, CoinPathEnv
from training.coin_path_rules import (
    CHAIN_WINDOW_FRAMES,
    HIT_LOSS,
    WALL_FINE,
    advance_coin_chains,
    collect_coins,
    kill_bonus,
    neighbors,
)
from training.mpc_agent import CANDIDATES, make_sandbox
from training.opportunity_distill import _shot_event
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.survival_expert_iter_530 import apply_action


PREPASS_COIN_GAIN = 180.0
PREPASS_OPPONENT_GAIN = 65.0
PREPASS_ROUTE_PROGRESS = 120.0
MAX_ROUTE_MULTIPLIER = 32.0
WALL_FRAME_PENALTY = 0.0
DEATH_SCORE = -12_000.0
SAFE_SUCCESS_SCORE = 12_000.0
OPPONENT_SELF_SCORE = 2_500.0
POSTPASS_OPPORTUNITY_GAIN = 8.0
POSTPASS_RISK = 400.0
GOOD_FIRE_BONUS = 500.0
PRESSURE_FIRE_BONUS = 120.0
BAD_FIRE_PENALTY = 220.0
SUICIDE_FIRE_PENALTY = 1500.0


def tank_cell(game, tank_index=0):
    tank = game.tanks[tank_index]
    return int(tank.x // game.scale), int(tank.y // game.scale)


def path_distance(game, target):
    """Continuous shortest-path distance to a cell centre."""
    if target is None:
        return 0.0
    tank = game.tanks[0]
    current = tank_cell(game)
    distances = game.dist_map(target[0], target[1])
    if distances is None:
        return 0.0
    steps = distances[current[0]][current[1]]
    if steps is None or steps != steps:
        return 0.0
    if current == target:
        centre_x = (target[0] + 0.5) * game.scale
        centre_y = (target[1] + 0.5) * game.scale
        return math.hypot(tank.x - centre_x, tank.y - centre_y) / game.scale
    candidates = []
    for adjacent in neighbors(game, current):
        value = distances[adjacent[0]][adjacent[1]]
        if value is not None and value == value and value < steps:
            candidates.append((float(value), adjacent[1], adjacent[0]))
    waypoint = target if not candidates else (
        min(candidates)[2], min(candidates)[1])
    centre_x = (waypoint[0] + 0.5) * game.scale
    centre_y = (waypoint[1] + 0.5) * game.scale
    local = math.hypot(tank.x - centre_x, tank.y - centre_y) / game.scale
    return max(float(steps) - 1.0, 0.0) + local


def select_coin_target(game, coins):
    current = tank_cell(game)
    candidates = []
    for cell, value in coins.items():
        distances = game.dist_map(cell[0], cell[1])
        if distances is None:
            continue
        distance = distances[current[0]][current[1]]
        if distance is not None and distance == distance:
            candidates.append((
                (float(distance) + 0.5) / max(value, 1.0),
                -value, cell[1], cell[0]))
    return None if not candidates else (min(candidates)[3], min(candidates)[2])


def simulate_action(env, action, analyzer, rng_seed, horizon=48, hold=16):
    sandbox = make_sandbox(env.game, "L2", rng_seed=rng_seed)
    coins = dict(env.coins)
    banks = list(env.banks)
    counts = list(env.counts)
    picked_values = list(env.picked_values)
    chain_counts = list(env.chain_counts)
    chain_timers = list(env.chain_timers)
    start_banks = list(banks)
    start_chain = chain_counts[0]
    start_chain_timer = chain_timers[0]
    target = env.coin_target if env.coin_target in coins else select_coin_target(
        sandbox, coins)
    start_route = path_distance(sandbox, target)
    start_metrics = analyzer.metrics(sandbox)
    shot = _shot_event(sandbox) if action[2] == 1 else None
    fired = False
    wall_frames = 0
    kill_credit = False
    enemy_death_frame = None
    frame = 0
    finish_frame = min(horizon, max(1, env.cap - env.ledger.frames))
    while frame < finish_frame:
        me = sandbox.tanks[0]
        if frame == 0:
            apply_action(sandbox, action)
        elif frame == hold:
            me.fire = False
        events = sandbox.step()
        fired = fired or any(
            event[0] == "fire" and event[1] == 0 for event in events)
        advance_coin_chains(chain_counts, chain_timers)
        collect_coins(
            sandbox, coins, banks, counts, picked_values,
            chain_counts, chain_timers)
        for index, tank in enumerate(sandbox.tanks):
            if tank.hit_something:
                fine = min(banks[index], WALL_FINE)
                banks[index] -= fine
                if index == 0:
                    wall_frames += 1
        for event in events:
            if event[0] != "hit":
                continue
            attacker, victim = event[1], event[2]
            loss = min(banks[victim], HIT_LOSS)
            banks[victim] -= loss
            if attacker != victim:
                banks[attacker] += kill_bonus(
                    sandbox, sandbox.tanks[attacker], sandbox.tanks[victim])
            if attacker == 0 and victim == 1:
                kill_credit = True
        alive = [tank.alive for tank in sandbox.tanks]
        if not alive[0]:
            return DEATH_SCORE + frame
        if not alive[1] and enemy_death_frame is None:
            enemy_death_frame = frame
            finish_frame = min(
                max(finish_frame, frame + SAFE_SETTLE_FRAMES),
                max(1, env.cap - env.ledger.frames))
        frame += 1

    if enemy_death_frame is not None:
        if kill_credit:
            return SAFE_SUCCESS_SCORE + 20.0 * banks[0] - enemy_death_frame
        return OPPONENT_SELF_SCORE + 5.0 * banks[0] - enemy_death_frame

    own_gain = banks[0] - start_banks[0]
    opponent_gain = banks[1] - start_banks[1]
    score = -WALL_FRAME_PENALTY * wall_frames
    score += PREPASS_COIN_GAIN * own_gain
    score -= PREPASS_OPPONENT_GAIN * opponent_gain
    if target in coins:
        next_multiplier = min(2.0 ** start_chain, MAX_ROUTE_MULTIPLIER)
        if start_chain_timer > 0:
            urgency = 1.0 + 2.0 * (
                1.0 - start_chain_timer / CHAIN_WINDOW_FRAMES)
        else:
            urgency = 1.0
        score += PREPASS_ROUTE_PROGRESS * next_multiplier * urgency * (
            start_route - path_distance(sandbox, target))
    end_metrics = analyzer.metrics(sandbox)
    score += POSTPASS_OPPORTUNITY_GAIN * (
        analyzer.potential(end_metrics) - analyzer.potential(start_metrics))
    score -= POSTPASS_RISK * float(end_metrics[2])
    if fired:
        if shot is None:
            score -= BAD_FIRE_PENALTY
        elif shot["result"] == "HIT" and start_metrics[0] >= 0.60:
            score += GOOD_FIRE_BONUS
        elif shot["result"] == "SUICIDE":
            score -= SUICIDE_FIRE_PENALTY
        elif shot.get("closest", float("inf")) <= 0.75 * sandbox.scale:
            score += PRESSURE_FIRE_BONUS
        else:
            score -= BAD_FIRE_PENALTY
    return score


class CoinPathMPC:
    name = "p35_coin_path_mpc"

    def __init__(self, seed=0, horizon=48, hold=16, samples=1):
        self.rng = random.Random(seed)
        self.horizon = horizon
        self.hold = hold
        self.samples = samples
        self.game = None
        self.analyzer = None

    def reset(self):
        self.game = None
        self.analyzer = None

    def scores(self, env):
        if env.game is not self.game:
            self.game = env.game
            self.analyzer = OpportunityAnalyzer360(env.game)
        seeds = [self.rng.randrange(1 << 30) for _ in range(self.samples)]
        values = np.empty(len(CANDIDATES), dtype=np.float32)
        for index, action in enumerate(CANDIDATES):
            values[index] = np.mean([
                simulate_action(
                    env, action, self.analyzer, seed,
                    self.horizon, self.hold)
                for seed in seeds
            ])
        return values

    def act_index(self, env):
        return int(np.argmax(self.scores(env)))


class ChainCoinMPC(CoinPathMPC):
    """Named P36 viewer/training preset for the chain-coin rules."""

    name = "P36 三秒指数连吃 MPC 老师"


def run_round(seed, horizon, hold, samples, cap_seconds):
    env = CoinPathEnv(seed, cap_seconds)
    env.reset()
    teacher = CoinPathMPC(seed ^ 0x35C0FFEE, horizon, hold, samples)
    decisions = 0
    while True:
        action = teacher.act_index(env)
        _, _, done, info = env.step(action)
        decisions += 1
        if done:
            return {**info, "seed": seed, "decisions": decisions}


def summarize(rounds):
    count = len(rounds)
    return {
        "rounds": count,
        "course_success_pct": sum(
            item["course_success"] for item in rounds) / count,
        "qualified_pct": sum(item["qualified"] for item in rounds) / count,
        "safe_kill_pct": sum(item["safe_kill"] for item in rounds) / count,
        "double_death_pct": sum(
            item["double_death"] for item in rounds) / count,
        "mean_bank": statistics.mean(item["bank"] for item in rounds),
        "median_bank": statistics.median(item["bank"] for item in rounds),
        "mean_cells": statistics.mean(
            item["unique_cells"] for item in rounds),
        "mean_shots": statistics.mean(item["shots"] for item in rounds),
        "wasted_shot_pct": sum(item["wasted_shots"] for item in rounds)
        / max(sum(item["shots"] for item in rounds), 1),
        "mean_seconds": statistics.mean(
            item["frames"] for item in rounds) / 25.0,
        "ends": {
            end: sum(item["end"] == end for item in rounds)
            for end in ("kill", "opponent_self", "death", "double", "cap")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "eval"])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=35_500_001)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--hold", type=int, default=16)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--cap-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.command == "smoke":
        env = CoinPathEnv(args.seed, args.cap_seconds)
        env.reset()
        teacher = CoinPathMPC(args.seed, args.horizon, args.hold, args.samples)
        scores = teacher.scores(env)
        best = int(np.argmax(scores))
        print(json.dumps({
            "scores": scores.tolist(),
            "best_index": best,
            "best_action": CANDIDATES[best],
            "bank": env.banks[0],
            "target": env.coin_target,
        }, ensure_ascii=False))
        return
    started = time.time()
    rounds = []
    for index in range(args.n):
        result = run_round(
            args.seed + index, args.horizon, args.hold,
            args.samples, args.cap_seconds)
        rounds.append(result)
        print(
            f"[{index + 1}/{args.n}] {result['end']} "
            f"bank={result['bank']:.0f} cells={result['unique_cells']} "
            f"shots={result['shots']} {time.time() - started:.0f}s",
            flush=True)
    output = summarize(rounds)
    output["elapsed_seconds"] = time.time() - started
    output["seeds"] = [args.seed, args.seed + args.n - 1]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
