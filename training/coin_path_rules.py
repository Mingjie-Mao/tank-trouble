"""P34: one-shot path coins and Laika baseline calibration."""

import argparse
import json
import math
import os
import statistics
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.laika import LaikaAI
from tank_trouble_original.maze import h_open, v_open


FPS = 25
COIN_RADIUS = 0.35
BASE_COIN_VALUE = 1.0
CHAIN_WINDOW_SECONDS = 3.0
CHAIN_WINDOW_FRAMES = int(CHAIN_WINDOW_SECONDS * FPS)
WALL_FINE = 0.0
HIT_LOSS = 25.0
BASE_KILL_BONUS = 10.0
CLOSE_KILL_DISTANCE = 6.0
CLOSE_KILL_STEP = 5.0


def tank_cell(game, tank):
    return int(tank.x // game.scale), int(tank.y // game.scale)


def neighbors(game, cell):
    x, y = cell
    width, height = len(game.maze), len(game.maze[0])
    result = []
    if x > 0 and v_open(game.maze, x, y):
        result.append((x - 1, y))
    if x < width - 1 and v_open(game.maze, x + 1, y):
        result.append((x + 1, y))
    if y > 0 and h_open(game.maze, x, y - 1):
        result.append((x, y - 1))
    if y < height - 1 and h_open(game.maze, x, y):
        result.append((x, y + 1))
    return result


def shortest_path(game, start, goal):
    queue = deque([start])
    previous = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for adjacent in neighbors(game, cell):
            if adjacent not in previous:
                previous[adjacent] = cell
                queue.append(adjacent)
    if goal not in previous:
        return [start]
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def path_distances(game, path):
    distances = {cell: 0 for cell in path}
    queue = deque(path)
    while queue:
        cell = queue.popleft()
        for adjacent in neighbors(game, cell):
            if adjacent not in distances:
                distances[adjacent] = distances[cell] + 1
                queue.append(adjacent)
    return distances


def build_coins(game):
    start = tank_cell(game, game.tanks[0])
    goal = tank_cell(game, game.tanks[1])
    path = shortest_path(game, start, goal)
    coins = {}
    for item in game.reachable:
        cell = (int(item["x"]), int(item["y"]))
        coins[cell] = BASE_COIN_VALUE
    return coins, path


def maze_distance(game, first, second):
    distances = game.dist_map(second[0], second[1])
    if distances is None:
        return CLOSE_KILL_DISTANCE
    value = distances[first[0]][first[1]]
    return CLOSE_KILL_DISTANCE if value is None or value != value else float(value)


def kill_bonus(game, attacker, victim):
    distance = maze_distance(
        game, tank_cell(game, attacker), tank_cell(game, victim))
    return BASE_KILL_BONUS + CLOSE_KILL_STEP * max(
        0.0, CLOSE_KILL_DISTANCE - distance)


def advance_coin_chains(chain_counts, chain_timers):
    """Advance the public per-player combo timers by one game frame."""
    for index in range(len(chain_timers)):
        if chain_timers[index] <= 0:
            chain_counts[index] = 0
            chain_timers[index] = 0
            continue
        chain_timers[index] -= 1
        if chain_timers[index] == 0:
            chain_counts[index] = 0


def collect_coins(game, coins, banks, counts, picked_values,
                  chain_counts=None, chain_timers=None, max_chains=None):
    radius = COIN_RADIUS * game.scale
    for cell, value in list(coins.items()):
        center_x = (cell[0] + 0.5) * game.scale
        center_y = (cell[1] + 0.5) * game.scale
        candidates = []
        for index, tank in enumerate(game.tanks):
            if tank.alive:
                distance = math.hypot(tank.x - center_x, tank.y - center_y)
                if distance <= radius:
                    candidates.append((distance, index))
        if candidates:
            _, winner = min(candidates)
            multiplier = 1.0
            if chain_counts is not None and chain_timers is not None:
                multiplier = 2.0 ** chain_counts[winner]
                chain_counts[winner] += 1
                chain_timers[winner] = CHAIN_WINDOW_FRAMES
                if max_chains is not None:
                    max_chains[winner] = max(
                        max_chains[winner], chain_counts[winner])
            earned = value * multiplier
            banks[winner] += earned
            counts[winner] += 1
            picked_values[winner] += earned
            del coins[cell]


def run_round(seed, seconds=30, wall_mode="fine", combat=False):
    from tank_trouble_original.game import Game

    invincible = set() if combat else {0, 1}
    game = Game(seed=seed, ai_enabled=True, invincible=invincible,
                hit_immunity_frames={0: 0, 1: 0})
    game.tanks[0].ai = LaikaAI(game, game.tanks[0])
    coins, path = build_coins(game)
    initial_pool = sum(coins.values())
    banks = [0.0, 0.0]
    counts = [0, 0]
    picked_values = [0.0, 0.0]
    chain_counts = [0, 0]
    chain_timers = [0, 0]
    max_chains = [0, 0]
    collisions = [0, 0]
    wall_fines = [0.0, 0.0]
    wall_death = None
    hits = [0, 0]
    kill_earned = [0.0, 0.0]
    hit_lost = [0.0, 0.0]
    collect_coins(
        game, coins, banks, counts, picked_values,
        chain_counts, chain_timers, max_chains)
    frames = 0
    for frame in range(seconds * FPS):
        events = game.step()
        frames = frame + 1
        advance_coin_chains(chain_counts, chain_timers)
        collect_coins(
            game, coins, banks, counts, picked_values,
            chain_counts, chain_timers, max_chains)
        for index, tank in enumerate(game.tanks):
            if tank.hit_something:
                collisions[index] += 1
                if wall_mode == "death":
                    banks[index] = 0.0
                    wall_death = index
                else:
                    fine = min(banks[index], WALL_FINE)
                    banks[index] -= fine
                    wall_fines[index] += fine
        if combat:
            for event in events:
                if event[0] != "hit":
                    continue
                attacker, victim = event[1], event[2]
                loss = min(banks[victim], HIT_LOSS)
                banks[victim] -= loss
                hit_lost[victim] += loss
                hits[attacker] += 1
                if attacker != victim:
                    bonus = kill_bonus(
                        game, game.tanks[attacker], game.tanks[victim])
                    banks[attacker] += bonus
                    kill_earned[attacker] += bonus
        if wall_death is not None:
            break
        if combat and sum(tank.alive for tank in game.tanks) <= 1:
            break
    return {
        "seed": seed,
        "frames": frames,
        "path_length": len(path),
        "initial_pool": initial_pool,
        "remaining_pool": sum(coins.values()),
        "banks": banks,
        "counts": counts,
        "picked_values": picked_values,
        "max_chains": max_chains,
        "collisions": collisions,
        "wall_fines": wall_fines,
        "wall_death": wall_death,
        "hits": hits,
        "kill_earned": kill_earned,
        "hit_lost": hit_lost,
    }


def percentile(values, level):
    return float(np.percentile(np.asarray(values, dtype=np.float64), level))


def summarize(rounds, wall_mode, seconds, combat):
    individual = [score for result in rounds for score in result["banks"]]
    totals = [sum(result["banks"]) for result in rounds]
    collected = [
        result["initial_pool"] - result["remaining_pool"] for result in rounds]
    collision_frames = [
        value for result in rounds for value in result["collisions"]]
    wall_deaths = sum(result["wall_death"] is not None for result in rounds)
    max_chains = [
        value for result in rounds for value in result["max_chains"]]
    return {
        "rounds": len(rounds),
        "seconds": seconds,
        "wall_mode": wall_mode,
        "combat": combat,
        "individual_coin_mean": statistics.mean(individual),
        "individual_coin_median": statistics.median(individual),
        "individual_coin_p25": percentile(individual, 25),
        "individual_coin_p75": percentile(individual, 75),
        "individual_coin_max": max(individual),
        "recommended_pass_score": percentile(individual, 75),
        "shared_collected_mean": statistics.mean(collected),
        "shared_bank_mean": statistics.mean(totals),
        "picked_coin_mean_per_agent": statistics.mean(
            value for result in rounds for value in result["picked_values"]),
        "wall_fine_mean_per_agent": statistics.mean(
            value for result in rounds for value in result["wall_fines"]),
        "kill_bonus_mean_per_agent": statistics.mean(
            value for result in rounds for value in result["kill_earned"]),
        "hit_loss_mean_per_agent": statistics.mean(
            value for result in rounds for value in result["hit_lost"]),
        "hit_events_mean_per_agent": statistics.mean(
            value for result in rounds for value in result["hits"]),
        "mean_frames": statistics.mean(result["frames"] for result in rounds),
        "collision_frames_per_agent": statistics.mean(collision_frames),
        "wall_death_round_pct": wall_deaths / max(len(rounds), 1),
        "mean_path_length": statistics.mean(
            result["path_length"] for result in rounds),
        "mean_map_coin_pool": statistics.mean(
            result["initial_pool"] for result in rounds),
        "mean_max_chain": statistics.mean(max_chains),
        "p75_max_chain": percentile(max_chains, 75),
        "maximum_chain": max(max_chains),
    }


def measure(args, wall_mode):
    rounds = [run_round(
        args.seed + index, args.seconds, wall_mode, args.combat)
        for index in range(args.n)]
    return summarize(rounds, wall_mode, args.seconds, args.combat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["measure", "compare"])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=34_000_000)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--wall-mode", choices=["fine", "death"],
                        default="fine")
    parser.add_argument("--combat", action="store_true")
    args = parser.parse_args()
    if args.command == "compare":
        result = [measure(args, mode) for mode in ("fine", "death")]
    else:
        result = measure(args, args.wall_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
