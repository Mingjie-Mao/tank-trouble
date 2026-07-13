"""
P25v2 机会老师：360 度可信炮线扫描。

坦克物理朝向共有 32 档，因此每个状态扫描 32 条全周假想弹道。炮线势能同时
考虑最佳弹道本身的质量和炮口与该弹道的对准程度：背后存在反弹杀线时可以先
发现机会，但只有持续转向该角度才会获得完整势能。

用法：

  python3 training/opportunity_teacher_v2.py teacher-eval --n 20
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.opportunity_distill import (
    FIRE_DISTANCE_CAP, FIRE_POSITION_WEIGHT, HOLD, HORIZON,
    LINE_WEIGHT, RISK_WEIGHT, OpportunityAnalyzer, opportunity_rollout)

N_HEADINGS = 32
ALIGNMENT_FLOOR = 0.25


class OpportunityAnalyzer360(OpportunityAnalyzer):
    """扫描 32 个物理朝向，并给出最佳炮线及其相对炮口方向。"""

    def best_line(self, game):
        from tank_trouble_original import constants as constants
        from training.tt_gym_env import (
            HIT_RADIUS_SCALE, SHOT_SIM_FRAMES, _reflective_closest_batch)

        me, enemy = game.tanks[0], game.tanks[1]
        if not (me.alive and enemy.alive):
            return 0.0, 0.0
        angles = np.arange(N_HEADINGS, dtype=np.float64) * (
            2.0 * math.pi / N_HEADINGS)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        spawn_distance = game.scale * 4.5 / 16.0
        origins = np.stack([
            me.x + directions[:, 0] * spawn_distance,
            me.y + directions[:, 1] * spawn_distance,
        ], axis=1)
        speed = constants.BULLETSPEED * (game.scale / 50.0)
        result = _reflective_closest_batch(
            origins, directions, np.full(N_HEADINGS, speed),
            np.full(N_HEADINGS, float(SHOT_SIM_FRAMES)), 2,
            self.boxes, enemy.x, enemy.y)
        hit = result[:, 0] <= HIT_RADIUS_SCALE * game.scale
        if not hit.any():
            return 0.0, 0.0
        time_quality = 1.0 - np.minimum(
            result[:, 1] / SHOT_SIM_FRAMES, 1.0)
        bounce_quality = 1.0 - 0.10 * np.minimum(result[:, 2], 2.0)
        qualities = np.where(
            hit, (0.65 + 0.35 * time_quality) * bounce_quality, 0.0)
        best = int(np.argmax(qualities))
        return float(qualities[best]), float(angles[best])

    def line_quality(self, game):
        return self.best_line(game)[0]

    def metrics(self, game):
        me = game.tanks[0]
        line, best_angle = self.best_line(game)
        distance, target = self.nearest_firing_position(game)
        reach = 1.0 - min(distance, FIRE_DISTANCE_CAP) / FIRE_DISTANCE_CAP
        reach = max(reach, line)
        risk = self.incoming_risk(game)
        if line > 0.0:
            forward = (me.rotation - 90.0) * math.pi / 180.0
            delta = math.atan2(math.sin(best_angle - forward),
                               math.cos(best_angle - forward))
            direction = (math.cos(delta), math.sin(delta))
        else:
            direction = self._next_direction(game, target)
        return np.asarray([line, reach, risk, direction[0], direction[1]],
                          dtype=np.float32)

    def potential(self, metrics):
        line, reach, risk = [float(value) for value in metrics[:3]]
        direction_x, direction_y = [float(value) for value in metrics[3:5]]
        angle_error = abs(math.atan2(direction_y, direction_x)) if line > 0 else 0.0
        alignment = 1.0 - min(angle_error / math.pi, 1.0)
        ready_line = line * (
            ALIGNMENT_FLOOR + (1.0 - ALIGNMENT_FLOOR) * alignment)
        safety = 1.0 - 0.60 * risk
        return (LINE_WEIGHT * ready_line * safety
                + FIRE_POSITION_WEIGHT * reach * safety
                - RISK_WEIGHT * risk)


class OpportunityMPC360:
    name = "p25v2_opportunity_mpc_360"

    def __init__(self, seed=0, horizon=HORIZON, hold=HOLD):
        from training.mpc_agent import CANDIDATES
        self.candidates = CANDIDATES
        self.rng = random.Random(seed)
        self.horizon = horizon
        self.hold = hold
        self.game = None
        self.analyzer = None

    def reset(self):
        self.game = None
        self.analyzer = None

    def act(self, game):
        from training.mpc_agent import make_sandbox

        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer360(game)
        start = self.analyzer.metrics(game)
        step_seed = self.rng.randrange(1 << 30)
        scores = []
        for action in self.candidates:
            sandbox = make_sandbox(game, "L2", rng_seed=step_seed)
            scores.append(opportunity_rollout(
                sandbox, action, self.analyzer, start,
                self.hold, self.horizon))
        throttle, turn, fire = self.candidates[int(np.argmax(scores))]
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["teacher-eval"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    evaluate_teacher(args.n, args.seed, args.workers)


def _eval_worker(job):
    worker, seed, count = job
    from training.evaluate import play_round_dual_engine

    policy = OpportunityMPC360(seed=worker * 104729 + 131)
    return [play_round_dual_engine(policy, seed + index)
            for index in range(count)]


def evaluate_teacher(n, seed, workers):
    base, remainder = divmod(n, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((worker, seed + offset, count))
            offset += count
    started = __import__("time").time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rounds = [item for part in pool.map(_eval_worker, jobs)
                  for item in part]
    total = len(rounds)
    count = lambda key: sum(result["true_result"] == key
                            for result in rounds)
    deaths = [result["death_cause"] for result in rounds
              if result["death_cause"]]
    kills = [result["kill_type"] for result in rounds
             if result["kill_type"]]
    shots = sum(result["shots"] for result in rounds)
    elapsed = __import__("time").time() - started
    print(f"===== P25v2 360度老师 原版验收 {total}局 @{seed} "
          f"({elapsed:.0f}s) =====")
    print(f"  真胜率 {count('win')/total:.1%}  "
          f"负 {count('loss')/total:.1%}  "
          f"双亡 {count('double_death')/total:.1%}  "
          f"平 {count('draw')/total:.1%}")
    print(f"  自杀 {sum(cause == 'self' for cause in deaths)/total:.1%}  "
          f"场均开火 {shots/total:.1f}  "
          f"命中率 {sum(result['kills'] for result in rounds)/max(shots, 1):.1%}")
    print(f"  场均移动 {sum(result['move_cells'] for result in rounds)/total:.1f}格  "
          f"平均局长 {sum(result['frames'] for result in rounds)/total/25:.1f}s")
    if kills:
        print(f"  击杀方式: 直射 {sum(kind == 'direct' for kind in kills)/len(kills):.1%}  "
              f"反弹 {sum(kind == 'bounce' for kind in kills)/len(kills):.1%}  "
              f"未知 {sum(kind == 'unknown' for kind in kills)/len(kills):.1%}")
    return rounds


if __name__ == "__main__":
    main()
