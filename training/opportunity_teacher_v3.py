"""
P25v3 反演击杀场老师：跑图、瞄准和开火共享同一个 KillPlan。

从 Laika 当前可见位置反向追踪 32 条有限弹道，最多两次反弹；反演只负责
提出候选射击格，所有候选再用同一套正向反射几何验证并绑定真实炮口朝向。
老师离机会较远时 MPC 向预计击杀成本最低的格子移动，接近后转向绑定角度；
当前炮口经原版 Laika 弹道检查确认能命中时立即尝试开火。

用法：

  python3 training/opportunity_teacher_v3.py teacher-eval --n 20
"""

import argparse
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.opportunity_distill import (
    FIRE_DISTANCE_CAP, HOLD, HORIZON, MOVE_OPTIONS, OpportunityAnalyzer,
    _cell, _finite_distance, opportunity_rollout)
from training.opportunity_teacher_v2 import N_HEADINGS, OpportunityAnalyzer360


REVERSE_HORIZON = 75
MAX_BOUNCES = 2
PREALIGN_CELLS = 2.0
LINE_WEIGHT_V3 = 60.0
REACH_WEIGHT_V3 = 45.0
RISK_WEIGHT_V3 = 25.0


@dataclass(frozen=True)
class KillCandidate:
    cell: tuple
    aim_angle: float
    hit_frame: float
    bounce_count: int
    confidence: float
    clearance: float
    shot_quality: float
    source: str = "inverse"


@dataclass(frozen=True)
class KillPlan:
    candidate: KillCandidate
    bfs_distance: float
    next_direction: tuple
    aim_error: float
    incoming_risk: float
    expected_cost: float


def _angle_delta(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


class InverseKillFieldAnalyzer(OpportunityAnalyzer360):
    """以反演候选为唯一导航锚，并用正向几何绑定射击角。"""

    def _reverse_candidate_cells(self, game):
        from tank_trouble_original import constants as constants

        enemy = game.tanks[1]
        speed = constants.BULLETSPEED * (game.scale / 50.0)
        spawn = game.scale * 4.5 / 16.0
        width, height = len(game.maze), len(game.maze[0])
        reachable = {
            (item["x"], item["y"]) for item in game.reachable
        }
        cells = set()
        for heading in range(N_HEADINGS):
            angle = heading * 2.0 * math.pi / N_HEADINGS
            dx, dy = math.cos(angle), math.sin(angle)
            x, y = enemy.x, enemy.y
            bounces = 0
            for _ in range(REVERSE_HORIZON):
                previous_x, previous_y = x, y
                x += dx * speed
                y += dy * speed
                if game.wall_hit(x, y):
                    hit_x_invert = game.wall_hit(
                        previous_x - dx * speed, previous_y + dy * speed)
                    hit_y_invert = game.wall_hit(
                        previous_x + dx * speed, previous_y - dy * speed)
                    if hit_x_invert and not hit_y_invert:
                        dy = -dy
                    elif hit_y_invert and not hit_x_invert:
                        dx = -dx
                    else:
                        dx, dy = -dx, -dy
                    bounces += 1
                    if bounces > MAX_BOUNCES:
                        break
                    x = previous_x + dx * speed
                    y = previous_y + dy * speed

                # 反向局部方向为 r，正向发射方向为 -r，因此坦克中心约为
                # 反演轨迹点 + r * spawn。邻格一并交给正向几何消歧。
                center_x, center_y = x + dx * spawn, y + dy * spawn
                base_x = int(center_x // game.scale)
                base_y = int(center_y // game.scale)
                for offset_x in (-1, 0, 1):
                    for offset_y in (-1, 0, 1):
                        cell = (base_x + offset_x, base_y + offset_y)
                        if (0 <= cell[0] < width and 0 <= cell[1] < height
                                and cell in reachable):
                            cells.add(cell)
        cells.add(_cell(game, game.tanks[0]))
        return cells

    def _validate_cells(self, game, cells):
        from tank_trouble_original import constants as constants
        from training.tt_gym_env import (
            HIT_RADIUS_SCALE, SHOT_SIM_FRAMES, _reflective_closest_batch)

        enemy = game.tanks[1]
        cells = sorted(cells)
        if not cells:
            return []
        angles = np.arange(N_HEADINGS, dtype=np.float64) * (
            2.0 * math.pi / N_HEADINGS)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        centers = np.asarray([
            ((cell[0] + 0.5) * game.scale,
             (cell[1] + 0.5) * game.scale)
            for cell in cells
        ], dtype=np.float64)
        tiled_directions = np.tile(directions, (len(cells), 1))
        repeated_centers = np.repeat(centers, N_HEADINGS, axis=0)
        spawn = game.scale * 4.5 / 16.0
        origins = repeated_centers + tiled_directions * spawn
        speed = constants.BULLETSPEED * (game.scale / 50.0)
        result = _reflective_closest_batch(
            origins, tiled_directions,
            np.full(len(origins), speed),
            np.full(len(origins), float(SHOT_SIM_FRAMES)),
            MAX_BOUNCES, self.boxes, enemy.x, enemy.y)
        hit_radius = HIT_RADIUS_SCALE * game.scale
        hit = result[:, 0] <= hit_radius
        if not hit.any():
            return []

        time_quality = 1.0 - np.minimum(
            result[:, 1] / SHOT_SIM_FRAMES, 1.0)
        bounce_quality = 1.0 - 0.10 * np.minimum(result[:, 2], 2.0)
        clearance = np.clip(1.0 - result[:, 0] / hit_radius, 0.0, 1.0)
        qualities = ((0.65 + 0.35 * time_quality) * bounce_quality
                     * (0.85 + 0.15 * clearance))
        candidates = []
        for cell_index, cell in enumerate(cells):
            begin = cell_index * N_HEADINGS
            end = begin + N_HEADINGS
            local = np.where(hit[begin:end], qualities[begin:end], -1.0)
            heading = int(np.argmax(local))
            index = begin + heading
            if local[heading] < 0.0:
                continue
            candidates.append(KillCandidate(
                cell=cell,
                aim_angle=float(angles[heading]),
                hit_frame=float(result[index, 1]),
                bounce_count=int(result[index, 2]),
                confidence=1.0,
                clearance=float(clearance[index]),
                shot_quality=float(qualities[index]),
            ))
        return candidates

    def _fallback_candidate(self, game):
        me, enemy = game.tanks[0], game.tanks[1]
        _, cell = self.nearest_firing_position(game)
        x = (cell[0] + 0.5) * game.scale
        y = (cell[1] + 0.5) * game.scale
        angle = math.atan2(enemy.y - y, enemy.x - x)
        distance = math.hypot(enemy.x - x, enemy.y - y)
        from tank_trouble_original import constants as constants
        speed = constants.BULLETSPEED * (game.scale / 50.0)
        return KillCandidate(
            cell=cell, aim_angle=angle,
            hit_frame=min(REVERSE_HORIZON, distance / max(speed, 1e-9)),
            bounce_count=0, confidence=0.5, clearance=0.0,
            shot_quality=0.50, source="direct_fallback")

    def best_plan(self, game):
        me = game.tanks[0]
        me_cell = _cell(game, me)
        forward = (me.rotation - 90.0) * math.pi / 180.0
        current_quality, current_angle = self.best_line(game)
        if current_quality > 0.0:
            candidates = [KillCandidate(
                cell=me_cell, aim_angle=current_angle,
                hit_frame=0.0, bounce_count=0, confidence=1.0,
                clearance=1.0, shot_quality=current_quality,
                source="current_pose")]
        else:
            cells = self._reverse_candidate_cells(game)
            candidates = self._validate_cells(game, cells)
        if not candidates:
            candidates = [self._fallback_candidate(game)]

        risk = self.incoming_risk(game)
        plans = []
        for candidate in candidates:
            distance = _finite_distance(game, me_cell, candidate.cell)
            if distance is None:
                continue
            aim_error = _angle_delta(candidate.aim_angle, forward)
            turn_cost = abs(aim_error) / math.pi / (1.0 + distance)
            expected_cost = (
                distance
                + 0.60 * candidate.hit_frame / REVERSE_HORIZON
                + 0.25 * candidate.bounce_count
                + 0.40 * (1.0 - candidate.shot_quality)
                + 0.30 * turn_cost
                + 0.75 * risk)
            plans.append(KillPlan(
                candidate=candidate,
                bfs_distance=float(distance),
                next_direction=self._next_direction(game, candidate.cell),
                aim_error=aim_error,
                incoming_risk=risk,
                expected_cost=expected_cost,
            ))
        if not plans:
            fallback = self._fallback_candidate(game)
            return KillPlan(
                candidate=fallback, bfs_distance=FIRE_DISTANCE_CAP,
                next_direction=self._next_direction(game, fallback.cell),
                aim_error=_angle_delta(fallback.aim_angle, forward),
                incoming_risk=risk, expected_cost=FIRE_DISTANCE_CAP)
        return min(plans, key=lambda plan: plan.expected_cost)

    def verified_hit_now(self, game):
        from tank_trouble_original.laika import LaikaAI

        me = game.tanks[0]
        if not (me.alive and game.tanks[1].alive
                and me.trigger_released and game.weapon_ready(me)):
            return None
        result = LaikaAI(game, me).check_bullet_path(me.rotation)
        return result if result["result"] == "HIT" else None

    def metrics(self, game):
        plan = self.best_plan(game)
        quality = plan.candidate.shot_quality
        distance = plan.bfs_distance
        reach = quality * (
            1.0 - min(distance, FIRE_DISTANCE_CAP) / FIRE_DISTANCE_CAP)
        proximity = 1.0 - min(distance, PREALIGN_CELLS) / PREALIGN_CELLS
        alignment = 1.0 - min(abs(plan.aim_error) / math.pi, 1.0)
        ready_line = quality * proximity * (0.20 + 0.80 * alignment)
        if distance <= 0.0:
            direction = (math.cos(plan.aim_error), math.sin(plan.aim_error))
        else:
            direction = plan.next_direction
        return np.asarray([
            ready_line, reach, plan.incoming_risk,
            direction[0], direction[1]], dtype=np.float32)

    def potential(self, metrics):
        ready_line, reach, risk = [float(value) for value in metrics[:3]]
        safety = 1.0 - 0.60 * risk
        return (LINE_WEIGHT_V3 * ready_line * safety
                + REACH_WEIGHT_V3 * reach * safety
                - RISK_WEIGHT_V3 * risk)


class OpportunityMPCInverse:
    name = "p25v3_inverse_killfield_mpc"

    def __init__(self, seed=0, horizon=HORIZON, hold=HOLD):
        self.rng = random.Random(seed)
        self.horizon = horizon
        self.hold = hold
        self.game = None
        self.analyzer = None
        self.verified_opportunities = 0
        self.forced_shots = 0

    def reset(self):
        self.game = None
        self.analyzer = None

    def act(self, game):
        from training.mpc_agent import make_sandbox

        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = InverseKillFieldAnalyzer(game)

        if self.analyzer.verified_hit_now(game) is not None:
            self.verified_opportunities += 1
            self.forced_shots += 1
            return {"forward": False, "backup": False,
                    "turn_left": False, "turn_right": False,
                    "fire": True}

        start = self.analyzer.metrics(game)
        step_seed = self.rng.randrange(1 << 30)
        scores = []
        for throttle, turn in MOVE_OPTIONS:
            sandbox = make_sandbox(game, "L2", rng_seed=step_seed)
            scores.append(opportunity_rollout(
                sandbox, (throttle, turn, 0), self.analyzer, start,
                self.hold, self.horizon))
        throttle, turn = MOVE_OPTIONS[int(np.argmax(scores))]
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": False}


def _eval_worker(job):
    worker, seed, count = job
    from training.evaluate import play_round_dual_engine

    policy = OpportunityMPCInverse(seed=worker * 104729 + 313)
    rounds = [
        play_round_dual_engine(policy, seed + index)
        for index in range(count)
    ]
    return rounds, policy.verified_opportunities, policy.forced_shots


def evaluate_teacher(n, seed, workers):
    base, remainder = divmod(n, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count:
            jobs.append((worker, seed + offset, count))
            offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_eval_worker, jobs)
    rounds = [round_result for part, _, _ in parts for round_result in part]
    opportunities = sum(part[1] for part in parts)
    forced_shots = sum(part[2] for part in parts)
    total = len(rounds)
    count = lambda key: sum(
        result["true_result"] == key for result in rounds)
    shots = sum(result["shots"] for result in rounds)
    print(f"===== P25v3 反演击杀场老师 原版验收 {total}局 @{seed} "
          f"({time.time() - started:.0f}s) =====")
    print(f"  真胜率 {count('win') / total:.1%}  "
          f"负 {count('loss') / total:.1%}  "
          f"双亡 {count('double_death') / total:.1%}  "
          f"平 {count('draw') / total:.1%}")
    print(f"  场均开火 {shots / total:.1f}  "
          f"命中率 {sum(result['kills'] for result in rounds) / max(shots, 1):.1%}  "
          f"平均局长 {sum(result['frames'] for result in rounds) / total / 25:.1f}s")
    print(f"  物理确认机会 {opportunities}  强制尝试 {forced_shots}  "
          f"转化 {forced_shots / max(opportunities, 1):.1%}")
    return rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["teacher-eval"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=984000)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()
    evaluate_teacher(args.n, args.seed, args.workers)


if __name__ == "__main__":
    main()
