"""P32: decisive kill-focused opportunity MPC.

P31 attacks, but replanning every frame makes it visibly indecisive.  P32 adds
continuous waypoint progress, explicit idle and wasted-shot costs, urgent fire
on a predicted hit, and a short real-action commitment window.
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.mpc_agent import CANDIDATES, make_sandbox
from training.opportunity_distill import opportunity_rollout
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.survival_distill_v2 import legacy_econ


FPS = 25
HORIZON = 24
HOLD = 8
COMMIT_MOVE_FRAMES = 5
COMMIT_TURN_FRAMES = 3
PROGRESS_WEIGHT = 24.0
ALIGNMENT_WEIGHT = 10.0
ADVANCE_BONUS = 3.0
IDLE_PENALTY = 10.0
WASTED_FIRE_PENALTY = 80.0
SUICIDE_FIRE_PENALTY = 500.0
URGENT_FIRE_BONUS = 24.0
READY_LINE = 0.55
BREAK_RISK = 0.65
ROUTE_LOCK_FRAMES = 2 * FPS
BLOCKED_MOVE_PENALTY = 45.0
BLOCKED_TURN_PENALTY = 35.0
ESCAPE_HORIZON = 24
ESCAPE_COMMIT_FRAMES = 3


def _world_direction(tank, metrics):
    local_x, local_y = [float(value) for value in metrics[3:5]]
    if math.hypot(local_x, local_y) < 1e-6:
        return None
    forward = (tank.rotation - 90.0) * math.pi / 180.0
    return (
        local_x * math.cos(forward) - local_y * math.sin(forward),
        local_x * math.sin(forward) + local_y * math.cos(forward),
    )


def decisive_rollout(sandbox, action, analyzer, start_metrics,
                     horizon=HORIZON, hold=HOLD):
    from tank_trouble_original.laika import LaikaAI

    tank = sandbox.tanks[0]
    start_x, start_y = tank.x, tank.y
    start_rotation = tank.rotation
    start_heading = (tank.rotation - 90.0) * math.pi / 180.0
    direction = _world_direction(tank, start_metrics)
    shot = None
    if action[2] == 1 and tank.trigger_released \
            and sandbox.weapon_ready(tank):
        shot = LaikaAI(sandbox, tank).check_bullet_path(tank.rotation)
    score = opportunity_rollout(
        sandbox, action, analyzer, start_metrics, hold, horizon)
    if abs(score) >= 900.0:
        return score

    if direction is not None and tank.alive:
        world_x, world_y = direction
        progress = (
            (tank.x - start_x) * world_x
            + (tank.y - start_y) * world_y) / sandbox.scale
        target_heading = math.atan2(world_y, world_x)
        end_heading = (tank.rotation - 90.0) * math.pi / 180.0
        start_alignment = math.cos(target_heading - start_heading)
        end_alignment = math.cos(target_heading - end_heading)
        score += PROGRESS_WEIGHT * progress
        score += ALIGNMENT_WEIGHT * (end_alignment - start_alignment)

    throttle, turn, fire = action
    travel = math.hypot(tank.x - start_x, tank.y - start_y) / sandbox.scale
    rotation_change = abs((tank.rotation - start_rotation + 180.0) % 360.0 - 180.0)
    if throttle != 1 and travel < 0.15:
        score = min(score - BLOCKED_MOVE_PENALTY, -800.0)
    if turn != 1 and rotation_change < 5.0:
        score = min(score - BLOCKED_TURN_PENALTY, -750.0)
    if tank.hit_something:
        score = min(score - 0.5 * BLOCKED_MOVE_PENALTY, -650.0)
    if float(start_metrics[0]) < READY_LINE:
        if throttle != 1:
            score += ADVANCE_BONUS
        elif turn == 1 and fire == 0:
            score -= IDLE_PENALTY
    if fire == 1:
        if shot is not None and shot.get("result") == "HIT":
            score += URGENT_FIRE_BONUS
        elif shot is not None and shot.get("result") == "SUICIDE":
            score -= SUICIDE_FIRE_PENALTY
        else:
            score -= WASTED_FIRE_PENALTY
    return score


class DecisiveAttackPolicy:
    name = "P32 果断击杀 MPC"

    def __init__(self, horizon=HORIZON, hold=HOLD, seed=0):
        self.horizon = horizon
        self.hold = hold
        self.rng = random.Random(seed)
        self.econ = dict(legacy_econ(), cap=12 * FPS, start=80.0)
        self.reset()

    def reset(self):
        self.game = None
        self.analyzer = None
        self.commit_remaining = 0
        self.committed_action = (1, 1, 0)
        self.last_action = None
        self.frames = 0
        self.moving_frames = 0
        self.idle_frames = 0
        self.turning_frames = 0
        self.fire_frames = 0
        self.action_switches = 0
        self.collision_frames = 0
        self.route_target = None
        self.route_lock_remaining = 0
        self.route_switches = 0
        self.collision_streak = 0
        self.escape_events = 0

    def _record(self, game, action):
        if self.last_action is not None and action != self.last_action:
            self.action_switches += 1
        self.last_action = action
        self.frames += 1
        self.moving_frames += int(action[0] != 1)
        self.turning_frames += int(action[1] != 1)
        self.fire_frames += int(action[2] == 1)
        self.idle_frames += int(action == (1, 1, 0))
        self.collision_frames += int(game.tanks[0].hit_something)

    @staticmethod
    def _dict(action):
        throttle, turn, fire = action
        return {
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": fire == 1,
        }

    def _urgent_fire(self, game):
        from tank_trouble_original.laika import LaikaAI

        tank = game.tanks[0]
        if not (tank.trigger_released and game.weapon_ready(tank)):
            return False
        shot = LaikaAI(game, tank).check_bullet_path(tank.rotation)
        return shot.get("result") == "HIT"

    def _escape_action(self, game, metrics):
        direction = _world_direction(game.tanks[0], metrics)
        step_seed = self.rng.randrange(1 << 30)
        scores = np.full(9, -1e9, dtype=np.float32)
        for movement, (throttle, turn) in enumerate(
                (item for item in ((throttle, turn)
                                   for throttle in (0, 1, 2)
                                   for turn in (0, 1, 2)))):
            if throttle == 1:
                continue
            sandbox = make_sandbox(game, "L2", rng_seed=step_seed)
            tank = sandbox.tanks[0]
            start_x, start_y = tank.x, tank.y
            tank.forward, tank.backup = throttle == 2, throttle == 0
            tank.turn_left, tank.turn_right = turn == 0, turn == 2
            tank.fire = False
            collisions = 0
            for _ in range(ESCAPE_HORIZON):
                sandbox.step()
                collisions += int(tank.hit_something)
                if not tank.alive:
                    break
            if not tank.alive:
                continue
            delta_x, delta_y = tank.x - start_x, tank.y - start_y
            travel = math.hypot(delta_x, delta_y) / game.scale
            progress = 0.0 if direction is None else (
                delta_x * direction[0] + delta_y * direction[1]) \
                / game.scale
            scores[movement] = (
                8.0 * travel + 2.0 * progress - 5.0 * collisions)
        return CANDIDATES[int(np.argmax(scores)) * 2]

    def _route_metrics(self, game):
        metrics = self.analyzer.metrics(game)
        if float(metrics[0]) >= READY_LINE:
            return metrics
        tank = game.tanks[0]
        current = (int(tank.x // game.scale), int(tank.y // game.scale))
        refresh = self.route_target is None \
            or self.route_lock_remaining <= 0 \
            or current == self.route_target
        if refresh:
            _, target = self.analyzer.nearest_firing_position(game)
            if target != self.route_target:
                self.route_switches += 1
            self.route_target = target
            self.route_lock_remaining = ROUTE_LOCK_FRAMES
        else:
            self.route_lock_remaining -= 1
        direction = self.analyzer._next_direction(game, self.route_target)
        routed = metrics.copy()
        routed[3], routed[4] = direction
        return routed

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer360(game)
            self.commit_remaining = 0
            self.committed_action = (1, 1, 0)
            self.route_target = None
            self.route_lock_remaining = 0
            self.collision_streak = 0

        metrics = self._route_metrics(game)
        if game.tanks[0].hit_something:
            self.collision_streak += 1
        else:
            self.collision_streak = 0
        if self._urgent_fire(game):
            action = (1, 1, 1)
            self.commit_remaining = 0
        elif self.collision_streak >= 2:
            action = self._escape_action(game, metrics)
            self.committed_action = action
            self.commit_remaining = ESCAPE_COMMIT_FRAMES
            self.collision_streak = 0
            self.escape_events += 1
        elif self.commit_remaining > 0 and float(metrics[2]) < BREAK_RISK \
                and not game.tanks[0].hit_something:
            action = self.committed_action
            self.commit_remaining -= 1
        else:
            step_seed = self.rng.randrange(1 << 30)
            scores = []
            for candidate in CANDIDATES:
                sandbox = make_sandbox(game, "L2", rng_seed=step_seed)
                scores.append(decisive_rollout(
                    sandbox, candidate, self.analyzer, metrics,
                    self.horizon, self.hold))
            action = CANDIDATES[int(np.argmax(scores))]
            if action[2] == 0:
                self.committed_action = action
                self.commit_remaining = (
                    COMMIT_MOVE_FRAMES if action[0] != 1
                    else COMMIT_TURN_FRAMES if action[1] != 1 else 0)
        self._record(game, action)
        return self._dict(action)

    def telemetry(self):
        frames = max(self.frames, 1)
        return {
            "moving_pct": self.moving_frames / frames,
            "idle_pct": self.idle_frames / frames,
            "turning_pct": self.turning_frames / frames,
            "fire_frames": self.fire_frames,
            "collision_pct": self.collision_frames / frames,
            "route_switches": self.route_switches,
            "escape_events": self.escape_events,
            "action_switches_per_second": self.action_switches
            / (frames / FPS),
        }


def probe_command(args):
    from training.evaluate import play_round_dual_engine

    policy = DecisiveAttackPolicy(
        horizon=args.horizon, hold=args.hold, seed=args.planner_seed)
    result = play_round_dual_engine(policy, args.seed)
    result.update(policy.telemetry())
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["probe"])
    parser.add_argument("--seed", type=int, default=30_000_001)
    parser.add_argument("--planner-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--hold", type=int, default=HOLD)
    args = parser.parse_args()
    probe_command(args)


if __name__ == "__main__":
    main()
