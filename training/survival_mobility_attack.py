"""P33: decisive attack teacher constrained by the hard mobility law."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.mpc_agent import CANDIDATES, make_sandbox
from training.survival_decisive_attack import (
    BREAK_RISK,
    COMMIT_MOVE_FRAMES,
    COMMIT_TURN_FRAMES,
    DecisiveAttackPolicy,
    _world_direction,
    decisive_rollout,
)
from training.survival_mobility_law import (
    MAX_CELL_FRAMES,
    MobilityLawLedger,
    tank_cell,
)


class MobilityAttackPolicy(DecisiveAttackPolicy):
    name = "P33 碰墙即死、同格超时即死的进攻老师"

    def reset(self):
        super().reset()
        self.masked_candidates = 0
        self.no_valid_decisions = 0
        self.cell_exit_events = 0

    def _cell_exit_action(self, game, metrics, cell_frames):
        direction = _world_direction(game.tanks[0], metrics)
        step_seed = self.rng.randrange(1 << 30)
        scores = np.full(9, -1e9, dtype=np.float64)
        remaining = max(1, MAX_CELL_FRAMES - cell_frames - 1)
        for movement, (throttle, turn) in enumerate(
                (item for item in ((throttle, turn)
                                   for throttle in (0, 1, 2)
                                   for turn in (0, 1, 2)))):
            if throttle == 1 and turn == 1:
                continue
            mid = make_sandbox(game, "L2", rng_seed=step_seed)
            tank = mid.tanks[0]
            start_x, start_y = tank.x, tank.y
            start_cell = tank_cell(mid)
            tank.forward, tank.backup = throttle == 2, throttle == 0
            tank.turn_left, tank.turn_right = turn == 0, turn == 2
            tank.fire = False
            first_frames = COMMIT_MOVE_FRAMES + 1 if throttle != 1 \
                else COMMIT_TURN_FRAMES + 1
            first_frames = min(first_frames, remaining)
            crossed_at = None
            legal = True
            for frame in range(first_frames):
                mid.step()
                if tank.hit_something or not tank.alive:
                    legal = False
                    break
                if tank_cell(mid) != start_cell:
                    crossed_at = frame
                    break
            if not legal:
                continue
            if crossed_at is None and remaining > first_frames:
                best_second = None
                for second_throttle in (0, 2):
                    for second_turn in (0, 1, 2):
                        sandbox = make_sandbox(mid, "L2", rng_seed=step_seed)
                        second = sandbox.tanks[0]
                        second.forward = second_throttle == 2
                        second.backup = second_throttle == 0
                        second.turn_left = second_turn == 0
                        second.turn_right = second_turn == 2
                        second.fire = False
                        for second_frame in range(remaining - first_frames):
                            sandbox.step()
                            if second.hit_something or not second.alive:
                                break
                            if tank_cell(sandbox) != start_cell:
                                total = first_frames + second_frame
                                best_second = total if best_second is None \
                                    else min(best_second, total)
                                break
                crossed_at = best_second
            delta_x, delta_y = tank.x - start_x, tank.y - start_y
            travel = np.hypot(delta_x, delta_y) / game.scale
            progress = 0.0 if direction is None else (
                delta_x * direction[0] + delta_y * direction[1]) \
                / game.scale
            scores[movement] = (
                (1000.0 - crossed_at if crossed_at is not None else 0.0)
                + 10.0 * travel + 3.0 * progress)
        return CANDIDATES[int(np.argmax(scores)) * 2]

    def _law_survival(self, game, action, cell_frames, step_seed):
        sandbox = make_sandbox(game, "L2", rng_seed=step_seed)
        tank = sandbox.tanks[0]
        enemy = sandbox.tanks[1]
        throttle, turn, fire = action
        tank.forward, tank.backup = throttle == 2, throttle == 0
        tank.turn_left, tank.turn_right = turn == 0, turn == 2
        tank.fire = fire == 1
        current_cell = tank_cell(sandbox)
        age = cell_frames
        action_frames = 1 if fire == 1 else \
            COMMIT_MOVE_FRAMES + 1 if throttle != 1 else \
            COMMIT_TURN_FRAMES + 1 if turn != 1 else 1
        crossed = False
        for frame in range(action_frames):
            if frame == self.hold:
                tank.fire = False
            sandbox.step()
            if tank.hit_something or not tank.alive:
                return False, frame, crossed
            if not enemy.alive:
                return True, action_frames, crossed
            new_cell = tank_cell(sandbox)
            if new_cell == current_cell:
                age += 1
            else:
                current_cell = new_cell
                age = 0
                crossed = True
            if age >= MAX_CELL_FRAMES:
                return False, frame, crossed
        return True, action_frames, crossed

    def _choose_lawful_action(self, game, metrics, cell_frames):
        step_seed = self.rng.randrange(1 << 30)
        scores = np.full(len(CANDIDATES), -1e9, dtype=np.float64)
        survival = np.zeros(len(CANDIDATES), dtype=np.int32)
        for index, candidate in enumerate(CANDIDATES):
            valid, survived, crossed = self._law_survival(
                game, candidate, cell_frames, step_seed)
            survival[index] = survived
            if not valid:
                self.masked_candidates += 1
                continue
            sandbox = make_sandbox(game, "L2", rng_seed=step_seed)
            scores[index] = decisive_rollout(
                sandbox, candidate, self.analyzer, metrics,
                self.horizon, self.hold)
            urgency = cell_frames / MAX_CELL_FRAMES
            scores[index] += urgency * (
                30.0 if crossed else 8.0 if candidate[0] != 1 else -12.0)
        if np.isfinite(scores).any() and np.max(scores) > -1e8:
            return CANDIDATES[int(np.argmax(scores))]
        self.no_valid_decisions += 1
        return CANDIDATES[int(np.argmax(survival))]

    def act_ctx(self, game, ledger):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            from training.opportunity_teacher_v2 import OpportunityAnalyzer360
            self.analyzer = OpportunityAnalyzer360(game)
            self.commit_remaining = 0
            self.committed_action = (1, 1, 0)
            self.route_target = None
            self.route_lock_remaining = 0
        metrics = self._route_metrics(game)
        if ledger.cell_frames >= MAX_CELL_FRAMES // 2:
            action = self._cell_exit_action(
                game, metrics, ledger.cell_frames)
            self.committed_action = action
            self.commit_remaining = 0
            self.cell_exit_events += 1
        elif self._urgent_fire(game):
            action = (1, 1, 1)
            self.commit_remaining = 0
        elif self.commit_remaining > 0 and float(metrics[2]) < BREAK_RISK:
            action = self.committed_action
            self.commit_remaining -= 1
        else:
            action = self._choose_lawful_action(
                game, metrics, ledger.cell_frames)
            if action[2] == 0:
                self.committed_action = action
                self.commit_remaining = (
                    COMMIT_MOVE_FRAMES if action[0] != 1
                    else COMMIT_TURN_FRAMES if action[1] != 1 else 0)
        self._record(game, action)
        return self._dict(action)

    def telemetry(self):
        result = super().telemetry()
        result.update({
            "masked_candidates": self.masked_candidates,
            "no_valid_decisions": self.no_valid_decisions,
            "cell_exit_events": self.cell_exit_events,
        })
        return result


def run_course(policy, seed):
    from tank_trouble_original.game import Game

    game = Game(seed=seed, ai_enabled=True, invincible={1},
                hit_immunity_frames={1: 0})
    ledger = MobilityLawLedger(game, policy.econ)
    shots = 0
    end = "alive"
    while end == "alive":
        inputs = policy.act_ctx(game, ledger)
        tank = game.tanks[0]
        tank.forward = bool(inputs.get("forward", False))
        tank.backup = bool(inputs.get("backup", False))
        tank.turn_left = bool(inputs.get("turn_left", False))
        tank.turn_right = bool(inputs.get("turn_right", False))
        tank.fire = bool(inputs.get("fire", False))
        events = game.step()
        shots += sum(event[0] == "fire" and event[1] == 0
                     for event in events)
        end = ledger.on_frame(game, events)
    return {
        "end": end,
        "mobility_death": ledger.mobility_death,
        "frames": ledger.frames,
        "hits": ledger.hits,
        "shots": shots,
        "cells": ledger.cells,
        "cell_frames": ledger.cell_frames,
        **policy.telemetry(),
    }


def probe_command(args):
    policy = MobilityAttackPolicy(
        horizon=args.horizon, hold=args.hold, seed=args.planner_seed)
    print(json.dumps(run_course(policy, args.seed), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["probe"])
    parser.add_argument("--seed", type=int, default=30_000_001)
    parser.add_argument("--planner-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--hold", type=int, default=8)
    args = parser.parse_args()
    probe_command(args)


if __name__ == "__main__":
    main()
