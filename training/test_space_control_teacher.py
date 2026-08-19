import unittest

from tank_trouble_original import Game
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.space_control_teacher import (
    evaluate_space_control_pair,
    space_control_score,
)


class SpaceControlTeacherTest(unittest.TestCase):
    def _row(self, **updates):
        row = {
            "allowed": True,
            "fired": True,
            "agent_kill": False,
            "enemy_unique_cells": 5,
            "enemy_min_exits": 2,
            "enemy_end_dead_end_depth": 0.0,
        }
        row.update(updates)
        return row

    def test_unsafe_fire_never_receives_pressure_credit(self):
        no_fire = self._row(enemy_unique_cells=5)
        fire = self._row(
            allowed=False, agent_kill=True, enemy_unique_cells=1)
        self.assertEqual(space_control_score(no_fire, fire), 0.0)

    def test_kill_or_route_denial_receives_credit(self):
        no_fire = self._row(enemy_unique_cells=5, enemy_min_exits=3)
        self.assertEqual(
            space_control_score(no_fire, self._row(agent_kill=True)), 1.0)
        pressure = self._row(
            enemy_unique_cells=3,
            enemy_min_exits=2,
            enemy_end_dead_end_depth=1.0,
        )
        self.assertGreaterEqual(space_control_score(no_fire, pressure), 0.2)

    def test_exact_pair_does_not_mutate_live_game(self):
        game = Game(seed=42, ai_enabled=True)
        analyzer = OpportunityAnalyzer360(game)
        metrics = analyzer.metrics(game)
        before = (
            game.frame, game.tanks[0].x, game.tanks[0].y,
            len(game.bullets), game.rng.getstate(),
        )
        result = evaluate_space_control_pair(
            game, analyzer, metrics, movement_index=4,
            score_horizon=24, fire_tail_horizon=40)
        after = (
            game.frame, game.tanks[0].x, game.tanks[0].y,
            len(game.bullets), game.rng.getstate(),
        )
        self.assertEqual(before, after)
        self.assertIn("space_control_score", result)
        self.assertIn("self_tail_safe", result)

    def test_movement_index_is_not_an_action_index(self):
        game = Game(seed=42, ai_enabled=True)
        analyzer = OpportunityAnalyzer360(game)
        result = evaluate_space_control_pair(
            game, analyzer, analyzer.metrics(game), movement_index=8,
            score_horizon=12, fire_tail_horizon=16)
        self.assertEqual(result["movement_index"], 8)
        self.assertEqual(result["no_fire"]["first_index"], 16)
        self.assertEqual(result["fire"]["first_index"], 17)


if __name__ == "__main__":
    unittest.main()
