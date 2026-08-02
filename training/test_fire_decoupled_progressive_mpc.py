import unittest
from types import SimpleNamespace

import numpy as np

from tank_trouble_original import Game
from training.mpc_agent import CANDIDATES
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.progressive_risk_mpc_teacher import (
    ProgressiveSearchEngine,
    _action_shot_event,
    _credible_root_fire,
    _movement_group,
    _movement_pair,
)


class FireDecoupledProgressiveMPCTest(unittest.TestCase):
    def test_candidate_pairs_share_movement(self):
        for group in range(9):
            no_fire, fire = _movement_pair(group)
            self.assertEqual(_movement_group(no_fire), group)
            self.assertEqual(_movement_group(fire), group)
            self.assertEqual(CANDIDATES[no_fire][:2], CANDIDATES[fire][:2])
            self.assertEqual((CANDIDATES[no_fire][2], CANDIDATES[fire][2]),
                             (0, 1))

    def test_diverse_pruning_keeps_base_and_each_throttle(self):
        order = [8, 7, 5, 4, 2, 1, 0, 3, 6]
        selected = ProgressiveSearchEngine._keep_diverse_groups(
            order, width=3, base_group=4)
        self.assertIn(4, selected)
        self.assertEqual({group // 3 for group in selected}, {0, 1, 2})

    def test_fire_gate_rejects_dead_enemy_and_unavailable_weapon(self):
        game = Game(seed=42, ai_enabled=True)
        metrics = OpportunityAnalyzer360(game).metrics(game)
        game.tanks[1].alive = False
        allowed, _, reason = _credible_root_fire(
            game, (1, 1, 1), metrics)
        self.assertFalse(allowed)
        self.assertEqual(reason, "unavailable")

        game = Game(seed=43, ai_enabled=True)
        metrics = OpportunityAnalyzer360(game).metrics(game)
        game.tanks[0].bullets_fired = game.settings_max_bullets
        self.assertIsNone(_action_shot_event(game, (1, 1, 1)))
        allowed, _, reason = _credible_root_fire(
            game, (1, 1, 1), metrics)
        self.assertFalse(allowed)
        self.assertEqual(reason, "unavailable")

    def test_pair_selection_requires_positive_safe_fire_gain(self):
        engine = ProgressiveSearchEngine(
            horizons=(2,), widths=(1,), final_samples=1,
            max_death=0.0, max_dd=0.0, fire_min_gain=0.015)
        aux = np.zeros(6, dtype=np.float32)
        final = {
            0: {"value": 0.10, "aux": aux},
            1: {"value": 0.11, "aux": aux},
        }
        branches = {
            1: SimpleNamespace(
                root_fire_allowed=True,
                root_fire_reason="predicted_hit"),
        }
        selected, row = engine._select_pair(0, final, branches)
        self.assertEqual(selected, 0)
        self.assertEqual(row["fire_reason"], "insufficient_paired_gain")

        final[1] = {"value": 0.12, "aux": aux}
        selected, row = engine._select_pair(0, final, branches)
        self.assertEqual(selected, 1)
        self.assertEqual(row["fire_reason"], "positive_paired_gain")

    def test_search_never_selects_a_blocked_fire_action(self):
        game = Game(seed=970002, ai_enabled=True)
        analyzer = OpportunityAnalyzer360(game)
        engine = ProgressiveSearchEngine(
            horizons=(2,), widths=(9,), final_samples=1,
            max_death=1.0, max_dd=1.0)
        action, diagnostics = engine.search(
            game, analyzer, base_index=8, root_seed=970002)
        if action[2] == 1:
            group = str(_movement_group(diagnostics["best_index"]))
            self.assertTrue(diagnostics["fire_pairs"][group]["fire_allowed"])


if __name__ == "__main__":
    unittest.main()
