import unittest

import numpy as np

from training.exact_teacher_distill import (
    PRIVILEGED_DIM,
    build_champion_label,
    build_consistent_label,
    privileged_state_features,
)
from training.mpc_agent import CANDIDATES
from tank_trouble_original import Game


def _rows():
    rows = []
    for index, action in enumerate(CANDIDATES):
        rows.append({
            "index": index,
            "action": action,
            "score": float(index),
            "value": float(index),
            "aux": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "kill": 0.0,
            "death": 0.0,
            "double_death": 0.0,
            "allowed": True,
        })
    return rows


class ExactTeacherDistillLabelTest(unittest.TestCase):
    def test_executed_action_is_policy_target_even_after_override(self):
        decision = {
            "rows": _rows(),
            "executed_index": 2,
            "safe_root_count": 18,
            "interventions": ("successor_shield_override",),
        }
        label = build_consistent_label(decision)
        self.assertTrue(label["action_valid"])
        self.assertEqual(2, int(label["Y_score"].argmax()))
        self.assertEqual(2, label["Y_action"])
        self.assertEqual("successor_shield_override", label["category"])

    def test_unsafe_action_has_exact_aux_and_cannot_be_valid(self):
        rows = _rows()
        rows[5]["allowed"] = False
        rows[5]["death"] = 1.0
        rows[5]["aux"][1] = 1.0
        decision = {
            "rows": rows,
            "executed_index": 4,
            "safe_root_count": 17,
            "interventions": (),
        }
        label = build_consistent_label(decision)
        self.assertFalse(label["allowed"][5])
        self.assertEqual(1.0, float(label["Y_aux"][5, 1]))
        self.assertLess(float(label["Y_score"][5]),
                        float(label["Y_score"][4]))

    def test_selected_movement_fire_label_matches_executed_action(self):
        decision = {
            "rows": _rows(),
            "executed_index": 7,
            "safe_root_count": 18,
            "interventions": (),
        }
        label = build_consistent_label(decision)
        movement = 7 // 2
        self.assertEqual(1.0, float(label["Y_fire_mask"][movement]))
        self.assertEqual(1.0, float(label["Y_fire"][movement]))

    def test_low_gain_no_kill_fire_is_suppressed(self):
        rows = _rows()
        rows[0]["value"] = 20.0
        rows[1]["value"] = 21.0
        decision = {
            "rows": rows,
            "executed_index": 0,
            "safe_root_count": 18,
            "interventions": ("low_gain_fire_suppressed",),
        }
        label = build_consistent_label(decision, fire_gain_margin=2.0)
        self.assertEqual(1.0, float(label["Y_fire_mask"][0]))
        self.assertEqual(0.0, float(label["Y_fire"][0]))
        self.assertTrue(np.isfinite(label["Y_score"]).all())

    def test_champion_label_records_bounded_override_target(self):
        rows = _rows()
        rows[2]["value"] = 8.0
        rows[4]["value"] = 20.0
        decision = {
            "rows": rows,
            "executed_index": 4,
            "safe_root_count": 18,
            "interventions": (),
            "search_indices": [2, 4, 6],
        }
        label = build_champion_label(decision, champion_index=2)
        self.assertTrue(label["teacher_override"])
        self.assertTrue(label["champion_action_allowed"])
        self.assertEqual(2, label["champion_action"])
        self.assertAlmostEqual(12.0 / 1000.0,
                               label["teacher_advantage"])
        self.assertAlmostEqual(12.0 / 1000.0,
                               label["teacher_margin"])

    def test_privileged_features_are_fixed_and_do_not_advance_rng(self):
        game = Game(seed=41, ai_enabled=True)
        before = game.rng.getstate()
        first = privileged_state_features(game)
        second = privileged_state_features(game)
        self.assertEqual((PRIVILEGED_DIM,), first.shape)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(before, game.rng.getstate())
        self.assertTrue(np.isfinite(first).all())

        game.tanks[1].ai.my_goal["dist"] = [[0, 1], [1, 0]]
        with_distance_map = privileged_state_features(game)
        self.assertEqual((PRIVILEGED_DIM,), with_distance_map.shape)
        self.assertTrue(np.isfinite(with_distance_map).all())


if __name__ == "__main__":
    unittest.main()
