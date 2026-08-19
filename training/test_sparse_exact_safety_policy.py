import unittest

from training.diagnose_sparse_exact_handoff import controls_to_action
from training.sparse_exact_safety_policy import (
    exceeded_nonwin_gate,
    movement_only_action,
    parse_seeds,
    should_trigger_key_search,
    should_arm_narrow_replan,
)


class SparseExactSafetyPolicyTest(unittest.TestCase):
    def test_controls_to_action(self):
        self.assertEqual(
            controls_to_action({
                "forward": True,
                "turn_left": True,
                "fire": True,
            }),
            (2, 0, 1),
        )
        self.assertEqual(controls_to_action({}), (1, 1, 0))

    def test_parse_seed_ranges(self):
        self.assertEqual(
            (10, 11, 12, 20),
            parse_seeds("10:3,20"),
        )

    def test_narrow_replan_gate(self):
        self.assertTrue(should_arm_narrow_replan(3, 3, "behavior"))
        self.assertFalse(should_arm_narrow_replan(4, 3, "behavior"))
        self.assertFalse(should_arm_narrow_replan(0, 3, "unsafe"))
        self.assertFalse(should_arm_narrow_replan(2, 3, "narrow"))

    def test_nonwin_early_stop_gate(self):
        rows = [
            {"true_result": "win"},
            {"true_result": "loss"},
            {"true_result": "double_death"},
        ]
        self.assertFalse(exceeded_nonwin_gate(rows, -1))
        self.assertFalse(exceeded_nonwin_gate(rows, 2))
        self.assertTrue(exceeded_nonwin_gate(rows, 1))

    def test_fire_is_not_part_of_committed_movement(self):
        self.assertEqual(movement_only_action((2, 0, 1)), (2, 0, 0))

    def test_key_search_is_limited_to_fire_or_danger(self):
        self.assertTrue(should_trigger_key_search((1, 1, 1), 0.0, 0.18))
        self.assertTrue(should_trigger_key_search((1, 1, 0), 0.20, 0.18))
        self.assertFalse(should_trigger_key_search((1, 1, 0), 0.10, 0.18))


if __name__ == "__main__":
    unittest.main()
