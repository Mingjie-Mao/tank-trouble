import unittest

import numpy as np

from training.exact_teacher_privileged_safety import (
    choose_replacements,
    safety_override_metrics,
)


class ExactTeacherPrivilegedSafetyTest(unittest.TestCase):
    def test_replacement_is_nonfire_and_ranked_by_champion_value(self):
        safe = np.zeros((1, 18), dtype=np.float32)
        safe[0, 2] = 0.91
        safe[0, 4] = 0.95
        safe[0, 5] = 0.99
        values = np.zeros((1, 18), dtype=np.float32)
        values[0, 2] = 3.0
        values[0, 4] = 2.0
        values[0, 5] = 10.0
        replacement = choose_replacements(
            safe, np.asarray([0]), values, 0.8, 0.9, 0.2)
        self.assertEqual(2, int(replacement[0]))

    def test_override_precision_requires_unsafe_champion_and_safe_replacement(self):
        safe = np.zeros((3, 18), dtype=np.float32)
        safe[:, 2] = 0.99
        values = np.zeros((3, 18), dtype=np.float32)
        values[:, 2] = 1.0
        allowed = np.zeros((3, 18), dtype=np.bool_)
        allowed[0, 2] = True
        allowed[1, 0] = True
        allowed[1, 2] = True
        data = {
            "allowed": allowed,
            "champion_action": np.asarray([0, 0, 0]),
            "round_seed": np.asarray([976001, 983001, 993001]),
        }
        metrics = safety_override_metrics(
            safe, values, data, np.arange(3), 0.8, 0.9, 0.2)
        self.assertEqual(3, metrics["predicted_overrides"])
        self.assertEqual(1, metrics["correct_overrides"])
        self.assertAlmostEqual(1 / 3, metrics["precision"])
        self.assertEqual(1, metrics["unnecessary_overrides"])
        self.assertEqual(1, metrics["unsafe_replacements"])

    def test_no_override_when_probability_gap_is_too_small(self):
        safe = np.zeros((1, 18), dtype=np.float32)
        safe[0, 0] = 0.15
        safe[0, 2] = 0.90
        values = np.zeros((1, 18), dtype=np.float32)
        replacement = choose_replacements(
            safe, np.asarray([0]), values, 0.8, 0.9, 0.80)
        self.assertEqual(-1, int(replacement[0]))


if __name__ == "__main__":
    unittest.main()
