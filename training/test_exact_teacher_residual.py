import unittest

import numpy as np

from training.exact_teacher_residual import (
    override_metrics,
    stratified_seed_holdout,
)


class ExactTeacherResidualTest(unittest.TestCase):
    def test_holdout_preserves_seed_bands_without_overlap(self):
        seeds = np.asarray([
            *range(976000, 976010),
            *range(983000, 983010),
            *range(993000, 993010),
        ])
        train, test = stratified_seed_holdout(seeds, 0.2, 17)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(map(int, seeds)), set(train) | set(test))
        self.assertEqual({976, 983, 993}, {seed // 1000 for seed in test})

    def test_override_precision_requires_gate_and_exact_action(self):
        data = {
            "champion_action": np.asarray([0, 0, 0]),
            "Y_action": np.asarray([2, 4, 6]),
            "residual_target": np.asarray([True, True, False]),
            "champion_action_allowed": np.asarray([False, True, True]),
            "round_seed": np.asarray([976001, 983001, 993001]),
        }
        gate = np.asarray([0.9, 0.9, 0.9])
        action = np.zeros((3, 18), dtype=np.float32)
        action[0, 2] = 1.0
        action[1, 5] = 1.0
        action[2, 6] = 1.0
        metrics = override_metrics(
            gate, action, data, np.arange(3), 0.8, 0.8)
        self.assertEqual(3, metrics["predicted_overrides"])
        self.assertEqual(1, metrics["correct_overrides"])
        self.assertAlmostEqual(1 / 3, metrics["precision"])


if __name__ == "__main__":
    unittest.main()
