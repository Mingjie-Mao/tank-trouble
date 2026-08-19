import unittest

import numpy as np

from training.search_gate_model import threshold_report


class SearchGateModelTests(unittest.TestCase):
    def test_threshold_prefers_lowest_trigger_at_required_recall(self):
        probability = np.asarray([0.9, 0.8, 0.7, 0.1])
        target = np.asarray([1, 1, 0, 0])
        selected, _ = threshold_report(probability, target, target_recall=1.0)
        self.assertGreater(selected["threshold"], 0.7)
        self.assertEqual(selected["recall"], 1.0)
        self.assertEqual(selected["trigger_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
