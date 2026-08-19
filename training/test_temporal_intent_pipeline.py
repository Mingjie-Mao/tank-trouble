import unittest

import numpy as np

from training.temporal_intent_pipeline import group_split, sequence_targets


class TemporalIntentPipelineTests(unittest.TestCase):
    def test_group_split_keeps_rounds_disjoint(self):
        seeds = np.repeat([10, 11, 12, 13], 4)
        train, validation, validation_seeds = group_split(
            seeds, validation_fraction=0.25, split_seed=1)
        self.assertFalse(set(seeds[train]) & set(seeds[validation]))
        self.assertEqual(set(seeds[validation]), set(validation_seeds))

    def test_sequence_targets_mark_end_of_run(self):
        remaining, hold, interrupt = sequence_targets([1, 1, 2, 2, 2])
        np.testing.assert_array_equal(remaining, [2, 1, 3, 2, 1])
        np.testing.assert_array_equal(interrupt, [0, 1, 0, 0, 0])
        self.assertEqual(len(hold), 5)


if __name__ == "__main__":
    unittest.main()
