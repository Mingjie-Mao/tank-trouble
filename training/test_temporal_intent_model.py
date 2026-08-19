import unittest

import numpy as np

from training.temporal_intent_model import (
    ACTION_COUNT,
    AUX_DIM,
    HOLD_BINS,
    MOVEMENT_COUNT,
    POLICY_FEATURE_DIM,
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_STATE_FEATURE_DIM,
    build_temporal_features,
    build_temporal_intent_net,
    movement_run_targets,
)


class TemporalIntentFeatureTests(unittest.TestCase):
    def test_features_are_bounded_and_have_stable_shape(self):
        features = build_temporal_features(
            np.arange(18) * 100.0,
            np.zeros((18, 6)),
            np.zeros(9),
            last_movement=4,
            frames_since_change=120,
        )
        self.assertEqual(features.shape, (TEMPORAL_FEATURE_DIM,))
        self.assertEqual(features.dtype, np.float32)
        movement_start = ACTION_COUNT + ACTION_COUNT * AUX_DIM + MOVEMENT_COUNT
        self.assertEqual(features[movement_start + 4], 1.0)
        self.assertEqual(features[POLICY_FEATURE_DIM - 1], 1.0)
        self.assertTrue(np.isfinite(features).all())

    def test_score_ranking_signal_is_not_scaled_away(self):
        features = build_temporal_features(
            np.linspace(-1.0, 0.0, 18),
            np.zeros((18, 6)), np.zeros(9),
            last_movement=None, frames_since_change=0,
        )
        self.assertGreater(float(np.ptp(features[:ACTION_COUNT])), 0.9)

    def test_topology_features_are_appended_without_reordering_policy_data(self):
        topology = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        features = build_temporal_features(
            np.zeros(18), np.zeros((18, 6)), np.zeros(9),
            last_movement=None, frames_since_change=0,
            topology_features=topology,
        )
        np.testing.assert_array_equal(features[POLICY_FEATURE_DIM:], topology)

    def test_raw_state_can_be_appended_for_search_gate(self):
        state = np.linspace(-1.0, 1.0, 440, dtype=np.float32)
        features = build_temporal_features(
            np.zeros(18), np.zeros((18, 6)), np.zeros(9),
            last_movement=None, frames_since_change=0,
            state_features=state,
        )
        self.assertEqual(features.shape, (TEMPORAL_STATE_FEATURE_DIM,))
        np.testing.assert_array_equal(features[-440:], state)

    def test_run_targets_count_down_within_each_intent(self):
        remaining, bins = movement_run_targets([2, 2, 2, 5, 5, 1])
        np.testing.assert_array_equal(remaining, [3, 2, 1, 2, 1, 1])
        self.assertEqual([HOLD_BINS[index] for index in bins],
                         [2, 2, 1, 2, 1, 1])

    def test_model_outputs_all_temporal_heads(self):
        import torch

        model = build_temporal_intent_net(hidden_dim=32)
        output = model(torch.zeros(2, 5, TEMPORAL_FEATURE_DIM))
        self.assertEqual(tuple(output["movement_delta"].shape), (2, 5, 9))
        self.assertEqual(tuple(output["hold"].shape), (2, 5, len(HOLD_BINS)))
        self.assertEqual(tuple(output["interrupt"].shape), (2, 5))
        self.assertEqual(tuple(output["progress"].shape), (2, 5))
        self.assertEqual(tuple(output["search_needed"].shape), (2, 5))
        self.assertEqual(tuple(output["hidden"].shape), (1, 2, 32))


if __name__ == "__main__":
    unittest.main()
