import unittest

from training.p27_guided_search import (
    action_to_controls,
    choose_movement_by_safety,
    controls_to_action,
    movement_candidates,
)


class P27GuidedSearchHelpersTest(unittest.TestCase):
    def test_action_controls_round_trip(self):
        for action in ((0, 0, 0), (1, 1, 1), (2, 2, 0)):
            self.assertEqual(action, controls_to_action(action_to_controls(action)))

    def test_candidates_are_distinct_movements_and_have_fallbacks(self):
        ranking = [(2, 2, 1), (2, 2, 0), (0, 0, 1), (1, 2, 0)]
        result = movement_candidates(ranking, top_k=3, current=(2, 1, 0))
        self.assertEqual(len(result), len(set(result)))
        self.assertTrue(all(action[2] == 0 for action in result))
        self.assertIn((1, 1, 0), result)
        self.assertIn((0, 1, 0), result)
        self.assertIn((2, 1, 0), result)

    def test_search_does_not_replace_a_safe_network_action_for_shaping(self):
        scored = [(-2.0, (2, 1, 0)), (0.0, (0, 1, 0))]
        self.assertEqual(
            ((2, 1, 0), False),
            choose_movement_by_safety(scored, (2, 1, 0)))

    def test_search_replaces_a_terminally_unsafe_network_action(self):
        scored = [(-990.0, (2, 1, 0)), (-1.0, (0, 1, 0))]
        self.assertEqual(
            ((0, 1, 0), True),
            choose_movement_by_safety(scored, (2, 1, 0)))


if __name__ == "__main__":
    unittest.main()
