import unittest

from training.mpc_agent import CANDIDATES
from training.exact_state_mpc_teacher import (
    candidate_movement,
    prefer_movement_continuity,
)


def row(index, value, allowed=True, kill=0.0):
    return {
        "index": index,
        "action": CANDIDATES[index],
        "score": value,
        "value": value,
        "aux": [kill, 0.0, 0.0],
        "kill": kill,
        "death": 0.0,
        "double_death": 0.0,
        "allowed": allowed,
    }


class CandidateMovementTest(unittest.TestCase):
    def test_index_layout_matches_candidates(self):
        # CANDIDATES = [(th, tu, f) for th in 0..2 for tu in 0..2 for f in 0,1]
        self.assertEqual(len(CANDIDATES), 18)
        for index, action in enumerate(CANDIDATES):
            self.assertEqual(candidate_movement(index), index // 2)
            self.assertEqual(action[2], index % 2)

    def test_actions_sharing_a_movement_differ_only_in_fire(self):
        for movement in range(9):
            nofire = CANDIDATES[movement * 2]
            fire = CANDIDATES[movement * 2 + 1]
            self.assertEqual(nofire[:2], fire[:2])
            self.assertEqual((nofire[2], fire[2]), (0, 1))


class PreferMovementContinuityTest(unittest.TestCase):
    def test_disabled_by_default_epsilon(self):
        rows = [row(0, 100.0), row(4, 99.9)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, current_movement=2), 0)

    def test_keeps_current_movement_when_near_tied(self):
        rows = [row(0, 100.0), row(4, 99.5)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, 2, epsilon=1.0), 4)

    def test_does_not_switch_when_gap_exceeds_epsilon(self):
        rows = [row(0, 100.0), row(4, 90.0)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, 2, epsilon=1.0), 0)

    def test_no_change_when_best_already_matches(self):
        rows = [row(4, 100.0), row(0, 99.9)]
        self.assertEqual(
            prefer_movement_continuity(rows, 4, 2, epsilon=1.0), 4)

    def test_never_promotes_a_disallowed_action(self):
        # The near-tied continuation is unsafe: must not be selected.
        rows = [row(0, 100.0), row(4, 99.9, allowed=False)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, 2, epsilon=1.0), 0)

    def test_never_changes_the_fire_decision(self):
        # best fires (index 1); the near-tied same-movement alternative does
        # not fire, so it must be rejected -- firing is not this function's
        # business.
        rows = [row(1, 100.0), row(4, 99.9)]
        self.assertEqual(
            prefer_movement_continuity(rows, 1, 2, epsilon=1.0), 1)

    def test_matches_fire_bit_when_available(self):
        # index 5 = movement 2 with fire=1, same fire bit as best index 1.
        rows = [row(1, 100.0), row(5, 99.9)]
        self.assertEqual(
            prefer_movement_continuity(rows, 1, 2, epsilon=1.0), 5)

    def test_picks_the_best_of_several_matching_rows(self):
        # Only one row per (movement, fire) pair normally exists, but the
        # function must not depend on that.
        rows = [row(0, 100.0), row(4, 99.5), row(4, 99.8)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, 2, epsilon=1.0), 4)

    def test_none_best_index_passes_through(self):
        rows = [row(0, 100.0)]
        self.assertIsNone(
            prefer_movement_continuity(rows, None, 2, epsilon=1.0))

    def test_none_current_movement_passes_through(self):
        rows = [row(0, 100.0), row(4, 99.9)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, None, epsilon=1.0), 0)

    def test_negative_epsilon_is_treated_as_disabled(self):
        rows = [row(0, 100.0), row(4, 99.9)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, 2, epsilon=-5.0), 0)

    def test_missing_continuation_row_passes_through(self):
        # The candidate order may not even contain the current movement.
        rows = [row(0, 100.0), row(6, 99.9)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, 2, epsilon=1.0), 0)

    def test_exact_boundary_is_inclusive(self):
        rows = [row(0, 100.0), row(4, 99.0)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, 2, epsilon=1.0), 4)

    def test_result_is_always_an_int_index_present_in_rows(self):
        rows = [row(0, 100.0), row(4, 99.9)]
        result = prefer_movement_continuity(rows, 0, 2, epsilon=1.0)
        self.assertIsInstance(result, int)
        self.assertIn(result, [int(item["index"]) for item in rows])

    def test_a_large_epsilon_still_cannot_pick_unsafe(self):
        # Guard the safety invariant under an absurd configuration.
        rows = [row(0, 100.0), row(4, -1e9, allowed=False)]
        self.assertEqual(
            prefer_movement_continuity(rows, 0, 2, epsilon=1e12), 0)

    def test_only_allowed_rows_are_ever_returned(self):
        rows = [row(index, 100.0 - index, allowed=(index % 2 == 0))
                for index in range(18)]
        for movement in range(9):
            result = prefer_movement_continuity(
                rows, 0, movement, epsilon=1e6)
            self.assertTrue(
                next(item for item in rows
                     if int(item["index"]) == result)["allowed"])


if __name__ == "__main__":
    unittest.main()
