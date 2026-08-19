import unittest

from training.mpc_agent import CANDIDATES
from training.exact_state_mpc_teacher import choose_unsafe_fallback


def row(index, value, kill=0.0, allowed=False):
    return {
        "index": index,
        "action": CANDIDATES[index],
        "score": value,
        "value": value,
        "aux": [kill, 1.0, 0.0],
        "kill": kill,
        "death": 1.0,
        "double_death": 0.0,
        "allowed": allowed,
    }


FIRE_ACTION = CANDIDATES[1]        # movement 0, fire
NOFIRE_ACTION = CANDIDATES[0]      # movement 0, no fire


class ProposedModeTest(unittest.TestCase):
    def test_proposed_mode_is_the_old_behaviour(self):
        rows = [row(4, 10.0), row(6, 50.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, FIRE_ACTION, mode="proposed"),
            FIRE_ACTION)

    def test_unknown_mode_falls_back_to_proposed(self):
        self.assertEqual(
            choose_unsafe_fallback([], FIRE_ACTION, mode="whatever"),
            FIRE_ACTION)


class StripFireModeTest(unittest.TestCase):
    """The minimal fix: touch the fire bit, leave the movement prior alone."""

    def test_nonfire_proposal_is_untouched(self):
        for index in range(0, 18, 2):
            proposal = CANDIDATES[index]
            self.assertEqual(
                choose_unsafe_fallback([], proposal, mode="strip_fire"),
                proposal)

    def test_fire_is_stripped_with_no_rows(self):
        self.assertEqual(
            choose_unsafe_fallback([], FIRE_ACTION, mode="strip_fire"),
            NOFIRE_ACTION)

    def test_fire_is_stripped_when_no_kill_predicted(self):
        rows = [row(1, 50.0, kill=0.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, FIRE_ACTION, mode="strip_fire"),
            NOFIRE_ACTION)

    def test_fire_survives_when_that_exact_action_predicts_a_kill(self):
        rows = [row(1, 50.0, kill=1.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, FIRE_ACTION, mode="strip_fire"),
            FIRE_ACTION)

    def test_a_kill_on_a_different_action_does_not_license_this_one(self):
        # Row 3 is a different movement; its kill must not keep index 1 firing.
        rows = [row(3, 90.0, kill=1.0), row(1, 50.0, kill=0.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, FIRE_ACTION, mode="strip_fire"),
            NOFIRE_ACTION)

    def test_movement_is_never_changed(self):
        for index in range(1, 18, 2):
            proposal = CANDIDATES[index]
            rows = [row(other, 999.0) for other in range(0, 18, 2)]
            result = choose_unsafe_fallback(
                rows, proposal, mode="strip_fire")
            self.assertEqual(result[:2], proposal[:2])

    def test_differs_from_least_bad_when_another_row_scores_higher(self):
        rows = [row(1, 10.0, kill=0.0), row(6, 90.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, FIRE_ACTION, mode="strip_fire"),
            NOFIRE_ACTION)
        self.assertEqual(
            choose_unsafe_fallback(rows, FIRE_ACTION, mode="least_bad"),
            CANDIDATES[6])


class LeastBadModeTest(unittest.TestCase):
    def test_picks_the_highest_value_evaluated_row(self):
        rows = [row(4, 10.0), row(6, 50.0), row(8, 20.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, NOFIRE_ACTION), CANDIDATES[6])

    def test_strips_fire_when_no_kill_predicted(self):
        # This is seed 970252: the proposal fired, nothing was safe, the
        # long-tail check never ran, and its own bullet came back.
        rows = [row(1, 50.0, kill=0.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, FIRE_ACTION), CANDIDATES[0])

    def test_keeps_fire_when_the_row_predicts_a_kill(self):
        rows = [row(1, 50.0, kill=1.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, NOFIRE_ACTION), CANDIDATES[1])

    def test_partial_kill_is_not_enough_to_keep_fire(self):
        rows = [row(1, 50.0, kill=0.9)]
        self.assertEqual(
            choose_unsafe_fallback(rows, NOFIRE_ACTION), CANDIDATES[0])

    def test_missing_kill_field_is_treated_as_no_kill(self):
        candidate = {"index": 1, "value": 50.0}
        self.assertEqual(
            choose_unsafe_fallback([candidate], NOFIRE_ACTION),
            CANDIDATES[0])

    def test_no_rows_strips_fire_from_the_proposal(self):
        # Nothing was evaluated, so we cannot be smarter than the proposal --
        # but blind-firing on it is never the right call.
        self.assertEqual(
            choose_unsafe_fallback([], FIRE_ACTION), NOFIRE_ACTION)

    def test_no_rows_keeps_a_nonfire_proposal_intact(self):
        proposal = CANDIDATES[8]
        self.assertEqual(choose_unsafe_fallback([], proposal), proposal)

    def test_rows_without_value_are_ignored(self):
        rows = [{"index": 6}, row(4, 10.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, NOFIRE_ACTION), CANDIDATES[4])

    def test_rows_without_index_are_ignored(self):
        rows = [{"value": 99.0}, row(4, 10.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, NOFIRE_ACTION), CANDIDATES[4])

    def test_all_rows_unusable_falls_back_to_movement_only(self):
        self.assertEqual(
            choose_unsafe_fallback([{"value": 1.0}], FIRE_ACTION),
            NOFIRE_ACTION)

    def test_never_returns_a_firing_action_without_a_kill_row(self):
        # Property: across every candidate set, fire survives only with kill.
        for index in range(18):
            for kill in (0.0, 0.5, 1.0):
                result = choose_unsafe_fallback(
                    [row(index, 1.0, kill=kill)], FIRE_ACTION)
                if result[2] == 1:
                    self.assertGreaterEqual(kill, 1.0)

    def test_result_is_always_a_valid_candidate(self):
        rows = [row(index, float(index)) for index in range(18)]
        result = choose_unsafe_fallback(rows, FIRE_ACTION)
        self.assertIn(result, CANDIDATES)

    def test_negative_values_are_handled(self):
        rows = [row(4, -100.0), row(6, -5.0)]
        self.assertEqual(
            choose_unsafe_fallback(rows, NOFIRE_ACTION), CANDIDATES[6])

    def test_allowed_flag_is_irrelevant_here(self):
        # By construction nothing is allowed on this path; selection is by
        # value alone, which is what "least bad" means.
        rows = [row(4, 10.0, allowed=False), row(6, 50.0, allowed=False)]
        self.assertEqual(
            choose_unsafe_fallback(rows, NOFIRE_ACTION), CANDIDATES[6])


if __name__ == "__main__":
    unittest.main()
