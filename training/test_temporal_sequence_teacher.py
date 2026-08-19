import unittest

from tank_trouble_original import Game
from training.exact_state import state_fingerprint
from training.exact_state_mpc_teacher import ALL_CANDIDATE_INDICES
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.temporal_sequence_teacher import (
    choose_smooth_safe_sequence,
    exact_two_stage_search,
    rollout_exact_sequence,
)


class TemporalSequenceTeacherTest(unittest.TestCase):
    def test_rollout_does_not_mutate_live_game(self):
        game = Game(seed=973034, ai_enabled=True)
        analyzer = OpportunityAnalyzer360(game)
        metrics = analyzer.metrics(game)
        before = state_fingerprint(game)

        row = rollout_exact_sequence(
            game, analyzer, metrics, 8, 10,
            chunk_frames=3, score_horizon=12, fire_tail_horizon=16)

        self.assertTrue(row["live_fingerprint_unchanged"])
        self.assertEqual(before, state_fingerprint(game))
        self.assertEqual(row["first_index"], 8)
        self.assertEqual(row["second_index"], 10)
        self.assertIn("agent_kill", row)
        self.assertIn("unassisted_enemy_death", row)
        self.assertGreaterEqual(row["movement_cells"], 0.0)

    def test_safety_is_hard_constraint(self):
        rows = [
            {"allowed": False, "value": 1000.0, "kill": True,
             "kill_frame": 1, "movement_switches": 0},
            {"allowed": True, "value": 1.0, "kill": False,
             "kill_frame": None, "movement_switches": 0},
        ]
        self.assertIs(choose_smooth_safe_sequence(rows), rows[1])

    def test_smoothness_only_breaks_near_value_ties(self):
        smooth = {"allowed": True, "value": 10.0, "kill": False,
                  "kill_frame": None, "movement_switches": 0}
        rough = {"allowed": True, "value": 10.5, "kill": False,
                 "kill_frame": None, "movement_switches": 1}
        self.assertIs(
            choose_smooth_safe_sequence([smooth, rough], 1.0), smooth)
        self.assertIs(
            choose_smooth_safe_sequence([smooth, rough], 0.1), rough)

    def test_agent_kill_beats_unassisted_death_within_tolerance(self):
        unassisted = {
            "allowed": True, "value": 10.0, "kill": True,
            "agent_kill": False, "kill_frame": 5,
            "movement_switches": 0,
        }
        agent_kill = {
            "allowed": True, "value": 9.5, "kill": True,
            "agent_kill": True, "kill_frame": 6,
            "movement_switches": 0,
        }
        self.assertIs(
            choose_smooth_safe_sequence(
                [unassisted, agent_kill], value_tolerance=1.0),
            agent_kill,
        )

    def test_two_stage_search_returns_valid_candidate_without_mutation(self):
        game = Game(seed=42, ai_enabled=True)
        analyzer = OpportunityAnalyzer360(game)
        metrics = analyzer.metrics(game)
        before = state_fingerprint(game)

        selected, rows = exact_two_stage_search(
            game, analyzer, metrics, ALL_CANDIDATE_INDICES,
            chunk_frames=2, score_horizon=8,
            root_beam=2, continuation_beam=1,
            fire_tail_horizon=10)

        self.assertTrue(rows)
        self.assertIsNotNone(selected)
        self.assertTrue(selected["allowed"])
        self.assertEqual(before, state_fingerprint(game))


if __name__ == "__main__":
    unittest.main()
