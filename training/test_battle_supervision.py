import tempfile
import unittest
from pathlib import Path

from training.battle_supervision import append_supervision, diagnose_battle


class BattleSupervisionTests(unittest.TestCase):
    def test_double_death_and_behavior_issues_are_independent(self):
        report = diagnose_battle({
            "true_result": "double_death",
            "frames": 100,
            "event_metrics": {"events": {
                "missed_fire_window": 3,
                "dead_end_stall": 2,
            }},
        })
        self.assertIn("double_death_risk", report["issue_categories"])
        self.assertIn("fire_opportunity_gap", report["issue_categories"])
        self.assertIn("dead_end_navigation", report["issue_categories"])

    def test_append_writes_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rounds.jsonl"
            append_supervision(str(path), {
                "result": "win", "frames": 50,
                "behavior_events": {"stutter_stall": 1},
            })
            self.assertIn("movement_stutter", path.read_text())


if __name__ == "__main__":
    unittest.main()
