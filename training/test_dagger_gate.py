import unittest

from training.dagger_gate import (
    GATE_LEVELS,
    HARD,
    METRICS,
    OBJECTIVE,
    PERMANENT_REGRESSION_SEEDS,
    WATCH,
    compare_metric,
    compare_summaries,
    format_table,
    gate_verdict,
    summarize_report,
)


def make_round(seed, result="win", frames=200, search_rate=0.10,
               elapsed=5.0, kills=1, events=None, action_frames=None,
               **extra):
    row = {
        "seed": seed,
        "true_result": result,
        "frames": frames,
        "search_frame_rate": search_rate,
        "elapsed_seconds": elapsed,
        "kills": kills,
        "event_metrics": {
            "action_frames": frames if action_frames is None else action_frames,
            "events": events or {},
        },
    }
    row.update(extra)
    return row


def make_report(rounds):
    return {"rounds": rounds}


class SummarizeTest(unittest.TestCase):
    def test_basic_rates(self):
        report = make_report([
            make_round(1), make_round(2), make_round(3, result="loss"),
            make_round(4, result="double_death"),
        ])
        summary = summarize_report(report)
        self.assertEqual(summary["rounds"], 4)
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertAlmostEqual(summary["nonwin_rate"], 0.5)
        self.assertAlmostEqual(summary["double_death_rate"], 0.25)
        self.assertEqual(summary["nonwin_seeds"], [3, 4])

    def test_empty_report_raises(self):
        with self.assertRaises(ValueError):
            summarize_report({"rounds": []})

    def test_high_search_round_rate(self):
        report = make_report([
            make_round(1, search_rate=0.10),
            make_round(2, search_rate=0.30),
            make_round(3, search_rate=0.26),
            make_round(4, search_rate=0.25),  # boundary is exclusive
        ])
        summary = summarize_report(report)
        self.assertAlmostEqual(summary["high_search_round_rate"], 0.5)
        self.assertAlmostEqual(summary["max_search_frame_rate"], 0.30)

    def test_event_rates_use_action_frames(self):
        report = make_report([
            make_round(1, action_frames=1000, events={
                "throttle_reversal": 40, "turn_reversal": 60,
                "stutter_stall": 7, "dead_end_stall": 6,
                "missed_fire_window": 5}),
        ])
        summary = summarize_report(report)
        self.assertAlmostEqual(summary["reversal_per_1000"], 100.0)
        self.assertAlmostEqual(summary["stutter_per_1000"], 7.0)
        self.assertAlmostEqual(summary["dead_end_per_1000"], 6.0)
        self.assertAlmostEqual(summary["missed_fire_per_1000"], 5.0)

    def test_fire_capture_rate(self):
        report = make_report([
            make_round(1, events={"fire_window": 8,
                                  "captured_fire_window": 2}),
            make_round(2, events={"fire_window": 2,
                                  "captured_fire_window": 1}),
        ])
        self.assertAlmostEqual(
            summarize_report(report)["fire_capture_rate"], 0.3)

    def test_zero_denominators_do_not_divide_by_zero(self):
        report = make_report([make_round(1)])
        summary = summarize_report(report)
        self.assertEqual(summary["fire_capture_rate"], 0.0)
        self.assertEqual(summary["long_tail_rejection_rate"], 0.0)
        self.assertEqual(summary["topology_abort_rate"], 0.0)

    def test_active_kill_rate_counts_rounds_not_kills(self):
        report = make_report([
            make_round(1, kills=1), make_round(2, kills=0),
            make_round(3, kills=1), make_round(4, kills=0),
        ])
        self.assertAlmostEqual(
            summarize_report(report)["active_kill_rate"], 0.5)

    def test_derived_rates_from_counters(self):
        report = make_report([
            make_round(1, long_tail_fire_checks=10,
                       long_tail_fire_rejections=6,
                       topology_requests=8, topology_aborts=2),
        ])
        summary = summarize_report(report)
        self.assertAlmostEqual(summary["long_tail_rejection_rate"], 0.6)
        self.assertAlmostEqual(summary["topology_abort_rate"], 0.25)

    def test_every_declared_metric_is_produced(self):
        summary = summarize_report(make_report([make_round(1)]))
        for name in METRICS:
            self.assertIn(name, summary)


class CompareMetricTest(unittest.TestCase):
    def test_lower_is_better_metric(self):
        row = compare_metric("mean_search_frame_rate", 0.20, 0.10)
        self.assertTrue(row["improved"])
        self.assertFalse(row["regressed"])

    def test_higher_is_better_metric(self):
        row = compare_metric("win_rate", 0.90, 0.95)
        self.assertTrue(row["improved"])
        row = compare_metric("win_rate", 0.95, 0.90)
        self.assertTrue(row["regressed"])

    def test_tolerance_absorbs_small_watch_regression(self):
        # mean_frames tolerance is 5%
        row = compare_metric("mean_frames", 100.0, 103.0)
        self.assertFalse(row["regressed"])
        self.assertTrue(row["within_tolerance"])
        row = compare_metric("mean_frames", 100.0, 120.0)
        self.assertTrue(row["regressed"])

    def test_hard_metrics_have_zero_tolerance(self):
        row = compare_metric("win_rate", 0.900, 0.899)
        self.assertTrue(row["regressed"])

    def test_zero_baseline_does_not_divide_by_zero(self):
        row = compare_metric("double_death_rate", 0.0, 0.05)
        self.assertTrue(row["regressed"])
        row = compare_metric("double_death_rate", 0.0, 0.0)
        self.assertFalse(row["regressed"])
        self.assertFalse(row["improved"])


class GateVerdictTest(unittest.TestCase):
    def _rows(self, baseline, candidate):
        return compare_summaries(baseline, candidate)

    def _summaries(self, **candidate_overrides):
        baseline = summarize_report(make_report([
            make_round(seed, search_rate=0.20, elapsed=8.0,
                       events={"fire_window": 4, "captured_fire_window": 1,
                               "throttle_reversal": 10, "turn_reversal": 10})
            for seed in (1, 2, 3, 4)]))
        candidate = dict(baseline)
        candidate.update(candidate_overrides)
        return baseline, candidate

    def test_pass_when_objective_improves(self):
        baseline, candidate = self._summaries(mean_search_frame_rate=0.10)
        verdict = gate_verdict(
            self._rows(baseline, candidate), "paired12")
        self.assertTrue(verdict["passed"])
        self.assertIn("mean_search_frame_rate", verdict["objective_gains"])
        self.assertEqual(verdict["next_level"], "unseen100")

    def test_fail_when_nothing_improves(self):
        baseline, candidate = self._summaries()
        verdict = gate_verdict(self._rows(baseline, candidate), "paired12")
        self.assertFalse(verdict["passed"])
        self.assertTrue(any("objective" in item
                            for item in verdict["failures"]))

    def test_hard_regression_fails_even_with_objective_gain(self):
        # The exact trap: search rate halves because the net stopped
        # proposing anything, and win rate drops.
        baseline, candidate = self._summaries(
            mean_search_frame_rate=0.05, win_rate=0.75, nonwin_rate=0.25)
        verdict = gate_verdict(self._rows(baseline, candidate), "unseen100")
        self.assertFalse(verdict["passed"])
        self.assertIn("win_rate", verdict["hard_regressions"])
        self.assertIsNone(verdict["next_level"])

    def test_permanent_level_fails_on_any_nonwin(self):
        baseline, candidate = self._summaries(mean_search_frame_rate=0.05)
        verdict = gate_verdict(
            self._rows(baseline, candidate), "permanent",
            candidate_nonwin_seeds=[996004])
        self.assertFalse(verdict["passed"])
        self.assertTrue(any("permanent" in item
                            for item in verdict["failures"]))

    def test_permanent_level_fails_on_a_new_seed_too(self):
        baseline, candidate = self._summaries(mean_search_frame_rate=0.05)
        verdict = gate_verdict(
            self._rows(baseline, candidate), "permanent",
            candidate_nonwin_seeds=[123456])
        self.assertFalse(verdict["passed"])

    def test_permanent_level_passes_clean_sweep(self):
        baseline, candidate = self._summaries(mean_search_frame_rate=0.05)
        verdict = gate_verdict(
            self._rows(baseline, candidate), "permanent",
            candidate_nonwin_seeds=[])
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["next_level"], "paired12")

    def test_watch_regression_is_reported_not_failed(self):
        baseline, candidate = self._summaries(
            mean_search_frame_rate=0.05, reversal_per_1000=300.0)
        verdict = gate_verdict(self._rows(baseline, candidate), "paired12")
        self.assertTrue(verdict["passed"])
        self.assertIn("reversal_per_1000", verdict["watch_regressions"])
        self.assertIn("reversal_per_1000", verdict["requires_explanation"])

    def test_conservative_collapse_is_visible(self):
        # Search rate down but the policy also stopped killing and stopped
        # capturing fire windows: passes the gate but flags for explanation.
        baseline, candidate = self._summaries(
            mean_search_frame_rate=0.05, active_kill_rate=0.20,
            fire_capture_rate=0.05)
        verdict = gate_verdict(self._rows(baseline, candidate), "paired12")
        self.assertIn("active_kill_rate", verdict["watch_regressions"])
        self.assertIn("fire_capture_rate", verdict["watch_regressions"])

    def test_final_level_has_no_next(self):
        baseline, candidate = self._summaries(mean_search_frame_rate=0.05)
        verdict = gate_verdict(self._rows(baseline, candidate), "official")
        self.assertTrue(verdict["passed"])
        self.assertIsNone(verdict["next_level"])


class MiscTest(unittest.TestCase):
    def test_permanent_seeds_include_known_failures(self):
        for seed in (996004, 998002, 979000, 970252, 970105, 970128):
            self.assertIn(seed, PERMANENT_REGRESSION_SEEDS)
        self.assertEqual(len(set(PERMANENT_REGRESSION_SEEDS)),
                         len(PERMANENT_REGRESSION_SEEDS))

    def test_metric_classes_are_known(self):
        for _, (direction, metric_class, tolerance) in METRICS.items():
            self.assertIn(direction, (1, -1))
            self.assertIn(metric_class, (HARD, OBJECTIVE, WATCH))
            self.assertGreaterEqual(tolerance, 0.0)

    def test_gate_levels_are_ordered(self):
        self.assertEqual(GATE_LEVELS[0], "permanent")
        self.assertEqual(GATE_LEVELS[-1], "official")

    def test_format_table_renders(self):
        rows = [compare_metric("win_rate", 0.9, 0.95)]
        text = format_table(rows)
        self.assertIn("win_rate", text)
        self.assertIn("improved", text)


if __name__ == "__main__":
    unittest.main()
