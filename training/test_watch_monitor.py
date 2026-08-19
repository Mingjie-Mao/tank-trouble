import json
import os
import tempfile
import unittest

from training.watch_monitor import (
    BAD_WIN_RULES,
    analyze,
    bad_wins,
    classify_bad_win,
    compare_groups,
    format_report,
    group_stats,
    load_rounds,
)


def make_row(seed, result="win", frames=200, search=0.10, shots=1, kills=1,
             death_cause=None, timestamp="2026-08-04T00:00:00+00:00",
             events=None, action_frames=None, issues=()):
    return {
        "seed": seed,
        "true_result": result,
        "death_cause": death_cause,
        "frames": frames,
        "search_frame_rate": search,
        "shots": shots,
        "kills": kills,
        "timestamp": timestamp,
        "event_metrics": {
            "action_frames": frames if action_frames is None else action_frames,
            "events": events or {},
        },
        "diagnosis": {"issue_categories": list(issues)},
    }


def write_log(rows):
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8")
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return handle.name


class LoadRoundsTest(unittest.TestCase):
    def test_deduplicates_by_seed_keeping_last(self):
        path = write_log([
            make_row(1, frames=100), make_row(2), make_row(1, frames=999)])
        try:
            raw, rounds, _ = load_rounds(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(raw), 3)
        self.assertEqual(len(rounds), 2)
        self.assertEqual(
            next(r for r in rounds if r["seed"] == 1)["frames"], 999)

    def test_rounds_are_sorted_by_seed(self):
        path = write_log([make_row(5), make_row(1), make_row(3)])
        try:
            _, rounds, _ = load_rounds(path)
        finally:
            os.unlink(path)
        self.assertEqual([r["seed"] for r in rounds], [1, 3, 5])

    def test_skips_a_truncated_last_line(self):
        # The GUI can be mid-write while we read.
        path = write_log([make_row(1)])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"seed": 2, "true_res')
        try:
            raw, rounds, _ = load_rounds(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(rounds), 1)

    def test_blank_lines_are_ignored(self):
        path = write_log([make_row(1)])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n\n")
        try:
            _, rounds, _ = load_rounds(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(rounds), 1)

    def test_since_filters_by_timestamp_prefix(self):
        path = write_log([
            make_row(1, timestamp="2026-08-03T10:00:00+00:00"),
            make_row(2, timestamp="2026-08-04T10:00:00+00:00"),
        ])
        try:
            _, rounds, _ = load_rounds(path, since="2026-08-04")
        finally:
            os.unlink(path)
        self.assertEqual([r["seed"] for r in rounds], [2])


def with_config(row, epsilon=0.0, policy="temporal-hybrid"):
    row = dict(row)
    row.update({
        "cfg_policy": policy,
        "cfg_movement_continuity_epsilon": epsilon,
        "cfg_temporal_intent_net": "net.pt",
        "cfg_top_k": 12,
        "cfg_search_horizon": 72,
        "cfg_temporal_confidence": 0.6,
    })
    return row


class RunSeparationTest(unittest.TestCase):
    def test_two_configs_do_not_overwrite_each_other(self):
        # The A/B trap: the second session replays the same seeds.
        path = write_log([
            with_config(make_row(1, result="win"), epsilon=0.0),
            with_config(make_row(2, result="loss"), epsilon=0.0),
            with_config(make_row(1, result="win",
                                 timestamp="2026-08-05T00:00:00+00:00"),
                        epsilon=5.0),
        ])
        try:
            _, rounds, runs = load_rounds(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(runs), 2)
        # Newest config is selected by default, and it has only one round.
        self.assertEqual(len(rounds), 1)

    def test_config_filter_selects_the_baseline(self):
        path = write_log([
            with_config(make_row(1), epsilon=0.0),
            with_config(make_row(2), epsilon=0.0),
            with_config(make_row(1, timestamp="2026-08-05T00:00:00+00:00"),
                        epsilon=5.0),
        ])
        try:
            _, rounds, _ = load_rounds(path, config="eps=0.0")
        finally:
            os.unlink(path)
        self.assertEqual(len(rounds), 2)

    def test_legacy_rows_without_config_stay_grouped(self):
        rows = [make_row(1), make_row(2)]
        for row in rows:
            row["policy"] = "old-hybrid"
        path = write_log(rows)
        try:
            _, rounds, runs = load_rounds(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(rounds), 2)

    def test_legacy_and_configured_rows_are_separated(self):
        legacy = make_row(1)
        legacy["policy"] = "old-hybrid"
        path = write_log([legacy, with_config(make_row(1), epsilon=5.0)])
        try:
            _, _, runs = load_rounds(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(runs), 2)

    def test_same_config_still_deduplicates_by_seed(self):
        path = write_log([
            with_config(make_row(1, frames=100), epsilon=5.0),
            with_config(make_row(1, frames=999,
                                 timestamp="2026-08-05T00:00:00+00:00"),
                        epsilon=5.0),
        ])
        try:
            _, rounds, runs = load_rounds(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0]["frames"], 999)


class GroupStatsTest(unittest.TestCase):
    def test_empty_group(self):
        self.assertEqual(group_stats([]), {"rounds": 0})

    def test_zero_shot_and_kill_counts(self):
        stats = group_stats([
            make_row(1, shots=0, kills=0), make_row(2, shots=2, kills=1)])
        self.assertEqual(stats["zero_shot_rounds"], 1)
        self.assertEqual(stats["active_kill_rounds"], 1)
        self.assertEqual(stats["mean_shots"], 1.0)

    def test_fire_capture_rate(self):
        stats = group_stats([
            make_row(1, events={"fire_window": 4,
                                "captured_fire_window": 1})])
        self.assertEqual(stats["fire_capture_rate"], 0.25)

    def test_no_fire_windows_does_not_divide_by_zero(self):
        self.assertEqual(group_stats([make_row(1)])["fire_capture_rate"], 0.0)

    def test_reversal_combines_throttle_and_turn(self):
        stats = group_stats([
            make_row(1, action_frames=1000,
                     events={"throttle_reversal": 30, "turn_reversal": 20})])
        self.assertEqual(stats["reversal_per_1000"], 50.0)

    def test_missing_event_metrics_is_tolerated(self):
        row = make_row(1)
        del row["event_metrics"]
        stats = group_stats([row])
        self.assertEqual(stats["rounds"], 1)
        self.assertEqual(stats["reversal_per_1000"], 0.0)


class BadWinTest(unittest.TestCase):
    def test_clean_win_has_no_tells(self):
        self.assertEqual(classify_bad_win(make_row(1)), [])

    def test_no_shot_win_is_flagged(self):
        self.assertIn("no_shot_win", classify_bad_win(make_row(1, shots=0)))

    def test_excessive_search_is_flagged(self):
        self.assertIn("excessive_search",
                      classify_bad_win(make_row(1, search=0.40)))
        self.assertNotIn("excessive_search",
                         classify_bad_win(make_row(1, search=0.25)))

    def test_very_long_round_is_flagged(self):
        self.assertIn("very_long_round",
                      classify_bad_win(make_row(1, frames=700)))

    def test_multiple_tells_accumulate(self):
        tells = classify_bad_win(
            make_row(1, shots=0, kills=0, search=0.5, frames=900))
        self.assertEqual(len(tells), len(BAD_WIN_RULES))

    def test_bad_wins_ignores_losses(self):
        rows = [make_row(1, result="loss", shots=0),
                make_row(2, result="win", shots=0)]
        self.assertEqual([row["seed"] for row in bad_wins(rows)], [2])

    def test_every_rule_has_a_description(self):
        for name, description, predicate in BAD_WIN_RULES:
            self.assertTrue(name)
            self.assertTrue(description)
            self.assertTrue(callable(predicate))


class AnalyzeTest(unittest.TestCase):
    def _rounds(self):
        return [
            make_row(1, result="win", shots=2, kills=1),
            make_row(2, result="win", shots=0, kills=0),
            make_row(3, result="loss", death_cause="self", shots=1, kills=0,
                     issues=["terminal_loss"]),
            make_row(4, result="loss", death_cause="laika_bounce", shots=0,
                     kills=0, issues=["terminal_loss", "dead_end_navigation"]),
            make_row(5, result="double_death", death_cause="self", shots=1,
                     kills=0),
        ]

    def test_win_rate_and_results(self):
        report = analyze(self._rounds())
        self.assertEqual(report["rounds"], 5)
        self.assertEqual(report["win_rate"], 0.4)
        self.assertEqual(report["results"]["loss"], 2)

    def test_self_kill_and_zero_shot_failures(self):
        report = analyze(self._rounds())
        self.assertEqual(report["self_kill_failures"], 2)
        self.assertEqual(report["zero_shot_failures"], 1)

    def test_death_causes_include_result(self):
        report = analyze(self._rounds())
        self.assertIn("loss/self", report["death_causes"])
        self.assertIn("double_death/self", report["death_causes"])

    def test_failures_exclude_wins(self):
        report = analyze(self._rounds())
        self.assertEqual(
            sorted(row["seed"] for row in report["failures"]), [3, 4, 5])

    def test_issue_rounds_are_counted(self):
        report = analyze(self._rounds())
        self.assertEqual(report["issue_rounds"]["terminal_loss"], 2)

    def test_comparison_ratio_direction(self):
        rounds = [
            make_row(1, result="win", search=0.10),
            make_row(2, result="loss", search=0.30, death_cause="self"),
        ]
        comparison = compare_groups(rounds)
        self.assertEqual(comparison["nonwin_over_win"]["mean_search_rate"], 3.0)

    def test_comparison_with_no_failures(self):
        comparison = compare_groups([make_row(1, result="win")])
        self.assertEqual(comparison["nonwin"], {"rounds": 0})

    def test_truncated_rounds_are_counted_and_not_wins(self):
        rounds = [make_row(1, result="win"),
                  make_row(2, result="truncated", death_cause=None,
                           shots=0, kills=0, frames=2500)]
        report = analyze(rounds)
        self.assertEqual(report["truncated_rounds"], 1)
        self.assertEqual(report["win_rate"], 0.5)
        self.assertIn(2, [row["seed"] for row in report["failures"]])

    def test_truncated_round_is_not_a_bad_win(self):
        self.assertEqual(bad_wins([make_row(1, result="truncated",
                                            shots=0, kills=0)]), [])

    def test_format_report_renders(self):
        text = format_report(analyze(self._rounds()))
        self.assertIn("胜率", text)
        self.assertIn("失败局", text)
        self.assertIn("赢了但打得不好", text)


if __name__ == "__main__":
    unittest.main()
