import unittest

import numpy as np

from training.dagger_correction_recorder import (
    DATASET_KEYS,
    DEFAULT_TAG_WEIGHTS,
    DISAGREEMENT_TAGS,
    TAG_ACCEPTED,
    TAG_FIRE_REJECTED,
    TAG_HARDCASE,
    TAG_SEARCH_OVERRIDE,
    TAG_SUCCESSOR_SHIELD_OVERRIDE,
    TAG_TERMINAL_WINDOW,
    TAG_TOPOLOGY_ABORT,
    TAG_UNSAFE_TEMPORAL,
    build_correction_record,
    build_dataset,
    classify_correction,
    correction_weight,
    disagreement_rate,
    round_report,
    sequence_targets,
    tag_round,
)


def make_record(**overrides):
    fields = dict(
        frame=0,
        features=np.zeros(8, dtype=np.float32),
        executed_movement=3,
        executed_fire=False,
        network_movement=3,
        temporal_movement=None,
        temporal_confidence=0.0,
        topology_movement=None,
        proposed_movement=3,
        proposed_fire=False,
        reason=None,
        category=None,
        full_search=False,
        audit_failed=False,
        safe_root_count=18,
        successor_shield_triggered=False,
        interventions=(),
        long_tail_fire_rejected=False,
        risk=0.0,
        topology_active=False,
        topology_kind=None,
        topology_started=False,
        topology_aborted=False,
    )
    fields.update(overrides)
    return build_correction_record(**fields)


class BuildRecordTest(unittest.TestCase):
    def test_defaults_are_normalised(self):
        record = make_record()
        self.assertEqual(record["reason"], "network")
        self.assertEqual(record["category"], "standard")
        self.assertEqual(record["topology_kind"], "none")
        self.assertFalse(record["temporal_used"])
        self.assertFalse(record["movement_corrected"])
        self.assertFalse(record["temporal_corrected"])

    def test_movement_correction_detected(self):
        record = make_record(proposed_movement=5, executed_movement=3)
        self.assertTrue(record["movement_corrected"])

    def test_temporal_correction_requires_temporal_proposal(self):
        # Movement changed but the GRU never proposed anything: not a GRU fault.
        record = make_record(proposed_movement=5, executed_movement=3)
        self.assertFalse(record["temporal_corrected"])
        record = make_record(
            temporal_movement=5, proposed_movement=5, executed_movement=3)
        self.assertTrue(record["temporal_corrected"])

    def test_temporal_accepted_is_not_a_correction(self):
        record = make_record(
            temporal_movement=3, proposed_movement=3, executed_movement=3)
        self.assertTrue(record["temporal_used"])
        self.assertFalse(record["temporal_corrected"])

    def test_fire_correction_detected(self):
        record = make_record(proposed_fire=True, executed_fire=False)
        self.assertTrue(record["fire_corrected"])


class ClassifyTest(unittest.TestCase):
    def test_accepted(self):
        self.assertEqual(classify_correction(make_record()), TAG_ACCEPTED)

    def test_terminal_window_wins(self):
        record = make_record(
            temporal_movement=5, proposed_movement=5, executed_movement=3)
        self.assertEqual(
            classify_correction(record, terminal_window=True),
            TAG_TERMINAL_WINDOW)

    def test_successor_shield_override(self):
        record = make_record(
            full_search=True,
            interventions=("successor_shield_checked",
                           "successor_shield_override"),
            proposed_movement=1, executed_movement=3)
        self.assertEqual(
            classify_correction(record), TAG_SUCCESSOR_SHIELD_OVERRIDE)

    def test_fire_rejected(self):
        record = make_record(long_tail_fire_rejected=True)
        self.assertEqual(classify_correction(record), TAG_FIRE_REJECTED)

    def test_unsafe_temporal_beats_search_override(self):
        record = make_record(
            temporal_movement=5, proposed_movement=5, executed_movement=3,
            full_search=True, reason="unsafe")
        self.assertEqual(classify_correction(record), TAG_UNSAFE_TEMPORAL)

    def test_search_override_without_temporal(self):
        record = make_record(
            proposed_movement=5, executed_movement=3,
            full_search=True, reason="unsafe")
        self.assertEqual(classify_correction(record), TAG_SEARCH_OVERRIDE)

    def test_search_override_requires_a_search(self):
        # Movement differs only because of the hysteresis/commit path.
        record = make_record(proposed_movement=5, executed_movement=3,
                             reason="committed")
        self.assertEqual(classify_correction(record), TAG_ACCEPTED)

    def test_topology_abort(self):
        record = make_record(topology_aborted=True)
        self.assertEqual(classify_correction(record), TAG_TOPOLOGY_ABORT)

    def test_hardcase_event(self):
        record = make_record(category="dead_end_stall")
        self.assertEqual(classify_correction(record), TAG_HARDCASE)

    def test_hardcase_does_not_mask_real_correction(self):
        record = make_record(
            category="stutter_stall", temporal_movement=5,
            proposed_movement=5, executed_movement=3)
        self.assertEqual(classify_correction(record), TAG_UNSAFE_TEMPORAL)


class WeightTest(unittest.TestCase):
    def test_accepted_baseline_is_one(self):
        record = make_record()
        self.assertAlmostEqual(
            correction_weight(TAG_ACCEPTED, record), 1.0)

    def test_narrow_safe_roots_increase_weight(self):
        base = make_record()
        narrow = make_record(safe_root_count=2)
        self.assertGreater(
            correction_weight(TAG_ACCEPTED, narrow),
            correction_weight(TAG_ACCEPTED, base))

    def test_zero_safe_roots_is_not_treated_as_narrow(self):
        # safe_root_count == 0 means the search found nothing; the multiplier
        # is guarded so it does not silently apply to the default path.
        record = make_record(safe_root_count=0)
        self.assertAlmostEqual(
            correction_weight(TAG_ACCEPTED, record), 1.0)

    def test_topology_multiplier(self):
        record = make_record(topology_active=True)
        self.assertAlmostEqual(
            correction_weight(TAG_ACCEPTED, record), 1.2)

    def test_corrections_outweigh_accepted(self):
        record = make_record()
        for tag in (TAG_UNSAFE_TEMPORAL, TAG_SUCCESSOR_SHIELD_OVERRIDE,
                    TAG_TERMINAL_WINDOW):
            self.assertGreater(
                correction_weight(tag, record),
                correction_weight(TAG_ACCEPTED, record))

    def test_weights_are_bounded(self):
        # Guard against the collapse mode: no state may dominate a batch.
        record = make_record(safe_root_count=1, topology_active=True)
        worst = max(
            correction_weight(tag, record) for tag in DEFAULT_TAG_WEIGHTS)
        self.assertLessEqual(worst, 12.0)


class TagRoundTest(unittest.TestCase):
    def test_win_round_has_no_terminal_window(self):
        records = [make_record(frame=index) for index in range(10)]
        tags, weights = tag_round(records, "win", terminal_window=5)
        self.assertNotIn(TAG_TERMINAL_WINDOW, tags)
        self.assertEqual(len(weights), 10)

    def test_loss_round_keeps_pre_death_window(self):
        records = [make_record(frame=index) for index in range(10)]
        tags, _ = tag_round(records, "loss", terminal_window=4)
        self.assertEqual(tags[-4:], [TAG_TERMINAL_WINDOW] * 4)
        self.assertEqual(tags[:6], [TAG_ACCEPTED] * 6)

    def test_double_death_is_also_terminal(self):
        records = [make_record(frame=index) for index in range(3)]
        tags, _ = tag_round(records, "double_death", terminal_window=2)
        self.assertEqual(tags[-2:], [TAG_TERMINAL_WINDOW] * 2)

    def test_window_longer_than_round_tags_everything(self):
        records = [make_record(frame=index) for index in range(3)]
        tags, _ = tag_round(records, "loss", terminal_window=90)
        self.assertEqual(tags, [TAG_TERMINAL_WINDOW] * 3)

    def test_empty_round(self):
        tags, weights = tag_round([], "loss")
        self.assertEqual(tags, [])
        self.assertEqual(weights, [])


class ReportTest(unittest.TestCase):
    def test_disagreement_rate(self):
        self.assertEqual(disagreement_rate([]), 0.0)
        self.assertEqual(
            disagreement_rate([TAG_ACCEPTED, TAG_UNSAFE_TEMPORAL]), 0.5)
        self.assertTrue(
            all(tag in DISAGREEMENT_TAGS for tag in
                (TAG_UNSAFE_TEMPORAL, TAG_SEARCH_OVERRIDE,
                 TAG_SUCCESSOR_SHIELD_OVERRIDE, TAG_TERMINAL_WINDOW)))
        self.assertNotIn(TAG_ACCEPTED, DISAGREEMENT_TAGS)
        self.assertNotIn(TAG_HARDCASE, DISAGREEMENT_TAGS)

    def test_round_report_rates(self):
        records = [
            make_record(temporal_movement=1, proposed_movement=1,
                        executed_movement=1),
            make_record(temporal_movement=1, proposed_movement=1,
                        executed_movement=2, full_search=True,
                        reason="unsafe"),
            make_record(),
        ]
        tags, weights = tag_round(records, "win")
        report = round_report(970000, "win", 3, records, tags, weights)
        self.assertEqual(report["seed"], 970000)
        self.assertEqual(report["states"], 3)
        self.assertEqual(report["temporal_frames"], 2)
        self.assertAlmostEqual(report["temporal_correction_rate"], 0.5)
        self.assertAlmostEqual(report["movement_correction_rate"], 1 / 3)
        self.assertAlmostEqual(report["disagreement_rate"], 1 / 3)
        self.assertGreater(report["mean_weight"], 0.0)


def make_round(seed, result, movements, frames=None, **row_overrides):
    rows = []
    for index, movement in enumerate(movements):
        row = make_record(frame=index, executed_movement=movement,
                          **row_overrides)
        row["progress"] = 0.0
        rows.append(row)
    return {
        "seed": seed,
        "result": result,
        "frames": len(movements) if frames is None else frames,
        "rows": rows,
    }


class SequenceTargetsTest(unittest.TestCase):
    def test_interrupt_marks_the_frame_before_a_change(self):
        _, _, interrupt = sequence_targets([1, 1, 2, 2])
        self.assertEqual(interrupt.tolist(), [0.0, 1.0, 0.0, 0.0])

    def test_single_frame(self):
        remaining, hold, interrupt = sequence_targets([4])
        self.assertEqual(remaining.tolist(), [1])
        self.assertEqual(len(hold), 1)
        self.assertEqual(interrupt.tolist(), [0.0])


class BuildDatasetTest(unittest.TestCase):
    def test_payload_has_every_declared_key(self):
        rounds = [make_round(970000, "win", [0, 1, 1, 2])]
        payload, _ = build_dataset(rounds)
        self.assertEqual(set(payload), set(DATASET_KEYS))
        for key in DATASET_KEYS:
            self.assertEqual(len(payload[key]), 4, key)

    def test_losing_rounds_are_kept(self):
        # The v1 pipeline drops these entirely; that is the bug being fixed.
        rounds = [make_round(970001, "loss", [1, 1, 1])]
        payload, reports = build_dataset(rounds, terminal_window=2)
        self.assertEqual(len(payload["movement"]), 3)
        self.assertFalse(payload["round_result_win"].any())
        self.assertEqual(reports[0]["result"], "loss")

    def test_terminal_window_is_upweighted(self):
        rounds = [make_round(970001, "loss", [1] * 5)]
        payload, _ = build_dataset(rounds, terminal_window=2)
        weights = payload["weight"].tolist()
        self.assertEqual(weights[:3], [1.0, 1.0, 1.0])
        self.assertTrue(all(value > 1.0 for value in weights[3:]))

    def test_empty_rounds_are_skipped_not_fatal(self):
        rounds = [make_round(970000, "win", [0, 1]),
                  {"seed": 970002, "result": "win", "frames": 0, "rows": []}]
        payload, reports = build_dataset(rounds)
        self.assertEqual(len(reports), 1)
        self.assertEqual(len(payload["movement"]), 2)

    def test_all_empty_raises(self):
        rounds = [{"seed": 1, "result": "win", "frames": 0, "rows": []}]
        with self.assertRaises(RuntimeError):
            build_dataset(rounds)

    def test_round_seed_grouping_is_preserved(self):
        rounds = [make_round(970000, "win", [0, 1]),
                  make_round(970001, "win", [2, 3, 4])]
        payload, _ = build_dataset(rounds)
        self.assertEqual(payload["round_seed"].tolist(),
                         [970000, 970000, 970001, 970001, 970001])
        self.assertEqual(payload["frame"].tolist(), [0, 1, 0, 1, 2])

    def test_hold_targets_are_computed_per_round(self):
        # A run must not bleed across the round boundary.
        rounds = [make_round(970000, "win", [1, 1]),
                  make_round(970001, "win", [1, 1])]
        payload, _ = build_dataset(rounds)
        self.assertEqual(payload["interrupt"].tolist(),
                         [0.0, 0.0, 0.0, 0.0])

    def test_keep_accepted_fraction_never_drops_disagreements(self):
        rounds = [
            make_round(970000, "win", [1] * 20),
            make_round(970001, "win", [1] * 20, temporal_movement=5,
                       proposed_movement=5),
        ]
        payload, _ = build_dataset(
            rounds, keep_accepted_fraction=0.0, rng_seed=7)
        tags = set(payload["tag"].tolist())
        self.assertEqual(tags, {TAG_UNSAFE_TEMPORAL})
        self.assertEqual(len(payload["movement"]), 20)

    def test_correction_columns_round_trip(self):
        rounds = [make_round(970000, "win", [3, 3], temporal_movement=5,
                             proposed_movement=5, full_search=True,
                             reason="unsafe", safe_root_count=2)]
        payload, _ = build_dataset(rounds)
        self.assertTrue(payload["temporal_corrected"].all())
        self.assertTrue(payload["movement_corrected"].all())
        self.assertTrue(payload["temporal_used"].all())
        self.assertTrue(payload["full_search"].all())
        self.assertEqual(payload["safe_root_count"].tolist(), [2, 2])
        self.assertEqual(set(payload["tag"].tolist()), {TAG_UNSAFE_TEMPORAL})

    def test_labels_are_the_executed_movement(self):
        # The temporal net must be trained toward what the exact teacher
        # actually executed, never toward its own rejected proposal.
        rounds = [make_round(970000, "win", [3], temporal_movement=5,
                             proposed_movement=5)]
        payload, _ = build_dataset(rounds)
        self.assertEqual(payload["movement"].tolist(), [3])

    def test_dtypes_are_npz_safe(self):
        rounds = [make_round(970000, "win", [0, 1])]
        payload, _ = build_dataset(rounds)
        for key, value in payload.items():
            self.assertNotEqual(value.dtype.kind, "O", key)


if __name__ == "__main__":
    unittest.main()
