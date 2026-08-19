import unittest

from training.self_kill_trace import (
    blind_window,
    continuation_deviation,
    fire_events,
    reaction_window,
    summarize,
    verdict,
)


def frame(index, movement=0, risk=0.0, own_bullets=0, fired=False,
          validated_movement=None, long_tail_checked=False,
          long_tail_rejected=False):
    return {
        "frame": index,
        "movement": movement,
        "risk": risk,
        "own_bullets": own_bullets,
        "fired": fired,
        "validated_movement": validated_movement,
        "long_tail_checked": long_tail_checked,
        "long_tail_rejected": long_tail_rejected,
    }


class ContinuationDeviationTest(unittest.TestCase):
    def test_immediate_departure_is_zero(self):
        frames = [frame(0, movement=3, fired=True, validated_movement=3),
                  frame(1, movement=5)]
        self.assertEqual(continuation_deviation(frames, 0, 3), 0)

    def test_counts_frames_that_held(self):
        frames = [frame(0, movement=3, fired=True, validated_movement=3),
                  frame(1, movement=3), frame(2, movement=3),
                  frame(3, movement=7)]
        self.assertEqual(continuation_deviation(frames, 0, 3), 2)

    def test_never_departs_returns_full_length(self):
        frames = [frame(index, movement=3) for index in range(4)]
        self.assertEqual(continuation_deviation(frames, 0, 3), 3)

    def test_fire_on_the_last_frame_returns_none(self):
        frames = [frame(0, movement=3, fired=True, validated_movement=3)]
        self.assertIsNone(continuation_deviation(frames, 0, 3))

    def test_only_frames_after_the_shot_count(self):
        frames = [frame(0, movement=9), frame(1, movement=3, fired=True,
                                              validated_movement=3),
                  frame(2, movement=3), frame(3, movement=1)]
        self.assertEqual(continuation_deviation(frames, 1, 3), 1)


class BlindWindowTest(unittest.TestCase):
    def test_no_own_bullets(self):
        result = blind_window([frame(0), frame(1)])
        self.assertEqual(result["airborne_frames"], 0)
        self.assertEqual(result["blind_fraction"], 0.0)

    def test_fully_blind_flight(self):
        frames = [frame(index, own_bullets=1) for index in range(10)]
        result = blind_window(frames)
        self.assertEqual(result["airborne_frames"], 10)
        self.assertEqual(result["blind_frames"], 10)
        self.assertEqual(result["blind_fraction"], 1.0)

    def test_partially_visible_flight(self):
        frames = [frame(index, own_bullets=1, risk=0.0 if index < 8 else 0.4)
                  for index in range(10)]
        result = blind_window(frames)
        self.assertEqual(result["blind_frames"], 8)
        self.assertAlmostEqual(result["blind_fraction"], 0.8)

    def test_frames_without_own_bullets_are_excluded(self):
        frames = [frame(0), frame(1, own_bullets=1), frame(2)]
        self.assertEqual(blind_window(frames)["airborne_frames"], 1)


class ReactionWindowTest(unittest.TestCase):
    def test_measures_the_final_elevated_run(self):
        frames = [frame(index, risk=0.0) for index in range(10)]
        for index in (7, 8, 9):
            frames[index]["risk"] = 0.5
        self.assertEqual(reaction_window(frames, death_frame=9), 2)

    def test_earlier_escaped_scare_is_ignored(self):
        frames = [frame(index, risk=0.0) for index in range(12)]
        for index in (1, 2, 3):
            frames[index]["risk"] = 0.9
        for index in (10, 11):
            frames[index]["risk"] = 0.5
        self.assertEqual(reaction_window(frames, death_frame=11), 1)

    def test_no_elevated_risk_returns_none(self):
        frames = [frame(index) for index in range(5)]
        self.assertIsNone(reaction_window(frames, death_frame=4))

    def test_no_death_frame_returns_none(self):
        self.assertIsNone(reaction_window([frame(0, risk=0.9)], None))

    def test_threshold_is_inclusive(self):
        frames = [frame(0, risk=0.0), frame(1, risk=0.18)]
        self.assertEqual(reaction_window(frames, 1, threshold=0.18), 0)


class FireEventsTest(unittest.TestCase):
    def test_only_frames_that_fired(self):
        frames = [frame(0), frame(1, fired=True, validated_movement=2),
                  frame(2)]
        events = fire_events(frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["frame"], 1)
        self.assertEqual(events[0]["validated_movement"], 2)


class SummarizeTest(unittest.TestCase):
    def _self_kill(self):
        frames = [frame(0, movement=3, fired=True, validated_movement=3,
                        long_tail_checked=True)]
        frames += [frame(index, movement=7, own_bullets=1)
                   for index in range(1, 40)]
        for index in (38, 39):
            frames[index]["risk"] = 0.6
        return frames

    def test_summary_fields(self):
        summary = summarize(self._self_kill(), death_frame=39,
                            death_cause="self")
        self.assertEqual(summary["shots"], 1)
        self.assertEqual(summary["min_held_frames"], 0)
        self.assertEqual(summary["death_cause"], "self")
        self.assertGreater(summary["blind_window"]["blind_fraction"], 0.9)
        self.assertEqual(summary["reaction_frames"], 1)
        self.assertAlmostEqual(summary["max_risk"], 0.6)

    def test_summary_without_fires(self):
        summary = summarize([frame(index) for index in range(5)],
                            death_frame=4, death_cause="laika_bounce")
        self.assertEqual(summary["shots"], 0)
        self.assertIsNone(summary["min_held_frames"])
        self.assertEqual(summary["continuation_deviation"], [])

    def test_only_no_safe_events_within_60_frames_of_death_are_kept(self):
        frames = [frame(index, movement=7, own_bullets=1)
                  for index in range(300)]
        summary = summarize(
            frames, death_frame=299, death_cause="self",
            no_safe_events=[{"frame": 290}, {"frame": 250}, {"frame": 10}])
        kept = [item["frame"] for item in summary["no_safe_events_near_death"]]
        self.assertEqual(kept, [290, 250])


class VerdictTest(unittest.TestCase):
    def test_all_links_hold_for_the_hypothesised_case(self):
        summary = summarize(SummarizeTest()._self_kill(), death_frame=39,
                            death_cause="self")
        links = verdict(summary)
        self.assertTrue(links["continuation_broken"])
        self.assertTrue(links["bullet_was_invisible"])
        self.assertTrue(links["no_time_to_escape"])
        self.assertTrue(links["all_links_hold"])

    def test_a_long_held_continuation_refutes_link_one(self):
        frames = [frame(0, movement=3, fired=True, validated_movement=3)]
        frames += [frame(index, movement=3, own_bullets=1)
                   for index in range(1, 60)]
        summary = summarize(frames, death_frame=59, death_cause="self")
        self.assertFalse(verdict(summary)["continuation_broken"])
        self.assertFalse(verdict(summary)["all_links_hold"])

    def test_a_visible_bullet_refutes_link_two(self):
        frames = [frame(0, movement=3, fired=True, validated_movement=3)]
        frames += [frame(index, movement=7, own_bullets=1, risk=0.5)
                   for index in range(1, 40)]
        summary = summarize(frames, death_frame=39, death_cause="self")
        self.assertFalse(verdict(summary)["bullet_was_invisible"])

    def test_ample_reaction_time_refutes_link_three(self):
        frames = [frame(0, movement=3, fired=True, validated_movement=3)]
        frames += [frame(index, movement=7, own_bullets=1)
                   for index in range(1, 200)]
        for index in range(100, 200):
            frames[index]["risk"] = 0.5
        summary = summarize(frames, death_frame=199, death_cause="self")
        self.assertFalse(verdict(summary)["no_time_to_escape"])

    def test_no_safe_event_alone_confirms_link_three(self):
        frames = [frame(0, movement=3, fired=True, validated_movement=3)]
        frames += [frame(index, movement=7, own_bullets=1)
                   for index in range(1, 200)]
        summary = summarize(frames, death_frame=199, death_cause="self",
                            no_safe_events=[{"frame": 190}])
        self.assertTrue(verdict(summary)["no_time_to_escape"])


if __name__ == "__main__":
    unittest.main()
