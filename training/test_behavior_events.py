import unittest

from training.behavior_events import BehaviorEventTracker


class BehaviorEventTrackerTest(unittest.TestCase):
    def test_contiguous_stall_is_one_event_with_duration(self):
        tracker = BehaviorEventTracker()
        for frame in range(4):
            tracker.update_episode("stutter_stall", True, frame)
        tracker.update_episode("stutter_stall", False, 4)
        tracker.finish(4)

        summary = tracker.summary()
        self.assertEqual(summary["events"]["stutter_stall"], 1)
        self.assertEqual(summary["durations"]["stutter_stall_frames"], 4)

    def test_short_false_gap_is_merged_into_same_episode(self):
        tracker = BehaviorEventTracker(episode_merge_gap=2)
        tracker.update_episode("stutter_stall", True, 0)
        tracker.update_episode("stutter_stall", False, 1)
        tracker.update_episode("stutter_stall", True, 2)
        tracker.finish(3)

        summary = tracker.summary()
        self.assertEqual(summary["events"]["stutter_stall"], 1)
        self.assertEqual(summary["durations"]["stutter_stall_frames"], 3)

    def test_fire_window_counts_once_and_records_capture_latency(self):
        tracker = BehaviorEventTracker(fire_window_min_frames=3)
        for frame in range(5):
            tracker.update_fire_window(
                frame, clear_line=True, fired=(frame == 2))
        tracker.update_fire_window(5, clear_line=False, fired=False)

        summary = tracker.summary()
        self.assertEqual(summary["events"]["fire_window"], 1)
        self.assertEqual(summary["events"]["captured_fire_window"], 1)
        self.assertNotIn("missed_fire_window", summary["events"])
        self.assertEqual(summary["mean_fire_response_frames"], 2)

    def test_short_window_is_not_reported_as_missed(self):
        tracker = BehaviorEventTracker(fire_window_min_frames=3)
        tracker.update_fire_window(0, clear_line=True, fired=False)
        tracker.update_fire_window(1, clear_line=False, fired=False)
        self.assertNotIn("fire_window", tracker.summary()["events"])

    def test_action_switches_and_true_reversals_are_separate(self):
        tracker = BehaviorEventTracker()
        for action in ((2, 1, 0), (2, 0, 0), (0, 2, 0), (1, 1, 0)):
            tracker.update_action(action)

        events = tracker.summary()["events"]
        self.assertEqual(events["movement_switch"], 3)
        self.assertEqual(events["throttle_reversal"], 1)
        self.assertEqual(events["turn_reversal"], 1)


if __name__ == "__main__":
    unittest.main()
