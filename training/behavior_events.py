"""Event-level behavior metrics for policy evaluation.

Frame counters overstate long-lived problems: one missed firing opportunity or
one stall can remain true for dozens of frames.  These trackers collapse
contiguous frames into independent episodes while retaining duration and
response-latency information.
"""

from __future__ import annotations

from collections import Counter


class BehaviorEventTracker:
    def __init__(self, fire_window_min_frames=8, episode_merge_gap=4):
        self.fire_window_min_frames = max(1, int(fire_window_min_frames))
        self.episode_merge_gap = max(0, int(episode_merge_gap))
        self.events = Counter()
        self.durations = Counter()
        self.active = {}
        self.fire_window = None
        self.fire_response_frames = []
        self.last_movement = None
        self.action_frames = 0

    def update_action(self, action):
        throttle, turn = int(action[0]), int(action[1])
        movement = (throttle, turn)
        self.action_frames += 1
        if movement == (1, 1):
            self.durations["stationary_frames"] += 1
        if self.last_movement is not None and movement != self.last_movement:
            self.events["movement_switch"] += 1
            if {movement[0], self.last_movement[0]} == {0, 2}:
                self.events["throttle_reversal"] += 1
            if {movement[1], self.last_movement[1]} == {0, 2}:
                self.events["turn_reversal"] += 1
        self.last_movement = movement

    def update_episode(self, name, active, frame):
        frame = int(frame)
        episode = self.active.get(name)
        if active and episode is None:
            self.active[name] = {"start": frame, "last_true": frame}
            self.events[name] += 1
        elif active:
            episode["last_true"] = frame
        elif (episode is not None
              and frame - episode["last_true"] > self.episode_merge_gap):
            self.durations[f"{name}_frames"] += max(
                1, episode["last_true"] + 1 - episode["start"])
            del self.active[name]

    def update_fire_window(self, frame, clear_line, fired, enemy_alive=True):
        frame = int(frame)
        clear_line = bool(clear_line and enemy_alive)
        if self.fire_window is None and clear_line:
            self.fire_window = {"start": frame, "fired": False}

        window = self.fire_window
        if window is None:
            return
        if fired and not window["fired"]:
            window["fired"] = True
            self.fire_response_frames.append(frame - window["start"])

        if not clear_line:
            self._close_fire_window(frame)

    def _close_fire_window(self, frame):
        window = self.fire_window
        if window is None:
            return
        duration = max(1, int(frame) - int(window["start"]))
        if duration >= self.fire_window_min_frames:
            self.events["fire_window"] += 1
            self.durations["fire_window_frames"] += duration
            key = "captured_fire_window" if window["fired"] else "missed_fire_window"
            self.events[key] += 1
        self.fire_window = None

    def finish(self, frame):
        frame = int(frame)
        self._close_fire_window(frame)
        for name, episode in list(self.active.items()):
            self.durations[f"{name}_frames"] += max(
                1, min(frame, episode["last_true"] + 1)
                - episode["start"])
            del self.active[name]

    def summary(self):
        windows = int(self.events["fire_window"])
        captured = int(self.events["captured_fire_window"])
        response = self.fire_response_frames
        return {
            "events": dict(self.events),
            "durations": dict(self.durations),
            "fire_window_capture_rate": (
                captured / windows if windows else None),
            "mean_fire_response_frames": (
                sum(response) / len(response) if response else None),
            "fire_response_count": len(response),
            "fire_response_total_frames": sum(response),
            "action_frames": int(self.action_frames),
        }
