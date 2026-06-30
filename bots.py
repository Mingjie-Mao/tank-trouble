from __future__ import annotations

from collections import deque
import math
from typing import Any

import numpy as np


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class TeacherTankBot:
    """
    规则型老师 AI。

    优先级顺序：
    1. 卡住恢复
    2. 躲避明显来弹
    3. 有直线射界时瞄准并开火
    4. 否则沿导航图追击目标
    """

    def __init__(self, allow_fire: bool = True) -> None:
        self.allow_fire = allow_fire
        self.recent_positions: deque[tuple[float, float]] = deque(maxlen=16)
        self.unstuck_turn_dir = 1

    def act(self, info: dict[str, Any]) -> np.ndarray:
        self_state = info["enemy_tank"]
        target_state = info["player_tank"]
        bullets = info["bullets"]
        walls = info["walls"]
        world = info["world"]
        nav_points = info.get("nav_points", [])
        nav_graph = info.get("nav_graph", [])
        rules = info.get("rules", {})

        self.recent_positions.append((self_state["x"], self_state["y"]))

        if self._is_stuck():
            return self._unstick_action()

        dodge = self._dodge_if_needed(self_state, bullets)
        if dodge is not None:
            return dodge

        if info.get("enemy_has_line_of_sight", False):
            return self._fight_action(self_state, target_state)

        next_anchor = self._next_chase_anchor(self_state, target_state, nav_points, nav_graph)
        if next_anchor is not None:
            return self._move_to_point(self_state, next_anchor, allow_fire=False)

        # Fallback if graph data is unavailable.
        return self._move_to_point(self_state, (target_state["x"], target_state["y"]), allow_fire=False)

    def _fight_action(self, self_state: dict[str, float], target_state: dict[str, float]) -> np.ndarray:
        target_angle = math.atan2(target_state["y"] - self_state["y"], target_state["x"] - self_state["x"])
        angle_delta = wrap_angle(target_angle - self_state["angle"])
        distance = math.hypot(target_state["x"] - self_state["x"], target_state["y"] - self_state["y"])

        if angle_delta > 0.08:
            turn = 2
        elif angle_delta < -0.08:
            turn = 0
        else:
            turn = 1

        if distance > 230.0:
            throttle = 2
        elif distance < 110.0:
            throttle = 0
        else:
            throttle = 2 if abs(angle_delta) < 0.35 else 1

        fire = 1 if self.allow_fire and abs(angle_delta) < 0.12 and self_state["cooldown"] <= 0.0 else 0
        return np.array([throttle, turn, fire], dtype=np.int64)

    def _move_to_point(
        self,
        self_state: dict[str, float],
        point: tuple[float, float],
        allow_fire: bool,
    ) -> np.ndarray:
        dx = point[0] - self_state["x"]
        dy = point[1] - self_state["y"]
        target_angle = math.atan2(dy, dx)
        angle_delta = wrap_angle(target_angle - self_state["angle"])
        distance = math.hypot(dx, dy)

        if angle_delta > 0.10:
            turn = 2
        elif angle_delta < -0.10:
            turn = 0
        else:
            turn = 1

        if abs(angle_delta) < 1.0:
            throttle = 2 if distance > 18.0 else 1
        else:
            throttle = 1

        fire = 0
        if self.allow_fire and allow_fire and abs(angle_delta) < 0.12 and self_state["cooldown"] <= 0.0:
            fire = 1

        return np.array([throttle, turn, fire], dtype=np.int64)

    def _dodge_if_needed(
        self,
        self_state: dict[str, float],
        bullets: list[dict[str, float]],
    ) -> np.ndarray | None:
        sx = self_state["x"]
        sy = self_state["y"]
        facing = self_state["angle"]

        best_threat: tuple[float, dict[str, float]] | None = None
        for bullet in bullets:
            if bullet["owner"] != 0:
                continue

            rel_x = sx - bullet["x"]
            rel_y = sy - bullet["y"]
            vel_x = bullet["vx"]
            vel_y = bullet["vy"]
            speed_sq = vel_x * vel_x + vel_y * vel_y
            if speed_sq <= 1e-6:
                continue

            t_closest = (rel_x * vel_x + rel_y * vel_y) / speed_sq
            if t_closest < 0.0 or t_closest > 0.9:
                continue

            closest_x = bullet["x"] + vel_x * t_closest
            closest_y = bullet["y"] + vel_y * t_closest
            miss_distance = math.hypot(closest_x - sx, closest_y - sy)
            if miss_distance > 34.0:
                continue

            if best_threat is None or t_closest < best_threat[0]:
                best_threat = (
                    t_closest,
                    {"vx": vel_x, "vy": vel_y},
                )

        if best_threat is None:
            return None

        threat = best_threat[1]
        dodge_heading = math.atan2(threat["vy"], threat["vx"]) + math.pi / 2.0
        alt_heading = dodge_heading + math.pi

        main_delta = wrap_angle(dodge_heading - facing)
        alt_delta = wrap_angle(alt_heading - facing)
        chosen_delta = main_delta if abs(main_delta) <= abs(alt_delta) else alt_delta

        if chosen_delta > 0.10:
            turn = 2
        elif chosen_delta < -0.10:
            turn = 0
        else:
            turn = 1

        return np.array([2, turn, 0], dtype=np.int64)

    def _next_chase_anchor(
        self,
        self_state: dict[str, float],
        target_state: dict[str, float],
        nav_points: list[tuple[float, float]],
        nav_graph: list[list[int]],
    ) -> tuple[float, float] | None:
        if not nav_points:
            return None

        start_idx = self._nearest_anchor_index((self_state["x"], self_state["y"]), nav_points)
        goal_idx = self._nearest_anchor_index((target_state["x"], target_state["y"]), nav_points)
        if start_idx == goal_idx:
            return nav_points[goal_idx]

        if not nav_graph or len(nav_graph) != len(nav_points):
            return nav_points[goal_idx]

        prev = self._bfs_prev(start_idx, goal_idx, nav_graph)
        if goal_idx not in prev:
            return nav_points[goal_idx]

        path = [goal_idx]
        current = goal_idx
        while prev[current] is not None:
            current = prev[current]
            path.append(current)
        path.reverse()

        if len(path) >= 2:
            return nav_points[path[1]]
        return nav_points[path[0]]

    def _bfs_prev(
        self,
        start_idx: int,
        goal_idx: int,
        nav_graph: list[list[int]],
    ) -> dict[int, int | None]:
        queue = deque([start_idx])
        prev: dict[int, int | None] = {start_idx: None}

        while queue:
            current = queue.popleft()
            if current == goal_idx:
                break
            for neighbor in nav_graph[current]:
                if neighbor in prev:
                    continue
                prev[neighbor] = current
                queue.append(neighbor)

        return prev

    @staticmethod
    def _nearest_anchor_index(
        point: tuple[float, float],
        anchors: list[tuple[float, float]],
    ) -> int:
        return min(range(len(anchors)), key=lambda idx: math.hypot(point[0] - anchors[idx][0], point[1] - anchors[idx][1]))

    def _unstick_action(self) -> np.ndarray:
        self.unstuck_turn_dir *= -1
        turn = 2 if self.unstuck_turn_dir > 0 else 0
        return np.array([2, turn, 0], dtype=np.int64)

    def _is_stuck(self) -> bool:
        if len(self.recent_positions) < self.recent_positions.maxlen:
            return False
        start = self.recent_positions[0]
        end = self.recent_positions[-1]
        return math.hypot(end[0] - start[0], end[1] - start[1]) < 10.0


HeuristicTankBot = TeacherTankBot
