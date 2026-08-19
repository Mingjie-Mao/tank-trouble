"""Maze-topology goals for persistent, inspectable movement intent."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from tank_trouble_original.maze import h_open, v_open


GOAL_KINDS = (
    "escape_dead_end",
    "seek_firing_position",
    "hold_position",
)
TOPOLOGY_FEATURE_DIM = 12


def tank_cell(game, tank):
    width, height = len(game.maze), len(game.maze[0])
    return (
        max(0, min(width - 1, int(tank.x // game.scale))),
        max(0, min(height - 1, int(tank.y // game.scale))),
    )


def cardinal_neighbors(maze, cell):
    x, y = cell
    width, height = len(maze), len(maze[0])
    neighbors = []
    if x > 0 and v_open(maze, x, y):
        neighbors.append((x - 1, y))
    if x < width - 1 and v_open(maze, x + 1, y):
        neighbors.append((x + 1, y))
    if y > 0 and h_open(maze, x, y - 1):
        neighbors.append((x, y - 1))
    if y < height - 1 and h_open(maze, x, y):
        neighbors.append((x, y + 1))
    return tuple(neighbors)


def dead_end_depth(game, cell):
    x, y = cell
    value = game.dead_ends[x][y]
    return 0.0 if value is None else float(value)


def shortest_cardinal_path(maze, start, goal):
    if start == goal:
        return ()
    frontier = deque([start])
    parent = {start: None}
    while frontier:
        cell = frontier.popleft()
        for neighbor in cardinal_neighbors(maze, cell):
            if neighbor in parent:
                continue
            parent[neighbor] = cell
            if neighbor == goal:
                frontier.clear()
                break
            frontier.append(neighbor)
    if goal not in parent:
        return ()
    path = []
    cell = goal
    while cell != start:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return tuple(path)


def nearest_dead_end_exit(game, start):
    frontier = deque([start])
    parent = {start: None}
    target = None
    while frontier:
        cell = frontier.popleft()
        if (cell != start and dead_end_depth(game, cell) <= 0.0
                and len(cardinal_neighbors(game.maze, cell)) >= 2):
            target = cell
            break
        neighbors = sorted(
            cardinal_neighbors(game.maze, cell),
            key=lambda item: dead_end_depth(game, item),
        )
        for neighbor in neighbors:
            if neighbor in parent:
                continue
            parent[neighbor] = cell
            frontier.append(neighbor)
    if target is None:
        return start, ()
    path = []
    cell = target
    while cell != start:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return target, tuple(path)


def _relative_direction(game, tank, target):
    tx = (target[0] + 0.5) * game.scale
    ty = (target[1] + 0.5) * game.scale
    dx, dy = tx - tank.x, ty - tank.y
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 0.0
    dx, dy = dx / length, dy / length
    forward = (tank.rotation - 90.0) * math.pi / 180.0
    cos_forward, sin_forward = math.cos(forward), math.sin(forward)
    return (
        dx * cos_forward + dy * sin_forward,
        -dx * sin_forward + dy * cos_forward,
    )


def drive_action_to_cell(game, tank, target, can_reverse=True):
    """Mirror Laika's driveToPos steering for one discrete control frame."""
    tx = (target[0] + 0.5) * game.scale
    ty = (target[1] + 0.5) * game.scale
    cur = float(tank.rotation)
    dx, dy = tx - tank.x, ty - tank.y
    if dx > 0:
        target_angle = 90.0 + math.degrees(math.atan(dy / dx))
    elif dx < 0:
        target_angle = -90.0 + math.degrees(math.atan(dy / dx))
    elif dy > 0:
        target_angle = 180.0
    elif dy < 0:
        target_angle = 0.0
    else:
        target_angle = cur
    target_angle = tank.turn_speed * round(target_angle / tank.turn_speed)
    reverse = False
    if can_reverse and 90.0 < abs(target_angle - cur) < 270.0:
        reverse = True
        target_angle += 180.0
        if target_angle > 180.0:
            target_angle -= 360.0

    turn = 1
    if target_angle > cur:
        if abs(target_angle - cur) > 180.0:
            if abs(target_angle - cur) < 360.0 - tank.turn_speed:
                turn = 0
        elif abs(target_angle - cur) > tank.turn_speed:
            turn = 2
    elif target_angle < cur:
        if abs(target_angle - cur) > 180.0:
            if abs(target_angle - cur) < 360.0 - tank.turn_speed:
                turn = 2
        elif abs(target_angle - cur) > tank.turn_speed:
            turn = 0

    throttle = 0 if reverse else 2
    if 45.0 < abs(target_angle - cur) < 315.0:
        throttle = 1
    return throttle, turn, 0


@dataclass(frozen=True)
class TopologyGoal:
    kind: str
    current: tuple[int, int]
    target: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    current_dead_end_depth: float
    target_dead_end_depth: float
    current_exits: int

    @property
    def next_cell(self):
        return self.path[0] if self.path else self.target


class MapTopologyPlanner:
    def choose_goal(self, game, analyzer):
        me = game.tanks[0]
        current = tank_cell(game, me)
        current_depth = dead_end_depth(game, current)
        exits = len(cardinal_neighbors(game.maze, current))
        if current_depth > 0.0 or exits <= 1:
            target, path = nearest_dead_end_exit(game, current)
            kind = "escape_dead_end" if path else "hold_position"
        else:
            _, target = analyzer.nearest_firing_position(game)
            target = tuple(target)
            path = shortest_cardinal_path(game.maze, current, target)
            kind = "seek_firing_position" if path else "hold_position"
        return TopologyGoal(
            kind=kind,
            current=current,
            target=target,
            path=path,
            current_dead_end_depth=current_depth,
            target_dead_end_depth=dead_end_depth(game, target),
            current_exits=exits,
        )

    def features(self, game, goal):
        me = game.tanks[0]
        kind = np.zeros(len(GOAL_KINDS), dtype=np.float32)
        kind[GOAL_KINDS.index(goal.kind)] = 1.0
        next_forward, next_turn = _relative_direction(
            game, me, goal.next_cell)
        goal_forward, goal_turn = _relative_direction(
            game, me, goal.target)
        path_length = min(len(goal.path), 20) / 20.0
        current_depth = min(goal.current_dead_end_depth, 8.0) / 8.0
        target_depth = min(goal.target_dead_end_depth, 8.0) / 8.0
        target_changed = float(goal.current != goal.target)
        return np.asarray((
            *kind,
            path_length,
            current_depth,
            target_depth,
            goal.current_exits / 4.0,
            next_forward,
            next_turn,
            goal_forward,
            goal_turn,
            target_changed,
        ), dtype=np.float32)


class MovementHysteresis:
    """Suppress frame-to-frame movement reversals without delaying interrupts."""

    def __init__(self, hold_frames=4):
        self.hold_frames = max(0, int(hold_frames))
        self.action = None
        self.remaining = 0
        self.suppressions = 0

    def reset(self):
        self.action = None
        self.remaining = 0

    def choose(self, action, interrupt=False):
        action = tuple(int(value) for value in action)
        if self.hold_frames <= 1 or interrupt or action[2] == 1:
            self.reset()
            return action
        movement = (action[0], action[1], 0)
        if self.action is not None and self.remaining > 0:
            self.remaining -= 1
            if movement != self.action:
                self.suppressions += 1
            return self.action
        self.action = movement
        self.remaining = self.hold_frames - 1
        return movement
