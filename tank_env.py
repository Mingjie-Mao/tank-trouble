from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from bots import TeacherTankBot


@dataclass
class TankState:
    x: float
    y: float
    angle: float
    cooldown: float = 0.0
    alive: bool = True


@dataclass
class BulletState:
    x: float
    y: float
    vx: float
    vy: float
    owner: int
    lifetime: float
    bounces_left: int
    age_steps: int = 0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class TankDuelEnv(gym.Env):
    """
    Tank Trouble-style duel environment with:
    - random line-maze generation
    - continuous 2D movement
    - bouncing bullets
    - a stronger heuristic opponent
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: str | None = None, arena_preset: str = "maze_random") -> None:
        super().__init__()
        self.render_mode = render_mode
        self.arena_preset = arena_preset

        if self.arena_preset == "micro_fixed":
            self.grid_cols = 4
            self.grid_rows = 3
            self.cell_size = 96.0
            self.extra_connection_chance = 0.0
            self.max_steps = 360
        elif self.arena_preset == "small_fixed":
            self.grid_cols = 5
            self.grid_rows = 4
            self.cell_size = 84.0
            self.extra_connection_chance = 0.0
            self.max_steps = 520
        elif self.arena_preset == "cover_fixed":
            self.grid_cols = 5
            self.grid_rows = 4
            self.cell_size = 84.0
            self.extra_connection_chance = 0.0
            self.max_steps = 520
        elif self.arena_preset == "maze_random":
            self.grid_cols = 13
            self.grid_rows = 9
            self.cell_size = 60.0
            self.extra_connection_chance = 0.14
            self.max_steps = 1200
        else:
            raise ValueError(f"Unsupported arena_preset: {self.arena_preset}")

        self.world_width = self.grid_cols * self.cell_size
        self.world_height = self.grid_rows * self.cell_size
        self.cell_w = self.cell_size
        self.cell_h = self.cell_size

        self.wall_render_width = 4
        self.wall_collision_padding = 3.0

        self.tank_radius = 16.0
        self.tank_body_w = 22.0
        self.tank_body_h = 28.0
        self.tank_speed = 126.0
        self.turn_speed = math.radians(185.0)
        self.bullet_speed = 280.0
        self.bullet_radius = 3.0
        self.fire_cooldown = 0.62
        self.max_bounces = 2
        self.max_bullets = 5
        self.dt = 1.0 / 20.0
        self.self_hit_grace_steps = 2

        self.ray_count = 24
        self.bullet_slots = 4
        obs_size = 10 + 1 + self.ray_count * 3 + self.bullet_slots * 6

        self.action_space = spaces.MultiDiscrete(np.array([3, 3, 2], dtype=np.int64))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.opponent_bot = TeacherTankBot()
        self.player_tank = TankState(0.0, 0.0, 0.0)
        self.enemy_tank = TankState(0.0, 0.0, math.pi)
        self.bullets: list[BulletState] = []
        self.steps = 0
        self.episode_index = 0
        self.maze_rows: list[str] = []
        self.cell_graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self.wall_segments: list[tuple[float, float, float, float]] = []
        self.nav_points: list[tuple[float, float]] = []
        self.nav_graph: list[list[int]] = []
        self.player_spawn_cell = (0, 0)
        self.enemy_spawn_cell = (self.grid_cols - 1, self.grid_rows - 1)
        self.player_spawn = (0.0, 0.0)
        self.enemy_spawn = (0.0, 0.0)
        self.prev_player_distance = 0.0
        self.prev_player_path_distance = 0.0
        self.prev_player_alignment = 1.0
        self.prev_player_pos = (0.0, 0.0)
        self.idle_steps = 0

        bootstrap_rng = np.random.default_rng(0)
        self._reset_arena(bootstrap_rng)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_index += 1
        self._reset_arena(self.np_random)
        return self._get_obs(), self._get_info()

    def _reset_arena(self, rng: np.random.Generator) -> None:
        if self.arena_preset in {"micro_fixed", "small_fixed"}:
            self.cell_graph = self._build_small_fixed_graph()
            self.player_spawn_cell, self.enemy_spawn_cell = self._choose_small_fixed_spawns(rng)
        elif self.arena_preset == "cover_fixed":
            self.cell_graph = self._build_cover_fixed_graph()
            self.player_spawn_cell, self.enemy_spawn_cell = self._choose_small_fixed_spawns(rng)
        else:
            self.cell_graph = self._generate_maze_graph(rng)
            self.player_spawn_cell, self.enemy_spawn_cell = self._choose_spawn_cells(rng)
        self.wall_segments, self.nav_points = self._build_maze_geometry()
        self.nav_graph = self._build_nav_graph()
        self.maze_rows = self._build_ascii_maze()

        self.player_spawn = self._cell_center(*self.player_spawn_cell)
        self.enemy_spawn = self._cell_center(*self.enemy_spawn_cell)
        self.player_tank = TankState(*self.player_spawn, self._spawn_angle(self.player_spawn_cell, self.enemy_spawn_cell))
        self.enemy_tank = TankState(*self.enemy_spawn, self._spawn_angle(self.enemy_spawn_cell, self.player_spawn_cell))
        self.bullets = []
        self.steps = 0
        self._init_reward_memory()

    def _build_open_graph(self) -> dict[tuple[int, int], set[tuple[int, int]]]:
        graph = {
            (col, row): set()
            for row in range(self.grid_rows)
            for col in range(self.grid_cols)
        }
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cell = (col, row)
                if col + 1 < self.grid_cols:
                    right = (col + 1, row)
                    graph[cell].add(right)
                    graph[right].add(cell)
                if row + 1 < self.grid_rows:
                    down = (col, row + 1)
                    graph[cell].add(down)
                    graph[down].add(cell)
        return graph

    @staticmethod
    def _unlink_cells(
        graph: dict[tuple[int, int], set[tuple[int, int]]],
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> None:
        graph[a].discard(b)
        graph[b].discard(a)

    def _build_small_fixed_graph(self) -> dict[tuple[int, int], set[tuple[int, int]]]:
        # Starter curriculum map: an open box with only outer walls so the
        # policy can first learn steering, chasing, and timed firing.
        return self._build_open_graph()

    def _build_cover_fixed_graph(self) -> dict[tuple[int, int], set[tuple[int, int]]]:
        graph = self._build_open_graph()
        cover_cells = [
            (self.grid_cols // 2, self.grid_rows // 2 - 1),
            (self.grid_cols // 2, self.grid_rows // 2),
        ]
        for cell in cover_cells:
            for neighbor in self._cell_neighbors(cell):
                self._unlink_cells(graph, cell, neighbor)
        return graph

    def _choose_small_fixed_spawns(
        self,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        pairs = [
            ((1, 1), (self.grid_cols - 2, self.grid_rows - 2)),
            ((1, self.grid_rows - 2), (self.grid_cols - 2, 1)),
        ]
        pair = pairs[int(rng.integers(len(pairs)))]
        if rng.random() < 0.5:
            return pair
        return pair[1], pair[0]

    def _generate_maze_graph(self, rng: np.random.Generator) -> dict[tuple[int, int], set[tuple[int, int]]]:
        graph = {
            (col, row): set()
            for row in range(self.grid_rows)
            for col in range(self.grid_cols)
        }
        start = (int(rng.integers(self.grid_cols)), int(rng.integers(self.grid_rows)))
        visited = {start}
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []

        def add_frontier(cell: tuple[int, int]) -> None:
            for neighbor in self._cell_neighbors(cell):
                if neighbor not in visited:
                    frontier.append((cell, neighbor))

        add_frontier(start)
        while frontier:
            idx = int(rng.integers(len(frontier)))
            src, dst = frontier.pop(idx)
            if dst in visited:
                continue
            graph[src].add(dst)
            graph[dst].add(src)
            visited.add(dst)
            add_frontier(dst)

        optional_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                cell = (col, row)
                if col + 1 < self.grid_cols:
                    right = (col + 1, row)
                    if right not in graph[cell]:
                        optional_edges.append((cell, right))
                if row + 1 < self.grid_rows:
                    down = (col, row + 1)
                    if down not in graph[cell]:
                        optional_edges.append((cell, down))

        for src, dst in optional_edges:
            if rng.random() < self.extra_connection_chance:
                graph[src].add(dst)
                graph[dst].add(src)

        return graph

    def _choose_spawn_cells(self, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
        all_cells = list(self.cell_graph.keys())
        seed_cell = all_cells[int(rng.integers(len(all_cells)))]
        first = self._farthest_cell(seed_cell)
        second = self._farthest_cell(first)

        if self._graph_distance(first, second) < max(self.grid_cols, self.grid_rows):
            second = all_cells[int(rng.integers(len(all_cells)))]

        return first, second

    def _build_maze_geometry(self) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]]]:
        wall_segments: list[tuple[float, float, float, float]] = [
            (0.0, 0.0, self.world_width, 0.0),
            (0.0, self.world_height, self.world_width, self.world_height),
            (0.0, 0.0, 0.0, self.world_height),
            (self.world_width, 0.0, self.world_width, self.world_height),
        ]
        nav_points = [
            self._cell_center(col, row)
            for row in range(self.grid_rows)
            for col in range(self.grid_cols)
        ]

        for boundary_col in range(1, self.grid_cols):
            start_row: int | None = None
            for row in range(self.grid_rows + 1):
                blocked = False
                if row < self.grid_rows:
                    left = (boundary_col - 1, row)
                    right = (boundary_col, row)
                    blocked = right not in self.cell_graph[left]

                if blocked and start_row is None:
                    start_row = row
                elif not blocked and start_row is not None:
                    x = boundary_col * self.cell_w
                    y0 = start_row * self.cell_h
                    y1 = row * self.cell_h
                    wall_segments.append((x, y0, x, y1))
                    start_row = None

        for boundary_row in range(1, self.grid_rows):
            start_col: int | None = None
            for col in range(self.grid_cols + 1):
                blocked = False
                if col < self.grid_cols:
                    top = (col, boundary_row - 1)
                    bottom = (col, boundary_row)
                    blocked = bottom not in self.cell_graph[top]

                if blocked and start_col is None:
                    start_col = col
                elif not blocked and start_col is not None:
                    y = boundary_row * self.cell_h
                    x0 = start_col * self.cell_w
                    x1 = col * self.cell_w
                    wall_segments.append((x0, y, x1, y))
                    start_col = None

        return wall_segments, nav_points

    def _build_nav_graph(self) -> list[list[int]]:
        graph: list[list[int]] = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                neighbors = [
                    neighbor[1] * self.grid_cols + neighbor[0]
                    for neighbor in self.cell_graph[(col, row)]
                ]
                graph.append(sorted(neighbors))
        return graph

    def _build_ascii_maze(self) -> list[str]:
        width = self.grid_cols * 2 + 1
        height = self.grid_rows * 2 + 1
        grid = [["#" for _ in range(width)] for _ in range(height)]

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                gx = col * 2 + 1
                gy = row * 2 + 1
                grid[gy][gx] = "."
                for neighbor in self.cell_graph[(col, row)]:
                    nx, ny = neighbor
                    if nx == col + 1:
                        grid[gy][gx + 1] = "."
                    elif nx == col - 1:
                        grid[gy][gx - 1] = "."
                    elif ny == row + 1:
                        grid[gy + 1][gx] = "."
                    elif ny == row - 1:
                        grid[gy - 1][gx] = "."

        px, py = self.player_spawn_cell
        ex, ey = self.enemy_spawn_cell
        grid[py * 2 + 1][px * 2 + 1] = "P"
        grid[ey * 2 + 1][ex * 2 + 1] = "E"
        return ["".join(row) for row in grid]

    def step(self, action: np.ndarray):
        enemy_action = self.opponent_bot.act(self._get_info())
        return self.step_both(action, enemy_action)

    def step_both(self, player_action: np.ndarray, enemy_action: np.ndarray):
        self.steps += 1
        reward = -0.002
        terminated = False
        truncated = False

        player_action = np.asarray(player_action, dtype=np.int64)
        enemy_action = np.asarray(enemy_action, dtype=np.int64)

        self._tick_tank(self.player_tank, player_action)
        self._tick_tank(self.enemy_tank, enemy_action)
        player_fired = self._try_fire(self.player_tank, 0, player_action)
        enemy_fired = self._try_fire(self.enemy_tank, 1, enemy_action)
        hit_owner = self._tick_bullets()

        if hit_owner == 0:
            reward += 1.5
            terminated = True
        elif hit_owner == 1:
            reward -= 1.5
            terminated = True

        if not terminated:
            reward += self._dense_reward(player_action, player_fired)

        if self.steps >= self.max_steps:
            truncated = True

        if terminated or truncated:
            self.player_tank.alive = hit_owner != 1
            self.enemy_tank.alive = hit_owner != 0

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _tick_tank(self, tank: TankState, action: np.ndarray) -> None:
        throttle_idx, turn_idx, _ = action.tolist()
        throttle = {-1: -1.0, 0: 0.0, 1: 1.0}[throttle_idx - 1]
        turn = {-1: -1.0, 0: 0.0, 1: 1.0}[turn_idx - 1]

        tank.angle = wrap_angle(tank.angle + turn * self.turn_speed * self.dt)
        tank.cooldown = max(0.0, tank.cooldown - self.dt)

        if throttle == 0.0:
            return

        dx = math.cos(tank.angle) * self.tank_speed * throttle * self.dt
        dy = math.sin(tank.angle) * self.tank_speed * throttle * self.dt
        new_x = tank.x + dx
        new_y = tank.y + dy

        if not self._tank_collides(new_x, new_y):
            tank.x = new_x
            tank.y = new_y
            return

        # Try axis-separated motion so tanks slide along walls instead of
        # feeling glued to corners.
        if not self._tank_collides(new_x, tank.y):
            tank.x = new_x
            return
        if not self._tank_collides(tank.x, new_y):
            tank.y = new_y

    def _try_fire(self, tank: TankState, owner: int, action: np.ndarray) -> bool:
        if action[2] != 1 or tank.cooldown > 0.0:
            return False
        if sum(1 for bullet in self.bullets if bullet.owner == owner) >= self.max_bullets:
            return False

        spawn_dist = self.tank_radius + self.bullet_radius + 2.0
        bullet = BulletState(
            x=tank.x + math.cos(tank.angle) * spawn_dist,
            y=tank.y + math.sin(tank.angle) * spawn_dist,
            vx=math.cos(tank.angle) * self.bullet_speed,
            vy=math.sin(tank.angle) * self.bullet_speed,
            owner=owner,
            lifetime=4.6,
            bounces_left=self.max_bounces,
            age_steps=0,
        )
        if self._bullet_hits_wall(bullet.x, bullet.y):
            return False

        self.bullets.append(bullet)
        tank.cooldown = self.fire_cooldown
        return True

    def _tick_bullets(self) -> int | None:
        survivors: list[BulletState] = []
        hit_owner: int | None = None

        for bullet in self.bullets:
            bullet.lifetime -= self.dt
            bullet.age_steps += 1
            if bullet.lifetime <= 0.0:
                continue

            next_x = bullet.x + bullet.vx * self.dt
            next_y = bullet.y + bullet.vy * self.dt
            bounced = False

            if self._bullet_hits_wall(next_x, bullet.y):
                bullet.vx *= -1.0
                bounced = True
            else:
                bullet.x = next_x

            if self._bullet_hits_wall(bullet.x, next_y):
                bullet.vy *= -1.0
                bounced = True
            else:
                bullet.y = next_y

            if bounced:
                bullet.bounces_left -= 1
                if bullet.bounces_left < 0:
                    continue

            player_can_take_hit = bullet.owner != 0 or bullet.age_steps > self.self_hit_grace_steps
            enemy_can_take_hit = bullet.owner != 1 or bullet.age_steps > self.self_hit_grace_steps

            if self._bullet_hits_tank(bullet, self.player_tank) and player_can_take_hit:
                hit_owner = 1
                break
            if self._bullet_hits_tank(bullet, self.enemy_tank) and enemy_can_take_hit:
                hit_owner = 0
                break

            if 0.0 <= bullet.x <= self.world_width and 0.0 <= bullet.y <= self.world_height:
                survivors.append(bullet)

        self.bullets = survivors
        return hit_owner

    def _tank_collides(self, x: float, y: float) -> bool:
        if x - self.tank_radius < 0.0 or x + self.tank_radius > self.world_width:
            return True
        if y - self.tank_radius < 0.0 or y + self.tank_radius > self.world_height:
            return True
        for segment in self.wall_segments:
            if self._circle_segment_collision(x, y, self.tank_radius + self.wall_collision_padding, segment):
                return True
        return False

    def _bullet_hits_wall(self, x: float, y: float) -> bool:
        if x - self.bullet_radius < 0.0 or x + self.bullet_radius > self.world_width:
            return True
        if y - self.bullet_radius < 0.0 or y + self.bullet_radius > self.world_height:
            return True
        for segment in self.wall_segments:
            if self._circle_segment_collision(x, y, self.bullet_radius + self.wall_collision_padding, segment):
                return True
        return False

    def _bullet_hits_tank(self, bullet: BulletState, tank: TankState) -> bool:
        return math.hypot(bullet.x - tank.x, bullet.y - tank.y) <= (self.bullet_radius + self.tank_radius)

    @staticmethod
    def _point_segment_distance(px: float, py: float, segment: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = segment
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = clamp(t, 0.0, 1.0)
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return math.hypot(px - closest_x, py - closest_y)

    def _circle_segment_collision(
        self,
        cx: float,
        cy: float,
        radius: float,
        segment: tuple[float, float, float, float],
    ) -> bool:
        return self._point_segment_distance(cx, cy, segment) <= radius

    def _line_of_sight(self, source: TankState, target: TankState) -> bool:
        dx = target.x - source.x
        dy = target.y - source.y
        steps = max(1, int(math.hypot(dx, dy) / 6.0))
        for i in range(1, steps):
            x = source.x + dx * (i / steps)
            y = source.y + dy * (i / steps)
            if self._bullet_hits_wall(x, y):
                return False
        return True

    def _dense_reward(self, player_action: np.ndarray, player_fired: bool) -> float:
        distance = math.hypot(self.enemy_tank.x - self.player_tank.x, self.enemy_tank.y - self.player_tank.y)
        max_distance = math.hypot(self.world_width, self.world_height)
        distance_norm = distance / max_distance
        player_cell = self._position_to_cell(self.player_tank.x, self.player_tank.y)
        enemy_cell = self._position_to_cell(self.enemy_tank.x, self.enemy_tank.y)
        path_distance = self._graph_distance(player_cell, enemy_cell) / float(self.grid_cols * self.grid_rows)

        bearing = math.atan2(self.enemy_tank.y - self.player_tank.y, self.enemy_tank.x - self.player_tank.x)
        align = abs(wrap_angle(bearing - self.player_tank.angle)) / math.pi
        move_dist = math.hypot(
            self.player_tank.x - self.prev_player_pos[0],
            self.player_tank.y - self.prev_player_pos[1],
        )
        throttle_idx = int(player_action[0])
        turn_idx = int(player_action[1])
        has_los = self._line_of_sight(self.player_tank, self.enemy_tank)

        reward = 0.0
        reward += 0.032 * (self.prev_player_distance - distance_norm)
        reward += 0.090 * (self.prev_player_path_distance - path_distance)
        if has_los:
            reward += 0.022 * (self.prev_player_alignment - align)
        else:
            reward += 0.004 * (self.prev_player_alignment - align)

        if move_dist < 0.75:
            self.idle_steps += 1
            reward -= 0.014 + min(self.idle_steps, 24) * 0.0020
        else:
            self.idle_steps = 0
            reward += min(move_dist / self.cell_size, 0.22) * 0.040

        if throttle_idx == 2:
            reward += 0.006
        elif throttle_idx == 0:
            reward -= 0.006

        if turn_idx != 1:
            reward -= 0.0015
            if move_dist < 0.75:
                reward -= 0.012
            elif throttle_idx != 2:
                reward -= 0.004

        if throttle_idx == 1 and turn_idx == 1 and move_dist < 0.75:
            reward -= 0.010

        if has_los:
            reward += 0.010
        shot_window = has_los and align < 0.12 and self.player_tank.cooldown <= 1e-6
        if player_fired:
            reward -= 0.004
            if has_los and align < 0.14:
                reward += 0.028
            if has_los and align < 0.08 and move_dist >= 0.75:
                reward += 0.018
            if move_dist < 0.75:
                reward -= 0.030
            if not has_los:
                reward -= 0.018
            if align > 0.20:
                reward -= 0.018
        elif shot_window:
            reward -= 0.010

        self.prev_player_distance = distance_norm
        self.prev_player_path_distance = path_distance
        self.prev_player_alignment = align
        self.prev_player_pos = (self.player_tank.x, self.player_tank.y)
        return reward

    def _init_reward_memory(self) -> None:
        distance = math.hypot(self.enemy_tank.x - self.player_tank.x, self.enemy_tank.y - self.player_tank.y)
        max_distance = math.hypot(self.world_width, self.world_height)
        self.prev_player_distance = distance / max_distance
        player_cell = self._position_to_cell(self.player_tank.x, self.player_tank.y)
        enemy_cell = self._position_to_cell(self.enemy_tank.x, self.enemy_tank.y)
        self.prev_player_path_distance = self._graph_distance(player_cell, enemy_cell) / float(self.grid_cols * self.grid_rows)
        bearing = math.atan2(self.enemy_tank.y - self.player_tank.y, self.enemy_tank.x - self.player_tank.x)
        self.prev_player_alignment = abs(wrap_angle(bearing - self.player_tank.angle)) / math.pi
        self.prev_player_pos = (self.player_tank.x, self.player_tank.y)
        self.idle_steps = 0

    def _ray_features(self, source: TankState, target: TankState) -> np.ndarray:
        max_dist = math.hypot(self.world_width, self.world_height)
        features: list[float] = []

        for ray_idx in range(self.ray_count):
            angle = source.angle + (2.0 * math.pi * ray_idx / self.ray_count)
            dx = math.cos(angle)
            dy = math.sin(angle)
            wall_d = max_dist
            enemy_d = max_dist
            bullet_d = max_dist

            for t in np.arange(0.0, max_dist, 4.0):
                px = source.x + dx * t
                py = source.y + dy * t
                if self._bullet_hits_wall(px, py):
                    wall_d = t
                    break
                if math.hypot(px - target.x, py - target.y) <= self.tank_radius:
                    enemy_d = min(enemy_d, t)
                for bullet in self.bullets:
                    if math.hypot(px - bullet.x, py - bullet.y) <= self.bullet_radius + 1.0:
                        bullet_d = min(bullet_d, t)

            features.extend(
                [
                    wall_d / max_dist,
                    1.0 if enemy_d == max_dist else enemy_d / max_dist,
                    1.0 if bullet_d == max_dist else bullet_d / max_dist,
                ]
            )

        return np.asarray(features, dtype=np.float32)

    def _bullet_features(self, source: TankState) -> np.ndarray:
        features = np.zeros((self.bullet_slots, 6), dtype=np.float32)
        bullets = sorted(
            self.bullets,
            key=lambda bullet: math.hypot(source.x - bullet.x, source.y - bullet.y),
        )[: self.bullet_slots]

        for idx, bullet in enumerate(bullets):
            features[idx] = np.array(
                [
                    bullet.x / self.world_width,
                    bullet.y / self.world_height,
                    bullet.vx / self.bullet_speed,
                    bullet.vy / self.bullet_speed,
                    -1.0 if bullet.owner == 0 else 1.0,
                    bullet.lifetime / 4.6,
                ],
                dtype=np.float32,
            )
        return features.flatten()

    def _tank_features(self, source: TankState, target: TankState) -> np.ndarray:
        return np.array(
            [
                source.x / self.world_width,
                source.y / self.world_height,
                math.cos(source.angle),
                math.sin(source.angle),
                source.cooldown / self.fire_cooldown,
                target.x / self.world_width,
                target.y / self.world_height,
                math.cos(target.angle),
                math.sin(target.angle),
                target.cooldown / self.fire_cooldown,
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        los_flag = np.array(
            [1.0 if self._line_of_sight(self.player_tank, self.enemy_tank) else 0.0],
            dtype=np.float32,
        )
        obs = np.concatenate(
            [
                self._tank_features(self.player_tank, self.enemy_tank),
                los_flag,
                self._ray_features(self.player_tank, self.enemy_tank),
                self._bullet_features(self.player_tank),
            ]
        )
        return (obs * 2.0 - 1.0).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        bullets = [
            {
                "x": bullet.x,
                "y": bullet.y,
                "vx": bullet.vx,
                "vy": bullet.vy,
                "owner": bullet.owner,
                "lifetime": bullet.lifetime,
                "bounces_left": bullet.bounces_left,
                "age_steps": bullet.age_steps,
            }
            for bullet in self.bullets
        ]
        walls = [
            {"x1": segment[0], "y1": segment[1], "x2": segment[2], "y2": segment[3]}
            for segment in self.wall_segments
        ]
        enemy_has_los = self._line_of_sight(self.enemy_tank, self.player_tank)
        player_has_los = self._line_of_sight(self.player_tank, self.enemy_tank)

        return {
            "steps": self.steps,
            "episode_index": self.episode_index,
            "player_tank": {
                "x": self.player_tank.x,
                "y": self.player_tank.y,
                "angle": self.player_tank.angle,
                "cooldown": self.player_tank.cooldown,
                "alive": self.player_tank.alive,
            },
            "enemy_tank": {
                "x": self.enemy_tank.x,
                "y": self.enemy_tank.y,
                "angle": self.enemy_tank.angle,
                "cooldown": self.enemy_tank.cooldown,
                "alive": self.enemy_tank.alive,
            },
            "world": {
                "width": self.world_width,
                "height": self.world_height,
                "cell_size": self.cell_size,
                "grid_cols": self.grid_cols,
                "grid_rows": self.grid_rows,
                "arena_preset": self.arena_preset,
            },
            "rules": {
                "tank_radius": self.tank_radius,
                "bullet_radius": self.bullet_radius,
                "bullet_speed": self.bullet_speed,
                "dt": self.dt,
                "max_bounces": self.max_bounces,
                "wall_padding": self.wall_collision_padding,
                "self_hit_grace_steps": self.self_hit_grace_steps,
            },
            "walls": walls,
            "nav_points": self.nav_points,
            "nav_graph": self.nav_graph,
            "bullets": bullets,
            "bullet_count": len(self.bullets),
            "player_spawn_cell": self.player_spawn_cell,
            "enemy_spawn_cell": self.enemy_spawn_cell,
            "enemy_has_line_of_sight": enemy_has_los,
            "player_has_line_of_sight": player_has_los,
            "line_of_sight": enemy_has_los,
        }

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_ansi(self) -> str:
        grid = [list(row) for row in self.maze_rows]
        scale_x = max(1, len(grid[0]) - 1)
        scale_y = max(1, len(grid) - 1)

        def to_cell(x: float, y: float) -> tuple[int, int]:
            gx = int(round((x / self.world_width) * scale_x))
            gy = int(round((y / self.world_height) * scale_y))
            return clamp(gx, 0, scale_x), clamp(gy, 0, scale_y)

        for bullet in self.bullets:
            gx, gy = to_cell(bullet.x, bullet.y)
            if grid[int(gy)][int(gx)] == ".":
                grid[int(gy)][int(gx)] = "*"

        pgx, pgy = to_cell(self.player_tank.x, self.player_tank.y)
        egx, egy = to_cell(self.enemy_tank.x, self.enemy_tank.y)
        grid[int(pgy)][int(pgx)] = "P"
        grid[int(egy)][int(egx)] = "E"
        return "\n".join("".join(row) for row in grid)

    def _render_rgb(self) -> np.ndarray:
        canvas = np.zeros((int(self.world_height), int(self.world_width), 3), dtype=np.uint8)
        canvas[:] = np.array([244, 241, 233], dtype=np.uint8)
        self._draw_background_pattern(canvas)

        wall_color = np.array([24, 24, 24], dtype=np.uint8)
        for x1, y1, x2, y2 in self.wall_segments:
            self._draw_line(canvas, x1, y1, x2, y2, wall_color, width=self.wall_render_width)

        self._draw_tank(canvas, self.player_tank, np.array([43, 110, 255], dtype=np.uint8))
        self._draw_tank(canvas, self.enemy_tank, np.array([232, 85, 56], dtype=np.uint8))

        for bullet in self.bullets:
            self._draw_circle(canvas, bullet.x, bullet.y, self.bullet_radius + 0.5, np.array([34, 34, 34], dtype=np.uint8))

        return canvas

    def _draw_background_pattern(self, canvas: np.ndarray) -> None:
        for row in range(self.grid_rows):
            y0 = int(row * self.cell_h)
            y1 = int((row + 1) * self.cell_h)
            for col in range(self.grid_cols):
                if (row + col) % 2 == 0:
                    continue
                x0 = int(col * self.cell_w)
                x1 = int((col + 1) * self.cell_w)
                canvas[y0:y1, x0:x1] = np.array([240, 237, 229], dtype=np.uint8)

    def _draw_tank(self, canvas: np.ndarray, tank: TankState, color: np.ndarray) -> None:
        corners = self._tank_corners(tank)
        self._draw_polygon(canvas, corners, color)
        self._draw_circle(canvas, tank.x, tank.y, 8.0, np.array([248, 246, 240], dtype=np.uint8))
        barrel_x = tank.x + math.cos(tank.angle) * 19.0
        barrel_y = tank.y + math.sin(tank.angle) * 19.0
        self._draw_line(canvas, tank.x, tank.y, barrel_x, barrel_y, np.array([28, 28, 28], dtype=np.uint8), width=5)

    def _tank_corners(self, tank: TankState) -> list[tuple[float, float]]:
        forward_x = math.cos(tank.angle)
        forward_y = math.sin(tank.angle)
        right_x = -forward_y
        right_y = forward_x
        half_w = self.tank_body_w / 2.0
        half_h = self.tank_body_h / 2.0
        return [
            (tank.x - forward_x * half_h - right_x * half_w, tank.y - forward_y * half_h - right_y * half_w),
            (tank.x - forward_x * half_h + right_x * half_w, tank.y - forward_y * half_h + right_y * half_w),
            (tank.x + forward_x * half_h + right_x * half_w, tank.y + forward_y * half_h + right_y * half_w),
            (tank.x + forward_x * half_h - right_x * half_w, tank.y + forward_y * half_h - right_y * half_w),
        ]

    def _draw_polygon(self, canvas: np.ndarray, points: list[tuple[float, float]], color: np.ndarray) -> None:
        min_x = max(0, int(min(point[0] for point in points)) - 1)
        max_x = min(canvas.shape[1] - 1, int(max(point[0] for point in points)) + 1)
        min_y = max(0, int(min(point[1] for point in points)) - 1)
        max_y = min(canvas.shape[0] - 1, int(max(point[1] for point in points)) + 1)

        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                if self._point_in_polygon(px + 0.5, py + 0.5, points):
                    canvas[py, px] = color

    @staticmethod
    def _point_in_polygon(px: float, py: float, points: list[tuple[float, float]]) -> bool:
        inside = False
        for idx, point in enumerate(points):
            x1, y1 = point
            x2, y2 = points[(idx + 1) % len(points)]
            intersects = ((y1 > py) != (y2 > py)) and (
                px < (x2 - x1) * (py - y1) / ((y2 - y1) or 1e-6) + x1
            )
            if intersects:
                inside = not inside
        return inside

    def _draw_circle(self, canvas: np.ndarray, cx: float, cy: float, radius: float, color: np.ndarray) -> None:
        x0 = max(0, int(cx - radius - 1))
        x1 = min(canvas.shape[1], int(cx + radius + 1))
        y0 = max(0, int(cy - radius - 1))
        y1 = min(canvas.shape[0], int(cy + radius + 1))
        for y in range(y0, y1):
            for x in range(x0, x1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    canvas[y, x] = color

    def _draw_line(
        self,
        canvas: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: np.ndarray,
        width: int = 1,
    ) -> None:
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0)))
        radius = max(0, width // 2)
        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            for oy in range(-radius, radius + 1):
                for ox in range(-radius, radius + 1):
                    px = x + ox
                    py = y + oy
                    if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                        canvas[py, px] = color

    def _cell_neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        col, row = cell
        neighbors: list[tuple[int, int]] = []
        if row > 0:
            neighbors.append((col, row - 1))
        if col > 0:
            neighbors.append((col - 1, row))
        if row + 1 < self.grid_rows:
            neighbors.append((col, row + 1))
        if col + 1 < self.grid_cols:
            neighbors.append((col + 1, row))
        return neighbors

    def _cell_center(self, col: int, row: int) -> tuple[float, float]:
        return (col + 0.5) * self.cell_w, (row + 0.5) * self.cell_h

    def _position_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(clamp(x / self.cell_w, 0, self.grid_cols - 1))
        row = int(clamp(y / self.cell_h, 0, self.grid_rows - 1))
        return col, row

    def _graph_distance(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        queue: deque[tuple[int, int]] = deque([start])
        distance = {start: 0}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                return distance[cell]
            for neighbor in self.cell_graph[cell]:
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[cell] + 1
                queue.append(neighbor)
        return 0

    def _farthest_cell(self, start: tuple[int, int]) -> tuple[int, int]:
        queue: deque[tuple[int, int]] = deque([start])
        distance = {start: 0}
        farthest = start
        while queue:
            cell = queue.popleft()
            if distance[cell] >= distance[farthest]:
                farthest = cell
            for neighbor in self.cell_graph[cell]:
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[cell] + 1
                queue.append(neighbor)
        return farthest

    def _spawn_angle(self, start: tuple[int, int], goal: tuple[int, int]) -> float:
        path = self._path_between_cells(start, goal)
        if len(path) < 2:
            return 0.0
        (sx, sy), (nx, ny) = path[0], path[1]
        return math.atan2(ny - sy, nx - sx)

    def _path_between_cells(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        queue: deque[tuple[int, int]] = deque([start])
        prev = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                break
            for neighbor in self.cell_graph[cell]:
                if neighbor in prev:
                    continue
                prev[neighbor] = cell
                queue.append(neighbor)

        if goal not in prev:
            return [start]

        path = [goal]
        current = goal
        while prev[current] is not None:
            current = prev[current]
            path.append(current)
        path.reverse()
        return path


def make_tank_duel_env(
    render_mode: str | None = None,
    arena_preset: str = "maze_random",
) -> TankDuelEnv:
    return TankDuelEnv(render_mode=render_mode, arena_preset=arena_preset)
