"""P37: inverse bullet-density field and decisive MPC teacher.

The field is built by launching a deterministic fan of reverse bullets from
Laika's current cell centre.  Every reverse ray votes at most once for each
grid cell that can contain the corresponding forward shooter's centre.  Raw
vote counts are retained for inspection.  A maze-distance envelope extends
timely firing density over the reachable map, and every new uphill cell feeds
a three-second hunt chain with the discrete values 1, 2, 4, ... .

This module deliberately separates two jobs:

* inverse continuous reflection creates a smooth, target-centred navigation
  field cheaply;
* the original forward bullet simulator remains the authority for firing.

The inverse model proposes where to go and how to turn.  It never declares a
kill on its own, because the original game's finite substeps and corner bounce
rules are not perfectly time reversible.
"""

import argparse
from dataclasses import dataclass, field as dataclass_field
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import constants as C
from tank_trouble_original.laika import LaikaAI
from training.mpc_agent import CANDIDATES, make_sandbox
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.survival_expert_iter_530 import apply_action


DEFAULT_RAYS = 2048
DEFAULT_BOUNCES = 2
# A geometrically possible ten-second ricochet is not a useful combat
# opportunity.  The field therefore contains only bullets arriving within the
# same three-second window that already produced a stable chain-coin basin.
DEFAULT_FLIGHT_FRAMES = 3 * C.FPS
FIELD_LEVELS = 7
AIM_BINS = 72
SAMPLE_STEP_CELLS = 0.20
MIN_SHOOTER_DISTANCE_CELLS = 0.70
GUIDANCE_DISTANCE_DECAY = 0.18
HUNT_CHAIN_WINDOW_FRAMES = 3 * C.FPS
HUNT_CHAIN_MAX_EXPONENT = 6

MPC_HORIZON = 36
MPC_HOLD = 8
COMMIT_MOVE_FRAMES = 4
COMMIT_TURN_FRAMES = 2

ACTIVE_KILL_SCORE = 12_000.0
OPPONENT_SELF_SCORE = 1_500.0
DEATH_SCORE = -12_000.0
FIELD_ASCENT_WEIGHT = 34.0
FIELD_PEAK_WEIGHT = 6.0
HUNT_CHAIN_GAIN_WEIGHT = 12.0
GUIDANCE_PROGRESS_WEIGHT = 120.0
ALIGNMENT_WEIGHT = 190.0
GOOD_FIRE_BONUS = 1_800.0
FAILED_FIRE_PENALTY = 260.0
SUICIDE_FIRE_PENALTY = 2_500.0
RISK_WEIGHT = 320.0
NO_EFFECT_REPEAT_PENALTY = 600.0
MOVING_FIRE_SCORE = -1.0e9
OWN_BULLET_GUARD_HORIZON = 24
VALUE_LEAF_WEIGHT = 300.0   # 与塑形项同量级；见 density_rollout 注释


def _cell(game, tank):
    return int(tank.x // game.scale), int(tank.y // game.scale)


def _angle_delta(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


# --- 走位：候选硬淘汰 ---------------------------------------------------
# P32 的教训：只在 reward 里扣碰撞分，进攻收益照样能"买"下顶墙。改成对
# 推演后确实动不了的候选直接淘汰，实测卡墙 33%/48% -> 4.9%/11.0%。
MIN_MOTION_CELLS = 0.15
MIN_ROTATION_DEGREES = 5.0
MOTION_PROBE_FRAMES = COMMIT_MOVE_FRAMES
FUTILE_ACTION_SCORE = -5.0e8      # 低于一切正常分，高于 MOVING_FIRE_SCORE

# --- 火控 ---------------------------------------------------------------
# 曾经加过"按预测飞行时间衰减命中奖励 + 置信度开火门"来治被骗光子弹，
# 已回退：那是手写战术规则，两个阈值都是拍脑袋的，等于把上限锁死在人对
# 这个游戏的理解上。弹匣纪律应该由价值网络从真实胜负里学出来。
#
# 但有一条结构性问题仍然存在：下面的强制开火是**硬编码反射**，绕过全部
# 评分，所以任何学出来的价值都影响不到它。真要让飞轮能学会火控，得把它
# 降级为普通候选——注意那是"移除硬编码"，不是"新增阈值"，两者性质不同。
FIRE_CONFIDENCE_TAU = 12.0        # 仅供 _fire_confidence 诊断用，不参与决策


def action_motion(game, action, frames=MOTION_PROBE_FRAMES):
    """在沙盒里执行候选若干帧，返回 (位移格数, 转角度数)。

    只推承诺时长，成本约为主推演的 1/9。
    """
    sandbox = make_sandbox(game, "L1", rng_seed=0)
    me = sandbox.tanks[0]
    enemy = sandbox.tanks[1]
    enemy.forward = enemy.backup = False
    enemy.turn_left = enemy.turn_right = enemy.fire = False
    start_x, start_y, start_rot = me.x, me.y, me.rotation
    apply_action(sandbox, (action[0], action[1], 0))
    for _ in range(max(1, int(frames))):
        sandbox.step()
        if not me.alive:
            break
    displacement = math.hypot(me.x - start_x, me.y - start_y) / max(
        sandbox.scale, 1e-6)
    rotation = abs(math.degrees(
        _angle_delta(me.rotation, start_rot)))
    return displacement, rotation


def _live_action_indices():
    """能被选中的候选。移动+开火的 8 列必被 mask 覆写，不需要前推。"""
    return tuple(
        index for index, action in enumerate(CANDIDATES)
        if not (action[2] and action[:2] != (1, 1)))


def mask_moving_fire_scores(scores):
    """Tank Trouble can self-hit when locomotion and firing share a frame.

    Firing is therefore a stationary atomic action.  Keep all 18 columns for
    compatibility with existing distillation data, but make the eight moving
    fire combinations unselectable.
    """
    for index, action in enumerate(CANDIDATES):
        if action[2] and action[:2] != (1, 1):
            scores[index] = MOVING_FIRE_SCORE
    return scores


LIVE_ACTION_INDICES = _live_action_indices()


def _own_bullet_ids(game):
    """Identity only: positions may age asynchronously, births/removals may not."""
    me = game.tanks[0]
    return tuple(sorted(
        bullet.name for bullet in game.bullets
        if not bullet.removed and bullet.owner is me))


def action_self_hits(game, action, horizon=OWN_BULLET_GUARD_HORIZON):
    """Exact short rollout for the temporal fire-then-chase failure mode."""
    if not _own_bullet_ids(game):
        return False
    sandbox = make_sandbox(game, "L1", rng_seed=0)
    enemy = sandbox.tanks[1]
    enemy.forward = enemy.backup = False
    enemy.turn_left = enemy.turn_right = enemy.fire = False
    apply_action(sandbox, (action[0], action[1], 0))
    for _ in range(max(1, int(horizon))):
        events = sandbox.step()
        if any(event[0] == "hit" and event[1:] == (0, 0)
               for event in events):
            return True
        if not sandbox.tanks[0].alive or sandbox.frozen:
            break
    return False


@dataclass
class DensityField:
    """Immutable result of one Laika-cell inverse simulation."""

    target_cell: tuple
    ray_count: int
    max_bounces: int
    max_flight_frames: int
    counts: np.ndarray
    counts_by_bounce: np.ndarray
    aim_histogram: np.ndarray
    min_frames: np.ndarray
    tiers: np.ndarray
    values: np.ndarray
    guidance: np.ndarray
    max_count: int

    def in_bounds(self, cell):
        return (0 <= cell[0] < self.counts.shape[0]
                and 0 <= cell[1] < self.counts.shape[1])

    def count_at(self, cell):
        return int(self.counts[cell]) if self.in_bounds(cell) else 0

    def tier_at(self, cell):
        return int(self.tiers[cell]) if self.in_bounds(cell) else 0

    def value_at(self, cell):
        return float(self.values[cell]) if self.in_bounds(cell) else 0.0

    def guidance_at(self, cell):
        return float(self.guidance[cell]) if self.in_bounds(cell) else 0.0

    def success_rate_at(self, cell):
        """Absolute angular coverage: reverse rays reaching this cell / N."""
        return self.count_at(cell) / max(self.ray_count, 1)

    def relative_success_at(self, cell):
        """Map-normalized coverage, used only to scale teacher urgency."""
        return self.count_at(cell) / max(self.max_count, 1)

    def best_aim_at(self, cell, current_heading=None):
        """Return (forward bullet angle, conditional histogram mass)."""
        if not self.in_bounds(cell):
            return None, 0.0
        histogram = self.aim_histogram[cell[0], cell[1]]
        peak = int(histogram.max(initial=0))
        total = int(histogram.sum())
        if peak <= 0 or total <= 0:
            return None, 0.0
        eligible = np.flatnonzero(histogram >= max(1, math.ceil(0.85 * peak)))
        angles = (eligible.astype(np.float64) + 0.5) * (
            2.0 * math.pi / AIM_BINS)
        if current_heading is None:
            choice = int(eligible[np.argmax(histogram[eligible])])
        else:
            errors = np.abs(np.arctan2(
                np.sin(angles - current_heading),
                np.cos(angles - current_heading)))
            choice = int(eligible[int(np.argmin(errors))])
        angle = (choice + 0.5) * (2.0 * math.pi / AIM_BINS)
        return float(angle), peak / total


class NullDensityField:
    """场：目标不存在。

    每个狩猎项取值前都先过 `in_bounds`，所以一个"没有任何格子在界内"的
    场会让 ascent / guidance / alignment / 猎杀链**同时恒为 0**——rollout
    里一个分支都不用加。剩下的目标函数正好只有死亡罚分和来袭火力风险，
    而这恰恰就是"熬过 75 帧结算窗口"的定义。

    这就是为什么击杀之后不需要单独的模式：同一个打分函数本来就描述了
    回合的两半。手写状态机不是在补能力，是在补一个被截断的视野。
    """

    ray_count = 0
    max_count = 0
    # 猎杀链拿它当"目标是否换了格"的键；没有目标就恒为 None。
    target_cell = None

    def in_bounds(self, cell):
        return False

    def count_at(self, cell):
        return 0

    def tier_at(self, cell):
        return 0

    def value_at(self, cell):
        return 0.0

    def guidance_at(self, cell):
        return 0.0

    def success_rate_at(self, cell):
        return 0.0

    def relative_success_at(self, cell):
        return 0.0

    def best_aim_at(self, cell, current_heading=None):
        return None, 0.0


NULL_FIELD = NullDensityField()


@dataclass
class HuntChainState:
    """Three-second one-shot chain over strictly ascending guidance cells."""

    count: int = 0
    timer: int = 0
    collected: set = dataclass_field(default_factory=set)

    def clone(self):
        return HuntChainState(
            count=self.count, timer=self.timer,
            collected=set(self.collected))

    def advance(self, frames=1):
        self.timer = max(0, self.timer - int(frames))
        if self.timer == 0:
            self.count = 0

    def collect_ascent(self, field, previous_cell, current_cell,
                       target_stable=True):
        """Return 1,2,4,... only for an attributable new uphill cell."""
        if not target_stable or previous_cell == current_cell:
            return 0.0
        previous = field.guidance_at(previous_cell)
        current = field.guidance_at(current_cell)
        if current <= previous + 1e-7:
            return 0.0
        key = (field.target_cell, tuple(current_cell))
        if key in self.collected:
            return 0.0
        reward = float(2 ** min(self.count, HUNT_CHAIN_MAX_EXPONENT))
        self.count = min(self.count + 1, HUNT_CHAIN_MAX_EXPONENT)
        self.timer = HUNT_CHAIN_WINDOW_FRAMES
        self.collected.add(key)
        return reward


class InverseDensityFieldBuilder:
    """Trace reversible AABB reflections and rasterise shooter-centre votes."""

    def __init__(self, game, ray_count=DEFAULT_RAYS,
                 max_bounces=DEFAULT_BOUNCES,
                 max_frames=DEFAULT_FLIGHT_FRAMES,
                 levels=FIELD_LEVELS):
        self.game = game
        self.ray_count = int(ray_count)
        self.max_bounces = int(max_bounces)
        self.max_frames = int(max_frames)
        self.levels = int(levels)
        thickness = float(game.wall_half_t)
        self.boxes = np.asarray([
            (min(x1, x2) - thickness, min(y1, y2) - thickness,
             max(x1, x2) + thickness, max(y1, y2) + thickness)
            for x1, y1, x2, y2 in game.walls
        ], dtype=np.float64)
        self.reachable = {
            (int(item["x"]), int(item["y"])) for item in game.reachable
        }

    def _nearest_wall(self, x, y, dx, dy):
        """Return nearest positive AABB entry distance and reflected axes."""
        boxes = self.boxes
        epsilon = max(1e-7, self.game.scale * 1e-8)

        if abs(dx) < 1e-14:
            inside = (boxes[:, 0] <= x) & (x <= boxes[:, 2])
            near_x = np.where(inside, -np.inf, np.inf)
            far_x = np.where(inside, np.inf, -np.inf)
        else:
            first = (boxes[:, 0] - x) / dx
            second = (boxes[:, 2] - x) / dx
            near_x, far_x = np.minimum(first, second), np.maximum(first, second)

        if abs(dy) < 1e-14:
            inside = (boxes[:, 1] <= y) & (y <= boxes[:, 3])
            near_y = np.where(inside, -np.inf, np.inf)
            far_y = np.where(inside, np.inf, -np.inf)
        else:
            first = (boxes[:, 1] - y) / dy
            second = (boxes[:, 3] - y) / dy
            near_y, far_y = np.minimum(first, second), np.maximum(first, second)

        entry = np.maximum(near_x, near_y)
        leave = np.minimum(far_x, far_y)
        valid = (leave >= np.maximum(entry, epsilon)) & (entry > epsilon)
        distances = np.where(valid, entry, np.inf)
        nearest = float(distances.min(initial=np.inf))
        if not math.isfinite(nearest):
            return math.inf, False, False

        # Several thick wall boxes may meet at the same corner.  Merge their
        # normals so a corner reverses both components, matching the intended
        # orthogonal billiard geometry.
        tolerance = max(1e-6, self.game.scale * 1e-6)
        hits = np.flatnonzero(np.abs(distances - nearest) <= tolerance)
        flip_x = False
        flip_y = False
        for index in hits:
            difference = near_x[index] - near_y[index]
            if difference > tolerance:
                flip_x = True
            elif difference < -tolerance:
                flip_y = True
            else:
                flip_x = True
                flip_y = True
        return nearest, flip_x, flip_y

    def _clear_centres(self, xs, ys):
        if len(xs) == 0:
            return np.zeros(0, dtype=bool)
        b = self.boxes
        inside = ((xs[:, None] >= b[None, :, 0])
                  & (xs[:, None] <= b[None, :, 2])
                  & (ys[:, None] >= b[None, :, 1])
                  & (ys[:, None] <= b[None, :, 3]))
        return ~inside.any(axis=1)

    def _guidance_envelope(self, counts, min_frames):
        """Max-product maze-distance envelope of timely firing cells.

        Every reachable cell receives a positive value when at least one
        timely firing cell exists.  Moving one shortest-path step toward the
        source currently dominating the max strictly increases that source's
        contribution, giving the chain a dense sequence of collectible events.
        """
        guidance = np.zeros_like(counts, dtype=np.float32)
        max_count = int(counts.max(initial=0))
        if max_count <= 0:
            return guidance
        denominator = math.log1p(max_count)
        for source in self.reachable:
            count = int(counts[source])
            if count <= 0:
                continue
            count_quality = math.log1p(count) / denominator
            flight = float(min_frames[source])
            time_quality = math.exp(-flight / max(self.max_frames, 1))
            source_quality = count_quality * (0.50 + 0.50 * time_quality)
            distances = self.game.dist_map(source[0], source[1])
            if distances is None:
                continue
            for cell in self.reachable:
                distance = distances[cell[0]][cell[1]]
                if distance is None or distance != distance:
                    continue
                candidate = source_quality * math.exp(
                    -GUIDANCE_DISTANCE_DECAY * float(distance))
                if candidate > guidance[cell]:
                    guidance[cell] = candidate
        maximum = float(guidance.max(initial=0.0))
        if maximum > 0.0:
            guidance /= maximum
        return guidance

    def trace_rays(self, target_cell, ray_range=None):
        """Trace a subset of the inverse rays; return the raw accumulators.

        ``ray_range`` selects which ray indices to emit (``None`` = all).
        Rays are independent and each votes at most once per cell, so tracing
        disjoint subsets and merging with sum / elementwise-min reproduces a
        single full pass exactly.
        """
        game = self.game
        width, height = len(game.maze), len(game.maze[0])
        counts = np.zeros((width, height), dtype=np.int32)
        by_bounce = np.zeros(
            (self.max_bounces + 1, width, height), dtype=np.int32)
        histogram = np.zeros((width, height, AIM_BINS), dtype=np.int32)
        min_frames = np.full((width, height), np.inf, dtype=np.float32)

        scale = float(game.scale)
        target_x = (target_cell[0] + 0.5) * scale
        target_y = (target_cell[1] + 0.5) * scale
        speed = C.BULLETSPEED * (scale / 50.0)
        max_distance = speed * self.max_frames
        muzzle_offset = scale * 4.5 / 16.0
        step = SAMPLE_STEP_CELLS * scale
        min_distance = MIN_SHOOTER_DISTANCE_CELLS * scale
        epsilon = max(1e-5, scale * 1e-5)

        rays = range(self.ray_count) if ray_range is None else ray_range
        for ray in rays:
            angle = 2.0 * math.pi * (ray + 0.5) / self.ray_count
            dx, dy = math.cos(angle), math.sin(angle)
            x, y = target_x, target_y
            remaining = max_distance
            travelled = 0.0
            bounces = 0
            ray_cells = set()
            ray_bounce_cells = set()
            ray_aim_bins = set()

            while remaining > epsilon and bounces <= self.max_bounces:
                wall_distance, flip_x, flip_y = self._nearest_wall(
                    x, y, dx, dy)
                segment = min(remaining, wall_distance)
                sample_count = max(1, int(math.ceil(segment / step)))
                distances = np.linspace(
                    0.0, segment, sample_count + 1, dtype=np.float64)
                reverse_x = x + distances * dx
                reverse_y = y + distances * dy
                centre_x = reverse_x + muzzle_offset * dx
                centre_y = reverse_y + muzzle_offset * dy
                clear = self._clear_centres(centre_x, centre_y)
                far_enough = travelled + distances >= min_distance
                cell_x = np.floor(centre_x / scale).astype(np.int32)
                cell_y = np.floor(centre_y / scale).astype(np.int32)
                valid = (clear & far_enough
                         & (cell_x >= 0) & (cell_x < width)
                         & (cell_y >= 0) & (cell_y < height))
                forward_angle = math.atan2(-dy, -dx) % (2.0 * math.pi)
                aim_bin = min(
                    AIM_BINS - 1,
                    int(forward_angle / (2.0 * math.pi) * AIM_BINS))
                for index in np.flatnonzero(valid):
                    cell = (int(cell_x[index]), int(cell_y[index]))
                    if cell not in self.reachable:
                        continue
                    ray_cells.add(cell)
                    ray_bounce_cells.add((bounces, cell))
                    ray_aim_bins.add((cell, aim_bin))
                    frame = (travelled + float(distances[index])) / speed
                    if frame < min_frames[cell]:
                        min_frames[cell] = frame

                travelled += segment
                remaining -= segment
                if not math.isfinite(wall_distance) or wall_distance >= segment + epsilon:
                    break
                if bounces >= self.max_bounces:
                    break
                hit_x, hit_y = x + wall_distance * dx, y + wall_distance * dy
                if flip_x:
                    dx = -dx
                if flip_y:
                    dy = -dy
                if not flip_x and not flip_y:
                    dx, dy = -dx, -dy
                bounces += 1
                x, y = hit_x + epsilon * dx, hit_y + epsilon * dy
                remaining = max(0.0, remaining - epsilon)
                travelled += epsilon

            for cell in ray_cells:
                counts[cell] += 1
            for bounce, cell in ray_bounce_cells:
                by_bounce[bounce, cell[0], cell[1]] += 1
            for cell, aim_bin in ray_aim_bins:
                histogram[cell[0], cell[1], aim_bin] += 1

        return counts, by_bounce, histogram, min_frames

    def finalise(self, target_cell, counts, by_bounce, histogram,
                 min_frames):
        """Derive tiers / values / guidance from merged ray accumulators."""
        max_count = int(counts.max(initial=0))
        tiers = np.zeros_like(counts, dtype=np.int8)
        if max_count > 0:
            positive = counts > 0
            scaled = (self.levels * np.log1p(counts[positive])
                      / math.log1p(max_count))
            tiers[positive] = np.clip(
                np.ceil(scaled), 1, self.levels).astype(np.int8)
        values = np.zeros_like(counts, dtype=np.float32)
        positive = tiers > 0
        values[positive] = np.exp2(tiers[positive] - 1).astype(np.float32)
        guidance = self._guidance_envelope(counts, min_frames)
        return DensityField(
            target_cell=tuple(target_cell), ray_count=self.ray_count,
            max_bounces=self.max_bounces,
            max_flight_frames=self.max_frames, counts=counts,
            counts_by_bounce=by_bounce, aim_histogram=histogram,
            min_frames=min_frames, tiers=tiers, values=values,
            guidance=guidance,
            max_count=max_count)

    def build(self, target_cell):
        """Full single-pass build.  Behaviour unchanged from before the
        trace/finalise split."""
        return self.finalise(target_cell, *self.trace_rays(target_cell))


def _alignment(field, game, tank):
    cell = _cell(game, tank)
    heading = (tank.rotation - 90.0) * math.pi / 180.0
    aim, concentration = field.best_aim_at(cell, heading)
    if aim is None:
        return 0.0, None, 0.0
    alignment = 0.5 + 0.5 * math.cos(_angle_delta(aim, heading))
    return alignment, aim, concentration


def density_rollout(game, action, field, rng_seed, chain_state=None,
                    horizon=MPC_HORIZON, hold=MPC_HOLD,
                    opp_model="L2", opponent_action=None, leaf_fn=None):
    """Score one macro-action under exponential field and combat rules."""
    sandbox = make_sandbox(game, opp_model, rng_seed=rng_seed)
    me, enemy = sandbox.tanks
    if opponent_action is not None and enemy.alive:
        throttle, turn, fire = opponent_action
        enemy.forward, enemy.backup = throttle == 2, throttle == 0
        enemy.turn_left, enemy.turn_right = turn == 0, turn == 2
        enemy.fire = fire == 1
    start_cell = _cell(sandbox, me)
    start_value = field.value_at(start_cell)
    start_relative = field.relative_success_at(start_cell)
    start_alignment, _, start_concentration = _alignment(field, sandbox, me)
    shot = None
    if action[2] == 1 and me.trigger_released and sandbox.weapon_ready(me):
        shot = LaikaAI(sandbox, me).check_bullet_path(me.rotation)

    previous_value = start_value
    field_ascent = 0.0
    peak_value = start_value
    previous_cell = start_cell
    previous_guidance = field.guidance_at(start_cell)
    guidance_ascent = 0.0
    chain_gain = 0.0
    chain = HuntChainState() if chain_state is None else chain_state.clone()
    fired = False
    active_hit = False
    kill_score = None
    for frame in range(horizon):
        if frame == 0:
            apply_action(sandbox, action)
        elif frame == hold:
            me.fire = False
        events = sandbox.step()
        fired = fired or any(
            event[0] == "fire" and event[1] == 0 for event in events)
        active_hit = active_hit or any(
            event[0] == "hit" and event[1] == 0 and event[2] == 1
            for event in events)
        if not me.alive:
            return DEATH_SCORE + frame
        if not enemy.alive:
            # 击杀已经记账，但回合没有结束：引擎还要再跑 75 帧，一颗已经
            # 在飞的子弹仍然能把这局变成双杀。所以不能在这里 return——
            # 一旦 return，"击杀后被打死"就会被当成"赢"来打分，而这正是
            # 原来必须外挂一个击杀后状态机去修的洞。继续推演即可。
            if kill_score is None:
                kill_score = (ACTIVE_KILL_SCORE - 8.0 * frame if active_hit
                              else OPPONENT_SELF_SCORE - 2.0 * frame)
            continue

        chain.advance()
        current_cell = _cell(sandbox, me)
        value = field.value_at(current_cell)
        field_ascent += value - previous_value
        previous_value = value
        peak_value = max(peak_value, value)
        current_guidance = field.guidance_at(current_cell)
        guidance_ascent += current_guidance - previous_guidance
        previous_guidance = current_guidance
        if current_cell != previous_cell:
            chain_gain += chain.collect_ascent(
                field, previous_cell, current_cell)
            previous_cell = current_cell

    if kill_score is not None:
        # 活着走完了窗口。击杀分照给，但仍要扣掉此刻承受的来袭火力——
        # 原来的早退把这一项整个跳过了，于是"杀完停在弹道上"和"杀完躲
        # 进安全格"得分完全一样，AI 自然没有理由去躲。
        analyzer = OpportunityAnalyzer360(sandbox)
        return float(kill_score
                     - RISK_WEIGHT * analyzer.incoming_risk(sandbox))

    end_alignment, _, end_concentration = _alignment(field, sandbox, me)
    score = FIELD_ASCENT_WEIGHT * field_ascent
    score += FIELD_PEAK_WEIGHT * max(0.0, peak_value - start_value)
    score += GUIDANCE_PROGRESS_WEIGHT * guidance_ascent
    score += HUNT_CHAIN_GAIN_WEIGHT * chain_gain
    alignment_gain = end_alignment - start_alignment
    opportunity_weight = start_relative * max(start_value, 1.0)
    concentration = max(start_concentration, end_concentration, 0.10)
    score += (ALIGNMENT_WEIGHT * opportunity_weight * concentration
              * alignment_gain)

    if fired:
        if shot is not None and shot.get("result") == "HIT":
            score += GOOD_FIRE_BONUS
        elif shot is not None and shot.get("result") == "SUICIDE":
            score -= SUICIDE_FIRE_PENALTY
        else:
            score -= FAILED_FIRE_PENALTY * (1.0 + start_relative)

    analyzer = OpportunityAnalyzer360(sandbox)
    score -= RISK_WEIGHT * analyzer.incoming_risk(sandbox)

    # 价值叶子。只在"活到 horizon"时生效——rollout 内真实发生的生死
    # (ACTIVE_KILL / DEATH) 永远优先，与 AlphaZero 同构：真终局用真值，
    # 未终局才用估计。leaf_fn 约定返回 [-1, 1]。
    #
    # 权重刻意压到塑形项量级(~±300)而非终局量级(±12000)：P23 的教训是
    # 价值一旦盖过塑形，风格会被冲掉、策略变被动(我↔敌 2.59 -> 3.19 格)。
    if leaf_fn is not None:
        score += VALUE_LEAF_WEIGHT * float(leaf_fn(sandbox))
    return float(score)


class KillFieldTeacher:
    """Receding-horizon teacher for the inverse density rule proposal."""

    name = "P37 反演弹道密度场 MPC 老师"

    def __init__(self, seed=0, ray_count=DEFAULT_RAYS,
                 max_bounces=DEFAULT_BOUNCES,
                 max_flight_frames=DEFAULT_FLIGHT_FRAMES,
                 horizon=MPC_HORIZON, hold=MPC_HOLD, leaf_fn=None):
        self.leaf_fn = leaf_fn
        self.rng = random.Random(seed)
        self.ray_count = int(ray_count)
        self.max_bounces = int(max_bounces)
        self.max_flight_frames = int(max_flight_frames)
        self.horizon = int(horizon)
        self.hold = int(hold)
        self.reset()

    def reset(self):
        self.game = None
        self.round_number = None
        self._builder = None
        self._field_cache = {}
        self.field = None
        self.commit_remaining = 0
        self.committed_action = (1, 1, 0)
        self.field_build_seconds = 0.0
        self.field_builds = 0
        self.chain = HuntChainState()
        self.chain_total = 0.0
        self.last_chain_gain = 0.0
        self._chain_round = None
        self._chain_target = None
        self._chain_cell = None
        self.last_decision_kind = "none"
        self.last_scores = None
        self.last_action = (1, 1, 0)
        self.last_action_index = CANDIDATES.index(self.last_action)
        self.last_motion_action = (1, 1, 0)
        self.failed_translation = False
        self.failed_turn = False
        self.action_no_effect = False
        self.no_effect_frames = 0
        self.no_effect_events = 0
        self.own_bullet_guard_events = 0
        self.last_displacement = 0.0
        self.last_rotation_delta = 0.0
        self.observed_commit_remaining = 0
        self.observed_committed_action = (1, 1, 0)
        self.observed_previous_action = (1, 1, 0)
        self._effect_round = None
        self._effect_frame = None
        self._effect_pose = None
        self._effect_action = None

    def _observe_action_effect(self, game):
        """Compare the previous command with the resulting real pose."""
        tank = game.tanks[0]
        if self._effect_round != game.round_number:
            self.failed_translation = False
            self.failed_turn = False
            self.action_no_effect = False
            self.no_effect_frames = 0
            self.last_displacement = 0.0
            self.last_rotation_delta = 0.0
            self._effect_round = game.round_number
            self._effect_frame = game.frame
            self._effect_pose = (tank.x, tank.y, tank.rotation)
            self._effect_action = None
            return
        if self._effect_frame is None or game.frame == self._effect_frame:
            return

        previous = self._effect_pose
        action = self._effect_action
        if previous is None or action is None:
            return
        displacement = math.hypot(tank.x - previous[0], tank.y - previous[1])
        rotation_delta = abs(
            (tank.rotation - previous[2] + 180.0) % 360.0 - 180.0)
        self.last_displacement = displacement
        self.last_rotation_delta = rotation_delta
        requested_translation = action[0] != 1
        requested_turn = action[1] != 1
        moved = displacement > max(1e-4, game.scale * 1e-4)
        turned = rotation_delta > 1e-3
        self.failed_translation = requested_translation and not moved
        self.failed_turn = requested_turn and not turned
        self.action_no_effect = (
            (requested_translation or requested_turn)
            and not moved and not turned)
        self.no_effect_frames = (
            self.no_effect_frames + 1 if self.action_no_effect else 0)
        self.no_effect_events += int(self.action_no_effect)

    def _emit_action(self, game, action, kind):
        if action[2] and action[:2] != (1, 1):
            action = (1, 1, 1)
        if (action[2] == 0 and action[:2] != (1, 1)
                and action_self_hits(game, action)):
            # The selected MPC plan can predate a newly created own bullet.
            # Re-evaluate all no-fire motions against the live bullet set and
            # execute the safest one.  This is a narrow physics shield, not a
            # new navigation objective.
            from training.killfield_fast_distill import \
                post_kill_survival_scores
            safety_scores = post_kill_survival_scores(
                game, OWN_BULLET_GUARD_HORIZON)
            action = CANDIDATES[int(np.argmax(safety_scores))]
            action = (action[0], action[1], 0)
            self.commit_remaining = 0
            self.committed_action = action
            self.own_bullet_guard_events += 1
            kind = f"{kind}:own_bullet_guard"
        self.last_decision_kind = kind
        self.last_action = action
        self.last_action_index = CANDIDATES.index(action)
        if action[0] != 1 or action[1] != 1:
            self.last_motion_action = (action[0], action[1], 0)
        tank = game.tanks[0]
        self._effect_round = game.round_number
        self._effect_frame = game.frame
        self._effect_pose = (tank.x, tank.y, tank.rotation)
        self._effect_action = action
        return self._action_dict(action)

    def _ensure_field(self, game):
        if game is not self.game or game.round_number != self.round_number:
            self.game = game
            self.round_number = game.round_number
            self._builder = InverseDensityFieldBuilder(
                game, self.ray_count, self.max_bounces,
                self.max_flight_frames)
            self._field_cache = {}
            self.commit_remaining = 0
        if not game.tanks[1].alive:
            # 对着尸体建场会让 AI 继续朝"能打到他的位置"走，那是在追一个
            # 不存在的目标。空场让狩猎项归零，求生自然浮出来。
            self.field = NULL_FIELD
            return self.field
        target = _cell(game, game.tanks[1])
        if target not in self._field_cache:
            started = time.perf_counter()
            self._field_cache[target] = self._builder.build(target)
            self.field_build_seconds += time.perf_counter() - started
            self.field_builds += 1
            self.commit_remaining = 0
        self.field = self._field_cache[target]
        return self.field

    def _update_live_chain(self, game, field):
        current_cell = _cell(game, game.tanks[0])
        target = field.target_cell
        if self._chain_round != game.round_number:
            self.chain = HuntChainState()
            self._chain_round = game.round_number
            self._chain_target = target
            self._chain_cell = current_cell
            self.last_chain_gain = 0.0
            return
        self.chain.advance()
        target_stable = target == self._chain_target
        gain = self.chain.collect_ascent(
            field, self._chain_cell, current_cell,
            target_stable=target_stable)
        self.last_chain_gain = gain
        self.chain_total += gain
        self._chain_target = target
        self._chain_cell = current_cell

    @staticmethod
    def _action_dict(action):
        throttle, turn, fire = action
        return {
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": fire == 1,
        }

    @staticmethod
    def _verified_hit(game):
        """保留旧接口（realtime 等处仍在用）。"""
        return KillFieldTeacher._fire_confidence(game)[0]

    @staticmethod
    def _fire_confidence(game):
        """返回 (预测必中?, 置信度)。

        check_bullet_path 全程假设对手静止。飞行越久对手挪开的机会越大，
        所以置信度按预测飞行帧数指数衰减——这正是"骗光子弹"的机理。
        """
        me = game.tanks[0]
        if not (me.alive and game.tanks[1].alive
                and me.trigger_released and game.weapon_ready(me)):
            return False, 0.0
        result = LaikaAI(game, me).check_bullet_path(me.rotation)
        if result.get("result") != "HIT":
            return False, 0.0
        flight = float(result.get("time", 0.0))
        return True, math.exp(-flight / FIRE_CONFIDENCE_TAU)

    def scores(self, game):
        field = self._ensure_field(game)
        seed = self.rng.randrange(1 << 30)
        # 看多远，取决于"结局还要多久才定下来"。对手已死时未定的部分正是
        # 那 75 帧结算窗口，看到 36 帧就收手会漏掉一半飞行中的子弹。
        # 注意这调的是搜索深度，不是目标函数——目标函数自始至终只有一个。
        horizon = self.horizon
        if not game.tanks[1].alive:
            horizon = max(horizon, C.NUMBEROFFRAMESBEFOREEND
                          - C.NUMBEROFFRAMESFROZEN)
        # mask_moving_fire_scores 会把这 8 个"移动+开火"候选无条件覆写掉，
        # 所以不必为它们跑 36 帧前推。输出数组逐位不变，推演量减少 44%。
        scores = np.zeros(len(CANDIDATES), dtype=np.float32)
        # 候选硬淘汰已回退：判据是"位移不足 **且** 转向不足"，而贴墙时
        # 转速反而翻倍（实测 102.8°/帧 vs 自由时 52.5°），所以顶墙转向
        # 的候选根本通不过这个 and。实测卡墙 14.0% -> 16.5%，没有收益，
        # 每次决策却多花 9 个沙盒。action_motion 保留备用。
        for index in LIVE_ACTION_INDICES:
            scores[index] = density_rollout(
                game, CANDIDATES[index], field, seed, self.chain,
                horizon, self.hold, leaf_fn=self.leaf_fn)
        mask_moving_fire_scores(scores)
        if self.action_no_effect and self.observed_previous_action is not None:
            failed_movement = self.observed_previous_action[:2]
            for index, action in enumerate(CANDIDATES):
                if action[:2] == failed_movement:
                    scores[index] -= NO_EFFECT_REPEAT_PENALTY
        return scores

    def act(self, game):
        self.last_decision_kind = "none"
        self.last_scores = None
        if not game.tanks[0].alive:
            return {}
        self._observe_action_effect(game)
        # 这里原本有一整段"击杀后求生"状态机：换一套打分函数、换 horizon、
        # 强制不开火、单独的 commit 逻辑。它已被删除。
        #
        # 删它不是砍功能，而是因为它补的那个洞已经在源头堵上了——rollout
        # 不再在击杀帧 return，所以"杀完再被打死"本来就会被算成死亡。
        # 目标没了的时候，场是空的、狩猎项恒为 0，剩下的死亡罚分和火力
        # 风险自己就构成求生目标。求生现在是**同一个目标函数推出来的**，
        # 不是切过去的模式。删硬编码 != 加规则。
        field = self._ensure_field(game)
        self._update_live_chain(game, field)
        self.observed_previous_action = (
            self._effect_action if self._effect_action is not None
            else (1, 1, 0))
        self.observed_commit_remaining = self.commit_remaining
        self.observed_committed_action = self.committed_action

        if self.action_no_effect:
            self.commit_remaining = 0

        # No arbitrary firing threshold: an engine-verified shot is taken on
        # the first available frame and overrides movement commitment.
        # 硬编码扳机反射已删除。它绕过全部评分，且依据的
        # check_bullet_path 只有 16% 准确率。开火现在是普通候选，
        # 由 density_rollout 里的精确沙盒判据参与竞争。
        # 删硬编码 != 加规则：前者把决定权交还给系统。

        if self.commit_remaining > 0 and not game.tanks[0].hit_something:
            self.commit_remaining -= 1
            return self._emit_action(game, self.committed_action, "hold")

        values = self.scores(game)
        self.last_scores = values.copy()
        action = CANDIDATES[int(np.argmax(values))]
        if action[2] == 0:
            self.committed_action = action
            self.commit_remaining = (
                COMMIT_MOVE_FRAMES if action[0] != 1
                else COMMIT_TURN_FRAMES if action[1] != 1 else 0)
        return self._emit_action(game, action, "plan")

    def telemetry(self):
        mean_build = self.field_build_seconds / max(self.field_builds, 1)
        return {
            "field_builds": self.field_builds,
            "mean_field_build_seconds": mean_build,
            "cached_target_cells": len(self._field_cache),
            "hunt_chain": self.chain.count,
            "hunt_chain_timer": self.chain.timer,
            "hunt_chain_total": self.chain_total,
            "last_chain_gain": self.last_chain_gain,
            "failed_translation": self.failed_translation,
            "failed_turn": self.failed_turn,
            "action_no_effect": self.action_no_effect,
            "no_effect_frames": self.no_effect_frames,
            "no_effect_events": self.no_effect_events,
            "own_bullet_guard_events": self.own_bullet_guard_events,
            "last_displacement": self.last_displacement,
            "last_rotation_delta": self.last_rotation_delta,
            "observed_commit_remaining": self.observed_commit_remaining,
        }


def smoke(seed, rays, bounces, flight_frames, horizon, hold):
    from tank_trouble_original.game import Game

    game = Game(seed=seed, ai_enabled=True)
    teacher = KillFieldTeacher(
        seed=seed ^ 0x37D31517, ray_count=rays,
        max_bounces=bounces, max_flight_frames=flight_frames,
        horizon=horizon, hold=hold)
    started = time.perf_counter()
    field = teacher._ensure_field(game)
    build_seconds = time.perf_counter() - started
    scores = teacher.scores(game)
    best = int(np.argmax(scores))
    nonzero = field.counts[field.counts > 0]
    print(json.dumps({
        "seed": seed,
        "target_cell": field.target_cell,
        "rays": field.ray_count,
        "max_bounces": field.max_bounces,
        "max_flight_frames": field.max_flight_frames,
        "build_seconds": build_seconds,
        "nonzero_cells": int(np.count_nonzero(field.counts)),
        "max_count": field.max_count,
        "median_positive_count": (
            float(np.median(nonzero)) if len(nonzero) else 0.0),
        "tiers": sorted(int(value) for value in np.unique(field.tiers)),
        "best_action": CANDIDATES[best],
        "best_score": float(scores[best]),
        "finite_scores": bool(np.isfinite(scores).all()),
        "telemetry": teacher.telemetry(),
    }, ensure_ascii=False, indent=2))


def probe(seed, rays, bounces, flight_frames, horizon, hold, frames):
    """Short headless behaviour probe; not a policy evaluation gate."""
    from tank_trouble_original.game import Game

    game = Game(seed=seed, ai_enabled=True)
    teacher = KillFieldTeacher(
        seed=seed ^ 0x37D31517, ray_count=rays,
        max_bounces=bounces, max_flight_frames=flight_frames,
        horizon=horizon, hold=hold)
    stats = {
        "frames": 0, "moving_frames": 0, "turning_frames": 0,
        "fires": 0, "active_hits": 0, "self_hits": 0,
        "player_deaths": 0, "laika_deaths": 0, "new_rounds": 0,
    }
    started = time.perf_counter()
    for _ in range(frames):
        action = teacher.act(game)
        tank = game.tanks[0]
        tank.forward = bool(action.get("forward", False))
        tank.backup = bool(action.get("backup", False))
        tank.turn_left = bool(action.get("turn_left", False))
        tank.turn_right = bool(action.get("turn_right", False))
        tank.fire = bool(action.get("fire", False))
        stats["moving_frames"] += int(tank.forward or tank.backup)
        stats["turning_frames"] += int(tank.turn_left or tank.turn_right)
        events = game.step()
        stats["frames"] += 1
        stats["fires"] += sum(
            event[0] == "fire" and event[1] == 0 for event in events)
        stats["active_hits"] += sum(
            event[0] == "hit" and event[1] == 0 and event[2] == 1
            for event in events)
        stats["self_hits"] += sum(
            event[0] == "hit" and event[1] == 0 and event[2] == 0
            for event in events)
        stats["player_deaths"] += sum(
            event[0] == "destroy" and event[1] == 0 for event in events)
        stats["laika_deaths"] += sum(
            event[0] == "destroy" and event[1] == 1 for event in events)
        stats["new_rounds"] += sum(
            event[0] == "new_round" for event in events)
    elapsed = time.perf_counter() - started
    stats.update({
        "elapsed_seconds": elapsed,
        "mean_decision_ms": 1000.0 * elapsed / max(frames, 1),
        "moving_pct": stats["moving_frames"] / max(frames, 1),
        "turning_pct": stats["turning_frames"] / max(frames, 1),
        "scoreboard": game.scores,
        "field": teacher.telemetry(),
    })
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "probe"])
    parser.add_argument("--seed", type=int, default=37_500_001)
    parser.add_argument("--rays", type=int, default=DEFAULT_RAYS)
    parser.add_argument("--bounces", type=int, default=DEFAULT_BOUNCES)
    parser.add_argument("--flight-frames", type=int,
                        default=DEFAULT_FLIGHT_FRAMES)
    parser.add_argument("--horizon", type=int, default=MPC_HORIZON)
    parser.add_argument("--hold", type=int, default=MPC_HOLD)
    parser.add_argument("--frames", type=int, default=250)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.seed, args.rays, args.bounces, args.flight_frames,
              args.horizon, args.hold)
    else:
        probe(args.seed, args.rays, args.bounces, args.flight_frames,
              args.horizon, args.hold, args.frames)


if __name__ == "__main__":
    main()
