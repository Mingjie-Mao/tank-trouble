"""P37 老师的调度原型：邻格投机预建 + 击杀后导入预热。

只改"什么时候算"，不改"算什么"：

- 评分函数 ``density_rollout``、候选集 ``CANDIDATES``、承诺时长、
  强制开火判据、击杀后逻辑全部沿用 ``KillFieldTeacher``；
- 击杀场的数值结果与父类逐位相同，区别只是它在 Laika 走进该格之前
  就已经在后台算好了；
- 父类在"首次为某个目标格建场"时会把 ``commit_remaining`` 清零。
  预建会让这一步变成缓存命中，因此这里用"该格本回合是否首次成为
  目标"作为判据显式复现同一语义，避免行为漂移。

用法与父类一致，额外参数 ``prebuild_workers`` / ``prebuild_radius``。
"""

import concurrent.futures
import multiprocessing
import pickle
import time

import numpy as np

from training.killfield_teacher import (
    KillFieldTeacher,
    InverseDensityFieldBuilder,
    density_rollout,
    mask_moving_fire_scores,
    _cell,
    NO_EFFECT_REPEAT_PENALTY,
)
from training.mpc_agent import CANDIDATES

# mask_moving_fire_scores 会把"移动 + 开火"的 8 列无条件覆写成 -1e9，
# 无论推演返回什么。所以这 8 次前推是纯浪费，跳过它们后输出数组逐位相同。
LIVE_ACTION_INDICES = tuple(
    index for index, action in enumerate(CANDIDATES)
    if not (action[2] and action[:2] != (1, 1)))

# 注意：预热导入放在 __init__ 里而不是模块顶层。子进程要 import 本模块才能
# 找到 _prebuild_worker，若顶层拉 torch，每个 spawn worker 都要付一次。

_WORKER_BUILDERS = {}


def _prebuild_worker(maze_key, snapshot_bytes, target_cell,
                     rays, bounces, flight_frames):
    """子进程入口：按迷宫缓存 builder，返回一个 DensityField。"""
    key = (maze_key, rays, bounces, flight_frames)
    builder = _WORKER_BUILDERS.get(key)
    if builder is None:
        game = pickle.loads(snapshot_bytes)
        builder = InverseDensityFieldBuilder(
            game, rays, bounces, flight_frames)
        _WORKER_BUILDERS.clear()
        _WORKER_BUILDERS[key] = builder
    return builder.build(tuple(target_cell))


def _warm_worker(_):
    """空任务：只为强制 ProcessPoolExecutor 提前把子进程拉起来。"""
    time.sleep(0.05)
    return True


def _trace_worker(maze_key, snapshot_bytes, target_cell, ray_indices,
                  rays, bounces, flight_frames):
    """子进程入口：只追踪指定的射线子集，返回原始累加器。"""
    key = (maze_key, rays, bounces, flight_frames)
    builder = _WORKER_BUILDERS.get(key)
    if builder is None:
        game = pickle.loads(snapshot_bytes)
        builder = InverseDensityFieldBuilder(
            game, rays, bounces, flight_frames)
        _WORKER_BUILDERS.clear()
        _WORKER_BUILDERS[key] = builder
    return builder.trace_rays(tuple(target_cell), ray_indices)


class _ParallelBuilder:
    """把 trace_rays 按射线拆到多个进程上，再归并交给 finalise。

    射线彼此独立、每条对每格最多投一票，所以 sum / elementwise-min 归并
    与单趟全量追踪逐位相同（已验证）。
    """

    def __init__(self, base, executor, workers, game):
        self.base = base
        self.executor = executor
        self.workers = max(1, int(workers))
        self.ray_count = base.ray_count
        self.snapshot = pickle.dumps(
            game, protocol=pickle.HIGHEST_PROTOCOL)
        self.maze_key = (id(game), game.round_number)

    def build(self, target_cell):
        chunks = [
            list(range(i, self.ray_count, self.workers))
            for i in range(self.workers)
        ]
        futures = [
            self.executor.submit(
                _trace_worker, self.maze_key, self.snapshot,
                tuple(target_cell), chunk, self.ray_count,
                self.base.max_bounces, self.base.max_frames)
            for chunk in chunks if chunk
        ]
        parts = [future.result() for future in futures]
        counts = sum(part[0] for part in parts)
        by_bounce = sum(part[1] for part in parts)
        histogram = sum(part[2] for part in parts)
        min_frames = np.minimum.reduce([part[3] for part in parts])
        return self.base.finalise(
            target_cell, counts, by_bounce, histogram, min_frames)


def _rollout_worker(snapshot_bytes, field_bytes, indices, seed, chain,
                    horizon, hold):
    """子进程入口：算一批候选动作的评分，返回 (index, score) 列表。"""
    game = pickle.loads(snapshot_bytes)
    field = pickle.loads(field_bytes)
    return [
        (index, density_rollout(game, CANDIDATES[index], field, seed,
                                chain, horizon, hold))
        for index in indices
    ]


class FastKillFieldTeacher(KillFieldTeacher):
    """行为等价于 P37 老师，只改推演的调度方式。

    两个开关：

    ``skip_masked``
        不去算那 8 个必被屏蔽的"移动+开火"候选。输出逐位相同。

    ``parallel_workers``
        把剩余候选拆到多个进程上算。所有候选共用同一个 ``seed`` 和
        同一份 ``field``，``chain`` 在 ``density_rollout`` 内部各自
        ``clone()``，因此结果与串行完全一致。

    与投机预建不同，这里的并行发生在游戏线程**本来就在阻塞等待**的时候，
    所以不存在抢核问题——所有核都在服务当前这次决策。
    """

    name = "P37 击杀场老师 (跳过死列 + 并行推演)"

    def __init__(self, *args, skip_masked=True, parallel_workers=0,
                 parallel_field=True, **kwargs):
        import training.killfield_fast_distill  # noqa: F401  预热延迟导入
        self.skip_masked = bool(skip_masked)
        self.parallel_workers = max(0, int(parallel_workers))
        self.parallel_field = bool(parallel_field)
        self._executor = None
        self.executor_mode = "serial"
        self.rollouts_computed = 0
        self.rollouts_skipped = 0
        super().__init__(*args, **kwargs)
        # spawn 一个子进程要重新 import Python/numpy/本模块。若等到第一次
        # 规划才拉起, 那一帧会被拖到 ~370ms。构造期先热身。
        self._ensure_executor()
        if self._executor is not None:
            list(self._executor.map(
                _warm_worker, range(self.parallel_workers)))

    def _ensure_executor(self):
        if self._executor is not None or self.parallel_workers <= 0:
            return
        try:
            context = multiprocessing.get_context("spawn")
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.parallel_workers, mp_context=context)
            self.executor_mode = "process"
        except Exception:
            self._executor = None
            self.executor_mode = "serial"

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _make_builder(self, game):
        base = InverseDensityFieldBuilder(
            game, self.ray_count, self.max_bounces, self.max_flight_frames)
        if self._executor is None or not self.parallel_field:
            return base
        return _ParallelBuilder(
            base, self._executor, self.parallel_workers, game)

    def _ensure_field(self, game):
        # 与父类同构，唯一区别是 builder 可能是并行版本。
        if game is not self.game or game.round_number != self.round_number:
            self.game = game
            self.round_number = game.round_number
            self._builder = self._make_builder(game)
            self._field_cache = {}
            self.commit_remaining = 0
        target = _cell(game, game.tanks[1])
        if target not in self._field_cache:
            started = time.perf_counter()
            self._field_cache[target] = self._builder.build(target)
            self.field_build_seconds += time.perf_counter() - started
            self.field_builds += 1
            self.commit_remaining = 0
        self.field = self._field_cache[target]
        return self.field

    def scores(self, game):
        field = self._ensure_field(game)
        seed = self.rng.randrange(1 << 30)
        indices = (LIVE_ACTION_INDICES if self.skip_masked
                   else tuple(range(len(CANDIDATES))))
        self.rollouts_computed += len(indices)
        self.rollouts_skipped += len(CANDIDATES) - len(indices)

        values = np.zeros(len(CANDIDATES), dtype=np.float32)
        if self.parallel_workers > 0 and len(indices) > 1:
            self._ensure_executor()
        if self._executor is not None:
            snapshot = pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL)
            field_bytes = pickle.dumps(
                field, protocol=pickle.HIGHEST_PROTOCOL)
            chunks = [list(indices[i::self.parallel_workers])
                      for i in range(self.parallel_workers)]
            futures = [
                self._executor.submit(
                    _rollout_worker, snapshot, field_bytes, chunk, seed,
                    self.chain.clone(), self.horizon, self.hold)
                for chunk in chunks if chunk
            ]
            for future in futures:
                for index, score in future.result():
                    values[index] = score
        else:
            for index in indices:
                values[index] = density_rollout(
                    game, CANDIDATES[index], field, seed, self.chain,
                    self.horizon, self.hold)

        mask_moving_fire_scores(values)
        if self.action_no_effect and self.observed_previous_action is not None:
            failed_movement = self.observed_previous_action[:2]
            for index, action in enumerate(CANDIDATES):
                if action[:2] == failed_movement:
                    values[index] -= NO_EFFECT_REPEAT_PENALTY
        return values

    def telemetry(self):
        data = super().telemetry()
        data.update({
            "executor_mode": self.executor_mode,
            "rollouts_computed": self.rollouts_computed,
            "rollouts_skipped": self.rollouts_skipped,
        })
        return data


class PrebuildKillFieldTeacher(KillFieldTeacher):
    """行为等价于 P37 老师，但把建场挪出决策路径。"""

    name = "P37 击杀场老师 (邻格预建)"

    def __init__(self, *args, prebuild_workers=3, prebuild_radius=1,
                 **kwargs):
        # 击杀后分支在父类里是函数内延迟导入 (killfield_fast_distill 反向
        # 依赖 killfield_teacher, 顶层导入会循环)。在构造期先触发一次，
        # 让 torch 的导入代价不要落在击杀发生的那一帧。
        import training.killfield_fast_distill  # noqa: F401
        self.prebuild_workers = max(1, int(prebuild_workers))
        self.prebuild_radius = max(1, int(prebuild_radius))
        self._executor = None
        self.executor_mode = "none"
        self._pending = {}
        self._snapshot_bytes = None
        self._maze_key = None
        self._built_cells = set()
        self.prebuild_hits = 0
        self.prebuild_waits = 0
        self.prebuild_cold = 0
        self.prebuild_submitted = 0
        self.snapshot_seconds = 0.0
        self.blocking_seconds = []
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ 生命周期

    def reset(self):
        super().reset()
        self._cancel_pending()
        self._snapshot_bytes = None
        self._maze_key = None
        if hasattr(self, "_built_cells"):
            self._built_cells = set()

    def _cancel_pending(self):
        for future in getattr(self, "_pending", {}).values():
            future.cancel()
        self._pending = {}

    def _ensure_executor(self):
        if self._executor is not None:
            return
        try:
            context = multiprocessing.get_context("spawn")
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.prebuild_workers, mp_context=context)
            self.executor_mode = "process"
        except Exception:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.prebuild_workers)
            self.executor_mode = "thread"

    def close(self):
        self._cancel_pending()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ 预建

    def _harvest(self):
        """把已完成的后台结果收进缓存，不阻塞。"""
        for cell, future in list(self._pending.items()):
            if not future.done():
                continue
            self._pending.pop(cell, None)
            try:
                self._field_cache[cell] = future.result()
            except Exception:
                pass

    def _neighbour_cells(self, game, target):
        distances = game.dist_map(target[0], target[1])
        if distances is None:
            return []
        out = []
        for item in game.reachable:
            cell = (int(item["x"]), int(item["y"]))
            if cell == target or cell in self._field_cache:
                continue
            if cell in self._pending:
                continue
            distance = distances[cell[0]][cell[1]]
            if distance is None or distance != distance:
                continue
            if 1 <= int(distance) <= self.prebuild_radius:
                out.append((int(distance), cell))
        out.sort()
        return [cell for _, cell in out]

    def _speculate(self, game, target):
        candidates = self._neighbour_cells(game, target)
        if not candidates:
            return
        self._ensure_executor()
        if self._snapshot_bytes is None:
            started = time.perf_counter()
            self._snapshot_bytes = pickle.dumps(
                game, protocol=pickle.HIGHEST_PROTOCOL)
            self._maze_key = (id(game), game.round_number)
            self.snapshot_seconds += time.perf_counter() - started
        room = self.prebuild_workers - len(self._pending)
        for cell in candidates[:max(0, room)]:
            try:
                self._pending[cell] = self._executor.submit(
                    _prebuild_worker, self._maze_key, self._snapshot_bytes,
                    cell, self.ray_count, self.max_bounces,
                    self.max_flight_frames)
                self.prebuild_submitted += 1
            except Exception:
                break

    # ------------------------------------------------------------------ 覆盖点

    def _ensure_field(self, game):
        if game is not self.game or game.round_number != self.round_number:
            self.game = game
            self.round_number = game.round_number
            self._builder = InverseDensityFieldBuilder(
                game, self.ray_count, self.max_bounces,
                self.max_flight_frames)
            self._field_cache = {}
            self._cancel_pending()
            self._snapshot_bytes = None
            self._built_cells = set()
            self.commit_remaining = 0

        target = _cell(game, game.tanks[1])
        self._harvest()

        if target not in self._field_cache:
            started = time.perf_counter()
            field = None
            future = self._pending.get(target)
            # 只收割"已经算完"的结果。在途任务一律不等：它可能还排在
            # worker 队列里没开始，阻塞等待会比本地直接算更慢 (实测
            # 等待会把 max 从 80ms 抬到 180ms)。
            if future is not None and future.done():
                self._pending.pop(target, None)
                try:
                    field = future.result()
                    self.prebuild_waits += 1
                except Exception:
                    field = None
            if field is None:
                if future is not None and future.cancel():
                    self._pending.pop(target, None)
                field = self._builder.build(target)
                self.prebuild_cold += 1
            self._field_cache[target] = field
            blocked = time.perf_counter() - started
            self.field_build_seconds += blocked
            self.field_builds += 1
            self.blocking_seconds.append(blocked)
        else:
            self.prebuild_hits += 1

        # 复现父类语义：某格首次成为目标时打断承诺。父类靠"缓存未命中"
        # 隐式表达这一点，预建会让它变成命中，所以这里显式判定。
        if target not in self._built_cells:
            self._built_cells.add(target)
            self.commit_remaining = 0

        self.field = self._field_cache[target]
        self._speculate(game, target)
        return self.field

    def telemetry(self):
        data = super().telemetry()
        total = (self.prebuild_hits + self.prebuild_waits
                 + self.prebuild_cold)
        data.update({
            "executor_mode": self.executor_mode,
            "prebuild_hits": self.prebuild_hits,
            "prebuild_waits": self.prebuild_waits,
            "prebuild_cold": self.prebuild_cold,
            "prebuild_submitted": self.prebuild_submitted,
            "prebuild_hit_rate": self.prebuild_hits / max(total, 1),
            "snapshot_seconds": self.snapshot_seconds,
        })
        return data
