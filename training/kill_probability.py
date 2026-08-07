"""击杀概率：把"当前局面我多容易杀掉他"写成一个有物理量纲的数。

动机（用户拍板）：现在的评分是一堆没有共同量纲的手调权重——
`FIELD_ASCENT=34 / GUIDANCE=120 / ALIGNMENT=190 / RISK=320 / GOOD_FIRE=1800`，
每个都是拍的。有了 P_kill 之后，所有动作都能换算成同一个量：

    候选价值 = ΔP_kill(我) − ΔP_kill(对手)

朝射击位移动 = 降低 T_move；转炮口 = 降低 T_aim；躲子弹 = 抬高对手的
T_kill。五个拍脑袋的权重塌缩成一个可推导的物理量。

公式
----
    T_kill = T_move + T_aim + T_flight + T_weapon        （单位：帧）
    P_kill = exp(−T_kill / τ)

T_move 和 T_aim 是**相加**不是取 max：坦克只能沿着车头方向移动，所以
"走到射击位"和"把炮口转到射击角"没法并行——先开到位，再转。

实测常数（65.3 px/格）
----------------------
    坦克前进      5.22 px/帧 = 0.0800 格/帧  ->  12.5 帧/格
    坦克转向      10 °/帧                    ->  18 帧转 180°
    子弹          5.88 px/帧 = 0.0900 格/帧  ->  11.1 帧/格
    坦克半宽      0.168 格                   ->  横move开身位仅 2.1 帧

最后一行是这个游戏的关键尺度：**对手清出自己身位只要 2.1 帧，而实测
子弹飞行是 30–75 帧，差 15–35 倍。**这解释了为什么开火命中率只有
14–16%（对静止目标也一样）、为什么价值函数学不出来、为什么更深的搜索
没用——胜负取决于几十毫秒级的几何。

τ 不能拍
--------
* 理论上界 τ = 2.1 帧（对手全速横移躲避）→ P 恒为 0，因为最短 T_kill
  也有 30 帧；
* 实测拟合 τ ≈ 23 帧（命中率 14% / 平均飞行 45 帧反解）。

两者差 11 倍，说明对手并不在最优躲避——他们为自己的目的移动，碰巧留在
弹道上的概率远高于最坏情况。所以 τ 必须**从数据拟合**，`fit_tau()` 就是
干这个的。
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import constants as C  # noqa: E402

# 由引擎常数推导，不是调参
FRAMES_PER_CELL = None      # 运行时按 game.scale 算
TURN_DEGREES_PER_FRAME = float(C.TANK_TURN_SPEED)
DEFAULT_TAU = 23.0          # 帧；见模块 docstring，应由 fit_tau 覆盖


def frames_per_cell(game):
    """坦克走一格要多少帧。"""
    return game.scale / max(game.tanks[0].forward_speed, 1e-6)


def _cell_of(game, tank):
    return int(tank.x // game.scale), int(tank.y // game.scale)


def _weapon_wait(game, tank):
    """弹匣不可用时还要等多少帧。可用则 0。"""
    if game.weapon_ready(tank) and tank.trigger_released:
        return 0.0
    mine = [b for b in game.bullets if not b.removed and b.owner is tank]
    if not mine:
        return 0.0
    # 最早一颗子弹的剩余寿命 —— 它消失后弹槽才空出来
    return float(min(getattr(b, "lifetime", C.BULLETLIFETIME) for b in mine))


def kill_time(game, field, shooter, target, max_cells=None):
    """当前局面下，shooter 杀掉 target 所需的最短时间（帧）。

    假设 target 原地不动——这是"机会有多好"的定义，不是对未来的预测。
    返回 (T_kill, 最优射击格, 分项字典)。找不到任何射击位时返回 inf。
    """
    per_cell = frames_per_cell(game)
    my_cell = _cell_of(game, shooter)
    heading = (shooter.rotation - 90.0) * math.pi / 180.0
    distances = game.dist_map(my_cell[0], my_cell[1])
    wait = _weapon_wait(game, shooter)

    best = (float("inf"), None, None)
    cells = game.reachable if max_cells is None else game.reachable[:max_cells]
    for item in cells:
        cell = (int(item["x"]), int(item["y"]))
        flight = float(field.min_frames[cell]) if field.in_bounds(cell) else \
            float("inf")
        if not math.isfinite(flight):
            continue                      # 这个格打不到对方

        if distances is None:
            continue
        hops = distances[cell[0]][cell[1]]
        if hops is None or hops != hops:
            continue
        t_move = float(hops) * per_cell

        angle, _mass = field.best_aim_at(cell, heading)
        if angle is None:
            continue
        delta = abs(math.atan2(math.sin(angle - heading),
                               math.cos(angle - heading)))
        t_aim = math.degrees(delta) / TURN_DEGREES_PER_FRAME

        # 车头只能朝一个方向：先开到位，再转炮口，所以是相加不是 max
        total = t_move + t_aim + flight + wait
        if total < best[0]:
            best = (total, cell,
                    {"move": t_move, "aim": t_aim,
                     "flight": flight, "weapon": wait})
    return best


def kill_probability(kill_frames, tau=DEFAULT_TAU):
    """T_kill -> P_kill。时间越长，对手越有机会走开。"""
    if not math.isfinite(kill_frames):
        return 0.0
    return math.exp(-max(0.0, kill_frames) / max(tau, 1e-6))


class KillProbabilityMeter:
    """双方各算一次 P_kill。场按 (回合, 目标格) 缓存。"""

    def __init__(self, game, rays=256, bounces=2, flight=75,
                 tau=DEFAULT_TAU):
        from training.killfield_teacher import InverseDensityFieldBuilder
        self.builder = InverseDensityFieldBuilder(game, rays, bounces, flight)
        self.cache = {}
        self.tau = float(tau)
        self.round_number = game.round_number

    def _field(self, game, target_cell):
        if game.round_number != self.round_number:
            from training.killfield_teacher import InverseDensityFieldBuilder
            self.builder = InverseDensityFieldBuilder(
                game, self.builder.ray_count, self.builder.max_bounces,
                self.builder.max_frames)
            self.cache = {}
            self.round_number = game.round_number
        if target_cell not in self.cache:
            self.cache[target_cell] = self.builder.build(target_cell)
        return self.cache[target_cell]

    def measure(self, game, shooter_index):
        shooter = game.tanks[shooter_index]
        target = game.tanks[1 - shooter_index]
        if not (shooter.alive and target.alive):
            # 有人已经死了：不是"没有射击位"，是这一局结束了。
            # 两者必须分开，否则 HUD 上看起来像公式失效。
            return 0.0, float("inf"), "round_over"
        field = self._field(game, _cell_of(game, target))
        frames, cell, parts = kill_time(game, field, shooter, target)
        return kill_probability(frames, self.tau), frames, parts


def fit_tau(samples):
    """从 (T_kill, 是否真的击杀) 样本里极大似然拟合 τ。

    P(kill) = exp(−T/τ)。对数似然对 τ 求导没有闭式解，直接网格搜索。
    """
    if not samples:
        return DEFAULT_TAU
    times = np.asarray([t for t, _ in samples], dtype=np.float64)
    hits = np.asarray([1.0 if k else 0.0 for _, k in samples])
    grid = np.linspace(1.0, 400.0, 4000)
    best, best_ll = DEFAULT_TAU, -1e18
    for tau in grid:
        p = np.clip(np.exp(-times / tau), 1e-9, 1 - 1e-9)
        ll = float((hits * np.log(p) + (1 - hits) * np.log(1 - p)).sum())
        if ll > best_ll:
            best_ll, best = ll, float(tau)
    return best


def _probe(args):
    """在真实对局里打印双方的 P_kill，看它跳动得合不合理。"""
    from tank_trouble_original.game import Game
    from training.killfield_prebuild import FastKillFieldTeacher

    game = Game(seed=args.seed, ai_enabled=True)
    teacher = FastKillFieldTeacher(
        seed=373, ray_count=256, max_bounces=2, max_flight_frames=75,
        horizon=36, skip_masked=True, parallel_workers=0)
    meter = KillProbabilityMeter(game, rays=args.rays, tau=args.tau)
    me = game.tanks[0]
    print(f"{'帧':>5}{'我P':>8}{'敌P':>8}{'T_kill':>9}"
          f"{'走':>7}{'瞄':>7}{'飞':>7}{'弹':>7}")
    print("-" * 60)
    for frame in range(args.frames):
        if game.frozen or not me.alive:
            break
        if frame % args.every == 0:
            mine, t_mine, parts = meter.measure(game, 0)
            theirs, _t, _p = meter.measure(game, 1)
            if parts == "round_over":
                print(f"{frame:>5}   —— 回合已结束（有人被击杀）——")
            elif parts:
                print(f"{frame:>5}{mine:>8.3f}{theirs:>8.3f}{t_mine:>9.1f}"
                      f"{parts['move']:>7.1f}{parts['aim']:>7.1f}"
                      f"{parts['flight']:>7.1f}{parts['weapon']:>7.1f}")
            else:
                print(f"{frame:>5}{mine:>8.3f}{theirs:>8.3f}      无射击位")
        action = teacher.act(game)
        me.forward = bool(action.get("forward", False))
        me.backup = bool(action.get("backup", False))
        me.turn_left = bool(action.get("turn_left", False))
        me.turn_right = bool(action.get("turn_right", False))
        me.fire = bool(action.get("fire", False))
        game.step()
    teacher.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["probe"])
    parser.add_argument("--seed", type=int, default=37_500_001)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--rays", type=int, default=256)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = parser.parse_args()
    _probe(args)


if __name__ == "__main__":
    main()


def collect_tau_samples(seeds, rays=256, cap=750, horizon=36, window=25):
    """采 (T_kill, 固定窗口内是否击杀, 实际到击杀还有多久) 样本。

    **窗口必须固定。** 第一版用"T_kill 帧之内是否击杀"当判据，结果
    校准完全反了（T_kill 200+ 帧的实际击杀率 51.7%，0-30 帧只有 8.0%）
    —— 因为 T_kill 越大观察窗口越宽，长窗口自然更容易套住一次击杀。
    那是把"机会好坏"和"观察时长"混在了一起，不是公式错，是验证方法错。
    """
    from tank_trouble_original.game import Game
    from training.killfield_prebuild import FastKillFieldTeacher

    samples = []
    for seed in seeds:
        game = Game(seed=seed, ai_enabled=True)
        teacher = FastKillFieldTeacher(
            seed=373, ray_count=rays, max_bounces=2, max_flight_frames=75,
            horizon=horizon, skip_masked=True, parallel_workers=0)
        meter = KillProbabilityMeter(game, rays=rays)
        me = game.tanks[0]
        pending = []          # (采样帧, T_kill)
        kill_frame = None
        for frame in range(cap):
            if game.frozen or not me.alive:
                break
            if game.tanks[1].alive and frame % 5 == 0:
                _p, t_kill, parts = meter.measure(game, 0)
                if parts and parts != "round_over" and math.isfinite(t_kill):
                    pending.append((frame, t_kill))
            action = teacher.act(game)
            me.forward = bool(action.get("forward", False))
            me.backup = bool(action.get("backup", False))
            me.turn_left = bool(action.get("turn_left", False))
            me.turn_right = bool(action.get("turn_right", False))
            me.fire = bool(action.get("fire", False))
            for event in game.step():
                if event[0] == "destroy" and event[1] == 1:
                    kill_frame = frame
            if kill_frame is not None:
                break
        teacher.close()
        for sampled_at, t_kill in pending:
            hit = (kill_frame is not None
                   and sampled_at <= kill_frame <= sampled_at + window)
            actual = (kill_frame - sampled_at) if kill_frame is not None \
                else None
            samples.append((t_kill, hit, actual))
    return samples
