"""
正式训练环境 — Gymnasium 接口, 射线+向量观测, 轻势能塑形

设计要点:
  - 一局 episode = 新迷宫 -> 一方死亡 (destroy 事件立即 terminate,
    不跑原版 180 帧结算动画; reset 直接新建 Game, 最快最干净)
  - 超过 TRUNCATE_FRAMES 帧截断为平局 (存在双方僵持的死局, 必须截断)
  - action repeat: 每个决策重复 FRAME_SKIP 帧
  - 观测以自车为中心 (相对坐标旋转到自车朝向系), 天然免疫迷宫尺寸变化
  - 奖励: 击杀 +1 / 死亡 -1 (含自杀) / 平局 0
         + 势能塑形 gamma*phi(s') - phi(s), phi = -K_SHAPE * 迷宫路径距离
"""

import math
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import Game, constants as C  # noqa: E402

# ---- 环境参数 ----
FRAME_SKIP = 2            # 每个决策重复的帧数
TRUNCATE_FRAMES = 2500    # 单局最长帧数 (100 秒游戏时间)
N_RAYS = 24               # 射线数 (相对自车朝向均匀分布)
N_BULLET_SLOTS = 6        # 观测的最近子弹数
K_SHAPE = 0.02            # 势能塑形系数 (每格路径距离)
GAMMA = 0.995             # 与 PPO 的 gamma 保持一致

OBS_DIM = 6 + 8 + N_RAYS + N_BULLET_SLOTS * 6 + 2

# ---- v2 奖励参数 (打破"苟活躲弹"局部最优, 激励果断且精准的开火) ----
# v1 诊断: 每局仅打 1.1 发弹(弹匣5发), 靠躲到截断苟平; 走位已学会但不敢开火。
R_GOOD_SHOT = 0.30        # 开火时刻弹道模拟为 HIT -> 奖励 (瞄准后果断开火)
R_BAD_SHOT = -0.15        # 开火时刻弹道模拟为 SUICIDE -> 惩罚 (送头回头弹)
R_WASTE_SHOT = -0.02      # 开火但模拟 NOTHING 且无逼近价值 -> 轻惩 (乱打浪费弹)
R_TIME_PENALTY = -0.002   # 每决策步的存活惩罚 (逼它进攻, 削弱苟平吸引力)
R_DRAW = -0.5             # 平局(截断)明确判负向 (苟平不再是安全港)
NEAR_MISS_CELL = 2        # check_bullet_path 的 closest 折算 <= 该格数视为"有威胁"

# ---- v3 奖励参数 (打破 v2 的 ~52% 平台; 归因: 87% 败局死于对射) ----
# 核心: 奖励"化解来袭弹" — 上一步判定会命中我、这一步因移动变不会命中 = 成功闪避。
R_DODGE = 0.15           # 成功化解一颗本会命中的来袭弹
R_IN_DANGER = -0.01      # 当前有来袭弹锁定我 (持续压力, 逼它主动脱离)
DODGE_LOOKAHEAD = 22     # 来袭弹威胁预判帧数 (约 BULLETLIFETIME 的量级下限)


class TankTroubleGym(gym.Env):
    """vs Laika 训练环境。tank0 = 智能体, tank1 = Laika。"""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, reward_version: int = 2):
        super().__init__()
        self._base_seed = seed
        self._episode = 0
        self._rv = reward_version    # 1 = 纯胜负+弱塑形; 2 = 加开火塑形
        self.game = None
        self._sim = None             # Laika 弹道模拟器 (绑定到 tank0, 每局重建)
        self._wall_boxes = None      # (M,4) AABB 数组
        self._ray_angles = np.linspace(0, 2 * math.pi, N_RAYS,
                                       endpoint=False)
        self._max_ray = 1.0
        self._prev_phi = 0.0

        self.action_space = spaces.MultiDiscrete([3, 3, 2])
        self.observation_space = spaces.Box(
            low=-4.0, high=4.0, shape=(OBS_DIM,), dtype=np.float32)

    # ------------------------------------------------ 核心流程

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._base_seed = seed
        ep_seed = self._base_seed * 1_000_003 + self._episode
        self._episode += 1
        self.game = Game(seed=ep_seed, ai_enabled=True)
        self._build_wall_boxes()
        self._frames = 0
        self._prev_phi = self._phi()
        # v2: 复用 Laika 弹道模拟器判定"这一发好不好"
        if self._rv >= 2:
            from tank_trouble_original.laika import LaikaAI
            self._sim = LaikaAI(self.game, self.game.tanks[0])
        obs = self._obs()
        return obs, {"seed": ep_seed}

    def step(self, action):
        g = self.game
        t0 = g.tanks[0]
        throttle, turn, fire = int(action[0]), int(action[1]), int(action[2])
        t0.forward = throttle == 2
        t0.backup = throttle == 0
        t0.turn_left = turn == 0
        t0.turn_right = turn == 2
        t0.fire = fire == 1

        reward = 0.0
        terminated = False
        info = {}
        # v3: 决策步开始时, 我是否已被来袭弹锁定?
        danger_before = (self._rv >= 3
                         and self._incoming_will_hit(t0))
        for _ in range(FRAME_SKIP):
            # v2: 开火前先判"若此帧真能发弹, 这一发好不好"
            shot_reward = 0.0
            if self._rv >= 2 and t0.fire and g.weapon_ready(t0):
                shot_reward = self._shot_quality(t0)
            bf_before = t0.bullets_fired

            events = g.step()
            self._frames += 1

            # 只有真的打出弹(弹数增加)才结算开火奖励
            if self._rv >= 2 and t0.bullets_fired > bf_before:
                reward += shot_reward

            for ev in events:
                if ev[0] == "destroy":
                    if ev[1] == 0:
                        reward -= 1.0
                        info["result"] = "loss"
                    else:
                        reward += 1.0
                        info["result"] = "win"
                    terminated = True
            if terminated:
                break

        # v3: 闪避结算 — 决策步前被锁定, 步后存活且不再被锁 = 成功化解
        if self._rv >= 3 and not terminated:
            danger_after = self._incoming_will_hit(t0)
            if danger_before and not danger_after:
                reward += R_DODGE
            elif danger_after:
                reward += R_IN_DANGER

        truncated = False
        if not terminated and self._frames >= TRUNCATE_FRAMES:
            truncated = True
            info["result"] = "draw"
            if self._rv >= 2:
                reward += R_DRAW      # 苟平不再是安全港

        # 势能塑形 + 时间惩罚 (只在回合进行中)
        if not terminated:
            phi = self._phi()
            reward += GAMMA * phi - self._prev_phi
            self._prev_phi = phi
            if self._rv >= 2:
                reward += R_TIME_PENALTY

        return self._obs(), reward, terminated, truncated, info

    def _incoming_will_hit(self, tank):
        """是否有敌方子弹在未来 DODGE_LOOKAHEAD 帧内会命中我当前位置。

        复刻引擎子弹步进(每帧 7 子步 + 遇墙反弹), 逐帧检测是否进入我的
        命中形状。deadly==0 的子弹出膛即可命中, 故不设生效延迟。
        只看当前几何: 我不动、子弹按物理走, 会不会打到我。
        """
        g = self.game
        hci = C.BULLETHITCHECKINTERVALS
        for b in g.bullets:
            if b.owner is tank:
                continue          # 只躲敌方弹 (自己的回头弹由 v2 的开火惩罚管)
            x, y = b.x, b.y
            vx, vy = b.x_speed, b.y_speed
            life = b.lifetime
            for _ in range(DODGE_LOOKAHEAD):
                if life <= 0:
                    break
                for _ in range(hci):
                    px, py = x, y
                    x += vx
                    y += vy
                    if g.wall_hit(x, y):
                        # 与引擎一致: 独立测 X 反转 / Y 反转
                        hit_x_inv = g.wall_hit(px - vx, py + vy)
                        hit_y_inv = g.wall_hit(px + vx, py - vy)
                        if hit_x_inv and not hit_y_inv:
                            vy = -vy
                        elif hit_y_inv and not hit_x_inv:
                            vx = -vx
                        else:
                            vx = -vx
                            vy = -vy
                        x = px + vx
                        y = py + vy
                    if tank.point_in_shape(x, y):
                        return True
                life -= 1
        return False

    def _shot_quality(self, tank):
        """用 Laika 弹道模拟器评估当前朝向开火的这一发好坏。

        HIT     -> 会打中敌人, 大奖励
        SUICIDE -> 反弹回来打死自己, 惩罚
        NOTHING -> 没打中: 看 closest 是否逼近敌人(有威胁)给微奖, 否则轻惩浪费
        """
        res = self._sim.check_bullet_path(tank.rotation)
        r = res["result"]
        if r == "HIT":
            return R_GOOD_SHOT
        if r == "SUICIDE":
            return R_BAD_SHOT
        # NOTHING: check_bullet_path 的 closest 是像素曼哈顿距离(有威胁时)
        closest = res.get("closest", 1e9)
        if closest <= NEAR_MISS_CELL * self.game.scale:
            return R_GOOD_SHOT * 0.25    # 擦身而过也算施压
        return R_WASTE_SHOT

    # ------------------------------------------------ 势能

    def _phi(self):
        """phi = -K * 迷宫路径距离(自车格 -> 敌车格)"""
        g = self.game
        me = g.tank_fields[0]
        en = g.tank_fields[1]
        dm = g.dist_map(me["x"], me["y"])
        if dm is None:
            return 0.0
        ex, ey = en["x"], en["y"]
        if 0 <= ex < len(dm) and 0 <= ey < len(dm[ex]):
            v = dm[ex][ey]
            if v is not None and v == v:
                return -K_SHAPE * float(v)
        return 0.0

    # ------------------------------------------------ 射线 (numpy 向量化)

    def _build_wall_boxes(self):
        """墙线段 -> AABB (含厚度与方头端帽), 供 slab 法求交"""
        g = self.game
        t = float(g.wall_half_t)
        boxes = []
        for (x1, y1, x2, y2) in g.walls:
            xmin, xmax = min(x1, x2) - t, max(x1, x2) + t
            ymin, ymax = min(y1, y2) - t, max(y1, y2) + t
            boxes.append((xmin, ymin, xmax, ymax))
        self._wall_boxes = np.asarray(boxes, dtype=np.float64)
        w = len(g.maze) * g.scale
        h = len(g.maze[0]) * g.scale
        self._max_ray = math.hypot(w, h)

    def _raycast(self, ox, oy, angles):
        """从 (ox,oy) 沿各角度求最近墙距. 返回 shape=(len(angles),)"""
        b = self._wall_boxes
        dx = np.cos(angles)[:, None]          # (R,1)
        dy = np.sin(angles)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dx = np.where(dx != 0, 1.0 / dx, np.inf)
            inv_dy = np.where(dy != 0, 1.0 / dy, np.inf)
            tx1 = (b[:, 0] - ox) * inv_dx     # (R,M)
            tx2 = (b[:, 2] - ox) * inv_dx
            ty1 = (b[:, 1] - oy) * inv_dy
            ty2 = (b[:, 3] - oy) * inv_dy
        # dx==0 时: 若 ox 在盒 x 范围内, slab 为 (-inf, inf), 否则无交
        x_in = (b[:, 0] <= ox) & (ox <= b[:, 2])
        y_in = (b[:, 1] <= oy) & (oy <= b[:, 3])
        txmin = np.minimum(tx1, tx2)
        txmax = np.maximum(tx1, tx2)
        tymin = np.minimum(ty1, ty2)
        tymax = np.maximum(ty1, ty2)
        txmin = np.where(np.isnan(txmin), np.where(x_in, -np.inf, np.inf), txmin)
        txmax = np.where(np.isnan(txmax), np.where(x_in, np.inf, -np.inf), txmax)
        tymin = np.where(np.isnan(tymin), np.where(y_in, -np.inf, np.inf), tymin)
        tymax = np.where(np.isnan(tymax), np.where(y_in, np.inf, -np.inf), tymax)
        tmin = np.maximum(txmin, tymin)
        tmax = np.minimum(txmax, tymax)
        hit = tmax >= np.maximum(tmin, 0.0)
        dist = np.where(hit, np.maximum(tmin, 0.0), np.inf)   # (R,M)
        d = dist.min(axis=1)
        return np.minimum(d, self._max_ray)

    def _line_of_sight(self, x0, y0, x1, y1):
        """两点间是否无墙 (解析线段 vs AABB)"""
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-9:
            return True
        ang = np.asarray([math.atan2(y1 - y0, x1 - x0)])
        return float(self._raycast(x0, y0, ang)[0]) >= d

    # ------------------------------------------------ 观测

    def _obs(self):
        g = self.game
        me = g.tanks[0]
        en = g.tanks[1]
        scale = g.scale
        w = len(g.maze) * scale
        h = len(g.maze[0]) * scale
        rot = me.rotation * math.pi / 180.0
        # 自车"前方"为 rotation-90° 方向
        fwd = rot - math.pi / 2
        cos_f, sin_f = math.cos(fwd), math.sin(fwd)

        def to_local(dx, dy):
            """世界向量 -> 自车坐标系 (x=前, y=左手侧)"""
            lx = dx * cos_f + dy * sin_f
            ly = -dx * sin_f + dy * cos_f
            return lx, ly

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        i = 0
        # ---- 自车 6 ----
        obs[i:i + 6] = [
            me.x / w, me.y / h,
            math.cos(rot), math.sin(rot),
            (g.settings_max_bullets - me.bullets_fired)
            / g.settings_max_bullets,
            scale / 100.0,
        ]
        i += 6
        # ---- 敌车 8 ----
        dx, dy = en.x - me.x, en.y - me.y
        lx, ly = to_local(dx, dy)
        dist = math.hypot(dx, dy)
        en_rot = en.rotation * math.pi / 180.0
        # 迷宫路径距离
        pd = -self._prev_phi / K_SHAPE if K_SHAPE else 0.0
        los = self._line_of_sight(me.x, me.y, en.x, en.y)
        obs[i:i + 8] = [
            np.clip(lx / scale, -8, 8), np.clip(ly / scale, -8, 8),
            min(dist / scale, 16.0) / 4.0,
            min(pd, 20.0) / 10.0,
            math.cos(en_rot - rot), math.sin(en_rot - rot),
            (g.settings_max_bullets - en.bullets_fired)
            / g.settings_max_bullets,
            1.0 if los else 0.0,
        ]
        i += 8
        # ---- 射线 N_RAYS (相对自车前方) ----
        angles = fwd + self._ray_angles
        rays = self._raycast(me.x, me.y, angles)
        obs[i:i + N_RAYS] = np.minimum(rays / (4.0 * scale), 1.0)
        i += N_RAYS
        # ---- 子弹 N_BULLET_SLOTS x 6 ----
        bullets = sorted(
            g.bullets,
            key=lambda b: (b.x - me.x) ** 2 + (b.y - me.y) ** 2,
        )[:N_BULLET_SLOTS]
        spf = C.BULLETSPEED * (scale / 50.0)   # 每帧弹速
        for b in bullets:
            bx, by = to_local(b.x - me.x, b.y - me.y)
            vx, vy = to_local(b.x_speed * C.BULLETHITCHECKINTERVALS,
                              b.y_speed * C.BULLETHITCHECKINTERVALS)
            obs[i:i + 6] = [
                np.clip(bx / scale, -8, 8), np.clip(by / scale, -8, 8),
                vx / spf if spf else 0.0, vy / spf if spf else 0.0,
                1.0 if b.owner is me else -1.0,
                b.lifetime / C.BULLETLIFETIME,
            ]
            i += 6
        i = 6 + 8 + N_RAYS + N_BULLET_SLOTS * 6
        # ---- 其他 2 ----
        obs[i] = self._frames / TRUNCATE_FRAMES
        obs[i + 1] = 1.0 if me.trigger_released else 0.0
        return obs


def make_env(rank: int, base_seed: int = 0, reward_version: int = 2):
    """SubprocVecEnv 工厂 (需模块级可 pickle)"""
    def _init():
        return TankTroubleGym(seed=base_seed + rank,
                              reward_version=reward_version)
    return _init
