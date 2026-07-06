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

# ---- 弹道预演观测 (obs_traj=True 时附加; 治"看不见反弹几何") ----
# 每颗子弹 +4: 未来轨迹距我最近逼近/时刻/会否命中我/命中前是否反弹;
# 射击扇 7 角 x 3: 该角度开火的模拟弹道能否命中 Laika/多快/几次反弹。
TRAJ_BULLET_FEATS = 4
SHOT_FAN_DEG = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
TRAJ_DIM = N_BULLET_SLOTS * TRAJ_BULLET_FEATS + len(SHOT_FAN_DEG) * 3
BULLET_SIM_FRAMES = 50    # 子弹前瞻 2 秒
SHOT_SIM_FRAMES = 75      # 假想炮弹前瞻 3 秒
HIT_RADIUS_SCALE = 0.25   # 命中判定半径 (格), 近似坦克有效尺寸


def obs_dim(obs_traj: bool) -> int:
    return OBS_DIM + (TRAJ_DIM if obs_traj else 0)


def _reflective_closest_batch(origins, dirs, speeds, horizons,
                              max_bounces, aabbs, tx, ty):
    """M 条弹道的批量"反射折线 + 目标最近逼近"。

    沿方向匀速直线飞行, 撞墙按轴反射(角点双反), 按反弹深度分轮做矩阵运算。
    与引擎逐子步模拟相比是几何近似(端点精确), 仅供观测特征使用。
    返回 (M,3): [dmin, t_at_min(帧), bounces_before]。
    """
    m = origins.shape[0]
    px = origins[:, 0].astype(np.float64).copy()
    py = origins[:, 1].astype(np.float64).copy()
    dx = dirs[:, 0].astype(np.float64).copy()
    dy = dirs[:, 1].astype(np.float64).copy()
    t_used = np.zeros(m)
    active = np.ones(m, dtype=bool)
    best = np.full((m, 3), np.inf)
    best[:, 1] = 0.0
    best[:, 2] = 0.0

    for bounce in range(max_bounces + 1):
        if not active.any():
            break
        if aabbs.shape[0] == 0:
            # 无墙: 全部直飞到时限
            t_wall = np.full(m, np.inf)
            jx = jy = np.zeros(m)
        else:
            sdx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)[:, None]
            sdy = np.where(np.abs(dy) < 1e-12, 1e-12, dy)[:, None]
            t1 = (aabbs[None, :, 0] - px[:, None]) / sdx
            t2 = (aabbs[None, :, 2] - px[:, None]) / sdx
            t3 = (aabbs[None, :, 1] - py[:, None]) / sdy
            t4 = (aabbs[None, :, 3] - py[:, None]) / sdy
            tx_lo = np.minimum(t1, t2)
            ty_lo = np.minimum(t3, t4)
            tnear = np.maximum(tx_lo, ty_lo)
            tfar = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
            valid = (tnear <= tfar) & (tfar >= 0.0) & (tnear > 1e-9)
            tmat = np.where(valid, tnear, np.inf)
            j = tmat.argmin(axis=1)
            t_wall = tmat[np.arange(m), j]
            jx = tx_lo[np.arange(m), j]
            jy = ty_lo[np.arange(m), j]

        dist_left = (horizons - t_used) * speeds
        seg_len = np.minimum(t_wall, dist_left)
        seg_len = np.where(active, seg_len, 0.0)

        # 目标点到本轮各线段的最近逼近
        ex, ey = px + dx * seg_len, py + dy * seg_len
        sx, sy = ex - px, ey - py
        ll = sx * sx + sy * sy
        u = np.where(ll < 1e-12, 0.0,
                     ((tx - px) * sx + (ty - py) * sy)
                     / np.where(ll < 1e-12, 1.0, ll))
        u = np.clip(u, 0.0, 1.0)
        d = np.hypot(tx - (px + u * sx), ty - (py + u * sy))
        # 距离打平时保留更早的逼近 (首过命中才有意义), 防止反弹回程以
        # 浮点级差距抢走直射记录
        upd = active & (d < best[:, 0] - 1e-9)
        seg_frames = np.where(speeds > 1e-12, seg_len / speeds, 0.0)
        best[upd, 0] = d[upd]
        best[upd, 1] = (t_used + u * seg_frames)[upd]
        best[upd, 2] = bounce

        # 到达时限的弹道结束; 撞墙的反射进入下一轮
        hit_wall = active & (t_wall < dist_left)
        active = hit_wall
        if not active.any():
            break
        t_used = np.where(hit_wall,
                          t_used + t_wall / np.maximum(speeds, 1e-12), t_used)
        px = np.where(hit_wall, ex, px)
        py = np.where(hit_wall, ey, py)
        # 反射轴: 命中竖直面反x, 水平面反y, 角点双反
        corner = np.abs(jx - jy) < 1e-9
        flip_x = hit_wall & ((jx > jy) | corner)
        flip_y = hit_wall & ((jy > jx) | corner)
        dx = np.where(flip_x, -dx, dx)
        dy = np.where(flip_y, -dy, dy)
        # 沿新方向脱离命中面, 防止原地重复命中
        px = np.where(hit_wall, px + dx * 0.5, px)
        py = np.where(hit_wall, py + dy * 0.5, py)
    return best

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

# ---- v4 奖励参数 (真规则目标: 修复 destroy 口径把 v2 训成神风换子流) ----
# 阶段0实测: v2 先杀率 51.2% 但原版计分真胜率仅 22.1%, 双亡率 40.8%。
# v4 = v2 的开火塑形 + 终局改在原版计分点(endCount==50), 先杀后死不再给 +1:
#   击杀/死亡当帧给 ±R_EVENT 即时信用, 终局按真结果结算——
#   干净赢 = +0.5+1.0, 主动换子 = +0.5-0.5-0.2, 纯输 = -0.5-1.0。
R_EVENT = 0.5             # destroy 事件即时奖励幅度 (击杀 + / 被杀 -)
R_DOUBLE_DEATH = -0.2     # 双亡终局奖励 (换子略亏, 但远好于纯输)

# ---- v5 奖励参数 (v4 + 密集闪避压力; README 思路 B 的正确修法) ----
# v3 失败教训: 稀疏 +0.15 闪避奖励被 PPO 忽略。v5 改为逐步连续压力:
# 站在"预演判定会命中我"的弹道上, 每步按命中紧迫度扣分——信号密集,
# 且 obs_traj 已把同一几何摆在观测里, 策略看得见怎么脱离。
# 上限校核: 满紧迫连续锁定 50 帧(25 决策步) ~ -0.5, 不盖过终局 ±1。
R_THREAT_PRESSURE = -0.02  # 每决策步 x 命中紧迫度(0..1)

# ---- P13 空枪纪律 (录像观察: 泼弹不瞄准, 命中率 12% 横盘) ----
# 默认值即旧行为; 探针用 --waste-shot / --near-miss 收紧。
NEAR_MISS_BONUS = R_GOOD_SHOT * 0.25   # 旧: 擦身弹 +0.075

# ---- P14 闪避特训营 (dodge_drill=True) ----
# 禁用开火, 只能靠走位在 Laika 火力下活命; 先特训身法再回全规则微调。
# 奖励: 死 -1 / 活满 +0.5 / 密集威胁压力 x2 (需 obs_traj)。
DRILL_TRUNCATE_FRAMES = 1500          # 特训单局 60 秒
DRILL_SURVIVE_BONUS = 0.5             # 活满一局
DRILL_PRESSURE_MULT = 2.0             # 威胁压力加倍 (身法是唯一课题)



class TankTroubleGym(gym.Env):
    """vs Laika 训练环境。tank0 = 智能体, tank1 = Laika。"""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, reward_version: int = 2,
                 terminal_mode: str = "destroy", obs_traj: bool = False,
                 min_spawn_cells: int = 0,
                 dd_reward: float = R_DOUBLE_DEATH,
                 frame_skip: int = FRAME_SKIP,
                 bad_shot: float = R_BAD_SHOT,
                 time_escalate: bool = False,
                 waste_shot: float = R_WASTE_SHOT,
                 near_miss: float = NEAR_MISS_BONUS,
                 dodge_drill: bool = False):
        """terminal_mode:
        - "destroy": 首个 destroy 事件立即终局 (训练默认, 先杀即判胜)
        - "score":   跑到原版计分点 endCount==50 (round_end 事件), 先杀后死
                     = 双亡不得分 —— 与原版 Flash 规则一致, 用于真胜率评估
        obs_traj: 附加弹道预演特征 (观测 76 -> 121 维)
        min_spawn_cells: >0 时重掷出生路径距离小于该格数的局(训练去偏用,
                         逼策略学跑图; 评估协议禁用以保持无偏)
        """
        super().__init__()
        if terminal_mode not in ("destroy", "score"):
            raise ValueError(f"未知 terminal_mode: {terminal_mode}")
        if reward_version >= 4:
            terminal_mode = "score"   # v4 的目标定义在原版计分点上
        self._base_seed = seed
        self._episode = 0
        self._rv = reward_version    # 1 纯胜负; 2 开火塑形; 3 闪避实验; 4 真规则
        if reward_version >= 5 and not obs_traj:
            raise ValueError("v5 闪避压力依赖弹道预演观测 (obs_traj=True)")
        self._terminal_mode = terminal_mode
        self._obs_traj = obs_traj
        self._min_spawn_cells = min_spawn_cells
        self._dd_reward = dd_reward
        self.frame_skip = frame_skip
        self._bad_shot = bad_shot
        self._time_escalate = time_escalate
        self._waste_shot = waste_shot
        self._near_miss = near_miss
        self._dodge_drill = dodge_drill
        if dodge_drill and not obs_traj:
            raise ValueError("闪避特训依赖弹道预演观测 (obs_traj=True)")
        self.game = None
        self._sim = None             # Laika 弹道模拟器 (绑定到 tank0, 每局重建)
        self._wall_boxes = None      # (M,4) AABB 数组
        self._ray_angles = np.linspace(0, 2 * math.pi, N_RAYS,
                                       endpoint=False)
        self._max_ray = 1.0
        self._prev_phi = 0.0

        self.action_space = spaces.MultiDiscrete([3, 3, 2])
        self.observation_space = spaces.Box(
            low=-4.0, high=4.0, shape=(obs_dim(obs_traj),), dtype=np.float32)

    # ------------------------------------------------ 核心流程

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._base_seed = seed
        for _ in range(20):   # 去偏最多重掷 20 次, 掷不出就用最后一局
            ep_seed = self._base_seed * 1_000_003 + self._episode
            self._episode += 1
            self.game = Game(seed=ep_seed, ai_enabled=True)
            if (self._min_spawn_cells <= 0
                    or self._spawn_path_cells() >= self._min_spawn_cells):
                break
        self._build_wall_boxes()
        self._frames = 0
        self._first_destroy = None   # 首个被摧毁的坦克号 (0/1/"both")
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
        if self._dodge_drill:
            t0.fire = False        # 特训营缴械: 唯一课题是活下来

        reward = 0.0
        terminated = False
        info = {}
        all_events = []
        # v3: 决策步开始时, 我是否已被来袭弹锁定? (仅 v3 实验用, v4 不含)
        danger_before = (self._rv == 3
                         and self._incoming_will_hit(t0))
        for _ in range(self.frame_skip):
            # v2: 开火前先判"若此帧真能发弹, 这一发好不好"
            shot_reward = 0.0
            if (self._rv >= 2 and not self._dodge_drill
                    and t0.fire and g.weapon_ready(t0)):
                shot_reward = self._shot_quality(t0)
            bf_before = t0.bullets_fired

            events = g.step()
            self._frames += 1
            all_events.extend(events)

            # 只有真的打出弹(弹数增加)才结算开火奖励
            if self._rv >= 2 and t0.bullets_fired > bf_before:
                reward += shot_reward

            destroyed_now = [ev[1] for ev in events if ev[0] == "destroy"]
            # 先杀归属只看首个 destroy 帧; 同帧双杀记 "both", 之后不再改
            if destroyed_now and self._first_destroy is None:
                self._first_destroy = (destroyed_now[0]
                                       if len(set(destroyed_now)) == 1
                                       else "both")
            event_mag = R_EVENT if self._rv >= 4 else 1.0
            for victim in destroyed_now:
                if victim == 0:
                    reward -= event_mag
                elif not self._dodge_drill:
                    reward += event_mag   # 特训营: Laika 自杀不算我方功劳
                if self._terminal_mode == "destroy":
                    info["result"] = "loss" if victim == 0 else "win"
                    terminated = True

            if self._terminal_mode == "score":
                for ev in events:
                    if ev[0] == "round_end":
                        winner = ev[1]
                        result = ("win" if winner == 0 else
                                  "loss" if winner == 1 else "double_death")
                        info["result"] = result
                        terminated = True
                        if self._rv >= 4:
                            reward += {"win": 1.0, "loss": -1.0,
                                       "double_death": self._dd_reward}[result]
            if terminated:
                break
        info["events"] = all_events
        info["first_destroy"] = self._first_destroy

        # v3: 闪避结算 — 决策步前被锁定, 步后存活且不再被锁 = 成功化解
        if self._rv == 3 and not terminated:
            danger_after = self._incoming_will_hit(t0)
            if danger_before and not danger_after:
                reward += R_DODGE
            elif danger_after:
                reward += R_IN_DANGER

        truncated = False
        # score 模式下已有人死亡时不截断 (计分点在 destroy 后 75 帧内必到)
        trunc_at = (DRILL_TRUNCATE_FRAMES if self._dodge_drill
                    else TRUNCATE_FRAMES)
        if (not terminated and self._frames >= trunc_at
                and (self._terminal_mode == "destroy"
                     or self._first_destroy is None)):
            truncated = True
            info["result"] = "draw"   # 特训营: draw = 活满全场 (即特训成功)
            if self._dodge_drill:
                reward += DRILL_SURVIVE_BONUS
            elif self._rv >= 2:
                reward += R_DRAW      # 苟平不再是安全港

        # 势能塑形 + 时间惩罚 (只在回合进行中; 特训营不逼近不催时)
        if not terminated and not self._dodge_drill:
            phi = self._phi()
            reward += GAMMA * phi - self._prev_phi
            self._prev_phi = phi
            if self._rv >= 2:
                # 递增模式: 前期宽松后期加倍, 压"拖长局"(败局均长 141 步 vs 胜局 109)
                if self._time_escalate:
                    reward += R_TIME_PENALTY * 2.0 * (self._frames / TRUNCATE_FRAMES + 0.25)
                else:
                    reward += R_TIME_PENALTY

        obs = self._obs()
        # v5: 闪避压力 — 站在预演命中弹道上, 按紧迫度逐步扣分
        # 特训营: 无论奖励版本都启用, 且加倍 (身法是唯一课题)
        if (self._rv >= 5 or self._dodge_drill) and not terminated:
            block = obs[OBS_DIM:OBS_DIM + N_BULLET_SLOTS * TRAJ_BULLET_FEATS]
            block = block.reshape(N_BULLET_SLOTS, TRAJ_BULLET_FEATS)
            threat = float((block[:, 2] * (1.0 - block[:, 1])).max())
            mult = DRILL_PRESSURE_MULT if self._dodge_drill else 1.0
            reward += R_THREAT_PRESSURE * mult * threat
        return obs, reward, terminated, truncated, info

    def _spawn_path_cells(self):
        """双方出生格的迷宫路径距离(格数); 不可达按无穷远处理。"""
        g = self.game
        me = g.tank_fields[0]
        dm = g.dist_map(me["x"], me["y"])
        if dm is None:
            return 10 ** 9
        ex, ey = g.tank_fields[1]["x"], g.tank_fields[1]["y"]
        if 0 <= ex < len(dm) and 0 <= ey < len(dm[ex]):
            v = dm[ex][ey]
            if v is not None and v == v:
                return float(v)
        return 10 ** 9

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
            return self._bad_shot
        # NOTHING: check_bullet_path 的 closest 是像素曼哈顿距离(有威胁时)
        closest = res.get("closest", 1e9)
        if closest <= NEAR_MISS_CELL * self.game.scale:
            return self._near_miss       # 擦身施压 (P13 起可收紧)
        return self._waste_shot

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

        obs = np.zeros(obs_dim(self._obs_traj), dtype=np.float32)
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
        if not self._obs_traj:
            return obs
        i += 2

        # ================ 弹道预演特征 ================
        hit_r = HIT_RADIUS_SCALE * scale
        boxes = self._wall_boxes

        # 每颗子弹的未来轨迹(2 秒, 最多 3 反弹), 槽序与上方子弹块一致;
        # 空槽默认"无威胁": dmin=远, t=远
        for k in range(N_BULLET_SLOTS):
            obs[i + k * TRAJ_BULLET_FEATS + 0] = 1.0
            obs[i + k * TRAJ_BULLET_FEATS + 1] = 1.0
        if bullets:
            origins, dirs, speeds, horizons = [], [], [], []
            for b in bullets:
                sub = math.hypot(b.x_speed, b.y_speed)
                if sub < 1e-9:
                    sub = 1e-9
                origins.append((b.x, b.y))
                dirs.append((b.x_speed / sub, b.y_speed / sub))
                speeds.append(sub * C.BULLETHITCHECKINTERVALS)
                horizons.append(min(float(BULLET_SIM_FRAMES),
                                    float(max(b.lifetime, 0))))
            res = _reflective_closest_batch(
                np.asarray(origins), np.asarray(dirs), np.asarray(speeds),
                np.asarray(horizons), 3, boxes, me.x, me.y)
            for k in range(len(bullets)):
                dmin, t_min, bounces = res[k]
                base = i + k * TRAJ_BULLET_FEATS
                obs[base + 0] = min(dmin / (2.0 * scale), 1.0)
                obs[base + 1] = min(t_min / BULLET_SIM_FRAMES, 1.0)
                will_hit = dmin <= hit_r
                obs[base + 2] = 1.0 if will_hit else 0.0
                obs[base + 3] = 1.0 if (will_hit and bounces > 0) else 0.0
        i += N_BULLET_SLOTS * TRAJ_BULLET_FEATS

        # 射击扇: 当前朝向 ±30° 每 10° 一条假想弹道(3 秒, 最多 2 反弹)
        if en.alive:
            shot_speed = C.BULLETSPEED * (scale / 50.0)
            spawn_d = scale * 4.5 / 16.0
            angs = fwd + np.asarray(SHOT_FAN_DEG) * math.pi / 180.0
            cds, sds = np.cos(angs), np.sin(angs)
            origins = np.stack([me.x + cds * spawn_d,
                                me.y + sds * spawn_d], axis=1)
            dirs = np.stack([cds, sds], axis=1)
            m = len(SHOT_FAN_DEG)
            res = _reflective_closest_batch(
                origins, dirs, np.full(m, shot_speed),
                np.full(m, float(SHOT_SIM_FRAMES)), 2, boxes, en.x, en.y)
            for k in range(m):
                dmin, t_min, bounces = res[k]
                base = i + k * 3
                if dmin <= hit_r:
                    obs[base + 0] = 1.0
                    obs[base + 1] = 1.0 - min(t_min / SHOT_SIM_FRAMES, 1.0)
                    obs[base + 2] = 1.0 - bounces / 2.0   # 直射=1
        return obs


def make_env(rank: int, base_seed: int = 0, reward_version: int = 2,
             obs_traj: bool = False, min_spawn_cells: int = 0,
             dd_reward: float = R_DOUBLE_DEATH, frame_skip: int = FRAME_SKIP,
             bad_shot: float = R_BAD_SHOT, time_escalate: bool = False,
             waste_shot: float = R_WASTE_SHOT, near_miss: float = NEAR_MISS_BONUS,
             dodge_drill: bool = False):
    """SubprocVecEnv 工厂 (需模块级可 pickle)"""
    def _init():
        return TankTroubleGym(seed=base_seed + rank,
                              reward_version=reward_version,
                              obs_traj=obs_traj,
                              min_spawn_cells=min_spawn_cells,
                              dd_reward=dd_reward,
                              frame_skip=frame_skip,
                              bad_shot=bad_shot,
                              time_escalate=time_escalate,
                              waste_shot=waste_shot,
                              near_miss=near_miss,
                              dodge_drill=dodge_drill)
    return _init
