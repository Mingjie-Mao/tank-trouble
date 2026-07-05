"""
AI 训练数据接口

设计原则:
  - 引擎 (game.py) 完全无依赖、可无头高速运行
  - get_state() 返回完整的结构化真值 (坦克/子弹/迷宫/回合状态)
  - step(action) 按帧推进, 动作为 5 个布尔量 (与原版键盘输入等价)
  - 事件流 (fire/bounce/hit/destroy/round_end/new_round) 供奖励设计
  - 可选的 gymnasium 包装 (见文件末尾, 未安装 gymnasium 也不影响核心)

用法:
    env = TankTroubleEnv(seed=42)
    state = env.reset()
    state, events = env.step({"forward": True, "fire": True})
"""

from .game import Game
from . import constants as C

# 离散动作编码 (可选便捷映射): [throttle(0/1/2), turn(0/1/2), fire(0/1)]
#   throttle: 0=倒车, 1=停, 2=前进;  turn: 0=左转, 1=不转, 2=右转
def discrete_to_input(throttle, turn, fire):
    return {
        "forward": throttle == 2,
        "backup": throttle == 0,
        "turn_left": turn == 0,
        "turn_right": turn == 2,
        "fire": bool(fire),
    }


class TankTroubleEnv:
    """vs Laika 环境: tank0 = 受控方(玩家/训练智能体), tank1 = Laika。

    帧率恒为 25FPS 的离散帧; step() 一次 = 一帧。
    """

    def __init__(self, seed=None, ai_enabled=True):
        self._seed = seed
        self.ai_enabled = ai_enabled
        self.game = None
        self.reset(seed)

    # ------------------------------------------------ 控制流

    def reset(self, seed=None):
        if seed is not None:
            self._seed = seed
        self.game = Game(seed=self._seed, ai_enabled=self.ai_enabled)
        return self.get_state()

    def step(self, action=None, actions=None):
        """推进一帧。

        action:  tank0 的输入 dict (键: forward/backup/turn_left/turn_right/fire)
        actions: {tank_index: input_dict} 多坦克控制 (被 AI 控制的坦克会被
                 AI 在帧内覆盖, 与原版一致)
        返回 (state, events)
        """
        g = self.game
        all_inputs = {}
        if action is not None:
            all_inputs[0] = action
        if actions:
            all_inputs.update(actions)
        for idx, inp in all_inputs.items():
            t = g.tanks[idx]
            t.forward = bool(inp.get("forward", False))
            t.backup = bool(inp.get("backup", False))
            t.turn_left = bool(inp.get("turn_left", False))
            t.turn_right = bool(inp.get("turn_right", False))
            t.fire = bool(inp.get("fire", False))
        events = g.step()
        return self.get_state(), events

    def run_frames(self, n, action=None):
        """连续跑 n 帧 (保持同一输入), 返回 (state, 所有事件)"""
        all_events = []
        state = None
        for _ in range(n):
            state, ev = self.step(action)
            all_events.extend(ev)
        return state, all_events

    # ------------------------------------------------ 状态导出

    def get_state(self):
        """完整真值状态 (每帧均可安全调用, 纯数据无引用)"""
        g = self.game
        return {
            "frame": g.frame,
            "round": g.round_number,
            "fps": C.FPS,
            "scale": g.scale,
            "maze_width": len(g.maze),
            "maze_height": len(g.maze[0]),
            # maze[x][y] = [ground, 下墙, 左墙]
            "maze": [[list(cell) for cell in col] for col in g.maze],
            "walls": list(g.walls),               # 轴对齐线段 (x1,y1,x2,y2)
            "wall_half_thickness": g.wall_half_t,  # 碰撞半厚
            "world_width": len(g.maze) * g.scale,
            "world_height": len(g.maze[0]) * g.scale,
            "frozen": g.frozen,
            "end_count": g.end_count,
            "reset_count": g.reset_count,
            "alive_count": g.alive_count,
            "scores": list(g.scores),
            "settings_max_bullets": g.settings_max_bullets,
            "tanks": [
                {
                    "index": t.number,
                    "x": t.x,
                    "y": t.y,
                    "rotation": t.rotation,       # 度, (-180, 180]
                    "alive": t.alive,
                    "bullets_fired": t.bullets_fired,
                    "cell": dict(g.tank_fields[t.number]),
                    "is_ai": t.ai is not None,
                    "hit_something": t.hit_something,
                    "input": {
                        "forward": t.forward, "backup": t.backup,
                        "turn_left": t.turn_left, "turn_right": t.turn_right,
                        "fire": t.fire,
                    },
                }
                for t in g.tanks
            ],
            "bullets": [
                {
                    "name": b.name,
                    "x": b.x,
                    "y": b.y,
                    # 注意: x_speed 为"每子步"速度, 每帧移动 = x_speed * 7
                    "x_speed": b.x_speed,
                    "y_speed": b.y_speed,
                    "speed_per_frame": (b.x_speed * C.BULLETHITCHECKINTERVALS,
                                        b.y_speed * C.BULLETHITCHECKINTERVALS),
                    "lifetime": b.lifetime,
                    "owner": b.owner.number,
                }
                for b in g.bullets
            ],
            # Laika 内部状态 (调试/模仿学习用)
            "laika_goal": (dict(
                (k, v) for k, v in g.tanks[1].ai.my_goal.items()
                if k in ("goal", "priority", "period", "id"))
                if g.tanks[1].ai is not None else None),
        }

    # ------------------------------------------------ 距离图 (奖励设计用)

    def cell_distance(self, from_cell, to_cell):
        """两格间的迷宫距离 (原版 distancesForMaze); 不可达返回 None"""
        dm = self.game.dist_map(from_cell[0], from_cell[1])
        if dm is None:
            return None
        x, y = to_cell
        if 0 <= x < len(dm) and 0 <= y < len(dm[x]):
            v = dm[x][y]
            if v is None or v != v:  # None/NaN
                return None
            return v
        return None


# ================================================================
# 可选: gymnasium 包装 (pip install gymnasium 后可用)
# ================================================================

try:
    import gymnasium as _gym
    import numpy as _np

    class TankTroubleGymEnv(_gym.Env):
        """回合制 gym 环境: 一局 = 从新迷宫到有一方死亡(计分瞬间)。

        动作: MultiDiscrete([3,3,2]) = [油门, 转向, 开火]
        观测: 结构化向量 (自身/敌方/最近子弹), 也可直接改用 get_state()
        奖励: 击杀 +1, 被杀 -1 (含自杀), 其余 0 — 需要塑形请用 events
        """

        metadata = {"render_modes": []}
        MAX_BULLET_SLOTS = 10

        def __init__(self, seed=None):
            super().__init__()
            self.core = TankTroubleEnv(seed=seed)
            self.action_space = _gym.spaces.MultiDiscrete([3, 3, 2])
            n = 8 + 6 + self.MAX_BULLET_SLOTS * 5
            self.observation_space = _gym.spaces.Box(
                low=-2.0, high=2.0, shape=(n,), dtype=_np.float32)

        def _obs(self, state):
            import math as _m
            w = state["world_width"]
            h = state["world_height"]
            me = state["tanks"][0]
            en = state["tanks"][1]
            rot0 = me["rotation"] * _m.pi / 180
            rot1 = en["rotation"] * _m.pi / 180
            vec = [
                me["x"] / w, me["y"] / h,
                _m.cos(rot0), _m.sin(rot0),
                me["bullets_fired"] / state["settings_max_bullets"],
                1.0 if me["alive"] else 0.0,
                state["scale"] / 100.0,
                1.0 if state["frozen"] else 0.0,
                en["x"] / w, en["y"] / h,
                _m.cos(rot1), _m.sin(rot1),
                en["bullets_fired"] / state["settings_max_bullets"],
                1.0 if en["alive"] else 0.0,
            ]
            bullets = sorted(
                state["bullets"],
                key=lambda b: (b["x"] - me["x"]) ** 2 + (b["y"] - me["y"]) ** 2
            )[: self.MAX_BULLET_SLOTS]
            for b in bullets:
                spf = b["speed_per_frame"]
                vec.extend([b["x"] / w, b["y"] / h,
                            spf[0] / 10.0, spf[1] / 10.0,
                            b["lifetime"] / 250.0])
            while len(vec) < 14 + self.MAX_BULLET_SLOTS * 5:
                vec.append(0.0)
            return _np.asarray(vec, dtype=_np.float32)

        def reset(self, *, seed=None, options=None):
            state = self.core.reset(seed)
            return self._obs(state), {"state": state}

        def step(self, action):
            inp = discrete_to_input(int(action[0]), int(action[1]),
                                    int(action[2]))
            reward = 0.0
            terminated = False
            state, events = self.core.step(inp)
            for ev in events:
                if ev[0] == "destroy":
                    if ev[1] == 0:
                        reward -= 1.0
                    else:
                        reward += 1.0
                    terminated = True
            return (self._obs(state), reward, terminated, False,
                    {"state": state, "events": events})

except ImportError:
    TankTroubleGymEnv = None
