"""
基线策略 — 用于确立评估基准线

三个基线 (由弱到强):
  IdlePolicy    静止不动 (下限: 看 Laika 多快杀死靶子)
  RandomPolicy  随机输入
  HunterPolicy  手写猎杀脚本 — 利用 Laika 的已知弱点:
                  1. 弹尽必逃 (打完 5 发就手无寸铁)
                  2. 瞄准时站定 (turnTo 期间是静止靶)
                  3. 近距离反应延迟 (决策周期 10 帧)
                战术: 有视线且对准 -> 开火; 否则沿最短路接近;
                      有子弹逼近 -> 垂直闪避。

基线策略直接读引擎内部状态 (评估对照用, 不走观测向量)。
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import Game, constants as C  # noqa: E402
from tank_trouble_original.maze import (  # noqa: E402
    get_shortest_path_with_distances,
)

DEG = math.pi / 180.0


def _norm180(a):
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a


class IdlePolicy:
    name = "idle"

    def reset(self):
        pass

    def act(self, game):
        return {}


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        import random
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def act(self, game):
        r = self.rng
        return {
            "forward": r.random() < 0.6,
            "backup": r.random() < 0.1,
            "turn_left": r.random() < 0.25,
            "turn_right": r.random() < 0.25,
            "fire": r.random() < 0.08,
        }


class ExploitBot:
    """漏洞验证脚本 — 量化"静止窗口"漏洞的利用上限 (白盒读引擎, 上界估计)。

    2000 局挖掘结论: Laika 的稳定弱点不是弹药/位置, 而是静止状态 ——
    卡墙 lift 2.87, 刚开火 1s 内 lift 1.57, idle/shootAfter 死前画像 1.2-1.4。

    战术: 继承 Hunter (躲弹/最短路接敌/弹道模拟开火), 在 Laika 处于
    静止窗口时切换强攻 — 放宽开火角度阈值抢时机, 全速逼近。
    """
    name = "exploit"

    FIRE_WINDOW = 25     # Laika 开火后的僵直窗口 (帧)
    STILL_GOALS = ("idle", "shootAfter", "fireWeapon", "turnTo")

    def __init__(self):
        self._hunter = HunterPolicy()
        self._laika_last_fire = None
        self._prev_laika_fired = 0

    def reset(self):
        self._hunter.reset()
        self._laika_last_fire = None
        self._prev_laika_fired = 0

    def _laika_stationary(self, game):
        """Laika 是否处于静止窗口: 卡墙 / 刚开火 / 瞄准类目标"""
        en = game.tanks[1]
        if en.hit_something:
            return True
        # 用弹数增量近似"它刚开火"(白盒也可读事件, 这里保持 act(game) 接口)
        if en.bullets_fired > self._prev_laika_fired:
            self._laika_last_fire = game.frame
        self._prev_laika_fired = en.bullets_fired
        if (self._laika_last_fire is not None
                and game.frame - self._laika_last_fire <= self.FIRE_WINDOW):
            return True
        if en.ai is not None:
            goal = en.ai.my_goal.get("goal", "")
            if goal in self.STILL_GOALS:
                return True
            # 动作栈顶是瞄准/开火类动作 = 正在站定
            if en.ai.my_actions:
                act = en.ai.my_actions[-1].get("action", "")
                if act in ("turnTo", "fireWeapon"):
                    return True
        return False

    def act(self, game):
        me = game.tanks[0]
        en = game.tanks[1]
        if not me.alive or not en.alive:
            return {}
        stationary = self._laika_stationary(game)
        out = self._hunter.act(game)

        if stationary and not out.get("fire"):
            # 静止窗口强攻: 放宽 Hunter 的角度阈值, 只要模拟命中就开火
            dx, dy = en.x - me.x, en.y - me.y
            dist = math.hypot(dx, dy)
            has_los = HunterPolicy._los(game, me.x, me.y, en.x, en.y)
            if (has_los and abs(_norm180(
                    math.degrees(math.atan2(dy, dx)) + 90 - me.rotation)) < 25
                    and me.bullets_fired < game.settings_max_bullets):
                res = self._hunter._bullet_sim(game).check_bullet_path(me.rotation)
                if res["result"] == "HIT":
                    out["fire"] = True
        return out


class HunterPolicy:
    """手写猎杀脚本 (RL 的及格线)

    开火前用 LaikaAI.check_bullet_path 做弹道模拟:
    只有模拟结果为 HIT (且非 SUICIDE) 才扣扳机 — 消灭吃反弹弹自杀。
    """
    name = "hunter"

    def __init__(self):
        self.fire_hold = 0
        self._sim = None       # 复用 Laika 的弹道模拟器
        self._sim_game = None

    def reset(self):
        self.fire_hold = 0
        self._sim = None
        self._sim_game = None

    def _bullet_sim(self, game):
        """绑定到 tank0 的弹道模拟器 (每局重建)"""
        if self._sim is None or self._sim_game is not game:
            from tank_trouble_original.laika import LaikaAI
            self._sim = LaikaAI(game, game.tanks[0])
            self._sim_game = game
        return self._sim

    # ---- 工具 ----

    @staticmethod
    def _los(game, x0, y0, x1, y1):
        """两点无墙视线 (逐步采样 wall_hit)"""
        d = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(d / 4.0))
        for k in range(1, steps):
            t = k / steps
            if game.wall_hit(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
                return False
        return True

    @staticmethod
    def _incoming_threat(game, me):
        """最危险的逼近子弹 -> (t_closest, miss_dist, vx, vy) 或 None"""
        best = None
        for b in game.bullets:
            vx = b.x_speed * C.BULLETHITCHECKINTERVALS
            vy = b.y_speed * C.BULLETHITCHECKINTERVALS
            sp2 = vx * vx + vy * vy
            if sp2 < 1e-9:
                continue
            rx, ry = me.x - b.x, me.y - b.y
            t = (rx * vx + ry * vy) / sp2       # 以帧为单位
            if t < 0 or t > 30:
                continue
            cx, cy = b.x + vx * t, b.y + vy * t
            miss = math.hypot(cx - me.x, cy - me.y)
            if miss > game.scale * 0.7:
                continue
            if best is None or t < best[0]:
                best = (t, miss, vx, vy)
        return best

    def _steer_to_angle(self, me, target_deg, out):
        delta = _norm180(target_deg - me.rotation)
        out["turn_left"] = delta < -5
        out["turn_right"] = delta > 5
        return delta

    # ---- 主逻辑 ----

    def act(self, game):
        me = game.tanks[0]
        en = game.tanks[1]
        out = {"forward": False, "backup": False,
               "turn_left": False, "turn_right": False, "fire": False}
        if not me.alive or not en.alive:
            return out
        scale = game.scale

        # 1) 躲避逼近的子弹 (最高优先)
        threat = self._incoming_threat(game, me)
        if threat is not None:
            t, miss, vx, vy = threat
            # 垂直于弹道方向闪避
            perp = math.degrees(math.atan2(vx, -vy))  # 弹道法线朝向角
            delta = self._steer_to_angle(me, perp, out)
            # 已大致垂直则全速冲刺, 否则边转边动
            out["forward"] = True
            return out

        dx, dy = en.x - me.x, en.y - me.y
        dist = math.hypot(dx, dy)
        aim = math.degrees(math.atan2(dy, dx)) + 90   # 原版朝向系
        aim = _norm180(aim)
        has_los = self._los(game, me.x, me.y, en.x, en.y)

        # 2) 有视线: 对准后先模拟弹道, 确认命中才开火
        if has_los:
            delta = self._steer_to_angle(me, aim, out)
            close = dist < scale * 2.2
            if (abs(delta) < (14 if close else 7)
                    and me.bullets_fired < game.settings_max_bullets):
                res = self._bullet_sim(game).check_bullet_path(me.rotation)
                if res["result"] == "HIT":
                    out["fire"] = True
            # 距离控制: 太远逼近, 贴脸减速
            if abs(delta) < 50 and dist > scale * 1.0:
                out["forward"] = True
            return out

        # 3) 无视线: 沿迷宫最短路接近
        fx = int(me.x // scale)
        fy = int(me.y // scale)
        ex, ey = game.tank_fields[1]["x"], game.tank_fields[1]["y"]
        dm = game.dist_map(fx, fy)
        if dm is not None:
            path = get_shortest_path_with_distances(
                game.maze, dm, fx, fy, ex, ey)
            if path:
                # 走向第一个路径点的格心
                tx = (path[0]["x"] + 0.5) * scale
                ty = (path[0]["y"] + 0.5) * scale
                # 若已接近该格心, 取下一个
                if (math.hypot(tx - me.x, ty - me.y) < scale * 0.35
                        and len(path) > 1):
                    tx = (path[1]["x"] + 0.5) * scale
                    ty = (path[1]["y"] + 0.5) * scale
                ang = _norm180(math.degrees(
                    math.atan2(ty - me.y, tx - me.x)) + 90)
                delta = self._steer_to_angle(me, ang, out)
                if abs(delta) < 60:
                    out["forward"] = True
                return out
        # 兜底: 原地转
        out["turn_right"] = True
        return out
