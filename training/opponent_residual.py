"""对手残差建模：以 Laika 为参照系描述对手。

设计要点（用户拍板）：不去绝对地描述"这个对手怎么打"，而是描述
"他和 Laika 差在哪"。

为什么锚在 Laika 而不是现成的强网络（P40/P25v2）——决定性理由是**成本**：
对手模型跑在 rollout **里面**，一次决策 10 候选 × 36 帧、对手每 4 帧决策
一次 = 约 90 次对手推理。真 Laika 是手写算法，微秒级；P40 那类网络单次
~1ms **而且需要以"我"为目标再建一张击杀场**，完全跑不动。

残差形式还有两个性质：
* `z = 0` 时**严格等于**今天的系统（不是它的近似），所以遇到没见过的
  对手最坏退回现状；
* 只学偏离量，数据需求远低于学一个完整策略。

注意这不是 LoRA——Laika 没有权重矩阵。它是**动作空间的残差策略**：
    对手动作 ~ softmax( onehot(Laika 的动作) + residual(s, z) )

本模块第一步只做一件事：**验证残差里确实有东西可学**。如果脚本对手和
Laika 在绝大多数帧上做一样的动作，那后面的编码器就没有意义。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.game import Game  # noqa: E402
from tank_trouble_original.laika import LaikaAI  # noqa: E402


def _action_of(tank):
    """把坦克当前输入压成 (throttle, turn, fire)，与 CANDIDATES 同构。"""
    throttle = 2 if tank.forward else (0 if tank.backup else 1)
    turn = 0 if tank.turn_left else (2 if tank.turn_right else 1)
    return throttle, turn, int(bool(tank.fire))


def laika_action_in(game, tank):
    """同一局面下 Laika 会做什么。用全新实例，不带内部状态。"""
    saved = (tank.forward, tank.backup, tank.turn_left,
             tank.turn_right, tank.fire, tank.ai)
    try:
        tank.ai = LaikaAI(game, tank)
        if tank.ai.make_decisions_and_update_goal():
            tank.ai.decide_actions_to_achieve_goal()
        tank.ai.set_input_to_do_actions()
        return _action_of(tank)
    finally:
        (tank.forward, tank.backup, tank.turn_left,
         tank.turn_right, tank.fire, tank.ai) = saved


# ------------------------------------------------------------------ 脚本对手

class ScriptedOpponent:
    """脚本对手基类。act() 返回 (throttle, turn, fire)。"""

    name = "base"

    def reset(self):
        pass

    def act(self, game, me, foe):
        raise NotImplementedError


class Camper(ScriptedOpponent):
    """苟住：原地不动，只在对手进入正前方时开火。"""

    name = "苟住"

    def act(self, game, me, foe):
        import math
        angle = math.atan2(foe.y - me.y, foe.x - me.x)
        facing = (me.rotation - 90) * math.pi / 180.0
        delta = math.atan2(math.sin(angle - facing),
                           math.cos(angle - facing))
        if abs(delta) < 0.15:
            return 1, 1, 1
        return 1, (0 if delta < 0 else 2), 0


class Rusher(ScriptedOpponent):
    """莽冲：始终朝对手推进，见缝就开火。"""

    name = "莽冲"

    def act(self, game, me, foe):
        import math
        angle = math.atan2(foe.y - me.y, foe.x - me.x)
        facing = (me.rotation - 90) * math.pi / 180.0
        delta = math.atan2(math.sin(angle - facing),
                           math.cos(angle - facing))
        turn = 1 if abs(delta) < 0.2 else (0 if delta < 0 else 2)
        fire = 1 if abs(delta) < 0.1 else 0
        return (2 if turn == 1 else 1), turn, fire


class Baiter(ScriptedOpponent):
    """骗弹：保持中距、持续横向移动、极少开火。

    这是群友用来打赢当前 AI 的那类打法的脚本化近似。
    """

    name = "骗弹"

    def reset(self):
        self.phase = 0

    def act(self, game, me, foe):
        import math
        self.phase = getattr(self, "phase", 0) + 1
        distance = math.hypot(foe.x - me.x, foe.y - me.y) / game.scale
        swing = 0 if (self.phase // 20) % 2 == 0 else 2
        throttle = 2 if distance > 3.0 else (0 if distance < 1.5 else 1)
        return throttle, swing, 0


class Flanker(ScriptedOpponent):
    """绕后：朝对手炮口的侧后方绕。"""

    name = "绕后"

    def act(self, game, me, foe):
        import math
        foe_facing = (foe.rotation - 90) * math.pi / 180.0
        target = foe_facing + math.pi
        angle = math.atan2(math.sin(target), math.cos(target))
        facing = (me.rotation - 90) * math.pi / 180.0
        delta = math.atan2(math.sin(angle - facing),
                           math.cos(angle - facing))
        turn = 1 if abs(delta) < 0.25 else (0 if delta < 0 else 2)
        return 2, turn, 0


POPULATION = [Camper, Rusher, Baiter, Flanker]


# ------------------------------------------------------------------ 差异测量

def measure_divergence(rounds=8, frames=500, seed_base=45_000_000):
    """对每个脚本对手，统计它的动作与 Laika 在同一局面下动作的差异。

    如果差异很稀疏，残差就没什么可学的——那是这条路的前置门槛。
    """
    from training.killfield_prebuild import FastKillFieldTeacher

    print(f"{'对手':<8}{'帧数':>7}{'动作不同':>10}{'油门不同':>10}"
          f"{'转向不同':>10}{'开火不同':>10}")
    print("-" * 57)
    results = {}
    for cls in POPULATION:
        total = diff = d_thr = d_turn = d_fire = 0
        for index in range(rounds):
            seed = seed_base + index
            game = Game(seed=seed, ai_enabled=False)
            teacher = FastKillFieldTeacher(
                seed=seed ^ 0x0FF,
                ray_count=256, max_bounces=2, max_flight_frames=75,
                horizon=36, skip_masked=True, parallel_workers=0)
            script = cls()
            script.reset()
            me, foe = game.tanks          # me=tank0 由老师控, foe=tank1 脚本
            for _ in range(frames):
                if game.frozen:
                    break
                if me.alive:
                    action = teacher.act(game)
                    me.forward = bool(action.get("forward", False))
                    me.backup = bool(action.get("backup", False))
                    me.turn_left = bool(action.get("turn_left", False))
                    me.turn_right = bool(action.get("turn_right", False))
                    me.fire = bool(action.get("fire", False))
                if foe.alive:
                    reference = laika_action_in(game, foe)
                    throttle, turn, fire = script.act(game, foe, me)
                    foe.forward = throttle == 2
                    foe.backup = throttle == 0
                    foe.turn_left = turn == 0
                    foe.turn_right = turn == 2
                    foe.fire = fire == 1
                    actual = (throttle, turn, fire)
                    total += 1
                    diff += int(actual != reference)
                    d_thr += int(actual[0] != reference[0])
                    d_turn += int(actual[1] != reference[1])
                    d_fire += int(actual[2] != reference[2])
                game.step()
            teacher.close()
        results[cls.name] = diff / max(total, 1)
        print(f"{cls.name:<8}{total:>7}{100*diff/max(total,1):>9.1f}%"
              f"{100*d_thr/max(total,1):>9.1f}%"
              f"{100*d_turn/max(total,1):>9.1f}%"
              f"{100*d_fire/max(total,1):>9.1f}%")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["divergence"])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()
    if args.command == "divergence":
        print("=== 脚本对手 vs Laika：同一局面下的动作差异 ===")
        print("    差异太稀疏 -> 残差没东西可学，这条路收益有限\n")
        measure_divergence(args.rounds, args.frames)


if __name__ == "__main__":
    main()
