"""
MPC 决策时搜索智能体 (B线, 无未来函数版)

公平性规范:
  可用: 当前墙体/双方位姿/子弹位置速度寿命/弹药 (全部屏幕可见或历史可推)
  物理前推: 合法 (确定性物理)
  禁止: 对手真实 RNG (沙盒独立采样) / 对手内部目标栈 (沙盒全新状态) /
        地图种子。自己的扳机状态可用 (自知)。

流程: 每个决策步(2帧)从可观测量构建沙盒 -> 枚举候选动作各前推 H 帧 ->
     打分选优。对手模型 L1=输入冻结 / L2=白盒 Laika 算法+新鲜状态+采样RNG。

用法:
  python3 training/mpc_agent.py --n 100 --opp-model L2 --horizon 48
"""

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.game import Game, Tank, Bullet  # noqa: E402
from tank_trouble_original.laika import LaikaAI  # noqa: E402

# 候选第一动作: 油门(倒/停/进) x 转向(左/无/右) x 开火(否/是) = 18
CANDIDATES = [(th, tu, f) for th in (0, 1, 2) for tu in (0, 1, 2)
              for f in (0, 1)]


# ================================================================ 沙盒

def _copy_tank(t, sandbox):
    nt = object.__new__(Tank)
    nt.__dict__.update(t.__dict__)     # 标量/不可变列表浅拷贝
    nt.game = sandbox
    nt.ai = None                       # 内部 AI 状态不带入 (公平性)
    return nt


def _copy_bullet(b, sandbox):
    nb = object.__new__(Bullet)
    nb.__dict__.update(b.__dict__)
    nb.game = sandbox
    nb.owner = sandbox.tanks[b.owner.number]   # 重定向到沙盒坦克
    return nb


def make_sandbox(game, opp_model="L2", rng_seed=0):
    """从可观测量构建前推沙盒。

    共享只读结构 (迷宫/墙体/距离场 — 当局全部可见);
    复制动态状态 (坦克位姿/子弹/计数器);
    洗掉隐藏信息 (RNG 重新播种, 对手 AI 全新实例)。
    """
    sb = object.__new__(Game)
    # ---- 共享只读 (当局可见) ----
    for attr in ("maze", "walls", "wall_half_t", "scale", "_wall_grid",
                 "distances_for_maze", "dead_ends", "reachable",
                 "reachable_index", "settings_max_bullets",
                 "settings_max_crates", "settings_crate_spawn_modifier",
                 "settings_active_weapons", "tanks_count", "ai_enabled",
                 "self_harm_immune", "invincible", "hit_immunity_duration"):
        if hasattr(game, attr):
            setattr(sb, attr, getattr(game, attr))
    # ---- 洗掉隐藏信息 ----
    sb.rng = random.Random(rng_seed)          # 独立采样, 非真实种子
    # ---- 复制动态状态 ----
    sb.tanks = [_copy_tank(t, sb) for t in game.tanks]
    sb.bullets = [_copy_bullet(b, sb) for b in game.bullets]
    sb.tank_fields = [dict(f) for f in game.tank_fields]
    sb.crates = {}
    sb.events = []
    for attr in ("frame", "alive_count", "end_count", "reset_count",
                 "frozen", "shake", "crate_timer", "_bullet_depth",
                 "round_number", "scores", "hit_immunity_remaining"):
        v = getattr(game, attr)
        setattr(sb, attr, list(v) if isinstance(v, list) else v)
    # ---- 对手模型 ----
    if opp_model == "L2":
        # 白盒算法 + 全新内部状态 (goal=idle) + 沙盒采样 RNG
        sb.tanks[1].ai = LaikaAI(sb, sb.tanks[1])
    # L1: ai=None, 输入保持当前值 (冻结模型)
    return sb


# ================================================================ 前推打分

def rollout(sandbox, first_action, hold=16, horizon=48, leaf_fn=None):
    """沙盒中执行: 前 hold 帧用候选动作, 之后松开火保持行驶, 共 horizon 帧。

    返回评分 (越大越好):
      我死 -1000+t (晚死略好) | 杀敌且存活 +1000-t (早杀略好)
      存活到期: leaf_fn(sandbox) 若给定 (学出来的价值), 否则温和启发式

    leaf_fn: 可选叶子评估器 (价值叶子)。仅在"存活到 horizon"时替换末端启发式;
      rollout 内真实发生的生/死 (±1000) 永远优先, 与 AlphaZero 结构一致
      (真终局用真值, 未终局才用价值估计)。约定返回同 ±1000 量纲 (= 1000*V)。
    """
    me = sandbox.tanks[0]
    th, tu, f = first_action
    for t in range(horizon):
        if t == 0:
            me.forward, me.backup = th == 2, th == 0
            me.turn_left, me.turn_right = tu == 0, tu == 2
            me.fire = f == 1
        elif t == hold:
            me.fire = False       # 之后不再开火, 只评估行驶延续
        events = sandbox.step()
        for ev in events:
            if ev[0] == "destroy":
                if ev[1] == 0:
                    return -1000.0 + t
                # 对手死, 但需自己活到最后才算数 (双亡窗口继续模拟)
        if not me.alive:
            return -1000.0 + t
        if not sandbox.tanks[1].alive and me.alive and t >= hold:
            return 1000.0 - t
    # 存活到期: 价值叶子优先, 否则温和启发式
    if leaf_fn is not None:
        return leaf_fn(sandbox)
    en = sandbox.tanks[1]
    score = 0.0
    if not en.alive:
        score += 800.0
    # 路径距离引导
    fx = int(me.x // sandbox.scale)
    fy = int(me.y // sandbox.scale)
    dm = sandbox.dist_map(fx, fy)
    if dm is not None:
        ex = int(en.x // sandbox.scale)
        ey = int(en.y // sandbox.scale)
        if 0 <= ex < len(dm) and 0 <= ey < len(dm[ex]):
            v = dm[ex][ey]
            if v is not None and v == v:
                score -= 0.5 * float(v)
    return score


class MPCPolicy:
    """每决策步: 对 18 个候选动作各做 n_samples 次沙盒前推, 取均分最优。"""
    name = "mpc"

    def __init__(self, opp_model="L2", horizon=48, hold=16, n_samples=1,
                 seed=0, leaf_fn=None):
        self.opp_model = opp_model
        self.horizon = horizon
        self.hold = hold
        self.n_samples = n_samples   # >1 时对 L2 的 RNG 多次采样取均值
        self.rng = random.Random(seed)
        self.leaf_fn = leaf_fn       # 价值叶子 (None = 用启发式末端)

    def reset(self):
        pass

    def act(self, game):
        me = game.tanks[0]
        if not me.alive:
            return {}
        best_a, best_s = (1, 1, 0), -1e18
        for a in CANDIDATES:
            total = 0.0
            for k in range(self.n_samples):
                sb = make_sandbox(game, self.opp_model,
                                  rng_seed=self.rng.randrange(1 << 30))
                total += rollout(sb, a, self.hold, self.horizon,
                                 leaf_fn=self.leaf_fn)
            s = total / self.n_samples
            if s > best_s:
                best_s, best_a = s, a
        th, tu, f = best_a
        return {"forward": th == 2, "backup": th == 0,
                "turn_left": tu == 0, "turn_right": tu == 2,
                "fire": f == 1}


# ================================================================ 评估入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=970000)
    ap.add_argument("--opp-model", choices=["L1", "L2"], default="L2")
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--hold", type=int, default=16)
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    from training.evaluate import play_round_dual_engine
    policy = MPCPolicy(args.opp_model, args.horizon, args.hold, args.samples)
    results = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    t0 = time.time()
    for i in range(args.n):
        r = play_round_dual_engine(policy, args.seed + i)
        results[r["true_result"]] = results.get(r["true_result"], 0) + 1
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{args.n}] 真胜率 {results['win']/(i+1):.1%} "
                  f"(负{results['loss']} 双亡{results['double_death']} "
                  f"平{results['draw']}) {el:.0f}s", flush=True)
    n = args.n
    print(f"\n===== MPC({args.opp_model}, H={args.horizon}) "
          f"{n} 局 @{args.seed} =====")
    print(f"  真胜率 {results['win']/n:.1%}  负 {results['loss']/n:.1%}  "
          f"双亡 {results['double_death']/n:.1%}  平 {results['draw']/n:.1%}")
    print(f"  用时 {time.time()-t0:.0f}s ({(time.time()-t0)/n:.1f}s/局)")
    print(f"  参照: 冠军 P17=36.4% | 镜像线 40.2%")


if __name__ == "__main__":
    main()
