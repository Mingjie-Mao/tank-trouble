"""
P24 生存模式 —— 逼出主动进攻的训练课程 (curriculum)

规则 (用户设计):
  - Laika 无敌 (打不死); 我方一发即死 (保留原版致命性 -> 身法全程有用)
  - 时间条倒计时; 每命中 Laika 一次 +Δt; 归零或我方死亡或触及上限则结束
  - 评分 = 存活帧数 (死亡终止 -> 鲁莽自罚, 堵住"不怕死硬刷分")

铁律 (防 reward hacking, 对齐 OpenAI CoastRunners 教训):
  **生存模式只是训练课程; 最终验收永远在原版游戏 (真胜率 + 进攻性)。**
  任何刷分漏洞 (如缩墙角安全刷分), 在原版验收上一测就露馅。

用法:
  python3 training/survival_mode.py measure --n 40 --workers 8   # 老师会不会狩猎
"""

import argparse
import math
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 时钟配置 (25fps): 起始10s, 每命中+5s, 上限100s。Δt/起始需实验调甜点。
DEFAULT_CFG = dict(start_frames=250, hit_bonus_frames=125, cap_frames=2500)
HIT_REWARD = 500.0          # rollout 里每次命中 Laika 的评分


def run_survival(policy, seed, cfg=DEFAULT_CFG):
    """跑一局生存模式, 返回 (存活帧数, 命中次数)。Laika 无敌, 我方可死。"""
    from tank_trouble_original.game import Game
    g = Game(seed=seed, ai_enabled=True, invincible={1})
    policy.reset()
    clock = cfg["start_frames"]
    frames, hits = 0, 0
    while frames < cfg["cap_frames"] and clock > 0:
        inp = policy.act(g)
        t0 = g.tanks[0]
        t0.forward = bool(inp.get("forward", False))
        t0.backup = bool(inp.get("backup", False))
        t0.turn_left = bool(inp.get("turn_left", False))
        t0.turn_right = bool(inp.get("turn_right", False))
        t0.fire = bool(inp.get("fire", False))
        events = g.step()
        frames += 1
        clock -= 1
        dead = False
        for ev in events:
            if ev[0] == "hit" and ev[1] == 0 and ev[2] == 1:
                clock += cfg["hit_bonus_frames"]
                hits += 1
            elif ev[0] == "destroy" and ev[1] == 0:
                dead = True
        if dead:
            break
    return frames, hits


def survival_rollout(sandbox, first_action, hold=16, horizon=48):
    """老师的生存目标: 奖励命中 Laika, 惩罚死亡, 温和引导靠近瞄准。
    我死 -1000+t | 存活: 命中数*HIT_REWARD - 0.5*距离(引导逼近以便命中)。"""
    me = sandbox.tanks[0]
    en = sandbox.tanks[1]
    th, tu, f = first_action
    hits = 0
    for t in range(horizon):
        if t == 0:
            me.forward, me.backup = th == 2, th == 0
            me.turn_left, me.turn_right = tu == 0, tu == 2
            me.fire = f == 1
        elif t == hold:
            me.fire = False
        events = sandbox.step()
        for ev in events:
            if ev[0] == "hit" and ev[1] == 0 and ev[2] == 1:
                hits += 1
        if not me.alive:
            return -1000.0 + t
    dist = math.hypot(me.x - en.x, me.y - en.y) / sandbox.scale
    return hits * HIT_REWARD - 0.5 * dist


class SurvivalMPC:
    """生存模式的 MPC 老师: 18 候选各前推, 用 survival_rollout 打分取最优。"""
    name = "survival_mpc"

    def __init__(self, horizon=48, hold=16, seed=0):
        import random
        from training.mpc_agent import CANDIDATES, make_sandbox
        self.horizon, self.hold = horizon, hold
        self.rng = random.Random(seed)
        self._cands = CANDIDATES
        self._make_sandbox = make_sandbox

    def reset(self):
        pass

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        best_a, best_s = (1, 1, 0), -1e18
        for a in self._cands:
            sb = self._make_sandbox(game, "L2",
                                    rng_seed=self.rng.randrange(1 << 30))
            s = survival_rollout(sb, a, self.hold, self.horizon)
            if s > best_s:
                best_s, best_a = s, a
        th, tu, f = best_a
        return {"forward": th == 2, "backup": th == 0,
                "turn_left": tu == 0, "turn_right": tu == 2,
                "fire": f == 1}


# ================================================================ 测量

def _baseline_idle():
    from training.baselines import IdlePolicy
    return IdlePolicy()


def _worker(job):
    kind, seed0, count = job
    import torch
    torch.set_num_threads(1)
    if kind == "survival_mpc":
        pol = SurvivalMPC()
    else:
        pol = _baseline_idle()
    surv_sum, hit_sum, n = 0, 0, 0
    for i in range(count):
        f, h = run_survival(pol, seed0 + i)
        surv_sum += f
        hit_sum += h
        n += 1
    return surv_sum, hit_sum, n


def measure(n, workers):
    """老师(生存目标) vs 空闲基线: 存活帧数 + 命中次数。
    老师存活远长于空闲 + 命中多 => 它学会了'主动打人续命'(会狩猎)。"""
    for kind, tag in [("idle", "空闲基线(不动不打)"),
                      ("survival_mpc", "生存老师(MPC+生存目标)")]:
        per = max(1, n // workers)
        jobs = [(kind, 880000 + w * per, per) for w in range(workers)]
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers) as pool:
            out = pool.map(_worker, jobs)
        sf, hs, tot = 0, 0, 0
        for s, h, c in out:
            sf += s
            hs += h
            tot += c
        print(f"\n{tag}: {tot}局 ({time.time()-t0:.0f}s)", flush=True)
        print(f"  平均存活 {sf/tot:.0f} 帧 ({sf/tot/25:.1f}s)  "
              f"平均命中 Laika {hs/tot:.1f} 次/局", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["measure"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int,
                    default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    print("===== [P24] 生存模式: 老师会不会主动狩猎 =====", flush=True)
    print(f"配置: 起始{DEFAULT_CFG['start_frames']}帧 每命中+"
          f"{DEFAULT_CFG['hit_bonus_frames']}帧 上限{DEFAULT_CFG['cap_frames']}帧",
          flush=True)
    measure(args.n, args.workers)
    print("\n提醒: 这只是训练课程的机制验证; 真验收永远在原版游戏。", flush=True)


if __name__ == "__main__":
    main()
