"""
组合型漏洞挖掘 — 击杀窗口分析 + 胜负对照 + 组合条件搜索。

目标: 找出 PPO 已在利用但未被显式总结的模式 (而非单变量相关)。

  [A] 击杀前 2 秒窗口画像: 双方位置/墙体关系/视线/反弹路径/Laika 意图与弹药
  [B] 胜局 vs 败局对照: 开火质量/移动/路径闭合/站桩/无效开火/自伤
  [C] 组合条件搜索: 2-3 个二值条件的合取, 按"出现该局面的局的真胜率"排序
      (支持度下限过滤, 防小样本假象)

用法:
  python3 training/kill_pattern_analysis.py --model training/models/p6_sharpen_best.zip --n 1500
"""

import argparse
import math
import os
import sys
from collections import Counter, deque
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from training.tt_gym_env import TankTroubleGym, OBS_DIM, N_BULLET_SLOTS, \
    TRAJ_BULLET_FEATS, SHOT_FAN_DEG  # noqa: E402
from training.exploit_analysis import laika_wall_class  # noqa: E402

PRE_KILL_STEPS = 25       # 击杀前窗口 (决策步; skip=2 时 = 50 帧 = 2 秒)
CAMP_STEP_CELLS = 0.02    # 单步移动小于该格数视为站桩步
FAN_BASE = OBS_DIM + N_BULLET_SLOTS * TRAJ_BULLET_FEATS


def norm180(a):
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a


def step_conditions(env, obs):
    """当前决策步的二值条件集 (组合搜索的原子)。"""
    g = env.game
    me, en = g.tanks[0], g.tanks[1]
    s = g.scale
    dist = math.hypot(en.x - me.x, en.y - me.y) / s

    # 射击扇 (来自观测, 与策略所见完全一致)
    fan = obs[FAN_BASE:FAN_BASE + len(SHOT_FAN_DEG) * 3].reshape(-1, 3)
    fan_hit = fan[:, 0] > 0.5
    fan_direct = bool((fan_hit & (fan[:, 2] > 0.99)).any())
    fan_bounce = bool((fan_hit & (fan[:, 2] < 0.99)).any())

    los = bool(env._line_of_sight(me.x, me.y, en.x, en.y))
    # 我是否暴露在 Laika 炮线上: 有视线且它炮口指向我 15° 内
    aim_to_me = norm180(math.degrees(math.atan2(me.y - en.y, me.x - en.x))
                        + 90 - en.rotation)
    exposed = los and abs(aim_to_me) < 15

    goal = en.ai.my_goal.get("goal", "none") if en.ai else "none"
    goal_class = ("dodge" if goal == "dodgeBullet" else
                  "aim" if goal in ("shootAfter", "idle") else "move")
    ammo = g.settings_max_bullets - en.bullets_fired
    wall = laika_wall_class(g, en)

    return {
        "L躲弹中": goal_class == "dodge",
        "L瞄准或发呆": goal_class == "aim",
        "L移动中": goal_class == "move",
        "L低弹药(<=1)": ammo <= 1,
        "L贴墙": wall in ("wall", "corner", "corridor"),
        "L卡墙": bool(en.hit_something),
        "有直射线": los,
        "有直射扇角": fan_direct,
        "仅反弹扇角": fan_bounce and not fan_direct,
        "我在炮线外": los and not exposed,
        "近距(<2.5格)": dist < 2.5,
        "中距(2.5-5格)": 2.5 <= dist < 5.0,
        "远距(>=5格)": dist >= 5.0,
    }


def run(model_path, n_rounds, base_seed, frame_skip):
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    assert model.observation_space.shape[0] == OBS_DIM + \
        N_BULLET_SLOTS * TRAJ_BULLET_FEATS + len(SHOT_FAN_DEG) * 3, \
        "组合挖掘需要弹道预演观测(121维)模型"
    env = TankTroubleGym(seed=0, terminal_mode="score", obs_traj=True,
                         frame_skip=frame_skip)

    cond_names = None
    base_step_counts = Counter()   # 条件 -> 全步出现数
    total_steps = 0
    kill_step_counts = Counter()   # 条件 -> 击杀步出现数
    kills = 0
    rounds = []                    # 每局聚合记录

    for ep in range(n_rounds):
        env._base_seed = base_seed + ep
        env._episode = 0
        obs, _ = env.reset()
        g = env.game
        ring = deque(maxlen=PRE_KILL_STEPS)
        seen_conds = Counter()     # 本局各条件出现步数
        steps = 0
        move_cells = 0.0
        camp_steps = 0
        shots = 0
        kill_conds = None          # 击杀那一步的条件
        path_start = env._spawn_path_cells()
        path_sum = 0.0
        result = None
        died_by_self = False

        while True:
            conds = step_conditions(env, obs)
            if cond_names is None:
                cond_names = list(conds)
            ring.append(conds)
            for k, v in conds.items():
                if v:
                    base_step_counts[k] += 1
                    seen_conds[k] += 1
            total_steps += 1
            steps += 1
            path_sum += min(env._spawn_path_cells(), 30)

            px, py = g.tanks[0].x, g.tanks[0].y
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            d = math.hypot(g.tanks[0].x - px, g.tanks[0].y - py) / g.scale
            move_cells += d
            if d < CAMP_STEP_CELLS:
                camp_steps += 1

            for ev in info.get("events", []):
                if ev[0] == "fire" and ev[1] == 0:
                    shots += 1
                elif ev[0] == "hit":
                    if ev[1] == 0 and ev[2] == 1:
                        kills += 1
                        kill_conds = dict(conds)
                        for k, v in conds.items():
                            if v:
                                kill_step_counts[k] += 1
                    if ev[2] == 0 and ev[1] == 0:
                        died_by_self = True

            if terminated or truncated:
                result = info.get("result", "draw")
                break

        rounds.append({
            "win": result == "win",
            "result": result,
            "steps": steps,
            "move": move_cells,
            "camp_frac": camp_steps / max(steps, 1),
            "shots": shots,
            "killed": kill_conds is not None,
            "kill_conds": kill_conds,
            "cond_frac": {k: seen_conds[k] / max(steps, 1) for k in seen_conds},
            "path_start": path_start,
            "path_mean": path_sum / max(steps, 1),
            "died_by_self": died_by_self,
        })
        if (ep + 1) % 250 == 0:
            w = sum(r["win"] for r in rounds) / len(rounds)
            print(f"  [{ep+1}/{n_rounds}] 真胜率 {w:.1%}", flush=True)

    # ================= 报告 =================
    n = len(rounds)
    wins = [r for r in rounds if r["win"]]
    losses = [r for r in rounds if r["result"] == "loss"]
    dd = [r for r in rounds if r["result"] == "double_death"]
    print(f"\n===== 组合型模式挖掘  {model_path}  ({n} 局) =====")
    print(f"真胜率 {len(wins)/n:.1%}  负 {len(losses)/n:.1%}  双亡 {len(dd)/n:.1%}  "
          f"击杀 {kills}")

    # [A] 击杀步条件画像 (lift vs 全步基线)
    print(f"\n[A] 击杀瞬间条件画像 (基线占比 -> 击杀步占比, lift):")
    lifts = []
    for k in cond_names:
        b = base_step_counts[k] / max(total_steps, 1)
        kk = kill_step_counts[k] / max(kills, 1)
        lifts.append((kk / b if b > 1e-9 else 0, k, b, kk))
    for lf, k, b, kk in sorted(lifts, reverse=True):
        print(f"  {k:<14} {b:6.1%} -> {kk:6.1%}   lift={lf:.2f}")

    # [B] 胜负对照
    def agg(rs, key):
        return np.mean([r[key] for r in rs]) if rs else float("nan")

    print(f"\n[B] 胜局({len(wins)}) vs 败局({len(losses)}) vs 双亡({len(dd)}) 对照:")
    for key, label in (("move", "场均移动(格)"), ("camp_frac", "站桩步占比"),
                       ("shots", "场均开火"), ("steps", "局长(决策步)"),
                       ("path_mean", "平均路径距离(格)"),
                       ("died_by_self", "死于自己子弹率")):
        print(f"  {label:<14} 胜 {agg(wins, key):6.2f}  "
              f"负 {agg(losses, key):6.2f}  双亡 {agg(dd, key):6.2f}")

    # [C] 组合条件搜索: 局内出现过该组合(任一步同时成立) -> 该局真胜率
    print(f"\n[C] 组合条件 -> 局真胜率 (基线 {len(wins)/n:.1%}, 支持度>=80 局):")
    combo_rows = []
    for r in rounds:
        # 每局的"组合出现"用击杀步或高暴露判定: 该组合在 >=10% 步数同时成立
        pass
    # 预计算: 每局每条件占比已在 cond_frac
    for size in (2, 3):
        for combo in combinations(cond_names, size):
            support_rounds = [
                r for r in rounds
                if all(r["cond_frac"].get(c, 0) >= 0.10 for c in combo)]
            if len(support_rounds) < 80:
                continue
            wr = sum(r["win"] for r in support_rounds) / len(support_rounds)
            combo_rows.append((wr, len(support_rounds), combo))
    combo_rows.sort(reverse=True)
    print("  -- 胜率最高的组合 --")
    for wr, cnt, combo in combo_rows[:12]:
        print(f"  {wr:6.1%} ({cnt:4d}局)  " + " + ".join(combo))
    print("  -- 胜率最低的组合 (危险局面) --")
    for wr, cnt, combo in combo_rows[-6:]:
        print(f"  {wr:6.1%} ({cnt:4d}局)  " + " + ".join(combo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="training/models/p6_sharpen_best.zip")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=970000)
    ap.add_argument("--frame-skip", type=int, default=2)
    args = ap.parse_args()
    run(args.model, args.n, args.seed, args.frame_skip)


if __name__ == "__main__":
    main()
