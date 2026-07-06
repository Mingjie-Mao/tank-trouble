"""
标准评估协议 — 所有策略统一在此测胜率

协议 (默认双口径 + 行为指标):
  - 固定种子序列 base_seed..base_seed+N-1, 每种子一局 (新迷宫)
  - 每局同时结算两个口径:
      先杀率  = 首个 destroy 的受害者是 Laika (队友训练环境的口径)
      真胜率  = 原版计分点 endCount==50 时我方存活且 Laika 死亡
               (先杀后死 = 双亡, 不得分 —— 与原版 Flash 规则一致)
  - 行为指标: 自杀率 / 双亡率 / 场均开火 / 命中率 / 场均移动 /
             死因分布(直射/反弹弹/自己子弹) / 平均局长
  - --legacy 回退到旧的单口径(destroy 即终局)输出

用法:
  python training/evaluate.py --policy hunter --n 500
  python training/evaluate.py --policy random --n 200
  python training/evaluate.py --policy model --model models/best_model.zip
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import Game  # noqa: E402
from training.baselines import (IdlePolicy, RandomPolicy, HunterPolicy,  # noqa: E402
                                ExploitBot)
from training.tt_gym_env import TRUNCATE_FRAMES, FRAME_SKIP  # noqa: E402


# ================================================================ 双口径评估

class RoundTracker:
    """逐步消费事件流, 汇总一局的行为指标。

    致命弹溯源: 每步前快照场上子弹, 步后若有 hit 事件, 本步消失且剩余寿命
    大于步长的子弹即命中弹; 结合 bounce 事件计数区分直射/反弹。同步内多弹
    命中(极罕见)归为 unknown。
    """

    def __init__(self, game):
        self.game = game
        self.scale = game.scale
        self.bounce_count: dict[str, int] = {}
        self.shots = 0
        self.kills = 0               # 我方子弹击杀 Laika 次数 (0/1)
        self.kill_type = None        # direct / bounce / unknown (我方击杀的弹道类型)
        self.death_cause = None      # self / laika_direct / laika_bounce / laika_unknown
        self.move_px = 0.0
        self._prev_pos = (game.tanks[0].x, game.tanks[0].y)
        self._snapshot = {}
        self.first_destroy = None    # 0 / 1 / "both"

    def pre_step(self):
        self._snapshot = {b.name: b.lifetime for b in self.game.bullets}
        t0 = self.game.tanks[0]
        self._prev_pos = (t0.x, t0.y)

    def post_step(self, events, window_frames):
        g = self.game
        t0 = g.tanks[0]
        self.move_px += math.hypot(t0.x - self._prev_pos[0],
                                   t0.y - self._prev_pos[1])
        # 先杀归属只看第一个 destroy; 同窗口内双杀才算 "both"
        if self.first_destroy is None:
            victims = [ev[1] for ev in events if ev[0] == "destroy"]
            if victims:
                self.first_destroy = victims[0] if len(set(victims)) == 1 else "both"
        for ev in events:
            if ev[0] == "bounce":
                self.bounce_count[ev[1]] = self.bounce_count.get(ev[1], 0) + 1
            elif ev[0] == "fire" and ev[1] == 0:
                self.shots += 1
            elif ev[0] == "hit":
                owner, victim = ev[1], ev[2]
                if victim == 1 and owner == 0:
                    self.kills += 1
                    self.kill_type = self._classify_bullet(window_frames,
                                                           "direct", "bounce")
                if victim == 0:
                    self.death_cause = self._classify_death(owner, window_frames)

    def _classify_bullet(self, window_frames, direct_label, bounce_label):
        """本窗口内命中弹溯源: 唯一消失的非到期子弹 + 反弹计数。"""
        alive_names = {b.name for b in self.game.bullets}
        killers = [name for name, life in self._snapshot.items()
                   if name not in alive_names and life > window_frames]
        if len(killers) == 1:
            return (bounce_label if self.bounce_count.get(killers[0], 0) > 0
                    else direct_label)
        return "unknown"

    def _classify_death(self, owner, window_frames):
        if owner == 0:
            return "self"
        kind = self._classify_bullet(window_frames, "laika_direct", "laika_bounce")
        return "laika_unknown" if kind == "unknown" else kind


def play_round_dual_engine(policy, seed, max_frames=TRUNCATE_FRAMES):
    """基线策略: 直接驱动引擎, 跑到原版计分点。返回指标 dict。"""
    game = Game(seed=seed, ai_enabled=True)
    policy.reset()
    tracker = RoundTracker(game)
    true_result = None
    frames = 0
    while frames < max_frames or tracker.first_destroy is not None:
        inp = policy.act(game)
        t0 = game.tanks[0]
        t0.forward = bool(inp.get("forward", False))
        t0.backup = bool(inp.get("backup", False))
        t0.turn_left = bool(inp.get("turn_left", False))
        t0.turn_right = bool(inp.get("turn_right", False))
        t0.fire = bool(inp.get("fire", False))
        tracker.pre_step()
        events = game.step()
        frames += 1
        tracker.post_step(events, 1)
        for ev in events:
            if ev[0] == "round_end":
                winner = ev[1]
                true_result = ("win" if winner == 0 else
                               "loss" if winner == 1 else "double_death")
        if true_result:
            break
    return _round_stats(tracker, true_result or "draw", frames)


def play_round_dual_model(model_policy, seed):
    """模型策略: 走 score 模式 gym 环境 (观测与训练完全一致)。"""
    env = model_policy.score_env
    env._base_seed = seed
    env._episode = 0
    obs, _ = env.reset()
    tracker = RoundTracker(env.game)
    while True:
        action, _ = model_policy.model.predict(obs, deterministic=True)
        tracker.pre_step()
        obs, _, terminated, truncated, info = env.step(action)
        tracker.post_step(info.get("events", []), env.frame_skip)
        if terminated or truncated:
            return _round_stats(tracker, info.get("result", "draw"),
                                env._frames)


def _round_stats(tracker, true_result, frames):
    kill_first = ("win" if tracker.first_destroy == 1 else
                  "loss" if tracker.first_destroy == 0 else
                  "both" if tracker.first_destroy == "both" else "none")
    return {
        "kill_first": kill_first,
        "true_result": true_result,
        "shots": tracker.shots,
        "kills": tracker.kills,
        "kill_type": tracker.kill_type,
        "death_cause": tracker.death_cause,
        "move_cells": tracker.move_px / tracker.scale,
        "frames": frames,
    }


def evaluate_dual(policy, n=500, base_seed=910000, verbose=True):
    rounds = []
    t0 = time.time()
    for i in range(n):
        seed = base_seed + i
        if isinstance(policy, ModelPolicy):
            r = play_round_dual_model(policy, seed)
        else:
            r = play_round_dual_engine(policy, seed)
        rounds.append(r)
        if verbose and (i + 1) % 100 == 0:
            kf = sum(1 for x in rounds if x["kill_first"] == "win") / len(rounds)
            tw = sum(1 for x in rounds if x["true_result"] == "win") / len(rounds)
            print(f"  [{i+1}/{n}] 先杀率 {kf:.1%}  真胜率 {tw:.1%}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    def rate(pred):
        return sum(1 for r in rounds if pred(r)) / n

    kf_win = rate(lambda r: r["kill_first"] == "win")
    true_win = rate(lambda r: r["true_result"] == "win")
    deaths = [r["death_cause"] for r in rounds if r["death_cause"]]
    n_death = max(len(deaths), 1)
    total_shots = sum(r["shots"] for r in rounds)
    stats = {
        "policy": policy.name, "n": n,
        "kill_first_rate": kf_win,
        "true_win_rate": true_win,
    }
    if verbose:
        el = time.time() - t0
        print(f"\n===== {policy.name} 双口径评估 ({n} 局, {el:.0f}s) =====")
        print(f"  先杀率(destroy口径): {kf_win:.1%}"
              f"   [同帧双杀 {rate(lambda r: r['kill_first'] == 'both'):.1%}]")
        print(f"  真胜率(原版计分):    {true_win:.1%}   ← 口径差 "
              f"{kf_win - true_win:+.1%}")
        print(f"  真负率 {rate(lambda r: r['true_result'] == 'loss'):.1%}  "
              f"双亡率 {rate(lambda r: r['true_result'] == 'double_death'):.1%}  "
              f"超时平局 {rate(lambda r: r['true_result'] == 'draw'):.1%}")
        print(f"  自杀率(死于自己子弹): {rate(lambda r: r['death_cause'] == 'self'):.1%}")
        print(f"  场均开火 {total_shots / n:.1f} 发  "
              f"命中率 {sum(r['kills'] for r in rounds) / max(total_shots, 1):.1%}  "
              f"零开火局 {rate(lambda r: r['shots'] == 0):.0%}")
        print(f"  场均移动 {sum(r['move_cells'] for r in rounds) / n:.1f} 格  "
              f"平均局长 {sum(r['frames'] for r in rounds) / n / 25:.1f} 秒")
        print(f"  死因分布({len(deaths)}次死亡): "
              f"Laika直射 {deaths.count('laika_direct') / n_death:.0%}  "
              f"Laika反弹弹 {deaths.count('laika_bounce') / n_death:.0%}  "
              f"自己子弹 {deaths.count('self') / n_death:.0%}  "
              f"未知 {deaths.count('laika_unknown') / n_death:.0%}")
        kt = [r["kill_type"] for r in rounds if r["kill_type"]]
        if kt:
            print(f"  击杀方式({len(kt)}次击杀): 直射 {kt.count('direct') / len(kt):.0%}  "
                  f"反弹 {kt.count('bounce') / len(kt):.0%}  "
                  f"未知 {kt.count('unknown') / len(kt):.0%}")
    return stats


# ================================================================ 旧单口径协议


def play_round(policy, seed, max_frames=TRUNCATE_FRAMES):
    """跑一局, 返回 (result, frames, suicide)
    result: 'win' | 'loss' | 'draw'  (win = Laika 死)"""
    game = Game(seed=seed, ai_enabled=True)
    policy.reset()
    suicide = False
    for frame in range(max_frames):
        inp = policy.act(game)
        t0 = game.tanks[0]
        t0.forward = bool(inp.get("forward", False))
        t0.backup = bool(inp.get("backup", False))
        t0.turn_left = bool(inp.get("turn_left", False))
        t0.turn_right = bool(inp.get("turn_right", False))
        t0.fire = bool(inp.get("fire", False))
        events = game.step()
        result = None
        for ev in events:
            if ev[0] == "hit" and ev[1] == 0 and ev[2] == 0:
                suicide = True
            if ev[0] == "destroy":
                result = "loss" if ev[1] == 0 else "win"
        if result:
            return result, frame + 1, suicide
    return "draw", max_frames, suicide


class ModelPolicy:
    """SB3 模型包装成引擎策略 (走观测向量, 与训练一致)"""
    name = "model"

    def __init__(self, model_path, frame_skip=2):
        from stable_baselines3 import PPO
        from training.tt_gym_env import TankTroubleGym, OBS_DIM
        self.model = PPO.load(model_path, device="cpu")
        # 按模型输入维度自动匹配观测版本 (76 基础 / 121 弹道预演)
        traj = self.model.observation_space.shape[0] != OBS_DIM
        self.frame_skip = frame_skip
        self.env = TankTroubleGym(seed=0, obs_traj=traj, frame_skip=frame_skip)
        self.score_env = TankTroubleGym(seed=0, terminal_mode="score",
                                        obs_traj=traj, frame_skip=frame_skip)

    def reset(self):
        pass

    def play_round(self, seed, max_frames=TRUNCATE_FRAMES):
        """模型策略用 gym 环境跑 (保证观测与训练完全一致)"""
        env = self.env
        env._base_seed = seed
        env._episode = 0
        obs, _ = env.reset()
        suicide = False
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                result = info.get("result", "draw")
                return result, env._frames, suicide


def evaluate(policy, n=200, base_seed=910000, verbose=True):
    wins = losses = draws = suicides = 0
    total_frames = 0
    t0 = time.time()
    for i in range(n):
        seed = base_seed + i
        if isinstance(policy, ModelPolicy):
            result, frames, sui = policy.play_round(seed)
        else:
            result, frames, sui = play_round(policy, seed)
        total_frames += frames
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1
        if sui:
            suicides += 1
        if verbose and (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{n}] 胜率 {wins/(i+1)*100:.1f}% "
                  f"(胜{wins} 负{losses} 平{draws}) {el:.0f}s")
    el = time.time() - t0
    stats = {
        "policy": policy.name,
        "n": n,
        "win_rate": wins / n,
        "wins": wins, "losses": losses, "draws": draws,
        "suicides": suicides,
        "avg_frames": total_frames / n,
        "elapsed": el,
    }
    if verbose:
        print(f"\n===== {policy.name} 评估结果 ({n} 局) =====")
        print(f"  胜率: {stats['win_rate']*100:.1f}%   "
              f"胜 {wins} / 负 {losses} / 平 {draws}")
        print(f"  自杀局数: {suicides}")
        print(f"  平均局长: {stats['avg_frames']:.0f} 帧 "
              f"({stats['avg_frames']/25:.1f} 秒)")
        print(f"  用时: {el:.1f}s")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="hunter",
                    choices=["idle", "random", "hunter", "exploit", "model"])
    ap.add_argument("--model", default="training/models/best_model.zip")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=910000)
    ap.add_argument("--legacy", action="store_true",
                    help="旧单口径协议(destroy 即终局), 仅胜/负/平")
    ap.add_argument("--frame-skip", type=int, default=2,
                    help="模型决策频率 (须与训练一致)")
    args = ap.parse_args()

    if args.policy == "idle":
        policy = IdlePolicy()
    elif args.policy == "random":
        policy = RandomPolicy(seed=1)
    elif args.policy == "hunter":
        policy = HunterPolicy()
    elif args.policy == "exploit":
        policy = ExploitBot()
    else:
        policy = ModelPolicy(args.model, frame_skip=args.frame_skip)

    if args.legacy:
        evaluate(policy, n=args.n, base_seed=args.seed)
    else:
        evaluate_dual(policy, n=args.n, base_seed=args.seed)


if __name__ == "__main__":
    main()
