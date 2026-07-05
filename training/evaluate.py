"""
标准评估协议 — 所有策略统一在此测胜率

协议:
  - 固定种子序列 base_seed..base_seed+N-1, 每种子一局 (新迷宫)
  - 一局 = 新迷宫 -> 一方死亡; 超过 TRUNCATE_FRAMES 帧算平局
  - 统计: 胜 / 负 / 平, 自杀数(己方子弹击杀自己), 平均局长

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
from training.baselines import IdlePolicy, RandomPolicy, HunterPolicy  # noqa: E402
from training.tt_gym_env import TRUNCATE_FRAMES, FRAME_SKIP  # noqa: E402


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

    def __init__(self, model_path):
        from stable_baselines3 import PPO
        from training.tt_gym_env import TankTroubleGym
        self.model = PPO.load(model_path, device="cpu")
        self.env = TankTroubleGym(seed=0)

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
                    choices=["idle", "random", "hunter", "model"])
    ap.add_argument("--model", default="training/models/best_model.zip")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=910000)
    args = ap.parse_args()

    if args.policy == "idle":
        policy = IdlePolicy()
    elif args.policy == "random":
        policy = RandomPolicy(seed=1)
    elif args.policy == "hunter":
        policy = HunterPolicy()
    else:
        policy = ModelPolicy(args.model)

    evaluate(policy, n=args.n, base_seed=args.seed)


if __name__ == "__main__":
    main()
