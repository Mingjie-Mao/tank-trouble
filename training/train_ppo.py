"""
PPO 训练脚本 — 训练智能体击败 Laika

用法:
  python training/train_ppo.py                       # 默认 1000 万步
  python training/train_ppo.py --steps 50000000      # 长训
  python training/train_ppo.py --resume training/models/latest.zip

监控:
  tensorboard --logdir training/tb_logs
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback, CheckpointCallback)

from training.tt_gym_env import TankTroubleGym, make_env, GAMMA  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_logs")


class WinRateCallback(BaseCallback):
    """每 rollout 统计训练局的胜/负/平, 写入 TensorBoard;
    定期做确定性评估并保存最优模型。"""

    def __init__(self, eval_every_steps=500_000, eval_rounds=100):
        super().__init__()
        self.results = []
        self.eval_every = eval_every_steps
        self.eval_rounds = eval_rounds
        self._next_eval = eval_every_steps
        self.best_win_rate = -1.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            r = info.get("result")
            if r:
                self.results.append(r)
        return True

    def _on_rollout_end(self) -> None:
        if self.results:
            recent = self.results[-400:]
            n = len(recent)
            self.logger.record("battle/win_rate",
                               recent.count("win") / n)
            self.logger.record("battle/loss_rate",
                               recent.count("loss") / n)
            self.logger.record("battle/draw_rate",
                               recent.count("draw") / n)
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            wr = self._deterministic_eval()
            self.logger.record("battle/eval_win_rate", wr)
            print(f"\n[eval @ {self.num_timesteps:,} steps] "
                  f"确定性评估胜率: {wr*100:.1f}%")
            if wr > self.best_win_rate:
                self.best_win_rate = wr
                path = os.path.join(MODELS_DIR, "best_model")
                self.model.save(path)
                print(f"  ✓ 新最优, 已保存 {path}.zip")

    def _deterministic_eval(self):
        env = TankTroubleGym(seed=880000)
        wins = 0
        for i in range(self.eval_rounds):
            env._base_seed = 880000 + i
            env._episode = 0
            obs, _ = env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    if info.get("result") == "win":
                        wins += 1
                    break
        return wins / self.eval_rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reward-version", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--tag", default="v2", help="TensorBoard 运行名前缀")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    vec_env = SubprocVecEnv([
        make_env(i, base_seed=1, reward_version=args.reward_version)
        for i in range(args.envs)])
    print(f"奖励版本: v{args.reward_version}   并行环境: {args.envs}")

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, device="cpu")
        print(f"从 {args.resume} 继续训练")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=512,               # 每环境每 rollout 步数
            batch_size=2048,
            n_epochs=6,
            gamma=GAMMA,               # 0.995, 与塑形一致
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=TB_DIR,
            device="cpu",              # MLP 小网络, CPU 快于 GPU 搬运
            verbose=1,
        )

    callbacks = [
        WinRateCallback(eval_every_steps=500_000, eval_rounds=100),
        CheckpointCallback(save_freq=max(1_000_000 // args.envs, 1),
                           save_path=MODELS_DIR,
                           name_prefix="ppo_tt"),
    ]

    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    progress_bar=True)
    except KeyboardInterrupt:
        print("\n[中断] 保存当前模型…")
    model.save(os.path.join(MODELS_DIR, "latest"))
    print(f"完成. 模型已保存到 {MODELS_DIR}/latest.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
