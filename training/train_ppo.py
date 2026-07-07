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

    def __init__(self, eval_every_steps=500_000, eval_rounds=100,
                 reward_version=2, obs_traj=False, frame_skip=2,
                 dodge_drill=False, obs_nav=False, obs_map=False):
        super().__init__()
        self.results = []
        self.eval_every = eval_every_steps
        self.eval_rounds = eval_rounds
        self.reward_version = reward_version   # 评估口径与训练目标一致
        self.obs_traj = obs_traj
        self.frame_skip = frame_skip
        self.dodge_drill = dodge_drill         # 特训营: 指标 = 存活率(draw)
        self.obs_nav = obs_nav
        self.obs_map = obs_map
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
        env = TankTroubleGym(seed=880000, reward_version=self.reward_version,
                             obs_traj=self.obs_traj, frame_skip=self.frame_skip,
                             dodge_drill=self.dodge_drill,
                             obs_nav=self.obs_nav, obs_map=self.obs_map)
        # 特训营的成功指标 = 活满一局(draw); 常规训练 = 击杀获胜(win)
        target = "draw" if self.dodge_drill else "win"
        wins = 0
        for i in range(self.eval_rounds):
            env._base_seed = 880000 + i
            env._episode = 0
            obs, _ = env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    if info.get("result") == target:
                        wins += 1
                    break
        return wins / self.eval_rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reward-version", type=int, default=2,
                    choices=[1, 2, 3, 4, 5],
                    help="1 纯胜负; 2 开火塑形; 3 闪避实验; "
                         "4 真规则(原版计分终局); 5 v4+密集闪避压力(需 --obs-traj)")
    ap.add_argument("--obs-traj", action="store_true",
                    help="附加弹道预演观测 (76 -> 121 维)")
    ap.add_argument("--min-spawn-dist", type=int, default=0,
                    help="训练去偏: 重掷出生路径距离小于该格数的局 (评估不受影响)")
    ap.add_argument("--dd-penalty", type=float, default=-0.2,
                    help="双亡终局奖励 (v4+; 更负 = 更不鼓励换子)")
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="覆盖熵系数 (默认沿用 checkpoint/0.01; 精修可降至 0.003)")
    ap.add_argument("--frame-skip", type=int, default=2,
                    help="每决策重复帧数 (1 = 每帧决策, 10° 瞄准粒度)")
    ap.add_argument("--bad-shot", type=float, default=-0.15,
                    help="开火时模拟为 SUICIDE 的即时惩罚 (自伤纪律)")
    ap.add_argument("--time-escalate", action="store_true",
                    help="时间惩罚随局长递增 (压拖长局)")
    ap.add_argument("--waste-shot", type=float, default=-0.02,
                    help="空枪惩罚: 模拟 NOTHING 且不逼近敌人的一发 (P13 收紧)")
    ap.add_argument("--near-miss", type=float, default=0.075,
                    help="擦身弹奖励 (P13 收紧防泼弹)")
    ap.add_argument("--dodge-drill", action="store_true",
                    help="闪避特训营: 禁用开火, 只练走位活命 (需 --obs-traj)")
    ap.add_argument("--value-warmup", type=int, default=0,
                    help="BC 热启动用: 先冻结策略网络训练价值头 N 步, "
                         "防止随机价值头毁掉克隆先验 (需 --resume)")
    ap.add_argument("--stuck-penalty", type=float, default=0.0,
                    help="顶墙每决策步惩罚 (P16: -0.01, 教它脱困)")
    ap.add_argument("--eval-every", type=int, default=500_000,
                    help="回调评估间隔 (长训调大以省算力)")
    ap.add_argument("--obs-nav", action="store_true",
                    help="附加导航观测: 最短路下一步/下下步方向 (+4 维)")
    ap.add_argument("--obs-map", action="store_true",
                    help="附加迷宫栅格观测 + CNN 地图头 (P18)")
    ap.add_argument("--tag", default="v2", help="TensorBoard 运行名前缀")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    vec_env = SubprocVecEnv([
        make_env(i, base_seed=1, reward_version=args.reward_version,
                 obs_traj=args.obs_traj, min_spawn_cells=args.min_spawn_dist,
                 dd_reward=args.dd_penalty, frame_skip=args.frame_skip,
                 bad_shot=args.bad_shot, time_escalate=args.time_escalate,
                 waste_shot=args.waste_shot, near_miss=args.near_miss,
                 dodge_drill=args.dodge_drill,
                 stuck_penalty=args.stuck_penalty,
                 obs_nav=args.obs_nav, obs_map=args.obs_map)
        for i in range(args.envs)])
    print(f"奖励版本: v{args.reward_version}   并行环境: {args.envs}   "
          f"弹道预演观测: {'开' if args.obs_traj else '关'}   "
          f"出生去偏: {args.min_spawn_dist} 格   双亡奖励: {args.dd_penalty}   "
          f"空枪/擦弹: {args.waste_shot}/{args.near_miss}   "
          f"特训营: {'开' if args.dodge_drill else '关'}")

    if args.resume:
        # 覆盖 checkpoint 内嵌的 tensorboard 路径与学习率
        # (长训实测: 恒定 3e-4 续训只在最优附近震荡, 精修需要降 lr)
        overrides = {"learning_rate": args.lr}
        if args.ent_coef is not None:
            overrides["ent_coef"] = args.ent_coef
        model = PPO.load(args.resume, env=vec_env, device="cpu",
                         tensorboard_log=TB_DIR, custom_objects=overrides)
        print(f"从 {args.resume} 继续训练 (覆盖 {overrides})")
    else:
        policy_kwargs = dict(net_arch=[256, 256])
        policy_name = "MlpPolicy"
        if args.obs_map:
            from training.map_extractor import TankMapExtractor
            policy_name = "MultiInputPolicy"
            policy_kwargs["features_extractor_class"] = TankMapExtractor
        model = PPO(
            policy_name,
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=TB_DIR,
            device="cpu",              # 小网络, CPU 快于 GPU 搬运
            verbose=1,
        )

    callbacks = [
        WinRateCallback(eval_every_steps=args.eval_every, eval_rounds=100,
                        reward_version=args.reward_version,
                        obs_traj=args.obs_traj, frame_skip=args.frame_skip,
                        dodge_drill=args.dodge_drill, obs_nav=args.obs_nav,
                        obs_map=args.obs_map),
        CheckpointCallback(save_freq=max(1_000_000 // args.envs, 1),
                           save_path=MODELS_DIR,
                           name_prefix="ppo_tt"),
    ]

    try:
        if args.value_warmup > 0 and args.resume:
            # 价值预热: 冻结策略侧 (mlp_extractor.policy_net + action_net),
            # 只让价值头在固定策略的轨迹上学会预测回报, 再放开联合训练。
            frozen = 0
            for name, p in model.policy.named_parameters():
                if "value" not in name:
                    p.requires_grad = False
                    frozen += 1
            print(f"[价值预热] 冻结 {frozen} 个策略参数张量, "
                  f"先训价值头 {args.value_warmup:,} 步")
            model.learn(total_timesteps=args.value_warmup,
                        callback=callbacks, progress_bar=True)
            for _, p in model.policy.named_parameters():
                p.requires_grad = True
            print("[价值预热] 完成, 解冻全部参数进入联合微调")
            model.learn(total_timesteps=args.steps, callback=callbacks,
                        progress_bar=True, reset_num_timesteps=False)
        else:
            model.learn(total_timesteps=args.steps, callback=callbacks,
                        progress_bar=True)
    except KeyboardInterrupt:
        print("\n[中断] 保存当前模型…")
    model.save(os.path.join(MODELS_DIR, "latest"))
    print(f"完成. 模型已保存到 {MODELS_DIR}/latest.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
