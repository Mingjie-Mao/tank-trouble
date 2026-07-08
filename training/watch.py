"""
回放器 — 在 tkinter 渲染器里观看策略对战 Laika

用法:
  python training/watch.py --policy hunter          # 观看手写猎杀脚本
  python training/watch.py --policy model           # 观看训练好的模型
  python training/watch.py --policy model --model training/models/best_model.zip
  python training/watch.py --policy hunter --seed 910007   # 复现指定局
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from play_tank_trouble import App  # noqa: E402
from training.baselines import IdlePolicy, RandomPolicy, HunterPolicy  # noqa: E402


class PolicyApp(App):
    """让策略接管 tank0 的渲染窗口 (R 键换局仍可用)"""

    def __init__(self, policy, seed=None, model_env=None, model=None,
                 self_harm_immune=None):
        self.policy = policy
        self.model_env = model_env    # ModelPolicy 用: 独立观测环境
        self.model = model
        super().__init__(seed=seed, two_players=False,
                         self_harm_immune=self_harm_immune)
        tag = " [Laika免疫自伤]" if self_harm_immune else ""
        self.root.title(f"Tank Trouble — {policy_name(policy)} vs Laika{tag}")

    def _tick(self):
        g = self.game
        if self.model is not None:
            # 模型策略: 用训练观测编码器
            self.model_env.game = g
            self.model_env._build_wall_boxes()
            if not hasattr(self, "_wframes"):
                self._wframes = 0
            self.model_env._frames = self._wframes
            obs = self.model_env._obs()
            action, _ = self.model.predict(obs, deterministic=True)
            t0 = g.tanks[0]
            t0.forward = int(action[0]) == 2
            t0.backup = int(action[0]) == 0
            t0.turn_left = int(action[1]) == 0
            t0.turn_right = int(action[1]) == 2
            t0.fire = int(action[2]) == 1
            self._wframes += 1
        else:
            inp = self.policy.act(g)
            t0 = g.tanks[0]
            t0.forward = bool(inp.get("forward", False))
            t0.backup = bool(inp.get("backup", False))
            t0.turn_left = bool(inp.get("turn_left", False))
            t0.turn_right = bool(inp.get("turn_right", False))
            t0.fire = bool(inp.get("fire", False))
        g.step()
        self._draw()
        self.root.after(40, self._tick)   # 25 FPS


def policy_name(p):
    return getattr(p, "name", type(p).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="hunter",
                    choices=["idle", "random", "hunter", "model",
                             "hybrid", "mpc", "scorenet"])
    ap.add_argument("--model", default="training/models/best_model.zip")
    ap.add_argument("--net", default=None,
                    help="scorenet 权重路径 (默认: scorenet_best 现任冠军, "
                         "无则回退 p21b_scorenet)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--k", type=int, default=5,
                    help="混合体候选数 (越大越强越慢)")
    ap.add_argument("--immune", action="store_true",
                    help="对手(Laika)对自己的子弹免疫, 不再神风自杀")
    args = ap.parse_args()

    model = model_env = None
    if args.policy == "idle":
        policy = IdlePolicy()
    elif args.policy == "random":
        policy = RandomPolicy(seed=1)
    elif args.policy == "hunter":
        policy = HunterPolicy()
    elif args.policy == "hybrid":
        from training.hybrid_agent import HybridPolicy
        policy = HybridPolicy(k=args.k)
        policy.name = "hybrid"
    elif args.policy == "mpc":
        from training.mpc_agent import MPCPolicy
        policy = MPCPolicy("L2", horizon=48, hold=16, n_samples=1)
        policy.name = "mpc"
    elif args.policy == "scorenet":
        from training.score_distill import ScoreNetPolicy
        net_path = args.net
        if net_path is None:
            best = "training/models/scorenet_best.pt"
            net_path = best if os.path.exists(best) else \
                "training/models/p21b_scorenet.pt"
        policy = ScoreNetPolicy(net_path)
        policy.name = f"scorenet ({os.path.basename(net_path)})"
    else:
        import gymnasium as _g
        from stable_baselines3 import PPO
        from training.tt_gym_env import TankTroubleGym, obs_dim
        model = PPO.load(args.model, device="cpu")
        # 与 evaluate.ModelPolicy 一致: 按模型观测空间自动识别观测版本
        space = model.observation_space
        obs_map = isinstance(space, _g.spaces.Dict)
        dim = (space["vec"] if obs_map else space).shape[0]
        traj, nav = next(
            (t, v) for t in (True, False) for v in (True, False)
            if obs_dim(t, v) == dim)
        model_env = TankTroubleGym(seed=0, obs_traj=traj, obs_nav=nav,
                                   obs_map=obs_map)
        policy = IdlePolicy()   # 占位, 实际由 model 控制
        policy.name = "model"

    immune = {1} if args.immune else None
    app = PolicyApp(policy, seed=args.seed, model_env=model_env, model=model,
                    self_harm_immune=immune)
    app.run()


if __name__ == "__main__":
    main()
