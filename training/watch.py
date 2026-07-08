"""
回放器 — 在 tkinter 渲染器里观看策略对战 Laika

用法:
  python training/watch.py --policy hunter          # 观看手写猎杀脚本
  python training/watch.py --policy model           # 观看训练好的模型
  python training/watch.py --policy model --model training/models/best_model.zip
  python training/watch.py --policy hunter --seed 910007   # 复现指定局
  python training/watch.py --policy survival               # P24 生存老师狩猎回放
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
                 self_harm_immune=None, invincible=None):
        self.policy = policy
        self.model_env = model_env    # ModelPolicy 用: 独立观测环境
        self.model = model
        super().__init__(seed=seed, two_players=False,
                         self_harm_immune=self_harm_immune,
                         invincible=invincible)
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


class SurvivalApp(PolicyApp):
    """P24 生存模式回放: Laika 无敌, 命中续命, 时间条归零换局。

    时间条规则与 survival_mode.run_survival 一致; 我方死亡走引擎原生
    回合循环 (new_round 事件重置时间条), 时间耗尽则整局重开。
    """

    def __init__(self, policy, seed=None):
        from training.survival_mode import DEFAULT_CFG
        self.cfg = dict(DEFAULT_CFG)
        self.clock = self.cfg["start_frames"]
        self.hits = 0
        self.round_frames = 0
        self.expire_count = -1        # >=0: 展示"时间耗尽"倒计时
        super().__init__(policy, seed=seed, invincible={1})
        self.root.title(
            f"Tank Trouble — 生存模式: {policy_name(policy)} vs 无敌Laika")

    def _tick(self):
        g = self.game
        if self.expire_count < 0:
            inp = self.policy.act(g)
            t0 = g.tanks[0]
            t0.forward = bool(inp.get("forward", False))
            t0.backup = bool(inp.get("backup", False))
            t0.turn_left = bool(inp.get("turn_left", False))
            t0.turn_right = bool(inp.get("turn_right", False))
            t0.fire = bool(inp.get("fire", False))
            events = g.step()
            self.round_frames += 1
            if g.tanks[0].alive and not g.frozen:
                self.clock -= 1
            for ev in events:
                if ev[0] == "hit" and ev[1] == 0 and ev[2] == 1:
                    self.clock += self.cfg["hit_bonus_frames"]
                    self.hits += 1
                elif ev[0] == "new_round":
                    self.clock = self.cfg["start_frames"]
                    self.hits = 0
                    self.round_frames = 0
            if self.clock <= 0 or self.round_frames >= self.cfg["cap_frames"]:
                self.expire_count = 37        # ~1.5s 提示后换局
        else:
            self.expire_count -= 1
            if self.expire_count == 0:
                from tank_trouble_original.game import Game
                self.game = Game(seed=None, ai_enabled=True, invincible={1})
                self.clock = self.cfg["start_frames"]
                self.hits = 0
                self.round_frames = 0
                self.expire_count = -1
        self._draw()
        self.root.after(40, self._tick)

    def _draw(self):
        super()._draw()
        cv = self.canvas
        secs = max(0, self.clock) / 25.0
        frac = min(1.0, max(0.0, self.clock / 1250.0))   # 满条 = 50s
        x0, y0, w, h = 12, 4, 220, 12
        cv.create_rectangle(x0, y0, x0 + w, y0 + h,
                            outline="#333333", fill="#DDDDDD")
        color = "#2E8B57" if self.clock > self.cfg["start_frames"] \
            else "#CC3322"
        cv.create_rectangle(x0, y0, x0 + w * frac, y0 + h,
                            outline="", fill=color)
        cv.create_text(x0 + w + 10, y0 + h / 2, anchor="w",
                       text=f"时间条 {secs:.1f}s   命中 {self.hits}",
                       font=("Helvetica", 12, "bold"), fill="#333333")
        if self.expire_count >= 0:
            cv.create_text(x0 + w / 2, y0 + h + 16, anchor="w",
                           text="时间耗尽 — 换局",
                           font=("Helvetica", 13, "bold"), fill="#CC3322")


def policy_name(p):
    return getattr(p, "name", type(p).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="hunter",
                    choices=["idle", "random", "hunter", "model",
                             "hybrid", "mpc", "scorenet", "survival"])
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
    elif args.policy == "survival":
        from training.survival_mode import SurvivalMPC
        policy = SurvivalMPC()
        policy.name = "生存老师(MPC)"
        app = SurvivalApp(policy, seed=args.seed)
        app.run()
        return
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
