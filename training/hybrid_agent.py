"""
混合体 (P20): P17 网络先验剪枝 + MPC 深推验证

每决策步:
  1. P17 冠军网络读 125 维观测, 给 18 个动作组合算联合概率 (0.3ms)
  2. 取概率前 K 名候选 (默认 5)
  3. MPC 沙盒只对这 K 个候选做 48 帧深推演 (~25ms, 全量 18 个要 91ms)
  4. 推演得分最高者执行
公平性与 MPC 相同 (无未来函数); 网络只是加速器, 不引入新信息。

用法:
  python3 training/hybrid_agent.py --n 100 --k 5
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.mpc_agent import make_sandbox, rollout, CANDIDATES  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class HybridPolicy:
    """网络剪枝 + MPC 验证"""
    name = "hybrid"

    def __init__(self, model_path=None, k=5, opp_model="L2",
                 horizon=48, hold=16, seed=0):
        import torch
        from stable_baselines3 import PPO
        from training.tt_gym_env import TankTroubleGym
        self._torch = torch
        self.model = PPO.load(
            model_path or os.path.join(MODELS_DIR, "p17_nav_best.zip"),
            device="cpu")
        self.k = k
        self.opp_model = opp_model
        self.horizon = horizon
        self.hold = hold
        import random
        self.rng = random.Random(seed)
        # 观测编码器 (绑定到真局, 只读)
        self._env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
        self._g = None
        self._frames = 0

    def reset(self):
        self._g = None
        self._frames = 0

    def _top_candidates(self, game):
        env = self._env
        if game is not self._g:
            env.game = game
            env._build_wall_boxes()
            self._g = game
            self._frames = 0
        env._frames = self._frames
        env._prev_phi = env._phi()     # 保持路径距离特征新鲜
        obs = env._obs()
        with self._torch.no_grad():
            dist = self.model.policy.get_distribution(
                self._torch.as_tensor(obs).unsqueeze(0))
            p_th, p_tu, p_f = [d.probs[0].numpy()
                               for d in dist.distribution]
        scored = [(p_th[th] * p_tu[tu] * p_f[f], (th, tu, f))
                  for (th, tu, f) in CANDIDATES]
        scored.sort(key=lambda x: -x[0])
        return [a for _, a in scored[:self.k]]

    def act(self, game):
        me = game.tanks[0]
        if not me.alive:
            return {}
        self._frames += 1
        best_a, best_s = (1, 1, 0), -1e18
        for a in self._top_candidates(game):
            sb = make_sandbox(game, self.opp_model,
                              rng_seed=self.rng.randrange(1 << 30))
            s = rollout(sb, a, self.hold, self.horizon)
            if s > best_s:
                best_s, best_a = s, a
        th, tu, f = best_a
        return {"forward": th == 2, "backup": th == 0,
                "turn_left": tu == 0, "turn_right": tu == 2,
                "fire": f == 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=970000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--opp-model", choices=["L1", "L2"], default="L2")
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--hold", type=int, default=16)
    args = ap.parse_args()

    from training.evaluate import play_round_dual_engine
    policy = HybridPolicy(args.model, k=args.k, opp_model=args.opp_model,
                          horizon=args.horizon, hold=args.hold)
    results = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    t0 = time.time()
    for i in range(args.n):
        r = play_round_dual_engine(policy, args.seed + i)
        results[r["true_result"]] = results.get(r["true_result"], 0) + 1
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{args.n}] 真胜率 {results['win']/(i+1):.1%} "
                  f"(负{results['loss']} 双亡{results['double_death']} "
                  f"平{results['draw']}) {el:.0f}s "
                  f"({el/(i+1):.1f}s/局)", flush=True)
    n = args.n
    el = time.time() - t0
    print(f"\n===== 混合体(K={args.k}, H={args.horizon}) {n} 局 @{args.seed} =====")
    print(f"  真胜率 {results['win']/n:.1%}  负 {results['loss']/n:.1%}  "
          f"双亡 {results['double_death']/n:.1%}  平 {results['draw']/n:.1%}")
    print(f"  用时 {el:.0f}s ({el/n:.1f}s/局)")
    print(f"  参照: 全量 MPC = 96.0% (21.6s/局) | P17 网络 = 36.4%")


if __name__ == "__main__":
    main()
