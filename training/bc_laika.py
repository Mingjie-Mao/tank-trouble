"""
P15 — 行为克隆 Laika (imitation warmup)

依据: Laika 镜像对战真胜率 40.2% (1000局@970000) > 冠军 PPO 33.4%
    => 完美克隆的起点已超过一切已训模型, 以克隆为热启动再 RL 微调。

流程 (一条命令跑完):
  1. 采集: LaikaAI 挂到 tank0 驱动, 记录 (121维观测, 它的输入) 对
  2. 训练: 在 SB3 PPO 的策略网络上做监督学习 (三头交叉熵)
  3. 保存: p15_bc_clone.zip (标准 PPO zip, 价值头未训 — 微调时需价值预热)

注意: Laika 是有状态策略 (目标记忆+动作栈), 无记忆 MLP 克隆必然有损;
     此处目标不是完美复刻, 而是给 RL 一个"完整战士"的先验。

用法:
  python3 training/bc_laika.py --samples 800000 --epochs 12
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.tt_gym_env import TankTroubleGym  # noqa: E402
from tank_trouble_original.laika import LaikaAI  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def collect(n_samples, seed0=3_000_000, obs_nav=False, obs_map=False):
    """跑 Laika 镜像局, 从 tank0 视角记录 (观测, 动作) 对。

    动作标签 = 该决策步(2帧)结束时 tank0 的输入状态 (Laika 的动作
    多为持续数帧的驾驶/转向指令, 帧内标签噪声很小)。
    """
    env = TankTroubleGym(seed=seed0, obs_traj=True, obs_nav=obs_nav,
                         obs_map=obs_map)
    if obs_map:
        from training.tt_gym_env import MAP_C, MAP_H, MAP_W
        vec_dim = env.observation_space['vec'].shape[0]
        Xm = np.zeros((n_samples, MAP_C, MAP_H, MAP_W), dtype=np.uint8)
    else:
        vec_dim = env.observation_space.shape[0]
        Xm = None
    X = np.zeros((n_samples, vec_dim), dtype=np.float32)
    Y = np.zeros((n_samples, 3), dtype=np.int64)
    k = 0
    ep = 0
    t0 = time.time()
    noop = np.array([1, 1, 0])
    while k < n_samples:
        env._base_seed = seed0 + ep
        env._episode = 0
        ep += 1
        obs, _ = env.reset()
        env.game.tanks[0].ai = LaikaAI(env.game, env.game.tanks[0])
        while k < n_samples:
            prev_obs = obs
            obs, _r, term, trunc, _info = env.step(noop)  # 输入被内挂 AI 覆盖
            t = env.game.tanks[0]
            if obs_map:
                X[k] = prev_obs['vec']
                Xm[k] = prev_obs['map'].astype(np.uint8)
            else:
                X[k] = prev_obs
            Y[k, 0] = 2 if t.forward else (0 if t.backup else 1)
            Y[k, 1] = 0 if t.turn_left else (2 if t.turn_right else 1)
            Y[k, 2] = 1 if t.fire else 0
            k += 1
            if term or trunc:
                break
        if ep % 500 == 0:
            print(f"  采集 {k}/{n_samples} ({ep} 局, {time.time()-t0:.0f}s)",
                  flush=True)
    print(f"采集完成: {k} 样本, {ep} 局, {time.time()-t0:.0f}s", flush=True)
    # 标签分布 (克隆质量的参照)
    for name, col, n_cls in (("油门", 0, 3), ("转向", 1, 3), ("开火", 2, 2)):
        dist = np.bincount(Y[:, col], minlength=n_cls) / k
        print(f"  {name}分布: {np.round(dist, 3)}", flush=True)
    return X, Xm, Y


def train_bc(X, Xm, Y, epochs=12, batch=4096, lr=3e-4, val_frac=0.03,
             obs_nav=False, obs_map=False):
    """在新建 PPO 的策略网络上做监督 BC; 返回 PPO 模型 (价值头未训)。"""
    from stable_baselines3 import PPO

    env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=obs_nav,
                         obs_map=obs_map)
    policy_kwargs = dict(net_arch=[256, 256])
    policy_name = "MlpPolicy"
    if obs_map:
        from training.map_extractor import TankMapExtractor
        policy_name = "MultiInputPolicy"
        policy_kwargs["features_extractor_class"] = TankMapExtractor
    model = PPO(policy_name, env, device="cpu",
                policy_kwargs=policy_kwargs, verbose=0)
    policy = model.policy

    n = X.shape[0]
    n_val = int(n * val_frac)
    perm = np.random.default_rng(0).permutation(n)
    Xv = torch.as_tensor(X[perm[:n_val]])
    Yv = torch.as_tensor(Y[perm[:n_val]])
    Xt = torch.as_tensor(X[perm[n_val:]])
    Yt = torch.as_tensor(Y[perm[n_val:]])
    if obs_map:
        Xmv = torch.as_tensor(Xm[perm[:n_val]]).float()
        Xmt = torch.as_tensor(Xm[perm[n_val:]])
    n_train = Xt.shape[0]

    def make_obs(idx=None, val=False):
        if val:
            return ({'vec': Xv, 'map': Xmv} if obs_map else Xv)
        return ({'vec': Xt[idx], 'map': Xmt[idx].float()} if obs_map
                else Xt[idx])

    # 只训策略侧 (价值头留给 RL 阶段的价值预热)
    params = [p for name, p in policy.named_parameters() if "value" not in name]
    opt = torch.optim.Adam(params, lr=lr)

    def val_acc():
        with torch.no_grad():
            dist = policy.get_distribution(make_obs(val=True))
            accs = []
            for head, d in enumerate(dist.distribution):
                pred = d.probs.argmax(dim=1)
                accs.append((pred == Yv[:, head]).float().mean().item())
        return accs

    t0 = time.time()
    for epoch in range(epochs):
        order = torch.randperm(n_train)
        total = 0.0
        nb = 0
        for i in range(0, n_train, batch):
            idx = order[i:i + batch]
            dist = policy.get_distribution(make_obs(idx))
            logp = dist.log_prob(Yt[idx])
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        a = val_acc()
        print(f"  epoch {epoch+1}/{epochs}  loss {total/nb:.4f}  "
              f"验证准确率 油门 {a[0]:.1%} 转向 {a[1]:.1%} 开火 {a[2]:.1%}  "
              f"{time.time()-t0:.0f}s", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=800_000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "p15_bc_clone"))
    ap.add_argument("--obs-nav", action="store_true",
                    help="附加导航观测 (+4 维)")
    ap.add_argument("--obs-map", action="store_true",
                    help="附加迷宫栅格观测 + CNN 地图头 (P18)")
    args = ap.parse_args()

    print(f"===== [P15-1] 采集 Laika 示范 {args.samples} 样本 =====", flush=True)
    X, Xm, Y = collect(args.samples, obs_nav=args.obs_nav,
                       obs_map=args.obs_map)
    print(f"===== [P15-2] 行为克隆训练 ({args.epochs} epochs) =====", flush=True)
    model = train_bc(X, Xm, Y, epochs=args.epochs, obs_nav=args.obs_nav,
                     obs_map=args.obs_map)
    model.save(args.out)
    print(f"===== [P15-3] 克隆已保存 {args.out}.zip =====", flush=True)


if __name__ == "__main__":
    main()
