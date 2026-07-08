"""
P23 价值叶子 (Value Leaf) —— 突破"老师短视"天花板的第一步

问题 (P22 实证): 纯 DAgger 蒸馏在 ~68% 见顶, 因为 MPC 老师只看 48 帧、末端用
-0.5*路径距离 这个短视启发式。进攻/占位的回报落在视野外, 被系统性误评。

方案 (AlphaZero 同构): 把 rollout 末端启发式换成一个**学在真实胜负上**的价值
V(s)。这样 48 帧搜索能"透过 V 看到视野之外"。V 锚定真实终局 (z ∈ {胜+1/负-1/
双亡·平0}), 因此**不被老师封顶** —— 这是能超越老师的关键。

第一步只验证杠杆 (不建完整闭环): MPC(价值叶子) vs MPC(启发式叶子), 比胜率 +
进攻性 (我↔敌平均距离)。杠杆有效再谈飞轮。

用法:
  python3 training/value_leaf.py collect --games 2000 --workers 8   # 采值+训V
  python3 training/value_leaf.py validate --n 200 --workers 8       # 验证杠杆
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.score_distill import extra_bullets, maze_bitmap  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
VDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "value_data")
CHAMPION = os.path.join(MODELS_DIR, "scorenet_best.pt")
VALUE_NET = os.path.join(MODELS_DIR, "value_leaf.pt")

# 价值观测 390 维 = 基础125 + 弹槽补全24 + 全迷宫240 + 卡墙1
# 刻意**砍掉动作预演18** (那需要 9 个子沙盘 rollout): 叶子评估要便宜,
# 否则每个 rollout 叶子都套 9 个子沙盘 = 灾难性慢。V 是状态价值, 不需动作条件特征。
VALUE_OBS_DIM = 125 + 24 + 240 + 1


def value_obs(env):
    """390 维便宜状态观测。调用前需: env.game 已挂、墙盒已建、_frames=0、
    _prev_phi=_phi()。时间特征恒为 0 (训练/叶子一致, 消除 train-serve 失配)。"""
    g = env.game
    env._frames = 0                        # 时间特征归零, 两端一致
    base = env._obs()                      # 125, 纯几何无子沙盘
    stuck = np.asarray([1.0 if g.tanks[0].hit_something else 0.0],
                       dtype=np.float32)
    return np.concatenate([base, extra_bullets(g), maze_bitmap(g), stuck])


def build_value_net(in_dim=VALUE_OBS_DIM, width=512):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, 1))


def _new_helper_env():
    from training.tt_gym_env import TankTroubleGym
    return TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)


# ================================================================ 采值

def _z_of(true_result):
    return {"win": 1.0, "loss": -1.0,
            "double_death": 0.0, "draw": 0.0}.get(true_result, 0.0)


def _collect_worker(job):
    (wid, n_games, seed0, champ, subsample) = job
    import torch
    torch.set_num_threads(1)
    from tank_trouble_original.game import Game
    from training.tt_gym_env import TRUNCATE_FRAMES
    from training.score_distill import ScoreNetPolicy

    policy = ScoreNetPolicy(champ)
    helper = _new_helper_env()
    xs, zs = [], []
    stats = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    for gi in range(n_games):
        game = Game(seed=seed0 + gi, ai_enabled=True)
        policy.reset()
        helper.game = game
        helper._build_wall_boxes()
        frame_obs = []
        true_result = "draw"
        frames = 0
        while frames < TRUNCATE_FRAMES:
            if frames % subsample == 0 and game.tanks[0].alive:
                helper._prev_phi = helper._phi()
                frame_obs.append(value_obs(helper))
            inp = policy.act(game)
            t0 = game.tanks[0]
            t0.forward = bool(inp.get("forward", False))
            t0.backup = bool(inp.get("backup", False))
            t0.turn_left = bool(inp.get("turn_left", False))
            t0.turn_right = bool(inp.get("turn_right", False))
            t0.fire = bool(inp.get("fire", False))
            events = game.step()
            frames += 1
            done = False
            for ev in events:
                if ev[0] == "round_end":
                    w = ev[1]
                    true_result = ("win" if w == 0 else
                                   "loss" if w == 1 else "double_death")
                    done = True
            if done:
                break
        z = _z_of(true_result)
        stats[true_result] = stats.get(true_result, 0) + 1
        for o in frame_obs:
            xs.append(o)
            zs.append(z)
    path = os.path.join(VDATA_DIR, f"vshard_{wid}.npz")
    np.savez_compressed(path, X=np.asarray(xs, np.float32),
                        Z=np.asarray(zs, np.float32))
    return stats, len(xs)


def collect(n_games, workers, champ=CHAMPION, seed_base=9_000_000,
            subsample=3):
    os.makedirs(VDATA_DIR, exist_ok=True)
    per = max(1, n_games // workers)
    jobs = [(w, per, seed_base + w * per, champ, subsample)
            for w in range(workers)]
    print(f"采值: {workers} 进程 x {per} 局, 冠军={os.path.basename(champ)}, "
          f"每 {subsample} 帧记一次真实回报", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(_collect_worker, jobs)
    xs, zs, stats, n = [], [], {}, 0
    for path in sorted(glob.glob(os.path.join(VDATA_DIR, "vshard_*.npz"))):
        d = np.load(path)
        xs.append(d["X"])
        zs.append(d["Z"])
    for st, cnt in results:
        for k, v in st.items():
            stats[k] = stats.get(k, 0) + v
        n += cnt
    X = np.concatenate(xs)
    Z = np.concatenate(zs)
    played = sum(stats.values())
    print(f"采值完成: {len(X)} 状态样本 / {played} 局 / {time.time()-t0:.0f}s",
          flush=True)
    print(f"  冠军战绩: 胜 {stats.get('win',0)/max(played,1):.1%} "
          f"负 {stats.get('loss',0)/max(played,1):.1%} "
          f"双亡 {stats.get('double_death',0)/max(played,1):.1%}", flush=True)
    return X, Z


# ================================================================ 训 V

def train_value(X, Z, epochs=25, batch=4096, lr=3e-4, val_frac=0.05):
    import torch
    n = len(X)
    n_val = int(n * val_frac)
    perm = np.random.default_rng(0).permutation(n)
    Xv = torch.as_tensor(X[perm[:n_val]])
    Zv = torch.as_tensor(Z[perm[:n_val]]).unsqueeze(1)
    Xt = torch.as_tensor(X[perm[n_val:]])
    Zt = torch.as_tensor(Z[perm[n_val:]]).unsqueeze(1)
    net = build_value_net(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    base_var = float(Zt.var())        # 基线: 预测常数均值的 MSE ≈ var
    t0 = time.time()
    for ep in range(epochs):
        order = torch.randperm(len(Xt))
        tot, nb = 0.0, 0
        for i in range(0, len(Xt), batch):
            idx = order[i:i + batch]
            loss = lossf(net(Xt[idx]), Zt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        with torch.no_grad():
            vmse = lossf(net(Xv), Zv).item()
        r2 = 1.0 - vmse / max(base_var, 1e-9)
        print(f"  epoch {ep+1}/{epochs}  训练MSE {tot/nb:.4f}  "
              f"验证MSE {vmse:.4f}  R²≈{r2:.3f}  {time.time()-t0:.0f}s",
              flush=True)
    return net, r2


# ================================================================ 价值叶子

class ValueLeaf:
    """rollout 末端评估器: 返回 1000*clip(V(叶子态), -1, 1), 与 ±1000 终局同量纲。
    墙盒按迷宫身份缓存 (换局才重建); 叶子态无子沙盘, 便宜。"""

    def __init__(self, net_path=VALUE_NET):
        import torch
        payload = torch.load(net_path, weights_only=True)
        in_dim = payload.get("in_dim", VALUE_OBS_DIM) if isinstance(
            payload, dict) else VALUE_OBS_DIM
        self.net = build_value_net(in_dim)
        state = payload["state_dict"] if isinstance(payload, dict) else payload
        self.net.load_state_dict(state)
        self.net.eval()
        self._torch = torch
        self._env = _new_helper_env()
        self._maze = None

    def __call__(self, sandbox):
        env = self._env
        env.game = sandbox
        if self._maze is not sandbox.maze:   # 换局才重建墙盒
            env._build_wall_boxes()
            self._maze = sandbox.maze
        if not sandbox.tanks[0].alive:
            return -1000.0
        env._prev_phi = env._phi()
        obs = value_obs(env)
        with self._torch.no_grad():
            v = float(self.net(self._torch.as_tensor(obs).unsqueeze(0))[0, 0])
        return 1000.0 * max(-1.0, min(1.0, v))


# ================================================================ 验证杠杆

def _validate_worker(job):
    (net_path, seed0, count, use_value) = job
    import torch
    torch.set_num_threads(1)
    from tank_trouble_original.game import Game
    from training.tt_gym_env import TRUNCATE_FRAMES
    from training.mpc_agent import MPCPolicy
    import math

    leaf = ValueLeaf(net_path) if use_value else None
    pol = MPCPolicy("L2", horizon=48, hold=16, n_samples=1,
                    seed=1234, leaf_fn=leaf)
    res = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    dist_sum, dist_n = 0.0, 0
    for i in range(count):
        game = Game(seed=seed0 + i, ai_enabled=True)
        pol.reset()
        true_result = "draw"
        frames = 0
        while frames < TRUNCATE_FRAMES:
            inp = pol.act(game)
            t0 = game.tanks[0]
            t0.forward = bool(inp.get("forward", False))
            t0.backup = bool(inp.get("backup", False))
            t0.turn_left = bool(inp.get("turn_left", False))
            t0.turn_right = bool(inp.get("turn_right", False))
            t0.fire = bool(inp.get("fire", False))
            if game.tanks[0].alive and game.tanks[1].alive:
                dx = game.tanks[0].x - game.tanks[1].x
                dy = game.tanks[0].y - game.tanks[1].y
                dist_sum += math.hypot(dx, dy) / game.scale
                dist_n += 1
            events = game.step()
            frames += 1
            done = False
            for ev in events:
                if ev[0] == "round_end":
                    w = ev[1]
                    true_result = ("win" if w == 0 else
                                   "loss" if w == 1 else "double_death")
                    done = True
            if done:
                break
        res[true_result] = res.get(true_result, 0) + 1
    return res, dist_sum, dist_n


def validate(n, workers, net_path=VALUE_NET):
    """MPC(价值叶子) vs MPC(启发式叶子): 同种子, 比胜率 + 我↔敌平均距离。"""
    per = max(1, n // workers)
    for use_value, tag in [(False, "启发式叶子(基线)"), (True, "价值叶子")]:
        jobs = [(net_path, 8_500_000 + w * per, per, use_value)
                for w in range(workers)]
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            out = pool.map(_validate_worker, jobs)
        agg, ds, dn = {}, 0.0, 0
        for r, s, c in out:
            for k, v in r.items():
                agg[k] = agg.get(k, 0) + v
            ds += s
            dn += c
        tot = sum(agg.values())
        print(f"\n{tag}: {tot} 局同种子 ({time.time()-t0:.0f}s)")
        print(f"  真胜率 {agg.get('win',0)/tot:.1%}  "
              f"负 {agg.get('loss',0)/tot:.1%}  "
              f"双亡 {agg.get('double_death',0)/tot:.1%}")
        print(f"  我↔敌平均距离 {ds/max(dn,1):.2f} 格 (越小越主动狩猎)")


# ================================================================ 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "validate"])
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int,
                    default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--champ", default=CHAMPION)
    args = ap.parse_args()

    if args.mode == "collect":
        print("===== [P23-1] 采真实回报值 =====", flush=True)
        X, Z = collect(args.games, args.workers, champ=args.champ,
                       subsample=args.subsample)
        print("===== [P23-2] 训价值网络 V(s) =====", flush=True)
        import torch
        net, r2 = train_value(X, Z, epochs=args.epochs)
        torch.save({"state_dict": net.state_dict(), "in_dim": X.shape[1]},
                   VALUE_NET)
        os.chmod(VALUE_NET, 0o644)
        print(f"===== V 已保存 {VALUE_NET} (R²≈{r2:.3f}) =====", flush=True)
        print("下一步: python3 training/value_leaf.py validate --n 200", flush=True)
    else:
        print("===== [P23-3] 验证价值叶子杠杆 =====", flush=True)
        validate(args.n, args.workers, net_path=VALUE_NET)


if __name__ == "__main__":
    main()
