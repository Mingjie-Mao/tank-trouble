"""
P21a 可行性探针: 评分蒸馏 (价值地形回归)

核心问题: 125 维有损观测能否预测 MPC 的 18 候选推演评分?
两个判决数字:
  - top-1 一致率: 网络 argmax == 老师 argmax 的比例 (>65% = 信息够)
  - 裸装 200 局: 贪心评分网络的真胜率 (>=50% = 路径验证)

对 P19 的三处修正:
  1. 标签 = 18 个评分 (价值地形) 而非 argmax — 打平不再是矛盾标签
  2. 同一决策步 18 个候选共用同一沙盒 RNG 种子 — 配对比较去抖动
  3. 执行噪声只动移动维, 永不强制开火 — 不再玩俄罗斯轮盘

用法:
  python3 training/score_distill.py probe --rounds 800 --workers 10
  python3 training/score_distill.py eval --net training/models/p21a_scorenet.pt --n 200
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "score_data")
SCORE_SCALE = 1000.0   # 回归目标 = 评分/1000

# ---- 观测 v2 (P21b 全知版): 125 基础 + 283 补全 = 408 维 ----
# 原则: 只给物理事实, 不给价值判断。老师评分是当前状态的纯函数,
# 故补全方向 = 当前态信息, 而非历史 (帧堆叠对该监督任务证明无增益)。
MOVE_OPTIONS = [(th, tu) for th in (0, 1, 2) for tu in (0, 1, 2)]
PREVIEW_FRAMES = 24
FULL_OBS_DIM = 125 + 18 + 24 + 240 + 1   # = 408


def action_preview(game):
    """动作条件预演 (18维): 9 种走法 x [会中弹?, 几帧后]。
    纯物理: 对手静止不补枪, 只推场上已有子弹 + 我的运动学。"""
    from training.mpc_agent import make_sandbox
    feats = np.zeros(18, dtype=np.float32)
    for i, (th, tu) in enumerate(MOVE_OPTIONS):
        sb = make_sandbox(game, "L1", rng_seed=0)
        opp = sb.tanks[1]
        opp.forward = opp.backup = opp.turn_left = opp.turn_right = False
        opp.fire = False
        me = sb.tanks[0]
        me.forward, me.backup = th == 2, th == 0
        me.turn_left, me.turn_right = tu == 0, tu == 2
        me.fire = False
        hit_t = -1
        for t in range(PREVIEW_FRAMES):
            sb.step()
            if not me.alive:
                hit_t = t
                break
        feats[i * 2] = 1.0 if hit_t >= 0 else 0.0
        feats[i * 2 + 1] = (hit_t / PREVIEW_FRAMES) if hit_t >= 0 else 1.0
    return feats


def extra_bullets(game):
    """第 7-10 颗子弹 (24维): 自车系位置/速度/归属/寿命"""
    import math as _m
    me = game.tanks[0]
    scale = game.scale
    rot = me.rotation * _m.pi / 180
    fwd = rot - _m.pi / 2
    c, s_ = _m.cos(fwd), _m.sin(fwd)
    bullets = sorted(game.bullets,
                     key=lambda b: (b.x - me.x) ** 2 + (b.y - me.y) ** 2)[6:10]
    out = np.zeros(24, dtype=np.float32)
    spf = 4.5 * (scale / 50.0)
    for k, b in enumerate(bullets):
        dx, dy = b.x - me.x, b.y - me.y
        vx, vy = b.x_speed * 7, b.y_speed * 7
        out[k * 6:(k + 1) * 6] = [
            np.clip((dx * c + dy * s_) / scale, -8, 8),
            np.clip((-dx * s_ + dy * c) / scale, -8, 8),
            (vx * c + vy * s_) / spf, (-vx * s_ + vy * c) / spf,
            1.0 if b.owner is me else -1.0, b.lifetime / 250.0]
    return out


def maze_bitmap(game):
    """全迷宫墙体位图 (240维): 12x10 x [下墙,左墙], padding=实心"""
    m = np.ones((2, 10, 12), dtype=np.float32)
    maze = game.maze
    for x in range(min(len(maze), 12)):
        for y in range(min(len(maze[0]), 10)):
            m[0, y, x] = float(maze[x][y][1])
            m[1, y, x] = float(maze[x][y][2])
    return m.ravel()


def full_obs(env):
    """408 维全知观测 = 原125 + 动作预演18 + 弹槽补全24 + 全图240 + 卡墙1"""
    g = env.game
    base = env._obs()
    stuck = np.asarray([1.0 if g.tanks[0].hit_something else 0.0],
                       dtype=np.float32)
    return np.concatenate([base, action_preview(g), extra_bullets(g),
                           maze_bitmap(g), stuck])


# ================================================================ 采集

def _worker(job):
    (wid, n_rounds, seed0, eps, horizon, hold) = job
    from training.tt_gym_env import TankTroubleGym
    from training.mpc_agent import make_sandbox, rollout, CANDIDATES

    env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True,
                         terminal_mode="score")
    rng = random.Random(wid * 104729 + 7)
    xs, ys = [], []
    stats = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    for r in range(n_rounds):
        env._base_seed = seed0 + r
        env._episode = 0
        obs, _ = env.reset()
        while True:
            # 同一决策步 18 候选共用一个沙盒种子 (配对比较)
            step_seed = rng.randrange(1 << 30)
            scores = np.empty(18, dtype=np.float32)
            for i, a in enumerate(CANDIDATES):
                sb = make_sandbox(env.game, "L2", rng_seed=step_seed)
                scores[i] = rollout(sb, a, hold, horizon)
            xs.append(full_obs(env))
            ys.append(scores / SCORE_SCALE)
            best = CANDIDATES[int(scores.argmax())]
            if rng.random() < eps:      # 噪声只动移动维, 开火位强制 0
                act = np.array([rng.randrange(3), rng.randrange(3), 0])
            else:
                act = np.array(best)
            obs, _r, term, trunc, info = env.step(act)
            if term or trunc:
                k = info.get("result", "draw")
                stats[k] = stats.get(k, 0) + 1
                break
    path = os.path.join(DATA_DIR, f"score_shard_{wid}.npz")
    np.savez_compressed(path, X=np.asarray(xs, np.float32),
                        Y=np.asarray(ys, np.float32))
    return path, stats, len(xs)


def collect(total_rounds, workers, eps=0.05, horizon=48, hold=16,
            seed_base=6_000_000):
    os.makedirs(DATA_DIR, exist_ok=True)
    per = total_rounds // workers
    jobs = [(w, per, seed_base + w * per, eps, horizon, hold)
            for w in range(workers)]
    t0 = time.time()
    print(f"采集: {workers} 进程 x {per} 局, 每决策步全量 18 候选评分",
          flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker, jobs)
    xs, ys = [], []
    stats = {}
    for path, st, n in results:
        d = np.load(path)
        xs.append(d["X"])
        ys.append(d["Y"])
        for k, v in st.items():
            stats[k] = stats.get(k, 0) + v
    played = sum(stats.values())
    X, Y = np.concatenate(xs), np.concatenate(ys)
    print(f"采集完成: {len(X)} 样本 / {played} 局 / {time.time()-t0:.0f}s",
          flush=True)
    print(f"  老师战绩(含{eps:.0%}移动噪声): "
          f"胜 {stats.get('win',0)/max(played,1):.1%} "
          f"双亡 {stats.get('double_death',0)/max(played,1):.1%}", flush=True)
    return X, Y


# ================================================================ 评分网络

def build_net(in_dim=FULL_OBS_DIM, width=1024):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, 18))


def train(X, Y, epochs=15, batch=4096, lr=3e-4, val_frac=0.05):
    import torch
    n = len(X)
    n_val = int(n * val_frac)
    perm = np.random.default_rng(0).permutation(n)
    Xv = torch.as_tensor(X[perm[:n_val]])
    Yv = torch.as_tensor(Y[perm[:n_val]])
    Xt = torch.as_tensor(X[perm[n_val:]])
    Yt = torch.as_tensor(Y[perm[n_val:]])
    net = build_net(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()

    def metrics():
        with torch.no_grad():
            pred = net(Xv)
            mse = lossf(pred, Yv).item()
            top1 = (pred.argmax(1) == Yv.argmax(1)).float().mean().item()
            # top-3: 老师最优是否在网络前三
            top3_idx = pred.topk(3, dim=1).indices
            hit3 = (top3_idx == Yv.argmax(1, keepdim=True)).any(1)
            top3 = hit3.float().mean().item()
        return mse, top1, top3

    t0 = time.time()
    n_train = len(Xt)
    for ep in range(epochs):
        order = torch.randperm(n_train)
        tot, nb = 0.0, 0
        for i in range(0, n_train, batch):
            idx = order[i:i + batch]
            loss = lossf(net(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        mse, top1, top3 = metrics()
        print(f"  epoch {ep+1}/{epochs}  训练MSE {tot/nb:.4f}  "
              f"验证MSE {mse:.4f}  top1一致 {top1:.1%}  "
              f"top3覆盖 {top3:.1%}  {time.time()-t0:.0f}s", flush=True)
    return net, (mse, top1, top3)


# ================================================================ 部署策略

class ScoreNetPolicy:
    """贪心评分网络: 一次前向 18 个评分, argmax 执行 (纯网络, 零搜索)"""
    name = "scorenet"

    def __init__(self, net_path):
        import torch
        from training.tt_gym_env import TankTroubleGym
        from training.mpc_agent import CANDIDATES
        self._torch = torch
        self._cands = CANDIDATES
        payload = torch.load(net_path, weights_only=True)
        in_dim = payload.get("in_dim", FULL_OBS_DIM) if isinstance(
            payload, dict) and "in_dim" in payload else FULL_OBS_DIM
        self.net = build_net(in_dim)
        state = payload["state_dict"] if isinstance(
            payload, dict) and "state_dict" in payload else payload
        self.net.load_state_dict(state)
        self.net.eval()
        self._env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
        self._g = None
        self._frames = 0

    def reset(self):
        self._g = None
        self._frames = 0

    def act(self, game):
        me = game.tanks[0]
        if not me.alive:
            return {}
        env = self._env
        if game is not self._g:
            env.game = game
            env._build_wall_boxes()
            self._g = game
            self._frames = 0
        self._frames += 1
        env._frames = self._frames
        env._prev_phi = env._phi()
        obs = full_obs(env)
        with self._torch.no_grad():
            scores = self.net(self._torch.as_tensor(obs).unsqueeze(0))[0]
        th, tu, f = self._cands[int(scores.argmax())]
        return {"forward": th == 2, "backup": th == 0,
                "turn_left": tu == 0, "turn_right": tu == 2,
                "fire": f == 1}


# ================================================================ 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "eval"])
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--net", default=os.path.join(MODELS_DIR,
                                                  "p21a_scorenet.pt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=970000)
    args = ap.parse_args()

    if args.mode == "probe":
        print("===== [P21a-1] 全量评分采集 =====", flush=True)
        X, Y = collect(args.rounds, args.workers)
        print("===== [P21a-2] 评分回归训练 =====", flush=True)
        import torch
        net, (mse, top1, top3) = train(X, Y, epochs=args.epochs)
        torch.save({"state_dict": net.state_dict(),
                    "in_dim": X.shape[1]}, args.net)
        print(f"===== [P21a-3] 评分网络已保存 {args.net} =====", flush=True)
        print(f"判决A: top1一致率 {top1:.1%} / top3覆盖 {top3:.1%} "
              f"(闸门: top1>65% 或 top3>85%)", flush=True)
    else:
        from training.evaluate import play_round_dual_engine
        policy = ScoreNetPolicy(args.net)
        results = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
        t0 = time.time()
        for i in range(args.n):
            r = play_round_dual_engine(policy, args.seed + i)
            results[r["true_result"]] = results.get(r["true_result"], 0) + 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{args.n}] 真胜率 "
                      f"{results['win']/(i+1):.1%} {time.time()-t0:.0f}s",
                      flush=True)
        n = args.n
        print(f"\n===== 评分网络裸装 {n} 局 @{args.seed} =====")
        print(f"  真胜率 {results['win']/n:.1%}  负 {results['loss']/n:.1%}  "
              f"双亡 {results['double_death']/n:.1%}")
        print(f"  判决B: >=50% = 路径验证通过 "
              f"(参照: P17=36.4% / 混合体K=8=95%)", flush=True)


if __name__ == "__main__":
    main()
