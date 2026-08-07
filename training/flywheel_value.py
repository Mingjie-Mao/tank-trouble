"""飞轮第一步：价值叶子（把 P23 重做对）。

P23 失败的两个原因，这里都修掉：

1. **泄漏**：`value_leaf.train_value` 用 `permutation(n)` 按**样本**切分，
   同一局的相邻帧同时进训练集和验证集；而迷宫位图一局内是常量，网络
   直接背下"迷宫→结局"，R² 虚高到 0.988，新图上其实是噪声。
   → 这里按**整局**切分（`G` 分组），验证集不含任何训练局的帧。

2. **单遍 = 绕当前策略的不动点**：V 学在防守型策略的对局上，"安全远位
   = 高价值"，插进搜索只会固化被动。
   → 这里把"迭代"做成一等公民：`collect --iteration N` 用**上一轮挂了
   V 的老师**去产生数据，形成闭环。第一轮才用裸老师启动。

学的是**真实胜负**，不是老师的评分——这是唯一能给系统带进新信息的通道。
拟合老师评分只是把它的信念抄一遍，环转多少圈都在原地。

叶子契约：返回 [-1, 1]，由 `density_rollout` 乘 `VALUE_LEAF_WEIGHT`
（刻意压到塑形量级，不是终局量级；见那里的注释）。
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.game import Game  # noqa: E402
from training.value_leaf import (  # noqa: E402
    VALUE_OBS_DIM, build_value_net, value_obs, _new_helper_env,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "flywheel_value_data")
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "models", "flywheel_value.pt")

# 结局 -> 价值目标，落在叶子契约的 [-1, 1] 内。
# 主动击杀并存活是满分；对手自杀只给一半（我们要奖励主动性）；
# 双亡明确为负（先杀后死不算赢，这是项目铁律）。
# 忠实于用户定的分级结局（击杀1.0 / 超时胜0.4 / 超时负0.2 / 双亡0.1 /
# 死0.0），线性映射到叶子契约的 [-1, 1]：v' = 2v - 1。
# 对手自杀取 0.7（介于击杀和超时胜之间）——赢了，但不是我打的。
OUTCOME_VALUE = {
    "active_win": 1.0,          # 1.0
    "opponent_self_win": 0.4,   # 0.7
    "timeout_ahead": -0.2,      # 0.4  ← 按链分裁决
    "timeout_behind": -0.6,     # 0.2
    "double": -0.8,             # 0.1
    "self_loss": -1.0,          # 0.0
}


def _classify(game, had_hit):
    me, enemy = game.tanks
    if me.alive and not enemy.alive:
        return "active_win" if had_hit else "opponent_self_win"
    if not me.alive and not enemy.alive:
        return "double"
    if not me.alive:
        return "self_loss"
    return "timeout"


def _selfplay_worker(job):
    """自博弈采集：老师打老师。

    对 Laika 采集的第一轮失败了——老师赢 83%，目标均值 0.646、方差 0.28，
    价值网络按整局切分后 R² 只有 +0.033（多数轮次为负）。**几乎必赢的对局
    里没有可学的价值信号。** 自博弈两边一样强，结局才接近 50/50。

    每局产出两份数据：tank0 用真实视角，tank1 用 arena.MirrorView 的
    镜像视角（所有老师都写死了 me = tanks[0]），结局互为反面。
    """
    (wid, rounds, seed0, subsample, rays, horizon, leaf_path,
     cap_frames) = job
    # 每个 worker 只用 1 个计算线程。漏掉这行会让 9 个 worker 各开 10 个
    # torch 线程去抢 10 个核（实测 load 冲到 131），而搜索本身是
    # numpy/纯 Python，那些线程一点用没有，纯粹制造上下文切换。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    from training.arena import ChainScoreboard, MirrorView, SideAdapter
    from training.killfield_prebuild import FastKillFieldTeacher

    leaf = ValueLeaf(leaf_path) if (leaf_path and
                                    os.path.exists(leaf_path)) else None
    helper = _new_helper_env()
    rows_x, rows_z, rows_g = [], [], []
    stats = {}

    def make(seed):
        return FastKillFieldTeacher(
            seed=seed, ray_count=rays, max_bounces=2, max_flight_frames=75,
            horizon=horizon, skip_masked=True, parallel_workers=0,
            leaf_fn=leaf)

    for index in range(rounds):
        seed = seed0 + index
        game = Game(seed=seed, ai_enabled=False)
        side_a = SideAdapter(make(seed ^ 0x51EED), 0)
        side_b = SideAdapter(make(seed ^ 0xA17FF), 1)
        side_a.reset()
        side_b.reset()
        mirror = MirrorView(game)
        # 30 秒到时按追猎链分裁决——不做这一步的话，自博弈约一半的局
        # 都会拿到同一个常数目标，最大的一块数据直接变成无信息样本。
        board = ChainScoreboard(game, rays=min(rays, 256))
        frames_a, frames_b = [], []
        hit_a = hit_b = False
        try:
            for frame in range(cap_frames):
                if game.frozen:
                    break
                for tank, side in ((game.tanks[0], side_a),
                                   (game.tanks[1], side_b)):
                    if not tank.alive:
                        continue
                    action = side.act(game)
                    tank.forward = bool(action.get("forward", False))
                    tank.backup = bool(action.get("backup", False))
                    tank.turn_left = bool(action.get("turn_left", False))
                    tank.turn_right = bool(action.get("turn_right", False))
                    tank.fire = bool(action.get("fire", False))
                if frame % subsample == 0:
                    if game.tanks[0].alive:
                        helper.game = game
                        helper._build_wall_boxes()
                        helper._prev_phi = helper._phi()
                        frames_a.append(value_obs(helper).copy())
                    if game.tanks[1].alive:
                        mirror.refresh()
                        helper.game = mirror
                        helper._build_wall_boxes()
                        helper._prev_phi = helper._phi()
                        frames_b.append(value_obs(helper).copy())
                for event in game.step():
                    if event[0] == "hit" and event[2] != event[1]:
                        if event[1] == 0:
                            hit_a = True
                        else:
                            hit_b = True
                board.on_frame(game)
        finally:
            side_a.close()
            side_b.close()

        if (index + 1) % 8 == 0:
            print(f"    [w{wid}] {index + 1}/{rounds} 局  "
                  f"{len(rows_x)} 样本", flush=True)
        for who, (frames, had_hit) in enumerate(
                ((frames_a, hit_a), (frames_b, hit_b))):
            outcome = _classify_side(game, who, had_hit, board.totals)
            stats[outcome] = stats.get(outcome, 0) + 1
            target = OUTCOME_VALUE[outcome]
            for obs in frames:
                rows_x.append(obs)
                rows_z.append(target)
                rows_g.append(seed)          # 同一局两方共用分组键

    path = os.path.join(DATA_DIR, f"vshard_{wid:02d}.npz")
    np.savez_compressed(
        path,
        X=np.asarray(rows_x, dtype=np.float32),
        Z=np.asarray(rows_z, dtype=np.float32),
        G=np.asarray(rows_g, dtype=np.int64))
    return path, stats, len(rows_x)


def _classify_side(game, who, had_hit, scores):
    """双方都活到 30 秒时按链分裁决，与 ranked 规则一致。"""
    me = game.tanks[who]
    enemy = game.tanks[1 - who]
    if me.alive and not enemy.alive:
        return "active_win" if had_hit else "opponent_self_win"
    if not me.alive and not enemy.alive:
        return "double"
    if not me.alive:
        return "self_loss"
    mine, theirs = scores[who], scores[1 - who]
    if mine > theirs:
        return "timeout_ahead"
    if mine < theirs:
        return "timeout_behind"
    return "timeout_ahead" if who == 0 else "timeout_behind"


def _worker(job):
    (wid, rounds, seed0, subsample, rays, horizon, leaf_path,
     cap_frames) = job
    from training.killfield_prebuild import FastKillFieldTeacher

    leaf = None
    if leaf_path and os.path.exists(leaf_path):
        leaf = ValueLeaf(leaf_path)

    helper = _new_helper_env()
    rows_x, rows_z, rows_g = [], [], []
    stats = {}
    for index in range(rounds):
        seed = seed0 + index
        game = Game(seed=seed, ai_enabled=True)
        teacher = FastKillFieldTeacher(
            seed=seed ^ 0x51EED, ray_count=rays, max_bounces=2,
            max_flight_frames=75, horizon=horizon,
            skip_masked=True, parallel_workers=0, leaf_fn=leaf)
        me = game.tanks[0]
        frames, had_hit = [], False
        for frame in range(cap_frames):
            if game.frozen:
                break
            if me.alive:
                action = teacher.act(game)
                me.forward = bool(action.get("forward", False))
                me.backup = bool(action.get("backup", False))
                me.turn_left = bool(action.get("turn_left", False))
                me.turn_right = bool(action.get("turn_right", False))
                me.fire = bool(action.get("fire", False))
            if frame % subsample == 0 and me.alive:
                helper.game = game
                helper._build_wall_boxes()
                helper._prev_phi = helper._phi()
                frames.append(value_obs(helper).copy())
            for event in game.step():
                if event[0] == "hit" and event[1] == 0 and event[2] == 1:
                    had_hit = True
        teacher.close()

        outcome = _classify(game, had_hit)
        stats[outcome] = stats.get(outcome, 0) + 1
        target = OUTCOME_VALUE[outcome]
        for obs in frames:
            rows_x.append(obs)
            rows_z.append(target)
            rows_g.append(seed)          # 局号 = 分组键

    path = os.path.join(DATA_DIR, f"vshard_{wid:02d}.npz")
    np.savez_compressed(
        path,
        X=np.asarray(rows_x, dtype=np.float32),
        Z=np.asarray(rows_z, dtype=np.float32),
        G=np.asarray(rows_g, dtype=np.int64))
    return path, stats, len(rows_x)


class ValueLeaf:
    """density_rollout 的 leaf_fn：吃沙盒状态，返回 [-1, 1]。"""

    def __init__(self, net_path=MODEL):
        import torch
        self.torch = torch
        payload = torch.load(net_path, map_location="cpu", weights_only=True)
        self.net = build_value_net(VALUE_OBS_DIM)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        self.env = _new_helper_env()
        self._maze = None

    def __call__(self, sandbox):
        if not sandbox.tanks[0].alive:
            return -1.0
        env = self.env
        env.game = sandbox
        if self._maze is not sandbox.maze:
            env._build_wall_boxes()
            self._maze = sandbox.maze
        env._prev_phi = env._phi()
        obs = value_obs(env)
        with self.torch.inference_mode():
            value = float(self.net(
                self.torch.as_tensor(obs).unsqueeze(0))[0, 0])
        return max(-1.0, min(1.0, value))


def collect(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(DATA_DIR, "vshard_*.npz")):
        os.remove(stale)
    per = max(1, args.rounds // args.workers)
    leaf_path = args.leaf if args.leaf else None
    jobs = [
        (wid, per, args.seed + wid * per * 7, args.subsample, args.rays,
         args.horizon, leaf_path, args.cap_frames)
        for wid in range(args.workers)
    ]
    worker_fn = _selfplay_worker if args.self_play else _worker
    started = time.time()
    print(f"===== 价值数据采集{'（自博弈）' if args.self_play else ''}: "
          f"{per * args.workers} 局 / "
          f"{args.workers} worker"
          f"{' / 挂载 ' + os.path.basename(leaf_path) if leaf_path else ' / 裸老师启动'}"
          f" =====", flush=True)
    total, merged = 0, {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for path, stats, count in pool.imap_unordered(worker_fn, jobs):
            total += count
            for key, value in stats.items():
                merged[key] = merged.get(key, 0) + value
            print(f"  {os.path.basename(path)}: {count} 样本", flush=True)
    print(f"===== 共 {total} 样本 / {time.time() - started:.0f}s / "
          f"结局 {merged} =====", flush=True)


def load():
    parts = sorted(glob.glob(os.path.join(DATA_DIR, "vshard_*.npz")))
    if not parts:
        raise SystemExit(f"{DATA_DIR} 里没有数据，先跑 collect")
    data = [np.load(p) for p in parts]
    return (np.concatenate([d["X"] for d in data]),
            np.concatenate([d["Z"] for d in data]),
            np.concatenate([d["G"] for d in data]))


def train(args):
    import torch

    X, Z, G = load()
    groups = np.unique(G)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * args.val_fraction))
    val_groups = set(groups[:n_val].tolist())
    is_val = np.isin(G, list(val_groups))

    # 按整局切分：验证集不含任何训练局的帧。P23 就是漏了这一步，
    # 迷宫位图一局内常量 -> R² 0.988 全是背下来的。
    Xt = torch.as_tensor(X[~is_val])
    Zt = torch.as_tensor(Z[~is_val]).unsqueeze(1)
    Xv = torch.as_tensor(X[is_val])
    Zv = torch.as_tensor(Z[is_val]).unsqueeze(1)
    print(f"===== 价值训练: {len(Xt)} 训练 / {len(Xv)} 验证 / "
          f"{len(groups)} 局（验证 {n_val} 局）=====", flush=True)

    net = build_value_net(X.shape[1])
    optimiser = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()
    baseline = float(Zv.var())      # 常数预测的 MSE ≈ 方差
    best = (float("inf"), None)
    for epoch in range(args.epochs):
        order = torch.randperm(len(Xt))
        net.train()
        for i in range(0, len(Xt), args.batch):
            index = order[i:i + args.batch]
            loss = loss_fn(net(Xt[index]), Zt[index])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        net.eval()
        with torch.inference_mode():
            val = float(loss_fn(net(Xv), Zv))
        r2 = 1.0 - val / max(baseline, 1e-9)
        marker = ""
        if val < best[0]:
            best = (val, {k: v.clone() for k, v in net.state_dict().items()})
            marker = "  <= best"
        print(f"  epoch {epoch:2d}  验证MSE {val:.4f}  "
              f"R² {r2:+.3f}{marker}", flush=True)

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    torch.save({"state_dict": best[1], "obs_dim": int(X.shape[1]),
                "val_mse": best[0], "baseline_var": baseline}, MODEL)
    print(f"===== 保存 {MODEL}  验证MSE {best[0]:.4f}  "
          f"R² {1.0 - best[0] / max(baseline, 1e-9):+.3f} =====", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "train"])
    parser.add_argument("--rounds", type=int, default=512)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--seed", type=int, default=41_000_000)
    parser.add_argument("--subsample", type=int, default=3)
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--cap-frames", type=int, default=700)
    parser.add_argument("--self-play", action="store_true",
                        help="老师打老师；对 Laika 采集结局方差太小")
    parser.add_argument("--leaf", default="",
                        help="挂载已有价值叶子做迭代采集（第一轮留空）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
