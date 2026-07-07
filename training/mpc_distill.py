"""
P19 蒸馏管线: MPC 老师 -> 神经网络学生

理论依据: MPC 自身近乎无记忆 (只依当前可观测状态推演), 老师本身就是
"观测->动作"的函数 — 与 P15 克隆有状态 Laika 失败的情形相反,
蒸馏保真率预期高得多。

流程:
  1. 采集: 多进程并行, MPC(95%) 打局, 每决策步记录 (125维观测, 老师动作);
     执行侧以 epsilon 概率注入随机动作 (DAgger式: 拓宽状态覆盖,
     标签始终是老师对当前状态的选择)
  2. 训练: 复用 bc_laika.train_bc (监督三头交叉熵)
  3. 定级在链脚本中进行

用法:
  python3 training/mpc_distill.py --rounds 3000 --workers 10 --epochs 12
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
                        "distill_data")


def _worker(job):
    """单进程采集: 独立种子段, MPC 驱动, score 模式跑完整回合(含双亡窗口)"""
    (wid, n_rounds, seed0, eps, horizon, hold, samples) = job
    from training.tt_gym_env import TankTroubleGym
    from training.mpc_agent import MPCPolicy

    env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True,
                         terminal_mode="score")
    pol = MPCPolicy("L2", horizon=horizon, hold=hold, n_samples=samples,
                    seed=wid * 7919 + 13)
    rng = random.Random(wid * 104729 + 1)
    xs, ys = [], []
    stats = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    for r in range(n_rounds):
        env._base_seed = seed0 + r
        env._episode = 0
        obs, _ = env.reset()
        while True:
            a = pol.act(env.game)          # 老师对当前状态的选择 = 标签
            label = (2 if a.get("forward") else (0 if a.get("backup") else 1),
                     0 if a.get("turn_left") else (2 if a.get("turn_right") else 1),
                     1 if a.get("fire") else 0)
            xs.append(obs.copy())
            ys.append(label)
            if rng.random() < eps:         # 执行侧探索噪声 (标签不变)
                act = np.array([rng.randrange(3), rng.randrange(3),
                                rng.randrange(2)])
            else:
                act = np.array(label)
            obs, _r, term, trunc, info = env.step(act)
            if term or trunc:
                stats[info.get("result", "draw")] = (
                    stats.get(info.get("result", "draw"), 0) + 1)
                break
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    path = os.path.join(DATA_DIR, f"shard_{wid}.npz")
    np.savez_compressed(path, X=x, Y=y)
    return path, stats, len(xs)


def collect_parallel(total_rounds, workers, eps, horizon, hold, samples,
                     seed_base=5_000_000):
    os.makedirs(DATA_DIR, exist_ok=True)
    per = total_rounds // workers
    jobs = [(w, per, seed_base + w * per, eps, horizon, hold, samples)
            for w in range(workers)]
    t0 = time.time()
    print(f"采集: {workers} 进程 x {per} 局 (eps={eps}, H={horizon})",
          flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker, jobs)
    total_stats = {}
    paths, total_n = [], 0
    for path, stats, n in results:
        paths.append(path)
        total_n += n
        for k, v in stats.items():
            total_stats[k] = total_stats.get(k, 0) + v
    played = sum(total_stats.values())
    print(f"采集完成: {total_n} 样本 / {played} 局 / "
          f"{time.time()-t0:.0f}s", flush=True)
    print(f"  老师战绩(含{eps:.0%}执行噪声): "
          f"胜 {total_stats.get('win',0)/max(played,1):.1%} "
          f"双亡 {total_stats.get('double_death',0)/max(played,1):.1%} "
          f"负 {total_stats.get('loss',0)/max(played,1):.1%}", flush=True)
    xs = []
    ys = []
    for p in paths:
        d = np.load(p)
        xs.append(d["X"])
        ys.append(d["Y"])
    return np.concatenate(xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--hold", type=int, default=16)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(MODELS_DIR,
                                                  "p19_mpc_student"))
    args = ap.parse_args()

    print("===== [P19-1] MPC 示范采集 =====", flush=True)
    X, Y = collect_parallel(args.rounds, args.workers, args.epsilon,
                            args.horizon, args.hold, args.samples)
    # 标签分布
    for name, col, n_cls in (("油门", 0, 3), ("转向", 1, 3), ("开火", 2, 2)):
        dist = np.bincount(Y[:, col], minlength=n_cls) / len(Y)
        print(f"  {name}分布: {np.round(dist, 3)}", flush=True)

    print(f"===== [P19-2] 蒸馏训练 ({args.epochs} epochs) =====", flush=True)
    from training.bc_laika import train_bc
    model = train_bc(X, None, Y, epochs=args.epochs, obs_nav=True)
    model.save(args.out)
    print(f"===== [P19-3] 学生已保存 {args.out}.zip =====", flush=True)


if __name__ == "__main__":
    main()
