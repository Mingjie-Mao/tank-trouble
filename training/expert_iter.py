"""
P22 专家迭代 (DAgger / Expert Iteration): 把 96% 的搜索老师内化成纯网络

循环 (一轮 = run 一次):
  1. 采集: 当前最强学生网络自己上场打 (含 5% 移动噪声), 每个决策步
     由 MPC 老师对 18 候选全量推演打分 —— 数据分布 = 学生自己会走到的局面,
     专治"学生会犯、但老师数据里没有"的错误 (6.4% 致死误判 / 9% 双亡)
  2. 聚合: 新数据 + 全部历史数据 (第0轮老师数据 + 之前各轮), 均匀混合
  3. 重训: 从零重训评分网络 (监督训练便宜, 从零避免漂移)
  4. 评测门: 新旧网络在同一批全新种子上配对对比, 赢了才晋升 scorenet_best

纪律:
  - 评测门用滚动新种子基 (8M+), 永不碰官方定级基 970000/990000 ——
    官方加冕仍需手动双基 1000+500 局 (score_distill.py eval)
  - 每轮所有产物落盘: 数据 score_data/iterNN/, 网络 models/iterNN_scorenet.pt,
    台账 score_data/iter_history.csv —— 任何一轮变差都可回滚

用法:
  python3 training/expert_iter.py run --rounds 3000 --workers 8      # 单轮
  python3 training/expert_iter.py run --count 3 --rounds 3000 --workers 60
  python3 training/expert_iter.py history                            # 看台账
"""

import argparse
import csv
import glob
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.score_distill import (  # noqa: E402
    DATA_DIR, MODELS_DIR, SCORE_SCALE, build_net, full_obs, train)

BEST_NET = os.path.join(MODELS_DIR, "scorenet_best.pt")
SEED_NET = os.path.join(MODELS_DIR, "p21b_scorenet.pt")
HISTORY = os.path.join(DATA_DIR, "iter_history.csv")
COLLECT_SEED_BASE = 7_000_000    # + 轮次*1M, 与第0轮 (6M) 和官方基隔离
EVAL_SEED_BASE = 8_000_000       # + 轮次*10_000, 每轮全新


def _load_net(path):
    import torch
    payload = torch.load(path, weights_only=True)
    in_dim = payload["in_dim"] if isinstance(payload, dict) else 408
    net = build_net(in_dim)
    net.load_state_dict(payload["state_dict"]
                        if isinstance(payload, dict) else payload)
    net.eval()
    return net


def current_student():
    """采集用的学生 = 最强网络: 晋升过的 best, 否则 P21b 种子网络"""
    return BEST_NET if os.path.exists(BEST_NET) else SEED_NET


# ================================================================ 采集

def _collect_worker(job):
    (wid, n_rounds, seed0, eps, horizon, hold, net_path, out_dir) = job
    import torch
    torch.set_num_threads(1)          # 多进程下禁止线程超订
    from training.tt_gym_env import TankTroubleGym
    from training.mpc_agent import make_sandbox, rollout, CANDIDATES

    net = _load_net(net_path)
    env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True,
                         terminal_mode="score")
    rng = random.Random(wid * 104729 + 31)
    xs, ys = [], []
    stats = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    regret_sum, lethal, decisions = 0.0, 0, 0
    for r in range(n_rounds):
        env._base_seed = seed0 + r
        env._episode = 0
        env.reset()
        while True:
            x = full_obs(env)
            # 老师标注: 同一决策步 18 候选共用一个沙盒种子 (配对比较)
            step_seed = rng.randrange(1 << 30)
            scores = np.empty(18, dtype=np.float32)
            for i, a in enumerate(CANDIDATES):
                sb = make_sandbox(env.game, "L2", rng_seed=step_seed)
                scores[i] = rollout(sb, a, hold, horizon)
            xs.append(x)
            ys.append(scores / SCORE_SCALE)
            # 学生决策 (DAgger: 数据分布跟着学生走, 标签跟着老师走)
            with torch.no_grad():
                pred = net(torch.as_tensor(x).unsqueeze(0))[0].numpy()
            pick = int(pred.argmax())
            # 残差遥测: 学生这一步在老师眼里亏多少
            regret_sum += float(scores.max() - scores[pick])
            if scores[pick] < -500 and scores.max() > -500:
                lethal += 1           # 有活路却选了死路
            decisions += 1
            if rng.random() < eps:    # 噪声只动移动维, 开火位强制 0
                act = np.array([rng.randrange(3), rng.randrange(3), 0])
            else:
                act = np.array(CANDIDATES[pick])
            _obs, _r, term, trunc, info = env.step(act)
            if term or trunc:
                k = info.get("result", "draw")
                stats[k] = stats.get(k, 0) + 1
                break
    path = os.path.join(out_dir, f"shard_{wid}.npz")
    np.savez_compressed(path, X=np.asarray(xs, np.float32),
                        Y=np.asarray(ys, np.float32))
    return stats, len(xs), regret_sum, lethal, decisions


def collect_student(iter_k, net_path, rounds, workers, eps=0.05,
                    horizon=48, hold=16):
    out_dir = os.path.join(DATA_DIR, f"iter{iter_k:02d}")
    os.makedirs(out_dir, exist_ok=True)
    per = max(1, rounds // workers)
    seed0 = COLLECT_SEED_BASE + iter_k * 1_000_000
    jobs = [(w, per, seed0 + w * per, eps, horizon, hold, net_path, out_dir)
            for w in range(workers)]
    print(f"  采集: {workers} 进程 x {per} 局, 学生={os.path.basename(net_path)}",
          flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(_collect_worker, jobs)
    stats, n_samples, regret_sum, lethal, decisions = {}, 0, 0.0, 0, 0
    for st, n, rs, le, de in results:
        for k, v in st.items():
            stats[k] = stats.get(k, 0) + v
        n_samples += n
        regret_sum += rs
        lethal += le
        decisions += de
    played = sum(stats.values())
    summary = {
        "collect_win": stats.get("win", 0) / max(played, 1),
        "collect_dd": stats.get("double_death", 0) / max(played, 1),
        "collect_regret_mean": regret_sum / max(decisions, 1),
        "collect_lethal_rate": lethal / max(decisions, 1),
    }
    print(f"  采集完成: {n_samples} 样本 / {played} 局 / "
          f"{time.time()-t0:.0f}s", flush=True)
    print(f"  学生现场体检(含{eps:.0%}噪声): "
          f"胜 {summary['collect_win']:.1%} "
          f"双亡 {summary['collect_dd']:.1%} "
          f"平均后悔 {summary['collect_regret_mean']:.1f}分 "
          f"致死误判 {summary['collect_lethal_rate']:.2%}", flush=True)
    return n_samples, summary


# ================================================================ 数据聚合

def load_all_data():
    """第0轮老师数据 + 全部迭代轮学生数据, 均匀混合 (标准 DAgger 聚合)"""
    patterns = [os.path.join(DATA_DIR, "score_shard_*.npz"),
                os.path.join(DATA_DIR, "iter*", "shard_*.npz")]
    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    xs, ys = [], []
    for p in paths:
        d = np.load(p)
        if d["X"].shape[1] != 408:
            print(f"  ! 跳过维度不符 shard: {p} ({d['X'].shape})", flush=True)
            continue
        xs.append(d["X"])
        ys.append(d["Y"])
    X, Y = np.concatenate(xs), np.concatenate(ys)
    print(f"  数据聚合: {len(paths)} shard -> {len(X)} 样本", flush=True)
    return X, Y


# ================================================================ 评测门

def _eval_worker(job):
    net_path, seed0, count = job
    import torch
    torch.set_num_threads(1)
    from training.score_distill import ScoreNetPolicy
    from training.evaluate import play_round_dual_engine
    policy = ScoreNetPolicy(net_path)
    res = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    for i in range(count):
        r = play_round_dual_engine(policy, seed0 + i)
        res[r["true_result"]] = res.get(r["true_result"], 0) + 1
    return res


def eval_net(net_path, n, seed0, workers):
    per = max(1, n // workers)
    jobs = [(net_path, seed0 + w * per, per) for w in range(workers)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(_eval_worker, jobs)
    agg = {}
    for r in results:
        for k, v in r.items():
            agg[k] = agg.get(k, 0) + v
    total = sum(agg.values())
    return {k: v / total for k, v in agg.items()}, total


# ================================================================ 台账

HIST_COLS = ["iter", "time", "rounds", "new_samples", "total_samples",
             "val_mse", "val_top1",
             "collect_win", "collect_dd", "collect_regret_mean",
             "collect_lethal_rate",
             "eval_n", "eval_seed", "new_win", "new_dd", "old_win",
             "promoted", "net"]


def append_history(row):
    new_file = not os.path.exists(HISTORY)
    with open(HISTORY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HIST_COLS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def print_history():
    if not os.path.exists(HISTORY):
        print("(还没有迭代记录)")
        return
    with open(HISTORY) as f:
        for line in f:
            print(line.rstrip())


def next_iter_index():
    dirs = glob.glob(os.path.join(DATA_DIR, "iter[0-9][0-9]"))
    if not dirs:
        return 1
    return max(int(os.path.basename(d)[4:]) for d in dirs) + 1


# ================================================================ 主流程

def run_one_iter(iter_k, args):
    import torch
    print(f"\n======== 专家迭代 第 {iter_k} 轮 ========", flush=True)
    student = args.net or current_student()

    # 1. 学生自打 + 老师重标
    print(f"---- [{iter_k}.1] 学生自打采集 ----", flush=True)
    new_samples, csum = collect_student(
        iter_k, student, args.rounds, args.workers, eps=args.eps)

    # 2+3. 聚合重训 (从零)
    print(f"---- [{iter_k}.2] 聚合重训 ----", flush=True)
    X, Y = load_all_data()
    net, (mse, top1, _top3) = train(X, Y, epochs=args.epochs)
    net_path = os.path.join(MODELS_DIR, f"iter{iter_k:02d}_scorenet.pt")
    torch.save({"state_dict": net.state_dict(), "in_dim": X.shape[1]},
               net_path)
    os.chmod(net_path, 0o644)
    print(f"  网络已保存 {net_path}", flush=True)

    # 4. 评测门: 新旧网络同种子配对对比
    print(f"---- [{iter_k}.3] 评测门 (配对新种子) ----", flush=True)
    eval_seed = EVAL_SEED_BASE + iter_k * 10_000
    t0 = time.time()
    new_res, n_eval = eval_net(net_path, args.eval_n, eval_seed, args.workers)
    old_res, _ = eval_net(student, args.eval_n, eval_seed, args.workers)
    print(f"  新网络: 胜 {new_res['win']:.1%} 双亡 "
          f"{new_res.get('double_death', 0):.1%}  |  "
          f"旧冠军: 胜 {old_res['win']:.1%}  "
          f"({n_eval}局同种子配对, {time.time()-t0:.0f}s)", flush=True)

    promoted = new_res["win"] >= old_res["win"]
    if promoted:
        shutil.copy2(net_path, BEST_NET)
        os.chmod(BEST_NET, 0o644)
        print(f"  ✅ 晋升: {os.path.basename(net_path)} -> scorenet_best.pt "
              f"({old_res['win']:.1%} -> {new_res['win']:.1%})", flush=True)
    else:
        print(f"  ❌ 不晋升: 新 {new_res['win']:.1%} < 旧 "
              f"{old_res['win']:.1%}, 数据保留, 下轮继续聚合", flush=True)

    append_history({
        "iter": iter_k, "time": time.strftime("%Y-%m-%d %H:%M"),
        "rounds": args.rounds, "new_samples": new_samples,
        "total_samples": len(X),
        "val_mse": round(mse, 5), "val_top1": round(top1, 4),
        "collect_win": round(csum["collect_win"], 4),
        "collect_dd": round(csum["collect_dd"], 4),
        "collect_regret_mean": round(csum["collect_regret_mean"], 2),
        "collect_lethal_rate": round(csum["collect_lethal_rate"], 5),
        "eval_n": n_eval, "eval_seed": eval_seed,
        "new_win": round(new_res["win"], 4),
        "new_dd": round(new_res.get("double_death", 0), 4),
        "old_win": round(old_res["win"], 4),
        "promoted": promoted, "net": os.path.basename(net_path)})
    return promoted, new_res["win"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "history"])
    ap.add_argument("--count", type=int, default=1, help="连跑几轮")
    ap.add_argument("--rounds", type=int, default=3000, help="每轮采集局数")
    ap.add_argument("--workers", type=int,
                    default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-n", type=int, default=400, help="评测门局数")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--net", default=None,
                    help="指定采集用学生 (默认: scorenet_best 或 p21b)")
    args = ap.parse_args()

    if args.mode == "history":
        print_history()
        return

    k0 = next_iter_index()
    print(f"专家迭代: 从第 {k0} 轮起连跑 {args.count} 轮, "
          f"{args.workers} 进程, 每轮 {args.rounds} 局采集 + "
          f"{args.eval_n} 局配对评测", flush=True)
    for k in range(k0, k0 + args.count):
        run_one_iter(k, args)
    print("\n======== 全部轮次完成 ========", flush=True)
    print_history()
    print("\n提醒: 官方加冕需手动双基定级 —— python3 training/score_distill.py "
          "eval --net training/models/scorenet_best.pt --n 1000 --seed 970000",
          flush=True)


if __name__ == "__main__":
    main()
