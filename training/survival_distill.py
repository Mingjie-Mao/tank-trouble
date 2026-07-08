"""
P24 蒸馏管线: 风格老师 (SurvivalMPC style) -> 评分网络

观测 = P21b 全知 408 维 + [分数池/300, 剩余时间比] = 410 维
  (池/时间是状态不是未来函数 —— 终局护分/贫困冲刺等时变策略靠这两维)
标签 = 18 候选 survival_rollout Δ池 (同决策步配对沙盒种子) / SCALE
纪律 = 采集噪声只动移动维不强制开火 (P19); 决策每 2 帧一次 (成本减半)

流程 (pipeline 一键串联):
  collect 采集 -> train 回归 -> eval 生存模式验收 -> eval-original 原版验收

用法:
  python3 training/survival_distill.py pipeline --rounds 1500 --workers 8
  python3 training/survival_distill.py eval --net training/models/p24_scorenet.pt
  python3 training/survival_distill.py eval-original --net ... --n 300
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
                        "survival_data")
SCALE = 300.0            # Δ池量纲: 命中+50 / 破产-(池+150) / 死亡-(池+200)
CTX_DIM = 2
DECIDE_EVERY = 2         # 决策间隔帧数 (间隔内保持动作)


def ctx_features(ledger, econ):
    # 钳位: 原版验收的虚拟账本可能出训练分布 (池<0 / 超时), 拉回域内
    pool = max(0.0, ledger.pool)
    remain = max(0.0, econ["cap"] - ledger.frames)
    return np.asarray([pool / 300.0, remain / econ["cap"]],
                      dtype=np.float32)


def bind_env(env, game, frames):
    """把外部 Game 绑到观测编码器 (ScoreNetPolicy 同款协议)"""
    if env.game is not game:
        env.game = game
        env._build_wall_boxes()
    env._frames = frames
    env._prev_phi = env._phi()


def build_obs(env, game, ledger, econ):
    from training.score_distill import full_obs
    bind_env(env, game, ledger.frames)
    return np.concatenate([full_obs(env), ctx_features(ledger, econ)])


def _apply(t0, th, tu, f):
    t0.forward, t0.backup = th == 2, th == 0
    t0.turn_left, t0.turn_right = tu == 0, tu == 2
    t0.fire = f == 1


# ================================================================ 采集

def _collect_worker(job):
    wid, n_games, seed0, eps = job
    from tank_trouble_original.game import Game
    from training.tt_gym_env import TankTroubleGym
    from training.mpc_agent import make_sandbox, CANDIDATES
    from training.survival_mode import Ledger, survival_rollout, ECON

    env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
    rng = random.Random(wid * 104729 + 13)
    xs, ys = [], []
    telem = dict(settle=0.0, hits=0, frames=0, stuck=0, style=0.0,
                 death=0, drain=0, cap=0)
    for r in range(n_games):
        g = Game(seed=seed0 + r, ai_enabled=True, invincible={1})
        ledger = Ledger(g)
        end = "cap"
        act = (1, 1, 0)
        while True:
            if ledger.frames % DECIDE_EVERY == 0:
                obs = build_obs(env, g, ledger, ECON)
                step_seed = rng.randrange(1 << 30)
                scores = np.empty(18, dtype=np.float32)
                for i, a in enumerate(CANDIDATES):
                    sb = make_sandbox(g, "L2", rng_seed=step_seed)
                    scores[i] = survival_rollout(
                        sb, a, ledger.pool, ledger.visited, ledger.frames)
                xs.append(obs)
                ys.append(scores / SCALE)
                best = CANDIDATES[int(scores.argmax())]
                if rng.random() < eps:   # 噪声只动移动维, 不强制开火
                    act = (rng.randrange(3), rng.randrange(3), 0)
                else:
                    act = best
            _apply(g.tanks[0], *act)
            events = g.step()
            end = ledger.on_frame(g, events)
            if end != "alive":
                break
        telem["settle"] += 0.0 if end == "death" else ledger.pool
        telem["hits"] += ledger.hits
        telem["frames"] += ledger.frames
        telem["stuck"] += ledger.stuck_frames
        telem["style"] += ledger.style
        telem[end] += 1
    path = os.path.join(DATA_DIR, f"sv_shard_{wid}.npz")
    np.savez_compressed(path, X=np.asarray(xs, np.float32),
                        Y=np.asarray(ys, np.float32))
    return path, telem, len(xs)


def collect(total_games, workers, eps=0.05, seed_base=7_000_000):
    os.makedirs(DATA_DIR, exist_ok=True)
    per = max(1, total_games // workers)
    jobs = [(w, per, seed_base + w * per, eps) for w in range(workers)]
    t0 = time.time()
    print(f"采集: {workers} 进程 x {per} 局 (决策每 {DECIDE_EVERY} 帧, "
          f"噪声 {eps:.0%})", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_collect_worker, jobs)
    xs, ys = [], []
    T = dict(settle=0.0, hits=0, frames=0, stuck=0, style=0.0,
             death=0, drain=0, cap=0)
    for path, t, n in results:
        d = np.load(path)
        xs.append(d["X"])
        ys.append(d["Y"])
        for k, v in t.items():
            T[k] += v
    X, Y = np.concatenate(xs), np.concatenate(ys)
    games = T["death"] + T["drain"] + T["cap"]
    alive_s = T["frames"] / 25.0
    print(f"采集完成: {len(X)} 样本 / {games} 局 / {time.time()-t0:.0f}s",
          flush=True)
    print(f"  老师遥测(含噪声): 结算 {T['settle']/games:.0f} "
          f"命中间隔 {alive_s/max(T['hits'],1):.1f}s "
          f"卡墙 {100.0*T['stuck']/max(T['frames'],1):.1f}% "
          f"风格速率 {T['style']/max(alive_s,1):+.2f}/s "
          f"终局 死{T['death']}/干{T['drain']}/满{T['cap']}", flush=True)
    return X, Y


# ================================================================ 学生

class SurvivalScoreNetPolicy:
    """P24 学生: 一次前向 18 评分 argmax。act_ctx 用真账本;
    act() 维护虚拟账本 (原版验收时提供池/时间特征)。"""
    name = "sv_scorenet"

    def __init__(self, net_path):
        import torch
        from training.tt_gym_env import TankTroubleGym
        from training.mpc_agent import CANDIDATES
        from training.score_distill import build_net, FULL_OBS_DIM
        from training.survival_mode import ECON
        self._torch = torch
        self._cands = CANDIDATES
        self._econ = ECON
        payload = torch.load(net_path, weights_only=True)
        in_dim = payload.get("in_dim", FULL_OBS_DIM + CTX_DIM)
        self.net = build_net(in_dim)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        self._env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
        self._g = None
        self._vledger = None

    def reset(self):
        self._g = None
        self._vledger = None

    def _decide(self, game, ledger):
        obs = build_obs(self._env, game, ledger, self._econ)
        with self._torch.no_grad():
            scores = self.net(self._torch.as_tensor(obs).unsqueeze(0))[0]
        th, tu, f = self._cands[int(scores.argmax())]
        return {"forward": th == 2, "backup": th == 0,
                "turn_left": tu == 0, "turn_right": tu == 2,
                "fire": f == 1}

    def act_ctx(self, game, ledger):
        if not game.tanks[0].alive:
            return {}
        return self._decide(game, ledger)

    def act(self, game):
        """原版验收入口: 内部虚拟账本喂池/时间特征"""
        from training.survival_mode import Ledger
        if not game.tanks[0].alive:
            return {}
        if game is not self._g:
            self._g = game
            self._vledger = Ledger(game, self._econ)
        else:
            self._vledger.on_frame(game, game.events)
        return self._decide(game, self._vledger)


# ================================================================ 评测

def _eval_worker(job):
    net_path, seed0, count = job
    from training.survival_mode import run_survival
    pol = SurvivalScoreNetPolicy(net_path)
    return [run_survival(pol, seed0 + i) for i in range(count)]


def eval_survival(net_path, n, workers, seed0=880000):
    from training.survival_mode import _agg
    per = max(1, n // workers)
    jobs = [(net_path, seed0 + w * per, per) for w in range(workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        outs = pool.map(_eval_worker, jobs)
    a = _agg([r for o in outs for r in o])
    print(f"\n===== 学生生存模式验收 ({a['n']}局, {time.time()-t0:.0f}s) "
          f"=====", flush=True)
    print(f"  结算分 {a['settle']:.1f}  存活 {a['alive_s']:.1f}s  "
          f"命中间隔 {a['hit_iv']:.1f}s  风格速率 {a['style_rate']:+.2f}/s  "
          f"卡墙 {a['stuck_pct']:.1f}%", flush=True)
    print(f"  终局: 死{a['end_death']:.0f}% 干{a['end_drain']:.0f}% "
          f"满{a['end_cap']:.0f}%  (老师参照见台账 P24v2)", flush=True)
    return a


def eval_original(net_path, n, seed=970000):
    from training.evaluate import play_round_dual_engine
    pol = SurvivalScoreNetPolicy(net_path)
    results = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    t0 = time.time()
    for i in range(n):
        pol.reset()
        r = play_round_dual_engine(pol, seed + i)
        results[r["true_result"]] = results.get(r["true_result"], 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] 真胜率 {results['win']/(i+1):.1%} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"\n===== 学生原版验收 {n} 局 @{seed} (铁律口径) =====", flush=True)
    print(f"  真胜率 {results['win']/n:.1%}  负 {results['loss']/n:.1%}  "
          f"双亡 {results['double_death']/n:.1%}  平 {results['draw']/n:.1%}",
          flush=True)
    print(f"  参照: P22 冠军 68.6/64.8 | P17 RL线 36.4 | Laika 镜像 40.2",
          flush=True)
    return results


# ================================================================ 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pipeline", "collect", "eval",
                                     "eval-original"])
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--workers", type=int,
                    default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--net", default=os.path.join(MODELS_DIR,
                                                  "p24_scorenet.pt"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=300)
    args = ap.parse_args()

    if args.mode in ("pipeline", "collect"):
        print("===== [P24-1] 生存课程采集 (风格老师) =====", flush=True)
        X, Y = collect(args.rounds, args.workers)
        if args.mode == "collect":
            return
        print("===== [P24-2] 评分回归训练 =====", flush=True)
        import torch
        from training.score_distill import train
        net, (mse, top1, top3) = train(X, Y, epochs=args.epochs)
        torch.save({"state_dict": net.state_dict(), "in_dim": X.shape[1]},
                   args.net)
        print(f"===== [P24-3] 已保存 {args.net} "
              f"(top1 {top1:.1%} top3 {top3:.1%}) =====", flush=True)
        print("===== [P24-4] 生存模式验收 =====", flush=True)
        eval_survival(args.net, args.n, args.workers)
        print("===== [P24-5] 原版验收 =====", flush=True)
        eval_original(args.net, args.eval_n)
        print("===== [P24] 管线完成 =====", flush=True)
    elif args.mode == "eval":
        eval_survival(args.net, args.n, args.workers)
    else:
        eval_original(args.net, args.eval_n)


if __name__ == "__main__":
    main()
