"""
P25v2 完整流水线：360 度机会老师 -> 独立蒸馏 -> 多轮 DAgger 纠正。

数据与 P25v1 完全隔离。流程先采老师分布和 P22 冠军分布，从零训练首个
P25v2 学生，再让当前最优 P25v2 学生自己上场两轮，由同一 360 度老师重标。
每轮在同一评测门种子上比较，最终回到原版规则做独立验收。

用法：

  python3 training/opportunity_distill_v2.py pipeline \
    --teacher-rounds 128 --bootstrap-rounds 128 \
    --dagger-rounds 2 --rounds-per-dagger 128 --workers 8
"""

import argparse
import glob
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.opportunity_distill import (
    FULL_OBS_DIM, OBS_DIM, SCORE_SCALE, bind_env, opportunity_rollout)
from training.opportunity_teacher_v2 import (
    ALIGNMENT_FLOOR, OpportunityAnalyzer360)
from training.score_distill import build_net, full_obs, train

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "opportunity_data_v2")
MODELS_DIR = os.path.join(HERE, "models")
BEST_NET = os.path.join(MODELS_DIR, "p25v2_opportunity_best.pt")
P22_NET = os.path.join(MODELS_DIR, "scorenet_best.pt")
FIRE_READY_THRESHOLD = 0.58


def build_observation(env, game, analyzer, frames):
    bind_env(env, game, frames)
    metrics = analyzer.metrics(game)
    previews = analyzer.action_previews(game, metrics)
    return np.concatenate([full_obs(env), metrics, previews]), metrics


def ready_line(metrics):
    line = float(metrics[0])
    angle_error = abs(np.arctan2(float(metrics[4]), float(metrics[3])))
    alignment = 1.0 - min(angle_error / np.pi, 1.0)
    return line * (ALIGNMENT_FLOOR
                   + (1.0 - ALIGNMENT_FLOOR) * alignment)


def gated_action(scores, metrics, candidates):
    paired = np.asarray(scores).reshape(9, 2)
    movement = int(np.max(paired, axis=1).argmax())
    fire_advantage = float(paired[movement, 1] - paired[movement, 0])
    fire = int(fire_advantage > 0.0
               and ready_line(metrics) >= FIRE_READY_THRESHOLD)
    return candidates[movement * 2 + fire]


def _load_network(path):
    import torch
    payload = torch.load(path, weights_only=True)
    input_dim = payload.get("in_dim", FULL_OBS_DIM)
    network = build_net(input_dim)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return network, input_dim


def _collect_worker(job):
    phase, worker, rounds, seed0, epsilon, actor_kind, actor_path = job
    import torch
    torch.set_num_threads(1)
    from training.mpc_agent import CANDIDATES, make_sandbox
    from training.tt_gym_env import TankTroubleGym

    actor = actor_dim = None
    if actor_kind != "teacher":
        actor, actor_dim = _load_network(actor_path)
    env = TankTroubleGym(seed=0, reward_version=1, terminal_mode="score",
                         obs_traj=True, obs_nav=True)
    rng = random.Random(worker * 104729 + seed0 + 173)
    observations, labels = [], []
    stats = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    regret, lethal, decisions = 0.0, 0, 0
    for episode in range(rounds):
        env._base_seed = seed0 + episode
        env._episode = 0
        env.reset()
        analyzer = OpportunityAnalyzer360(env.game)
        while True:
            observation, metrics = build_observation(
                env, env.game, analyzer, env._frames)
            step_seed = rng.randrange(1 << 30)
            scores = np.empty(len(CANDIDATES), dtype=np.float32)
            for index, action in enumerate(CANDIDATES):
                sandbox = make_sandbox(env.game, "L2", rng_seed=step_seed)
                scores[index] = opportunity_rollout(
                    sandbox, action, analyzer, metrics)
            observations.append(observation)
            labels.append(scores / SCORE_SCALE)
            if actor_kind == "teacher":
                chosen = int(scores.argmax())
                action = CANDIDATES[chosen]
            else:
                actor_input = observation[:actor_dim]
                with torch.no_grad():
                    predicted = actor(
                        torch.as_tensor(actor_input).unsqueeze(0))[0].numpy()
                if actor_dim == OBS_DIM:
                    action = gated_action(predicted, metrics, CANDIDATES)
                else:
                    action = CANDIDATES[int(predicted.argmax())]
                chosen = CANDIDATES.index(action)
            regret += float(scores.max() - scores[chosen])
            if scores[chosen] < -500.0 and scores.max() > -500.0:
                lethal += 1
            decisions += 1
            if rng.random() < epsilon:
                action = (rng.randrange(3), rng.randrange(3), 0)
            _, _, terminated, truncated, info = env.step(np.asarray(action))
            if terminated or truncated:
                result = info.get("result", "draw")
                stats[result] = stats.get(result, 0) + 1
                break
    phase_dir = os.path.join(DATA_DIR, phase)
    os.makedirs(phase_dir, exist_ok=True)
    path = os.path.join(phase_dir, f"shard_{worker}.npz")
    np.savez_compressed(path, X=np.asarray(observations, np.float32),
                        Y=np.asarray(labels, np.float32))
    return path, stats, regret, lethal, decisions


def collect_phase(phase, rounds, workers, seed_base, actor_kind,
                  actor_path=None, epsilon=0.03):
    base, remainder = divmod(rounds, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((phase, worker, count, seed_base + offset,
                         epsilon, actor_kind, actor_path))
            offset += count
    print(f"===== {phase}: {rounds}局 actor={actor_kind} eps={epsilon:.0%} =====",
          flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_collect_worker, jobs)
    stats, regret, lethal, decisions = {}, 0.0, 0, 0
    for _, worker_stats, worker_regret, worker_lethal, worker_decisions in results:
        for key, value in worker_stats.items():
            stats[key] = stats.get(key, 0) + value
        regret += worker_regret
        lethal += worker_lethal
        decisions += worker_decisions
    total = sum(stats.values())
    print(f"  完成 {decisions}样本 / {total}局 / {time.time()-started:.0f}s",
          flush=True)
    print(f"  现场真胜率 {stats.get('win', 0)/max(total, 1):.1%}  "
          f"双亡 {stats.get('double_death', 0)/max(total, 1):.1%}  "
          f"平均老师后悔 {regret/max(decisions, 1):.1f}  "
          f"致死误判 {lethal/max(decisions, 1):.2%}", flush=True)


def load_all_data():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*", "shard_*.npz")))
    xs, ys = [], []
    for path in paths:
        data = np.load(path)
        if data["X"].shape[1] != OBS_DIM:
            raise ValueError(f"P25v2数据维度错误: {path} {data['X'].shape}")
        xs.append(data["X"])
        ys.append(data["Y"])
    if not xs:
        raise RuntimeError("没有 P25v2 数据")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    print(f"数据聚合: {len(paths)} shard -> {len(X)}样本", flush=True)
    return X, Y


def train_candidate(index, epochs):
    import torch
    X, Y = load_all_data()
    network, metrics = train(X, Y, epochs=epochs)
    path = os.path.join(MODELS_DIR,
                        f"p25v2_opportunity_iter{index:02d}.pt")
    torch.save({"state_dict": network.state_dict(), "in_dim": X.shape[1]},
               path)
    mse, top1, top3 = metrics
    print(f"候选已保存 {path}: MSE {mse:.4f} "
          f"top1 {top1:.1%} top3 {top3:.1%}", flush=True)
    return path


class OpportunityScoreNetPolicyV2:
    name = "p25v2_opportunity_scorenet"

    def __init__(self, net_path=BEST_NET):
        import torch
        from training.mpc_agent import CANDIDATES
        from training.tt_gym_env import TankTroubleGym

        self.torch = torch
        self.candidates = CANDIDATES
        self.network, input_dim = _load_network(net_path)
        if input_dim != OBS_DIM:
            raise ValueError(f"P25v2网络应为{OBS_DIM}维，实际{input_dim}")
        self.env = TankTroubleGym(seed=0, reward_version=1,
                                  obs_traj=True, obs_nav=True)
        self.game = None
        self.analyzer = None
        self.frames = 0

    def reset(self):
        self.game = None
        self.analyzer = None
        self.frames = 0

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer360(game)
            self.frames = 0
        observation, metrics = build_observation(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        with self.torch.no_grad():
            scores = self.network(
                self.torch.as_tensor(observation).unsqueeze(0))[0].numpy()
        throttle, turn, fire = gated_action(
            scores, metrics, self.candidates)
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def _eval_worker(job):
    worker, net_path, seed, count = job
    import torch
    torch.set_num_threads(1)
    from training.evaluate import play_round_dual_engine

    policy = OpportunityScoreNetPolicyV2(net_path)
    return [play_round_dual_engine(policy, seed + index)
            for index in range(count)]


def evaluate(net_path, n, seed, workers):
    base, remainder = divmod(n, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((worker, net_path, seed + offset, count))
            offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rounds = [item for part in pool.map(_eval_worker, jobs)
                  for item in part]
    total = len(rounds)
    count = lambda key: sum(result["true_result"] == key
                            for result in rounds)
    shots = sum(result["shots"] for result in rounds)
    win = count("win") / total
    print(f"===== P25v2学生 {os.path.basename(net_path)} "
          f"{total}局 @{seed} ({time.time()-started:.0f}s) =====")
    print(f"  真胜率 {win:.1%}  负 {count('loss')/total:.1%}  "
          f"双亡 {count('double_death')/total:.1%}  "
          f"平 {count('draw')/total:.1%}")
    print(f"  场均开火 {shots/total:.1f}  "
          f"命中率 {sum(result['kills'] for result in rounds)/max(shots, 1):.1%}  "
          f"平均局长 {sum(result['frames'] for result in rounds)/total/25:.1f}s")
    return win


def run_pipeline(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    collect_phase("teacher", args.teacher_rounds, args.workers, 10_000_000,
                  "teacher", epsilon=args.epsilon)
    collect_phase("bootstrap_p22", args.bootstrap_rounds, args.workers,
                  10_200_000, "p22", P22_NET, args.epsilon)

    best_win, best_path = -1.0, None
    for iteration in range(args.dagger_rounds + 1):
        candidate = train_candidate(iteration, args.epochs)
        gate_win = evaluate(candidate, args.gate_n, args.gate_seed,
                            args.workers)
        if gate_win > best_win:
            best_win, best_path = gate_win, candidate
            shutil.copyfile(candidate, BEST_NET)
            print(f"  评测门晋升: {best_win:.1%} -> {BEST_NET}", flush=True)
        else:
            print(f"  评测门不晋升: 候选{gate_win:.1%} < 最佳{best_win:.1%}",
                  flush=True)
        if iteration < args.dagger_rounds:
            collect_phase(
                f"dagger_{iteration + 1:02d}", args.rounds_per_dagger,
                args.workers, 10_400_000 + iteration * 200_000,
                "p25v2", BEST_NET, args.epsilon)

    print(f"===== 最终冠军: {best_path} 门胜率{best_win:.1%} =====", flush=True)
    evaluate(BEST_NET, args.eval_n, args.eval_seed, args.workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pipeline", "eval"])
    parser.add_argument("--teacher-rounds", type=int, default=128)
    parser.add_argument("--bootstrap-rounds", type=int, default=128)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--rounds-per-dagger", type=int, default=128)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--gate-n", type=int, default=80)
    parser.add_argument("--gate-seed", type=int, default=9_700_000)
    parser.add_argument("--eval-n", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=973000)
    parser.add_argument("--net", default=BEST_NET)
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline(args)
    else:
        evaluate(args.net, args.eval_n, args.eval_seed, args.workers)


if __name__ == "__main__":
    main()
