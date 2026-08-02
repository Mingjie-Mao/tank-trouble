"""
P25v3 fire-gate probe.

This script keeps the P25v2 champion frozen and evaluates a post-network fire
gate over the existing 18-action score head. It is intentionally inference-only:
no data is modified and no model file is overwritten.

Usage:

  python3 training/opportunity_distill_v3.py gate-sweep \
    --net training/models/p25v2_opportunity_best.pt \
    --n 200 --seed 973000 --workers 8 \
    --margins 0 0.02 0.05 0.08 0.12 0.16
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.opportunity_distill import OBS_DIM  # noqa: E402
from training.opportunity_distill_v2 import (  # noqa: E402
    BEST_NET,
    FIRE_READY_THRESHOLD,
    _load_network,
    build_observation,
    ready_line,
)
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402


def margin_gated_action(scores, metrics, candidates, margin,
                        ready_threshold=FIRE_READY_THRESHOLD):
    """Choose P25v2 movement, but require extra score margin before firing."""
    paired = np.asarray(scores, dtype=np.float32).reshape(9, 2)
    movement = int(np.max(paired, axis=1).argmax())
    fire_advantage = float(paired[movement, 1] - paired[movement, 0])
    fire = int(fire_advantage > margin
               and ready_line(metrics) >= ready_threshold)
    return candidates[movement * 2 + fire]


class FireGatedOpportunityPolicy:
    name = "p25v3_firegate"

    def __init__(self, net_path=BEST_NET, margin=0.0,
                 ready_threshold=FIRE_READY_THRESHOLD):
        import torch
        from training.mpc_agent import CANDIDATES
        from training.tt_gym_env import TankTroubleGym

        self.torch = torch
        self.candidates = CANDIDATES
        self.margin = float(margin)
        self.ready_threshold = float(ready_threshold)
        self.network, input_dim = _load_network(net_path)
        if input_dim != OBS_DIM:
            raise ValueError(f"P25v3 fire gate expects {OBS_DIM} dims, got {input_dim}")
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
        throttle, turn, fire = margin_gated_action(
            scores, metrics, self.candidates, self.margin,
            self.ready_threshold)
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def _eval_worker(job):
    worker, net_path, margin, ready_threshold, seed, count = job
    import torch
    torch.set_num_threads(1)
    from training.evaluate import play_round_dual_engine

    policy = FireGatedOpportunityPolicy(net_path, margin, ready_threshold)
    return [play_round_dual_engine(policy, seed + index)
            for index in range(count)]


def _summarize(rounds):
    total = len(rounds)

    def count(key):
        return sum(result["true_result"] == key for result in rounds)

    shots = sum(result["shots"] for result in rounds)
    kills = sum(result["kills"] for result in rounds)
    frames = sum(result["frames"] for result in rounds)
    return {
        "n": total,
        "win": count("win") / total,
        "loss": count("loss") / total,
        "double_death": count("double_death") / total,
        "draw": count("draw") / total,
        "shots_per_game": shots / total,
        "hit_rate": kills / max(shots, 1),
        "avg_seconds": frames / total / 25,
    }


def evaluate(net_path, n, seed, workers, margin,
             ready_threshold=FIRE_READY_THRESHOLD):
    base, remainder = divmod(n, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((worker, net_path, margin, ready_threshold,
                         seed + offset, count))
            offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rounds = [item for part in pool.map(_eval_worker, jobs)
                  for item in part]
    stats = _summarize(rounds)
    print(f"===== P25v3-firegate {os.path.basename(net_path)} "
          f"margin={margin:g} ready={ready_threshold:g} "
          f"{stats['n']}局 @{seed} ({time.time()-started:.0f}s) =====",
          flush=True)
    print(f"  真胜率 {stats['win']:.1%}  负 {stats['loss']:.1%}  "
          f"双亡 {stats['double_death']:.1%}  平 {stats['draw']:.1%}",
          flush=True)
    print(f"  场均开火 {stats['shots_per_game']:.1f}  "
          f"命中率 {stats['hit_rate']:.1%}  "
          f"平均局长 {stats['avg_seconds']:.1f}s",
          flush=True)
    return stats


def gate_sweep(net_path, n, seed, workers, margins, ready_threshold):
    print("===== P25v3 fire-gate sweep =====", flush=True)
    print(f"  net={net_path}", flush=True)
    print(f"  n={n} seed={seed} workers={workers} "
          f"ready_threshold={ready_threshold:g}", flush=True)
    results = []
    for margin in margins:
        stats = evaluate(net_path, n, seed, workers, margin, ready_threshold)
        stats["margin"] = margin
        results.append(stats)

    print("===== sweep summary =====", flush=True)
    print("margin win loss dd draw shots hit avg_s", flush=True)
    for stats in results:
        print(f"{stats['margin']:g} "
              f"{stats['win']:.1%} {stats['loss']:.1%} "
              f"{stats['double_death']:.1%} {stats['draw']:.1%} "
              f"{stats['shots_per_game']:.1f} {stats['hit_rate']:.1%} "
              f"{stats['avg_seconds']:.1f}",
              flush=True)
    best = max(results, key=lambda item: (
        item["win"], -item["double_death"], -item["shots_per_game"]))
    print(f"===== best margin={best['margin']:g} "
          f"win={best['win']:.1%} dd={best['double_death']:.1%} "
          f"shots={best['shots_per_game']:.1f} hit={best['hit_rate']:.1%} =====",
          flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["eval", "gate-sweep"])
    parser.add_argument("--net", default=BEST_NET)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=973000)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--margins", type=float, nargs="+",
                        default=[0.0, 0.02, 0.05, 0.08, 0.12, 0.16])
    parser.add_argument("--ready-threshold", type=float,
                        default=FIRE_READY_THRESHOLD)
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate(args.net, args.n, args.seed, args.workers,
                 args.margin, args.ready_threshold)
    else:
        gate_sweep(args.net, args.n, args.seed, args.workers,
                   args.margins, args.ready_threshold)


if __name__ == "__main__":
    main()
