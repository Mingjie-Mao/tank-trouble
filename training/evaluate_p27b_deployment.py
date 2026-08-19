#!/usr/bin/env python3
"""Paired evaluation of frozen P27b and its action-repeat deployment."""

from __future__ import annotations

import argparse
from collections import Counter
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.deployment_policy import ActionRepeatPolicy  # noqa: E402
from training.evaluate import play_round_dual_engine  # noqa: E402
from training.p27_risk_value import P27BRiskValuePolicy  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = os.path.join(ROOT, "training/models/p26_amortized_mpc_iter05.pt")
DEFAULT_VALUE = os.path.join(ROOT, "training/models/p27b_risk_value_iter00.pt")


def make_policy(args):
    return P27BRiskValuePolicy(
        base_net=args.base_net,
        value_net=args.value_net,
        fire_margin=args.fire_margin,
    )


def _worker(job):
    seed, count, args = job
    import torch

    torch.set_num_threads(1)
    frozen = make_policy(args)
    deployed = ActionRepeatPolicy(frozen, interval=args.interval)
    pairs = []
    for index in range(count):
        round_seed = seed + index
        baseline = play_round_dual_engine(frozen, round_seed)
        repeat = play_round_dual_engine(deployed, round_seed)
        pairs.append((round_seed, baseline, repeat))
    return pairs


def summarize(rows, index):
    rounds = [row[index] for row in rows]
    results = Counter(item["true_result"] for item in rounds)
    total = len(rounds)
    return {
        "win": results["win"] / total,
        "loss": results["loss"] / total,
        "double_death": results["double_death"] / total,
        "draw": results["draw"] / total,
        "shots": sum(item["shots"] for item in rounds) / total,
        "seconds": sum(item["frames"] for item in rounds) / total / 25.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--workers", type=int,
                        default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--base-net", default=DEFAULT_BASE)
    parser.add_argument("--value-net", default=DEFAULT_VALUE)
    parser.add_argument("--fire-margin", type=float, default=0.16)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")

    workers = max(1, min(args.workers, args.n))
    base, remainder = divmod(args.n, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count:
            jobs.append((args.seed + offset, count, args))
            offset += count

    started = time.time()
    if len(jobs) == 1:
        parts = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
    rows = sorted((row for part in parts for row in part), key=lambda row: row[0])
    baseline = summarize(rows, 1)
    deployed = summarize(rows, 2)
    changed = sum(row[1]["true_result"] != row[2]["true_result"] for row in rows)

    print(f"===== P27b deployment paired eval: {args.n} games "
          f"@{args.seed} ({time.time() - started:.0f}s) =====")
    for name, stats in (("P27b every frame", baseline),
                        (f"P27b hold {args.interval}", deployed)):
        print(f"  {name:<18} win {stats['win']:.1%}  loss {stats['loss']:.1%}  "
              f"double {stats['double_death']:.1%}  draw {stats['draw']:.1%}  "
              f"shots {stats['shots']:.1f}  length {stats['seconds']:.1f}s")
    print(f"  paired outcome changed {changed}/{args.n} ({changed / args.n:.1%}); "
          f"win delta {deployed['win'] - baseline['win']:+.1%}")


if __name__ == "__main__":
    main()

