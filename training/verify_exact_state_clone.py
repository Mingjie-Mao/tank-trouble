"""Verify full-state clone fidelity across seeds and action schedules."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.game import Game  # noqa: E402
from training.exact_state import apply_controls, verify_clone_trajectory  # noqa: E402


def control_schedule(seed: int):
    rng = random.Random(seed)
    actions = []
    for _ in range(10000):
        throttle = rng.randrange(3)
        turn = rng.randrange(3)
        actions.append({
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": rng.random() < 0.08,
        })
    return lambda frame: actions[frame % len(actions)]


def verify_seed(seed: int, warmup_frames: int, verify_frames: int) -> dict:
    game = Game(seed=seed, ai_enabled=True)
    warmup = control_schedule(seed ^ 0x5A17)
    for frame in range(warmup_frames):
        apply_controls(game, warmup(frame))
        game.step()

    result = verify_clone_trajectory(
        game,
        control_schedule(seed ^ 0xA51D),
        verify_frames,
    )
    return {
        "seed": seed,
        "matched": result.matched,
        "frames_checked": result.frames_checked,
        "mismatch_frame": result.mismatch_frame,
        "source_events": result.source_events,
        "clone_events": result.clone_events,
        "source_fingerprint": result.source_fingerprint,
        "clone_fingerprint": result.clone_fingerprint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument("--verify-frames", type=int, default=1000)
    parser.add_argument(
        "--out",
        default="training/analysis/runs/exact_state_clone_fidelity.json",
    )
    args = parser.parse_args()

    started = time.time()
    results = []
    for offset in range(args.seeds):
        result = verify_seed(
            args.seed + offset,
            args.warmup_frames,
            args.verify_frames,
        )
        results.append(result)
        status = "match" if result["matched"] else "MISMATCH"
        print(
            f"{offset + 1:03d}/{args.seeds:03d} seed={result['seed']} "
            f"{status} frames={result['frames_checked']}",
            flush=True,
        )
        if not result["matched"]:
            break

    report = {
        "method": "exact_state_clone_fidelity",
        "all_matched": len(results) == args.seeds
        and all(result["matched"] for result in results),
        "seed": args.seed,
        "seeds_requested": args.seeds,
        "seeds_completed": len(results),
        "warmup_frames": args.warmup_frames,
        "verify_frames": args.verify_frames,
        "total_verified_frames": sum(r["frames_checked"] for r in results),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["all_matched"] else 1)


if __name__ == "__main__":
    main()
