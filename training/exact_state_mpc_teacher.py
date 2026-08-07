"""Privileged exact-state variant of the current prior-guided MPC teacher.

This is deliberately a controlled experiment.  Candidate ordering, horizon,
and objective match the current teacher; only sandbox construction changes
from sampled/reset Laika state to an exact clone of the live local engine.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluate import play_round_dual_engine  # noqa: E402
from training.exact_state import clone_exact_game, state_fingerprint  # noqa: E402
from training.mpc_agent import CANDIDATES  # noqa: E402
from training.p26_amortized_mpc import SCORE_SCALE, rollout_targets  # noqa: E402
from training.p28_hybrid_fallback import P28PriorSearchPolicy  # noqa: E402


DEFAULT_FAILURE_SEEDS = (970017, 970031, 990011, 990024, 990037, 973034)
ALL_CANDIDATE_INDICES = tuple(range(len(CANDIDATES)))


def _set_candidate_controls(game, action):
    throttle, turn, fire = action
    tank = game.tanks[0]
    tank.forward = throttle == 2
    tank.backup = throttle == 0
    tank.turn_left = turn == 0
    tank.turn_right = turn == 2
    tank.fire = fire == 1


def prefer_nonfire_secured_kill(rows, best_index, epsilon=1e-6):
    """Prefer an equal-value non-fire action once an exact kill is secured."""
    if best_index is None:
        return None
    best = next(row for row in rows if row["index"] == best_index)
    if CANDIDATES[int(best_index)][2] == 0 or best["kill"] < 1.0:
        return best_index
    alternatives = [
        row for row in rows
        if (row["allowed"]
            and CANDIDATES[int(row["index"])][2] == 0
            and row["kill"] >= best["kill"]
            and row["value"] >= best["value"] - float(epsilon))
    ]
    if not alternatives:
        return best_index
    return int(max(alternatives, key=lambda row: row["value"])["index"])


def prefer_nonfire_low_gain(rows, best_index, min_gain):
    """Charge a small persistence cost for fire that predicts no kill."""
    if best_index is None or min_gain <= 0.0:
        return best_index
    best = next(row for row in rows if row["index"] == best_index)
    if CANDIDATES[int(best_index)][2] == 0 or best["kill"] > 0.0:
        return best_index
    alternatives = [
        row for row in rows
        if row["allowed"] and CANDIDATES[int(row["index"])][2] == 0
    ]
    if not alternatives:
        return best_index
    nonfire = max(alternatives, key=lambda row: row["value"])
    if best["value"] - nonfire["value"] < float(min_gain):
        return int(nonfire["index"])
    return best_index


def exact_root_search(
        game,
        analyzer,
        metrics,
        indices,
        *,
        horizon=72,
        death_penalty=0.18,
        dd_penalty=0.45,
        kill_bonus=0.05,
        max_death=0.0,
        max_dd=0.0):
    """Evaluate root actions against the exact future state once each."""
    best_index = None
    best_value = -float("inf")
    rows = []
    for raw_index in indices:
        index = int(raw_index)
        sandbox = clone_exact_game(game)
        score, aux = rollout_targets(
            sandbox,
            CANDIDATES[index],
            analyzer,
            metrics,
            int(horizon),
        )
        allowed = bool(aux[1] <= max_death and aux[2] <= max_dd)
        value = (float(score)
                 - float(death_penalty) * SCORE_SCALE * float(aux[1])
                 - float(dd_penalty) * SCORE_SCALE * float(aux[2])
                 + float(kill_bonus) * SCORE_SCALE * float(aux[0]))
        rows.append({
            "index": index,
            "action": CANDIDATES[index],
            "score": float(score),
            "value": value,
            "aux": [float(item) for item in aux],
            "kill": float(aux[0]),
            "death": float(aux[1]),
            "double_death": float(aux[2]),
            "allowed": allowed,
        })
        if allowed and value > best_value:
            best_index = index
            best_value = value
    return best_index, rows


def exact_successor_viability(
        game,
        analyzer,
        root_rows,
        *,
        continuation_indices=ALL_CANDIDATE_INDICES,
        horizon=72,
        death_penalty=0.18,
        dd_penalty=0.45,
        kill_bonus=0.05,
        max_death=0.0,
        max_dd=0.0):
    """Check whether each safe root keeps a safe continuation after one frame.

    The live policy replans every frame, while the ordinary rollout holds one
    action for many frames.  This one-step viability check prevents a nominally
    safe root action from entering a state where every continuation is lethal.
    """
    viable = {}
    details = {}
    continuation_evaluations = 0
    safe_rows = sorted(
        (row for row in root_rows if row["allowed"]),
        key=lambda row: row["value"],
        reverse=True,
    )
    for row in safe_rows:
        root_index = int(row["index"])
        successor = clone_exact_game(game)
        _set_candidate_controls(successor, CANDIDATES[root_index])
        successor.step()
        successor_hash = state_fingerprint(successor)
        if not successor.tanks[0].alive:
            viable[root_index] = False
            details[root_index] = {
                "continuation_index": None,
                "successor_fingerprint": successor_hash,
            }
            continue
        successor_metrics = analyzer.metrics(successor)
        continuation, continuation_rows = exact_root_search(
            successor,
            analyzer,
            successor_metrics,
            continuation_indices,
            horizon=horizon,
            death_penalty=death_penalty,
            dd_penalty=dd_penalty,
            kill_bonus=kill_bonus,
            max_death=max_death,
            max_dd=max_dd,
        )
        continuation_evaluations += len(continuation_rows)
        viable[root_index] = continuation is not None
        details[root_index] = {
            "continuation_index": continuation,
            "successor_fingerprint": successor_hash,
        }
        if continuation is not None:
            break
    return viable, continuation_evaluations, details


class ExactStatePriorGuidedMPC(P28PriorSearchPolicy):
    name = "exact_state_prior_guided_mpc"

    def __init__(self, *args, successor_shield=True,
                 successor_horizon=None, successor_shield_max_safe_roots=2,
                 suppress_secured_fire=True, min_unsecured_fire_gain=2.0,
                 **kwargs):
        kwargs.setdefault("search_max_death", 0.0)
        kwargs.setdefault("search_max_dd", 0.0)
        super().__init__(*args, **kwargs)
        self.successor_shield = bool(successor_shield)
        self.successor_horizon = int(
            successor_horizon or self.search_horizon)
        self.successor_shield_max_safe_roots = int(
            successor_shield_max_safe_roots)
        self.suppress_secured_fire = bool(suppress_secured_fire)
        self.min_unsecured_fire_gain = float(min_unsecured_fire_gain)

    def reset(self):
        super().reset()
        self.exact_searches = 0
        self.exact_candidates = 0
        self.successor_shield_checks = 0
        self.successor_candidates = 0
        self.successor_rejections = 0
        self.safety_widenings = 0
        self.safety_widened_candidates = 0
        self.secured_fire_suppressions = 0
        self.low_gain_fire_suppressions = 0
        self.pending_successor_audit = None
        self.viability_audits = deque(maxlen=32)
        self.search_trace = deque(maxlen=96)
        self.last_search_decision = None

    def _search(self, game, metrics, indices):
        analyzer = self.analyzer
        prior_index = int(indices[0]) if len(indices) else None
        interventions = []
        audit = None
        if self.pending_successor_audit is not None:
            expected = self.pending_successor_audit
            audit = {
                "frame": int(game.frame),
                "state_matches": (
                    expected["successor_fingerprint"]
                    == state_fingerprint(game)),
                "expected_continuation_index": expected["continuation_index"],
            }
            self.viability_audits.append(audit)
            self.pending_successor_audit = None
        best_index, rows = exact_root_search(
            game,
            analyzer,
            metrics,
            indices,
            horizon=self.search_horizon,
            death_penalty=self.search_death_penalty,
            dd_penalty=self.search_dd_penalty,
            kill_bonus=self.search_kill_bonus,
            max_death=self.search_max_death,
            max_dd=self.search_max_dd,
        )
        self.exact_searches += 1
        self.exact_candidates += len(rows)
        if best_index is None and len(rows) < len(CANDIDATES):
            searched = {int(row["index"]) for row in rows}
            remaining = tuple(
                index for index in ALL_CANDIDATE_INDICES
                if index not in searched
            )
            widened_best, widened_rows = exact_root_search(
                game,
                analyzer,
                metrics,
                remaining,
                horizon=self.search_horizon,
                death_penalty=self.search_death_penalty,
                dd_penalty=self.search_dd_penalty,
                kill_bonus=self.search_kill_bonus,
                max_death=self.search_max_death,
                max_dd=self.search_max_dd,
            )
            rows.extend(widened_rows)
            self.exact_candidates += len(widened_rows)
            self.safety_widenings += 1
            self.safety_widened_candidates += len(widened_rows)
            interventions.append("safety_widening")
            if widened_best is not None:
                best_index = widened_best
                self._fb_count("safety_widening_recovered")
        if audit is not None:
            expected_index = audit["expected_continuation_index"]
            expected_row = next(
                (row for row in rows if row["index"] == expected_index), None)
            audit["expected_continuation_row"] = expected_row
        if self.suppress_secured_fire:
            original_best = best_index
            best_index = prefer_nonfire_secured_kill(rows, best_index)
            if best_index != original_best:
                self.secured_fire_suppressions += 1
                self._fb_count("secured_kill_fire_suppressed")
                interventions.append("secured_kill_fire_suppressed")
        original_best = best_index
        best_index = prefer_nonfire_low_gain(
            rows, best_index, self.min_unsecured_fire_gain)
        if best_index != original_best:
            self.low_gain_fire_suppressions += 1
            self._fb_count("low_gain_fire_suppressed")
            interventions.append("low_gain_fire_suppressed")
        shield_triggered = False
        viable = {}
        successor_details = {}
        safe_root_count = sum(row["allowed"] for row in rows)
        if (self.successor_shield and safe_root_count > 0
                and safe_root_count <= self.successor_shield_max_safe_roots):
            shield_triggered = True
            interventions.append("successor_shield_checked")
            self.successor_shield_checks += 1
            viable, evaluated, successor_details = exact_successor_viability(
                game,
                analyzer,
                rows,
                horizon=self.successor_horizon,
                death_penalty=self.search_death_penalty,
                dd_penalty=self.search_dd_penalty,
                kill_bonus=self.search_kill_bonus,
                max_death=self.search_max_death,
                max_dd=self.search_max_dd,
            )
            self.successor_candidates += evaluated
            original_best = best_index
            viable_rows = [
                row for row in rows
                if row["allowed"] and viable.get(int(row["index"]), False)
            ]
            best_index = (max(viable_rows, key=lambda row: row["value"])["index"]
                          if viable_rows else None)
            if best_index != original_best:
                self.successor_rejections += 1
                interventions.append("successor_shield_override")
        if best_index is None:
            self._fb_count("no_safe_search_action")
            if shield_triggered:
                self._fb_count("successor_shield_no_viable_root")
                interventions.append("successor_shield_no_viable_root")
            self.last_search_decision = {
                "frame": int(game.frame),
                "selected_index": None,
                "selected_action": None,
                "prior_index": prior_index,
                "prior_action": (None if prior_index is None
                                 else CANDIDATES[prior_index]),
                "executed_index": None,
                "executed_action": None,
                "safe_root_count": int(safe_root_count),
                "successor_shield_triggered": shield_triggered,
                "viable": {int(key): bool(value)
                           for key, value in viable.items()},
                "interventions": tuple(interventions),
                "rows": rows,
            }
            return None
        ordered = sorted(rows, key=lambda row: row["value"], reverse=True)
        if shield_triggered and best_index in successor_details:
            self.pending_successor_audit = successor_details[best_index]
        self.last_search_decision = {
            "frame": int(game.frame),
            "selected_index": int(best_index),
            "selected_action": CANDIDATES[best_index],
            "prior_index": prior_index,
            "prior_action": (None if prior_index is None
                             else CANDIDATES[prior_index]),
            "executed_index": None,
            "executed_action": None,
            "safe_root_count": int(safe_root_count),
            "successor_shield_triggered": shield_triggered,
            "viable_root_count": int(sum(viable.values())),
            "viable": {int(key): bool(value)
                       for key, value in viable.items()},
            "interventions": tuple(interventions),
            "rows": rows,
        }
        self.search_trace.append({
            **self.last_search_decision,
            "me": {
                "x": float(game.tanks[0].x),
                "y": float(game.tanks[0].y),
                "rotation": float(game.tanks[0].rotation),
                "bullets_fired": int(game.tanks[0].bullets_fired),
            },
            "enemy": {
                "x": float(game.tanks[1].x),
                "y": float(game.tanks[1].y),
                "rotation": float(game.tanks[1].rotation),
                "bullets_fired": int(game.tanks[1].bullets_fired),
            },
            "bullets": [{
                "name": bullet.name,
                "owner": int(bullet.owner.number),
                "x": float(bullet.x),
                "y": float(bullet.y),
                "x_speed": float(bullet.x_speed),
                "y_speed": float(bullet.y_speed),
                "lifetime": int(bullet.lifetime),
            } for bullet in game.bullets],
            "top": ordered[:3],
        })
        self._fb_count("searched")
        return CANDIDATES[best_index]

    def act(self, game):
        """Run the frozen teacher and expose the action actually executed."""
        controls = super().act(game)
        if self.last_search_decision is not None:
            action = (
                2 if controls.get("forward") else
                0 if controls.get("backup") else 1,
                0 if controls.get("turn_left") else
                2 if controls.get("turn_right") else 1,
                1 if controls.get("fire") else 0,
            )
            self.last_search_decision["executed_action"] = action
            self.last_search_decision["executed_index"] = (
                CANDIDATES.index(action) if action in CANDIDATES else None)
        return controls


def _make_policy(args):
    return ExactStatePriorGuidedMPC(
        base_net=args.base_net,
        value_net=args.value_net,
        fire_margin=args.fire_margin,
        top_k=args.top_k,
        search_horizon=args.search_horizon,
        search_samples=1,
        search_death_penalty=args.search_death_penalty,
        search_dd_penalty=args.search_dd_penalty,
        search_kill_bonus=args.search_kill_bonus,
        search_max_death=args.search_max_death,
        search_max_dd=args.search_max_dd,
        successor_shield=args.successor_shield,
        successor_horizon=args.successor_horizon,
        successor_shield_max_safe_roots=args.successor_shield_max_safe_roots,
        suppress_secured_fire=args.suppress_secured_fire,
        min_unsecured_fire_gain=args.min_unsecured_fire_gain,
    )


def _run_seed(job):
    seed, args = job
    import torch

    torch.set_num_threads(1)
    policy = _make_policy(args)
    policy.set_round_seed(seed)
    started = time.time()
    result = play_round_dual_engine(policy, seed)
    result.update({
        "seed": seed,
        "elapsed_seconds": time.time() - started,
        "exact_searches": policy.exact_searches,
        "exact_candidates": policy.exact_candidates,
        "successor_shield_checks": policy.successor_shield_checks,
        "successor_candidates": policy.successor_candidates,
        "successor_rejections": policy.successor_rejections,
        "safety_widenings": policy.safety_widenings,
        "safety_widened_candidates": policy.safety_widened_candidates,
        "secured_fire_suppressions": policy.secured_fire_suppressions,
        "low_gain_fire_suppressions": policy.low_gain_fire_suppressions,
        "viability_audit_count": len(policy.viability_audits),
        "viability_mismatch_count": sum(
            not item["state_matches"] for item in policy.viability_audits),
        "fallback_counts": dict(policy.fallback_counts),
    })
    if result["true_result"] != "win":
        result["search_trace"] = list(policy.search_trace)
        result["viability_audits"] = list(policy.viability_audits)
    return result


def _parse_seeds(value):
    seeds = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            start_text, count_text = item.split(":", 1)
            start, count = int(start_text), int(count_text)
            if count < 0:
                raise ValueError("seed range count must be non-negative")
            seeds.extend(range(start, start + count))
        else:
            seeds.append(int(item))
    return tuple(seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-net", default=(
        "training/models/p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--value-net", default=(
        "training/models/p27b_risk_value_iter00.pt"))
    parser.add_argument(
        "--seed-list",
        default=",".join(str(seed) for seed in DEFAULT_FAILURE_SEEDS),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--search-horizon", type=int, default=72)
    parser.add_argument("--search-death-penalty", type=float, default=0.18)
    parser.add_argument("--search-dd-penalty", type=float, default=0.45)
    parser.add_argument("--search-kill-bonus", type=float, default=0.05)
    parser.add_argument("--search-max-death", type=float, default=0.0)
    parser.add_argument("--search-max-dd", type=float, default=0.0)
    parser.add_argument("--successor-shield",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--successor-horizon", type=int, default=72)
    parser.add_argument("--successor-shield-max-safe-roots", type=int,
                        default=2)
    parser.add_argument("--suppress-secured-fire",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-unsecured-fire-gain", type=float, default=2.0)
    parser.add_argument(
        "--out",
        default="training/analysis/runs/exact_state_mpc_failure_replay.json",
    )
    parser.add_argument("--print-report", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seed_list)
    started = time.time()
    worker_count = max(1, min(args.workers, len(seeds)))
    if worker_count == 1:
        rounds = [_run_seed((seed, args)) for seed in seeds]
    else:
        with mp.Pool(worker_count) as pool:
            rounds = pool.map(_run_seed, [(seed, args) for seed in seeds])
    rounds.sort(key=lambda row: row["seed"])
    results = Counter(row["true_result"] for row in rounds)
    report = {
        "method": ("exact_state_safety_shielded_mpc" if args.successor_shield
                   else "exact_state_prior_guided_mpc"),
        "controlled_change": (
            "exact_state_plus_one_step_successor_viability"
            if args.successor_shield
            else "exact_rng_and_laika_internal_state_clone"),
        "base_net": args.base_net,
        "value_net": args.value_net,
        "top_k": args.top_k,
        "search_horizon": args.search_horizon,
        "successor_shield": args.successor_shield,
        "successor_horizon": args.successor_horizon,
        "successor_shield_max_safe_roots": (
            args.successor_shield_max_safe_roots),
        "suppress_secured_fire": args.suppress_secured_fire,
        "min_unsecured_fire_gain": args.min_unsecured_fire_gain,
        "seeds": seeds,
        "results": dict(results),
        "win_rate": results["win"] / max(1, len(rounds)),
        "elapsed_seconds": time.time() - started,
        "rounds": rounds,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for row in rounds:
        print(
            f"seed={row['seed']} result={row['true_result']} "
            f"death={row['death_cause']} frames={row['frames']} "
            f"searches={row['exact_searches']} "
            f"candidates={row['exact_candidates']} "
            f"widen={row['safety_widenings']} "
            f"shield={row['successor_shield_checks']}/"
            f"{row['successor_rejections']} "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    else:
        print(json.dumps({
            "method": report["method"],
            "results": report["results"],
            "win_rate": report["win_rate"],
            "elapsed_seconds": report["elapsed_seconds"],
            "out": args.out,
        }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
