"""
P28 hybrid fallback policy.

Default action comes from the P27b candidate champion. A small teacher search is
only used on hard frames: risky fire, missed safe fire windows, stutter/dead-end
stalls, passive map control, or low-confidence high-risk states.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.evaluate import play_round_dual_engine  # noqa: E402
from training.mpc_agent import CANDIDATES, make_sandbox  # noqa: E402
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402
from training.p26_amortized_mpc import (  # noqa: E402
    SCORE_SCALE,
    build_observation,
    label_actions,
    rollout_targets,
    select_action,
    stack_observation,
)
from training.p27_risk_value import (  # noqa: E402
    P27BRiskValuePolicy,
    _controls,
)


class P28HybridFallbackPolicy(P27BRiskValuePolicy):
    name = "p28_hybrid_fallback"

    def __init__(self, base_net, value_net, fire_margin=0.16,
                 fallback_horizon=48, fallback_samples=1,
                 fallback_cooldown=6, max_searches_per_round=80,
                 fallback_min_gain=0.03, fallback_death_penalty=0.18,
                 fallback_dd_penalty=0.45, fallback_kill_bonus=0.05,
                 fallback_max_death=1.10, fallback_max_dd=1.10,
                 require_fire_on_missed=False,
                 forbid_fire_on_risky=False,
                 risky_fire_death=0.55, risky_fire_dd=0.45,
                 missed_fire_line=0.72, missed_fire_max_risk=0.32,
                 uncertain_gap=0.055, uncertain_min_risk=0.48,
                 trigger_stalls=True, seed=0,
                 deterministic_search_seeds=False, **kwargs):
        super().__init__(
            base_net=base_net,
            value_net=value_net,
            fire_margin=fire_margin,
            **kwargs,
        )
        self.fallback_horizon = int(fallback_horizon)
        self.fallback_samples = int(fallback_samples)
        self.fallback_cooldown = int(fallback_cooldown)
        self.max_searches_per_round = int(max_searches_per_round)
        self.fallback_min_gain = float(fallback_min_gain)
        self.fallback_death_penalty = float(fallback_death_penalty)
        self.fallback_dd_penalty = float(fallback_dd_penalty)
        self.fallback_kill_bonus = float(fallback_kill_bonus)
        self.fallback_max_death = float(fallback_max_death)
        self.fallback_max_dd = float(fallback_max_dd)
        self.require_fire_on_missed = bool(require_fire_on_missed)
        self.forbid_fire_on_risky = bool(forbid_fire_on_risky)
        self.risky_fire_death = float(risky_fire_death)
        self.risky_fire_dd = float(risky_fire_dd)
        self.missed_fire_line = float(missed_fire_line)
        self.missed_fire_max_risk = float(missed_fire_max_risk)
        self.uncertain_gap = float(uncertain_gap)
        self.uncertain_min_risk = float(uncertain_min_risk)
        self.trigger_stalls = bool(trigger_stalls)
        self.rng = random.Random(seed)
        self.deterministic_search_seeds = bool(deterministic_search_seeds)
        self.round_seed = 0

    def set_round_seed(self, seed):
        self.round_seed = int(seed)

    def _sandbox_seed(self, index=0, sample=0):
        if not self.deterministic_search_seeds:
            return self.rng.randrange(1 << 30)
        value = (self.round_seed * 1_000_003
                 + self.frames * 9_176
                 + int(index) * 1_313
                 + int(sample) * 17_071
                 + 28_001)
        return int(value % (1 << 30))

    def reset(self):
        super().reset()
        self.search_cooldown = 0
        self.searches = 0
        self.fallback_counts = {}

    def _fb_count(self, name):
        self.fallback_counts[name] = self.fallback_counts.get(name, 0) + 1

    def _p27_gap(self, p27_value):
        order = np.sort(np.asarray(p27_value, dtype=np.float32))
        if len(order) < 2:
            return float("inf")
        return float(order[-1] - order[-2])

    def _trigger_reason(self, action, index, category, p27, metrics):
        if p27 is None:
            return None
        _, p27_aux, p27_value = p27
        line, _, risk = [float(value) for value in metrics[:3]]
        fire = action[2] == 1
        danger_death = float(p27_aux[index, 1])
        danger_dd = float(p27_aux[index, 2])

        if fire and (danger_death >= self.risky_fire_death
                     or danger_dd >= self.risky_fire_dd):
            return "risky_fire"
        if category in ("missed_fire_window", "missed_kill_line"):
            if line >= self.missed_fire_line and risk <= self.missed_fire_max_risk:
                return category
        if self.trigger_stalls and category in (
                "dead_end_stall", "stutter_stall", "passive_map_control"):
            return category
        if category in ("blind_fire", "unsafe_fire_death",
                        "double_death_risk", "fire_into_double_death",
                        "waste_or_unsafe_fire"):
            return category
        if risk >= self.uncertain_min_risk and self._p27_gap(p27_value) <= self.uncertain_gap:
            return "uncertain_risk"
        return None

    def _fallback_action(self, game, metrics, p27_index, reason):
        analyzer = self.analyzer or OpportunityAnalyzer360(game)
        scores, aux = label_actions(
            game, analyzer, metrics, self._sandbox_seed(p27_index, 0),
            self.fallback_horizon, score_samples=self.fallback_samples)
        value = (scores.astype(np.float32)
                 - self.fallback_death_penalty * SCORE_SCALE * aux[:, 1]
                 - self.fallback_dd_penalty * SCORE_SCALE * aux[:, 2]
                 + self.fallback_kill_bonus * SCORE_SCALE * aux[:, 0])
        for index, action in enumerate(CANDIDATES):
            if aux[index, 1] > self.fallback_max_death:
                value[index] = -1e9
            if aux[index, 2] > self.fallback_max_dd:
                value[index] = -1e9
            if (self.require_fire_on_missed
                    and reason in ("missed_fire_window", "missed_kill_line")
                    and action[2] != 1):
                value[index] = -1e9
            if (self.forbid_fire_on_risky and reason == "risky_fire"
                    and action[2] == 1):
                value[index] = -1e9
        best = int(value.argmax())
        if value[best] <= -1e8:
            self._fb_count(f"{reason}_no_safe_action")
            return None
        gain = float((value[best] - value[p27_index]) / SCORE_SCALE)
        if gain < self.fallback_min_gain:
            self._fb_count(f"{reason}_rejected")
            return None
        self._fb_count(reason)
        self._fb_count("accepted")
        return CANDIDATES[best]

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer360(game)
            self.frames = 0
            self.history = []
            self.pos_window.clear()
            self.input_window.clear()
            self.clear_fire_frames = 0
            self.context_positions.clear()
            self.context_distances.clear()
            self.last_context.fill(0.0)
            self.search_cooldown = 0
            self.searches = 0
        observation, metrics = build_observation(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        self.history.append(observation)
        stacked = stack_observation(self.history, self.frame_stack)
        with self.torch.no_grad():
            out = self.base_net(self.torch.as_tensor(stacked).unsqueeze(0))
        outputs = {
            "score": out["score"][0].numpy(),
            "aux": out["aux"][0].numpy(),
            "fire": out["fire"][0].numpy(),
        }
        base_action = select_action(
            outputs, self.candidates, self.fire_margin, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0)
        base_index = self.candidates.index(base_action)
        category = self._detect_category(game, _controls(base_action), metrics)
        context = self._update_context(game, metrics)
        p27 = self._p27_value(stacked, context)
        outputs = self._adjust_outputs(
            outputs, category, p27, base_index, metrics)
        action = select_action(
            outputs, self.candidates, self.fire_margin, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0)
        action_index = self.candidates.index(action)

        if self.search_cooldown > 0:
            self.search_cooldown -= 1
        reason = self._trigger_reason(action, action_index, category, p27, metrics)
        if (reason and self.search_cooldown <= 0
                and self.searches < self.max_searches_per_round):
            self._fb_count(f"{reason}_trigger")
            fallback = self._fallback_action(game, metrics, action_index, reason)
            self.searches += 1
            self.search_cooldown = self.fallback_cooldown
            if fallback is not None:
                action = fallback

        throttle, turn, fire = action
        if len(game.tanks) > 1 and not game.tanks[1].alive:
            fire = 0
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


class P28PriorSearchPolicy(P27BRiskValuePolicy):
    name = "p28_prior_search"

    def __init__(self, base_net, value_net, fire_margin=0.16, top_k=12,
                 search_horizon=48, search_samples=1,
                 search_death_penalty=0.18, search_dd_penalty=0.45,
                 search_kill_bonus=0.05, search_max_death=1.10,
                 search_max_dd=1.10, seed=0,
                 deterministic_search_seeds=False, **kwargs):
        super().__init__(
            base_net=base_net,
            value_net=value_net,
            fire_margin=fire_margin,
            **kwargs,
        )
        self.top_k = int(top_k)
        self.search_horizon = int(search_horizon)
        self.search_samples = int(search_samples)
        self.search_death_penalty = float(search_death_penalty)
        self.search_dd_penalty = float(search_dd_penalty)
        self.search_kill_bonus = float(search_kill_bonus)
        self.search_max_death = float(search_max_death)
        self.search_max_dd = float(search_max_dd)
        self.rng = random.Random(seed)
        self.deterministic_search_seeds = bool(deterministic_search_seeds)
        self.round_seed = 0

    def set_round_seed(self, seed):
        self.round_seed = int(seed)

    def reset(self):
        super().reset()
        self.fallback_counts = {}

    def _fb_count(self, name):
        self.fallback_counts[name] = self.fallback_counts.get(name, 0) + 1

    def _sandbox_seed(self, index, sample):
        if not self.deterministic_search_seeds:
            return self.rng.randrange(1 << 30)
        value = (self.round_seed * 1_000_003
                 + self.frames * 9_176
                 + int(index) * 1_313
                 + int(sample) * 17_071
                 + 28_101)
        return int(value % (1 << 30))

    def _candidate_order(self, outputs, p27, default_index, metrics):
        line, _, risk = [float(value) for value in metrics[:3]]
        if p27 is None:
            value = np.asarray(outputs["score"], dtype=np.float32)
        else:
            _, _, value = p27
            value = np.asarray(value, dtype=np.float32)
        order = list(np.argsort(value)[::-1])
        selected = []
        for index in [default_index] + order:
            index = int(index)
            if index not in selected:
                selected.append(index)
            if len(selected) >= max(1, self.top_k):
                break
        if line >= 0.78 and risk <= 0.30:
            fire_order = [i for i in order if CANDIDATES[int(i)][2] == 1]
            for index in fire_order[:3]:
                index = int(index)
                if index not in selected:
                    selected.append(index)
        return selected

    def _search(self, game, metrics, indices):
        analyzer = self.analyzer or OpportunityAnalyzer360(game)
        best_index, best_value = None, -float("inf")
        for index in indices:
            action = CANDIDATES[int(index)]
            total_score = 0.0
            total_aux = np.zeros(6, dtype=np.float32)
            for _ in range(max(1, self.search_samples)):
                sample = int(_)
                sandbox = make_sandbox(
                    game, "L2", rng_seed=self._sandbox_seed(index, sample))
                score, aux = rollout_targets(
                    sandbox, action, analyzer, metrics, self.search_horizon)
                total_score += score
                total_aux += aux
            samples = max(1, self.search_samples)
            score = total_score / samples
            aux = total_aux / samples
            if aux[1] > self.search_max_death or aux[2] > self.search_max_dd:
                continue
            value = (score
                     - self.search_death_penalty * SCORE_SCALE * aux[1]
                     - self.search_dd_penalty * SCORE_SCALE * aux[2]
                     + self.search_kill_bonus * SCORE_SCALE * aux[0])
            if value > best_value:
                best_value = value
                best_index = int(index)
        if best_index is None:
            self._fb_count("no_safe_search_action")
            return None
        self._fb_count("searched")
        return CANDIDATES[best_index]

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer360(game)
            self.frames = 0
            self.history = []
            self.pos_window.clear()
            self.input_window.clear()
            self.clear_fire_frames = 0
            self.context_positions.clear()
            self.context_distances.clear()
            self.last_context.fill(0.0)
        observation, metrics = build_observation(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        self.history.append(observation)
        stacked = stack_observation(self.history, self.frame_stack)
        with self.torch.no_grad():
            out = self.base_net(self.torch.as_tensor(stacked).unsqueeze(0))
        outputs = {
            "score": out["score"][0].numpy(),
            "aux": out["aux"][0].numpy(),
            "fire": out["fire"][0].numpy(),
        }
        default_action = select_action(
            outputs, self.candidates, self.fire_margin, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0)
        default_index = self.candidates.index(default_action)
        category = self._detect_category(
            game, _controls(default_action), metrics)
        context = self._update_context(game, metrics)
        p27 = self._p27_value(stacked, context)
        outputs = self._adjust_outputs(
            outputs, category, p27, default_index, metrics)
        p27_action = select_action(
            outputs, self.candidates, self.fire_margin, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0)
        p27_index = self.candidates.index(p27_action)
        indices = self._candidate_order(outputs, p27, p27_index, metrics)
        action = self._search(game, metrics, indices) or p27_action
        throttle, turn, fire = action
        if len(game.tanks) > 1 and not game.tanks[1].alive:
            fire = 0
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def _eval_worker(job):
    worker, seed, count, args = job
    import torch

    torch.set_num_threads(1)
    if args.search_mode == "prior":
        policy = P28PriorSearchPolicy(
            base_net=args.base_net,
            value_net=args.value_net,
            fire_margin=args.fire_margin,
            top_k=args.top_k,
            search_horizon=args.search_horizon,
            search_samples=args.search_samples,
            search_death_penalty=args.search_death_penalty,
            search_dd_penalty=args.search_dd_penalty,
            search_kill_bonus=args.search_kill_bonus,
            search_max_death=args.search_max_death,
            search_max_dd=args.search_max_dd,
            deterministic_search_seeds=args.deterministic_search_seeds,
            seed=args.seed + worker * 10007,
        )
    else:
        policy = P28HybridFallbackPolicy(
            base_net=args.base_net,
            value_net=args.value_net,
            fire_margin=args.fire_margin,
            fallback_horizon=args.fallback_horizon,
            fallback_samples=args.fallback_samples,
            fallback_cooldown=args.fallback_cooldown,
            max_searches_per_round=args.max_searches_per_round,
            fallback_min_gain=args.fallback_min_gain,
            fallback_death_penalty=args.fallback_death_penalty,
            fallback_dd_penalty=args.fallback_dd_penalty,
            fallback_kill_bonus=args.fallback_kill_bonus,
            fallback_max_death=args.fallback_max_death,
            fallback_max_dd=args.fallback_max_dd,
            require_fire_on_missed=args.require_fire_on_missed,
            forbid_fire_on_risky=args.forbid_fire_on_risky,
            risky_fire_death=args.risky_fire_death,
            risky_fire_dd=args.risky_fire_dd,
            missed_fire_line=args.missed_fire_line,
            missed_fire_max_risk=args.missed_fire_max_risk,
            uncertain_gap=args.uncertain_gap,
            uncertain_min_risk=args.uncertain_min_risk,
            trigger_stalls=not args.no_trigger_stalls,
            deterministic_search_seeds=args.deterministic_search_seeds,
            seed=args.seed + worker * 10007,
        )
    rounds = []
    counts = Counter()
    for index in range(count):
        round_seed = seed + index
        if hasattr(policy, "set_round_seed"):
            policy.set_round_seed(round_seed)
        result = play_round_dual_engine(policy, round_seed)
        result["seed"] = round_seed
        rounds.append(result)
        counts.update(policy.fallback_counts)
    return rounds, counts


def evaluate(args):
    workers = max(1, min(args.workers, args.n))
    base, remainder = divmod(args.n, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((worker, args.seed + offset, count, args))
            offset += count

    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        outputs = pool.map(_eval_worker, jobs)
    rounds = [item for part, _ in outputs for item in part]
    fallback_counts = Counter()
    for _, counts in outputs:
        fallback_counts.update(counts)
    total = max(1, len(rounds))
    count = lambda key: sum(item["true_result"] == key for item in rounds)
    shots = sum(item["shots"] for item in rounds)
    kills = sum(item["kills"] for item in rounds)
    elapsed = time.time() - started
    print(f"===== P28 hybrid fallback {total} games @{args.seed} "
          f"({elapsed:.0f}s) =====", flush=True)
    print(f"  true win {count('win')/total:.1%}  "
          f"loss {count('loss')/total:.1%}  "
          f"double death {count('double_death')/total:.1%}  "
          f"draw {count('draw')/total:.1%}", flush=True)
    print(f"  shots/game {shots/total:.1f}  "
          f"hit rate {kills/max(shots,1):.1%}  "
          f"avg length {sum(r['frames'] for r in rounds)/total/25:.1f}s  "
          f"wall {elapsed/total:.2f}s/game", flush=True)
    print(f"  fallback counts {dict(sorted(fallback_counts.items()))}",
          flush=True)
    if args.print_rounds or args.print_failures:
        for item in sorted(rounds, key=lambda row: row["seed"]):
            if args.print_failures and item["true_result"] == "win":
                continue
            print("  round "
                  f"seed={item['seed']} result={item['true_result']} "
                  f"shots={item['shots']} kills={item['kills']} "
                  f"death={item['death_cause']} "
                  f"move={item['move_cells']:.1f} "
                  f"frames={item['frames']}",
                  flush=True)
    if args.save_rounds:
        os.makedirs(os.path.dirname(args.save_rounds) or ".", exist_ok=True)
        with open(args.save_rounds, "w", encoding="utf-8") as handle:
            for item in sorted(rounds, key=lambda row: row["seed"]):
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(f"  saved rounds {args.save_rounds}", flush=True)
    return count("win") / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["eval"], nargs="?", default="eval")
    parser.add_argument("--base-net", default=(
        "training/models/p26_amortized_mpc_iter05.pt"))
    parser.add_argument("--value-net", default=(
        "training/models/p27b_risk_value_iter00.pt"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--fire-margin", type=float, default=0.16)
    parser.add_argument("--search-mode", choices=["fallback", "prior"],
                        default="fallback")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--search-horizon", type=int, default=48)
    parser.add_argument("--search-samples", type=int, default=1)
    parser.add_argument("--search-death-penalty", type=float, default=0.18)
    parser.add_argument("--search-dd-penalty", type=float, default=0.45)
    parser.add_argument("--search-kill-bonus", type=float, default=0.05)
    parser.add_argument("--search-max-death", type=float, default=1.10)
    parser.add_argument("--search-max-dd", type=float, default=1.10)
    parser.add_argument("--fallback-horizon", type=int, default=48)
    parser.add_argument("--fallback-samples", type=int, default=1)
    parser.add_argument("--fallback-cooldown", type=int, default=6)
    parser.add_argument("--max-searches-per-round", type=int, default=80)
    parser.add_argument("--fallback-min-gain", type=float, default=0.03)
    parser.add_argument("--fallback-death-penalty", type=float, default=0.18)
    parser.add_argument("--fallback-dd-penalty", type=float, default=0.45)
    parser.add_argument("--fallback-kill-bonus", type=float, default=0.05)
    parser.add_argument("--fallback-max-death", type=float, default=1.10)
    parser.add_argument("--fallback-max-dd", type=float, default=1.10)
    parser.add_argument("--require-fire-on-missed", action="store_true")
    parser.add_argument("--forbid-fire-on-risky", action="store_true")
    parser.add_argument("--risky-fire-death", type=float, default=0.55)
    parser.add_argument("--risky-fire-dd", type=float, default=0.45)
    parser.add_argument("--missed-fire-line", type=float, default=0.72)
    parser.add_argument("--missed-fire-max-risk", type=float, default=0.32)
    parser.add_argument("--uncertain-gap", type=float, default=0.055)
    parser.add_argument("--uncertain-min-risk", type=float, default=0.48)
    parser.add_argument("--no-trigger-stalls", action="store_true")
    parser.add_argument("--print-rounds", action="store_true")
    parser.add_argument("--print-failures", action="store_true")
    parser.add_argument("--save-rounds", default="")
    parser.add_argument("--deterministic-search-seeds", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
