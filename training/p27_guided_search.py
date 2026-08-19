"""Fair, deadline-aware online planning guided by the frozen P27b network.

This is the product path, not the privileged exact-state teacher:

* P27b ranks the legal controls and remains the timeout fallback.
* Only visible physics is copied into rollout sandboxes.
* Laika is reconstructed with fresh internal state and an independent RNG.
* Search is bounded by a wall-clock deadline and commits movement briefly.
* Firing is a stationary, separately verified action with a long-tail safety
  check, following the useful part of KillField's action-space reduction.

The first version deliberately uses a single 36-frame segment.  A two-stage
beam can be added behind the same interface after the baseline is measured.
"""

from __future__ import annotations

from collections import deque
import argparse
import os
import random
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


NEUTRAL = (1, 1, 0)


def action_to_controls(action):
    throttle, turn, fire = action
    return {
        "forward": throttle == 2,
        "backup": throttle == 0,
        "turn_left": turn == 0,
        "turn_right": turn == 2,
        "fire": fire == 1,
    }


def controls_to_action(controls):
    return (
        2 if controls.get("forward") else 0 if controls.get("backup") else 1,
        0 if controls.get("turn_left") else 2 if controls.get("turn_right") else 1,
        1 if controls.get("fire") else 0,
    )


def movement_candidates(ranking, top_k, current=NEUTRAL):
    """Return distinct no-fire movements plus deterministic safety fallbacks."""
    result = []

    def add(action):
        movement = (int(action[0]), int(action[1]), 0)
        if movement not in result:
            result.append(movement)

    for action in ranking:
        add(action)
        if len(result) >= max(1, int(top_k)):
            break
    add(current)
    add(NEUTRAL)
    # Emergency reverse-straight is cheap insurance when P27b's top-K is
    # overly concentrated around one direction.
    add((0, 1, 0))
    return result


def choose_movement_by_safety(scored, base_action, unsafe_score=-500.0):
    """Keep P27b unless its sampled rollout predicts a terminal loss.

    The cheap rollout's non-terminal path-distance heuristic is not strong
    enough to replace the learned policy.  It is only trusted for the large
    terminal separation around +/-1000.  This makes search a safety shield
    instead of a second, weaker navigation policy.
    """
    if not scored:
        return (base_action[0], base_action[1], 0), False
    base_movement = (base_action[0], base_action[1], 0)
    base_score = next((score for score, action in scored
                       if action == base_movement), None)
    if base_score is not None and base_score >= float(unsafe_score):
        return base_movement, False
    best_score, best_action = max(scored, key=lambda item: item[0])
    if best_score > (base_score if base_score is not None else -float("inf")):
        return best_action, best_action != base_movement
    return base_movement, False


class P27GuidedSearchPolicy:
    """P27b prior + fair short-horizon search + safe fallback."""

    name = "p27b-guided-search"

    def __init__(self, base_net="training/models/p26_amortized_mpc_iter05.pt",
                 value_net="training/models/p27b_risk_value_iter00.pt",
                 fire_margin=0.16, top_k=4, horizon=36, hold=8,
                 deadline_ms=30.0, action_hold_frames=6,
                 prior_refresh_frames=6, risk_threshold=0.18,
                 fire_safety_horizon=75, proactive_search_frames=24, seed=0):
        from training.p27_risk_value import P27BRiskValuePolicy

        self.base = P27BRiskValuePolicy(
            base_net=base_net, value_net=value_net, fire_margin=fire_margin)
        self.top_k = max(1, int(top_k))
        self.horizon = max(1, int(horizon))
        self.hold = max(1, min(int(hold), self.horizon))
        self.deadline_ms = max(1.0, float(deadline_ms))
        self.action_hold_frames = max(1, int(action_hold_frames))
        self.prior_refresh_frames = max(
            self.action_hold_frames, int(prior_refresh_frames))
        self.risk_threshold = float(risk_threshold)
        self.fire_safety_horizon = max(self.horizon, int(fire_safety_horizon))
        self.proactive_search_frames = max(
            self.action_hold_frames, int(proactive_search_frames))
        self.rng = random.Random(seed)
        self.latencies = deque(maxlen=500)
        self.reset()

    def reset(self):
        self.base.reset()
        self.committed_action = NEUTRAL
        self.commit_remaining = 0
        self.frames = 0
        self.searches = 0
        self.search_changes = 0
        self.deadline_hits = 0
        self.candidates_evaluated = 0
        self.fire_checks = 0
        self.fire_rejections = 0
        self.safety_overrides = 0
        self.base_safe_keeps = 0
        self.last_decision_ms = 0.0
        self.last_search_ms = 0.0
        self.last_search_reason = "startup"
        self.last_base_action = NEUTRAL
        self.last_selected_action = NEUTRAL
        self.last_candidate_scores = []
        self.wall_interrupt_cooldown = 0
        self.prior_age = self.prior_refresh_frames
        self.prior_refreshes = 0
        self.frames_since_search = self.proactive_search_frames

    def _deadline_hit(self, started):
        return (time.perf_counter() - started) * 1000.0 >= self.deadline_ms

    @staticmethod
    def _safe_fire_proposal(game):
        from tank_trouble_original.laika import LaikaAI

        me = game.tanks[0]
        if (not me.alive or not game.tanks[1].alive
                or not me.trigger_released or not game.weapon_ready(me)):
            return False
        return LaikaAI(game, me).check_bullet_path(me.rotation)["result"] == "HIT"

    def _fire_survives(self, game, action, started):
        """Reject a shot if its exact simulated tail kills the shooter.

        The sandbox still rebuilds Laika and reseeds its RNG; it does not copy
        hidden opponent intent from the live game.
        """
        from training.mpc_agent import make_sandbox

        if self._deadline_hit(started):
            return False
        sandbox = make_sandbox(
            game, "L2", rng_seed=self.rng.randrange(1 << 30))
        me = sandbox.tanks[0]
        throttle, turn, _ = action
        me.forward, me.backup = throttle == 2, throttle == 0
        me.turn_left, me.turn_right = turn == 0, turn == 2
        me.fire = True
        for frame in range(self.fire_safety_horizon):
            if frame == 1:
                me.fire = False
            if frame == self.action_hold_frames:
                me.forward = me.backup = False
                me.turn_left = me.turn_right = False
            sandbox.step()
            if not me.alive:
                return False
            if self._deadline_hit(started):
                return False
        return True

    def _verified_fire(self, game, base_action):
        """Keep learned fire, or add a clear shot, after a tail safety check."""
        proposal = (base_action if base_action[2] == 1 else
                    (1, 1, 1) if self._safe_fire_proposal(game) else None)
        if proposal is None:
            return None
        self.fire_checks += 1
        fire_started = time.perf_counter()
        if self._fire_survives(game, proposal, fire_started):
            return proposal
        self.fire_rejections += 1
        return None

    def _rollout_score(self, game, action, start_metrics, step_seed):
        from training.mpc_agent import make_sandbox, rollout

        sandbox = make_sandbox(game, "L2", rng_seed=step_seed)
        # P27b already provides the long-range learned prior.  Online search is
        # only the tactical physics verifier, so its leaf stays deliberately
        # cheap: terminal survival/kill first, path distance second.
        return float(rollout(
            sandbox, action, hold=self.hold, horizon=self.horizon))

    def _should_search(self, game, base_action, risk):
        if risk >= self.risk_threshold:
            return "risk"
        if game.tanks[0].hit_something:
            return "wall"
        if self.frames_since_search >= self.proactive_search_frames:
            return "proactive"
        return None

    def act(self, game):
        started = time.perf_counter()
        self.frames += 1
        self.prior_age += 1
        self.frames_since_search += 1
        if not game.tanks[0].alive:
            return {}

        # The whole point of amortised planning is that an accepted short plan
        # does not rebuild P27b's expensive action-preview observation on every
        # intervening frame.  A cheap visible-bullet risk check can still break
        # the commitment early.
        if (self.commit_remaining > 0
                and self.prior_age < self.prior_refresh_frames
                and self.base.analyzer is not None):
            risk = float(self.base.analyzer.incoming_risk(game))
            wall_interrupt = (
                game.tanks[0].hit_something
                and self.wall_interrupt_cooldown <= 0)
            if risk < self.risk_threshold and not wall_interrupt:
                self.commit_remaining -= 1
                self.wall_interrupt_cooldown = max(
                    0, self.wall_interrupt_cooldown - 1)
                chosen = (
                    self.committed_action[0], self.committed_action[1], 0)
                self.last_search_reason = "committed"
                self.last_selected_action = chosen
                self.last_decision_ms = (
                    time.perf_counter() - started) * 1000.0
                self.latencies.append(self.last_decision_ms)
                return action_to_controls(chosen)

        refresh_prior = (
            self.base.analyzer is None
            or not self.base.last_ranked_actions
            or self.prior_age >= self.prior_refresh_frames)
        if refresh_prior:
            # Keep the observation's round-progress feature tied to game time
            # even though we deliberately do not rebuild it every display frame.
            self.base.frames = max(self.base.frames, int(game.frame))
            base_controls = self.base.act(game)
            base_action = controls_to_action(base_controls)
            self.last_base_action = base_action
            self.prior_age = 0
            self.prior_refreshes += 1
        else:
            base_action = self.last_base_action
        metrics = self.base.last_metrics
        risk = (float(self.base.analyzer.incoming_risk(game))
                if self.base.analyzer is not None else 0.0)
        reason = self._should_search(game, base_action, risk)

        if reason is None:
            fire_action = self._verified_fire(game, base_action)
            chosen = (fire_action if fire_action is not None else
                      (base_action[0], base_action[1], 0))
            self.committed_action = (chosen[0], chosen[1], 0)
            self.commit_remaining = self.action_hold_frames
            self.last_search_reason = (
                "verified-fire" if fire_action is not None else "network")
            self.last_selected_action = chosen
            self.last_decision_ms = (time.perf_counter() - started) * 1000.0
            self.latencies.append(self.last_decision_ms)
            return action_to_controls(chosen)

        self.searches += 1
        self.frames_since_search = 0
        search_started = time.perf_counter()

        # Fire is an atomic decision.  A shot that is confirmed to hit and to
        # leave us alive through the post-kill tail dominates local movement
        # shaping, so execute it immediately instead of hiding it among highly
        # correlated moving+fire roots.
        fire_action = self._verified_fire(game, base_action)
        if fire_action is not None:
            self.committed_action = (fire_action[0], fire_action[1], 0)
            self.commit_remaining = self.action_hold_frames
            self.last_search_reason = "verified-fire"
            self.last_selected_action = fire_action
            self.last_candidate_scores = []
            self.last_search_ms = (
                time.perf_counter() - search_started) * 1000.0
            self.last_decision_ms = (
                time.perf_counter() - started) * 1000.0
            self.latencies.append(self.last_decision_ms)
            return action_to_controls(fire_action)

        current = (self.committed_action[0], self.committed_action[1], 0)
        ranking = self.base.last_ranked_actions or [base_action]
        # If one candidate costs most of the deadline on the current machine,
        # a nominal Top-K loop is no longer a real-time guarantee.  Estimate
        # how many can actually finish from the previous search and cap this
        # invocation before starting any irreversible rollout.
        if self.searches > 1 and self.last_search_ms > 0:
            previous_count = max(1, len(self.last_candidate_scores))
            per_candidate_ms = self.last_search_ms / previous_count
            budget_k = max(1, int((self.deadline_ms * 0.82) / max(
                0.25, per_candidate_ms)))
        else:
            budget_k = self.top_k
        candidates = movement_candidates(
            ranking, min(self.top_k, budget_k), current=current)
        step_seed = self.rng.randrange(1 << 30)
        scored = []
        for candidate in candidates:
            if scored and self._deadline_hit(search_started):
                self.deadline_hits += 1
                break
            score = self._rollout_score(game, candidate, metrics, step_seed)
            scored.append((score, candidate))
            self.candidates_evaluated += 1

        chosen, overrode = choose_movement_by_safety(scored, base_action)
        if overrode:
            self.safety_overrides += 1
        else:
            self.base_safe_keeps += 1
        scored.sort(key=lambda item: item[0], reverse=True)

        if chosen != base_action:
            self.search_changes += 1
        self.committed_action = (chosen[0], chosen[1], 0)
        self.commit_remaining = self.action_hold_frames
        self.wall_interrupt_cooldown = self.action_hold_frames
        self.last_search_reason = reason
        self.last_selected_action = chosen
        self.last_candidate_scores = [
            {"score": score, "action": list(action)} for score, action in scored]
        self.last_search_ms = (time.perf_counter() - search_started) * 1000.0
        self.last_decision_ms = (time.perf_counter() - started) * 1000.0
        self.latencies.append(self.last_decision_ms)
        return action_to_controls(chosen)

    def telemetry(self):
        samples = sorted(self.latencies)

        def percentile(frac):
            if not samples:
                return 0.0
            index = min(len(samples) - 1, int(frac * (len(samples) - 1)))
            return float(samples[index])

        return {
            "frames": self.frames,
            "prior_refreshes": self.prior_refreshes,
            "prior_refresh_rate": self.prior_refreshes / max(1, self.frames),
            "prior_age": self.prior_age,
            "searches": self.searches,
            "search_rate": self.searches / max(1, self.frames),
            "search_changes": self.search_changes,
            "change_rate": self.search_changes / max(1, self.searches),
            "deadline_hits": self.deadline_hits,
            "candidates_evaluated": self.candidates_evaluated,
            "fire_checks": self.fire_checks,
            "fire_rejections": self.fire_rejections,
            "safety_overrides": self.safety_overrides,
            "base_safe_keeps": self.base_safe_keeps,
            "last_decision_ms": self.last_decision_ms,
            "last_search_ms": self.last_search_ms,
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "last_reason": self.last_search_reason,
            "base_action": list(self.last_base_action),
            "selected_action": list(self.last_selected_action),
        }


def main():
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--deadline-ms", type=float, default=28.0)
    args = parser.parse_args()

    from training.evaluate import evaluate_dual

    policy = P27GuidedSearchPolicy(
        top_k=args.top_k, horizon=args.horizon,
        deadline_ms=args.deadline_ms, seed=args.seed)
    evaluate_dual(policy, n=args.n, base_seed=args.seed)
    print(policy.telemetry())


if __name__ == "__main__":
    main()
