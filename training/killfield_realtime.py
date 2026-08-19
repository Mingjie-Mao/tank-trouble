"""Realtime and opponent-robust wrapper for the P37 kill-field teacher.

The original teacher is behaviorally strong but synchronous: a target-cell
cache miss and 18 forward rollouts block the Tk game loop.  This module moves
the unchanged P37 plan into persistent worker processes.  ``act`` never waits
for a plan; it continues the last macro action and accepts only fresh results.

Profiles:

``laika``
    Exact original P37 L2 score landscape.  This is the behavior-preserving
    playback profile.

``mixed``
    Mostly original Laika scoring with a robust visible-action tail.

``human``
    No Laika algorithm in the rollout.  It scores the current opponent input
    and several paired plausible human inputs, combining their mean and worst
    case.  It therefore does not depend on a human opponent's private policy.
"""

import concurrent.futures
import math
import multiprocessing as mp
import pickle
import random
import time

import numpy as np

from training.killfield_post_kill import post_kill_survival_scores
from training.killfield_teacher import (
    COMMIT_MOVE_FRAMES,
    COMMIT_TURN_FRAMES,
    NO_EFFECT_REPEAT_PENALTY,
    CANDIDATES,
    HuntChainState,
    InverseDensityFieldBuilder,
    KillFieldTeacher,
    _cell,
    density_rollout,
    mask_moving_fire_scores,
    _own_bullet_ids,
)


_WORKER_MAZE_KEY = None
_WORKER_BUILDER = None
_WORKER_FIELDS = {}


def _maze_key(game):
    return (
        len(game.maze), len(game.maze[0]),
        tuple((cell[1], cell[2])
              for column in game.maze for cell in column),
    )


def _human_hypotheses(game, seed, count, include_current=False):
    """Paired action hypotheses; human mode leaves include_current false."""
    enemy = game.tanks[1]
    result = []
    current = None
    if include_current:
        current = (
            2 if enemy.forward else 0 if enemy.backup else 1,
            0 if enemy.turn_left else 2 if enemy.turn_right else 1,
            int(bool(enemy.fire)),
        )
        result.append(None)
    rng = random.Random(seed ^ 0x48A11CE)
    fire_ready = int(
        enemy.alive and enemy.trigger_released and game.weapon_ready(enemy))
    fixed = [
        (2, 0, fire_ready), (2, 2, fire_ready),
        (0, 0, fire_ready), (0, 2, fire_ready),
        (2, 1, fire_ready), (0, 1, fire_ready),
    ]
    if current is not None and current != (1, 1, 0):
        fixed.insert(0, current)
    rng.shuffle(fixed)
    result.extend(fixed[:max(1, int(count) - len(result))])
    return result


def _robust_action_score(game, action, field, seed, chain, settings,
                         profile):
    common = dict(
        chain_state=chain,
        horizon=settings["horizon"], hold=settings["hold"])
    if profile == "laika":
        return density_rollout(
            game, action, field, seed, opp_model="L2", **common)

    hypotheses = _human_hypotheses(
        game, seed, settings["human_samples"],
        include_current=(profile == "mixed"))
    human_scores = np.asarray([
        density_rollout(
            game, action, field, seed, opp_model="L1",
            opponent_action=hypothesis, **common)
        for hypothesis in hypotheses
    ], dtype=np.float64)
    robust_human = (
        settings["human_mean_weight"] * float(human_scores.mean())
        + (1.0 - settings["human_mean_weight"])
        * float(human_scores.min()))
    if profile == "human":
        return robust_human

    laika_score = density_rollout(
        game, action, field, seed, opp_model="L2", **common)
    return (settings["laika_weight"] * laika_score
            + (1.0 - settings["laika_weight"]) * robust_human)


def _plan_worker(game, settings, seed, chain, failed_movement, phase):
    """Process entry point.  Returns a complete paired score landscape."""
    started = time.perf_counter()
    if phase == "post_kill":
        scores = post_kill_survival_scores(
            game, settings["post_kill_horizon"])
        return {
            "phase": phase,
            "round": game.round_number,
            "frame": game.frame,
            "target": None,
            "field": None,
            "scores": scores,
            "own_bullets": _own_bullet_ids(game),
            "field_built": False,
            "elapsed": time.perf_counter() - started,
        }

    global _WORKER_MAZE_KEY, _WORKER_BUILDER, _WORKER_FIELDS
    key = _maze_key(game)
    if key != _WORKER_MAZE_KEY:
        _WORKER_MAZE_KEY = key
        _WORKER_BUILDER = InverseDensityFieldBuilder(
            game, settings["rays"], settings["bounces"],
            settings["flight_frames"])
        _WORKER_FIELDS = {}
    target = _cell(game, game.tanks[1])
    built = target not in _WORKER_FIELDS
    if built:
        _WORKER_FIELDS[target] = _WORKER_BUILDER.build(target)
    field = _WORKER_FIELDS[target]
    scores = np.asarray([
        _robust_action_score(
            game, action, field, seed, chain, settings,
            settings["opponent_profile"])
        for action in CANDIDATES
    ], dtype=np.float32)
    mask_moving_fire_scores(scores)
    if failed_movement is not None:
        for index, action in enumerate(CANDIDATES):
            if action[:2] == failed_movement:
                scores[index] -= NO_EFFECT_REPEAT_PENALTY
    return {
        "phase": phase,
        "round": game.round_number,
        "frame": game.frame,
        "target": target,
        "field": field,
        "scores": scores,
        "own_bullets": _own_bullet_ids(game),
        "field_built": built,
        "elapsed": time.perf_counter() - started,
    }


class RealtimeKillFieldTeacher(KillFieldTeacher):
    """P37 teacher with non-blocking, deadline-checked planning."""

    name = "P37 实时稳健击杀场老师"

    def __init__(self, *args, opponent_profile="laika",
                 max_plan_seconds=2.0, max_stale_frames=6,
                 human_samples=4, laika_weight=0.70,
                 human_mean_weight=0.65, worker_count=1,
                 post_kill_horizon=75, **kwargs):
        if opponent_profile not in ("laika", "mixed", "human"):
            raise ValueError(f"unknown opponent profile: {opponent_profile}")
        self.opponent_profile = opponent_profile
        self.max_plan_seconds = float(max_plan_seconds)
        self.max_stale_frames = int(max_stale_frames)
        self.human_samples = int(human_samples)
        self.laika_weight = float(laika_weight)
        self.human_mean_weight = float(human_mean_weight)
        self.worker_count = max(1, int(worker_count))
        self.post_kill_horizon = int(post_kill_horizon)
        self._executor = None
        self.executor_mode = None
        self._pending = None
        self._pending_phase = None
        self._ready = None
        self._submitted_at = None
        self._submitted_frame = None
        self.async_plans = 0
        self.async_results = 0
        self.stale_results = 0
        self.deadline_results = 0
        self.worker_seconds = []
        self.main_act_seconds = []
        self.result_age_frames = []
        self.async_wait_frames = 0
        self.post_fire_wait_frames = 0
        self.awaiting_post_fire_plan = False
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        pending = getattr(self, "_pending", None)
        if pending is not None:
            pending.cancel()
        self._pending = None
        self._pending_phase = None
        self._ready = None
        self._submitted_at = None
        self._submitted_frame = None
        self.awaiting_post_fire_plan = False

    def _ensure_executor(self):
        if self._executor is None:
            try:
                context = mp.get_context("spawn")
                self._executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.worker_count, mp_context=context)
                self.executor_mode = "process"
            except (OSError, PermissionError):
                # Restricted runners may disallow POSIX semaphores.  The game
                # snapshot still makes a thread worker race-free; process mode
                # remains preferred because it isolates Python-heavy tracing.
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.worker_count)
                self.executor_mode = "thread"

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _settings(self):
        return {
            "rays": self.ray_count,
            "bounces": self.max_bounces,
            "flight_frames": self.max_flight_frames,
            "horizon": self.horizon,
            "hold": self.hold,
            "post_kill_horizon": self.post_kill_horizon,
            "opponent_profile": self.opponent_profile,
            "executor_mode": self.executor_mode,
            "human_samples": self.human_samples,
            "laika_weight": self.laika_weight,
            "human_mean_weight": self.human_mean_weight,
        }

    def _submit(self, game, phase):
        if self._pending is not None:
            return
        self._ensure_executor()
        # Explicit snapshot prevents the executor's feeder thread from racing
        # the live Tk game while serialising a mutable Game object.
        snapshot = pickle.loads(
            pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL))
        failed = None
        if self.action_no_effect and self.observed_previous_action is not None:
            failed = self.observed_previous_action[:2]
        seed = self.rng.randrange(1 << 30)
        self._pending = self._executor.submit(
            _plan_worker, snapshot, self._settings(), seed,
            self.chain.clone(), failed, phase)
        self._pending_phase = phase
        self._submitted_at = time.perf_counter()
        self._submitted_frame = game.frame
        self.async_plans += 1

    def _poll(self, game, phase):
        future = self._pending
        if future is None or not future.done():
            return
        self._pending = None
        self._pending_phase = None
        try:
            result = future.result()
        except Exception:
            self.stale_results += 1
            return
        self.worker_seconds.append(float(result["elapsed"]))
        wall_seconds = time.perf_counter() - self._submitted_at
        if wall_seconds > self.max_plan_seconds:
            self.deadline_results += 1
            return
        if result["round"] != game.round_number or result["phase"] != phase:
            self.stale_results += 1
            return
        if result["own_bullets"] != _own_bullet_ids(game):
            # A plan made before our latest shot cannot safely choose the next
            # locomotion action: it has never simulated that new live bullet.
            self.stale_results += 1
            return
        if game.frame - result["frame"] > self.max_stale_frames:
            self.stale_results += 1
            return
        if phase == "combat" and result["target"] != _cell(
                game, game.tanks[1]):
            self.stale_results += 1
            return
        self._ready = result
        self.result_age_frames.append(game.frame - result["frame"])
        self.async_results += 1

    def _take_ready(self, game, phase):
        result = self._ready
        if result is None:
            return None
        self._ready = None
        scores = result["scores"]
        if phase == "combat":
            self.field = result["field"]
            self._field_cache[result["target"]] = self.field
            self.field_builds += int(result["field_built"])
            self.field_build_seconds += float(result["elapsed"]) \
                if result["field_built"] else 0.0
        self.last_scores = scores.copy()
        action = CANDIDATES[int(np.argmax(scores))]
        if phase == "post_kill":
            action = (action[0], action[1], 0)
        if action[2] == 0:
            self.committed_action = action
            self.commit_remaining = (
                COMMIT_MOVE_FRAMES if action[0] != 1 else
                COMMIT_TURN_FRAMES if action[1] != 1 else 0)
            if phase == "post_kill":
                self.commit_remaining = min(self.commit_remaining, 1)
        return action

    def act(self, game):
        started = time.perf_counter()
        try:
            self.last_decision_kind = "none"
            self.last_scores = None
            if not game.tanks[0].alive:
                return {}
            self._observe_action_effect(game)
            phase = "combat" if game.tanks[1].alive else "post_kill"
            self._poll(game, phase)

            if phase == "combat" and self.field is not None:
                target = _cell(game, game.tanks[1])
                if self.field.target_cell == target:
                    self._update_live_chain(game, self.field)
            self.observed_previous_action = (
                self._effect_action if self._effect_action is not None
                else (1, 1, 0))
            self.observed_commit_remaining = self.commit_remaining
            self.observed_committed_action = self.committed_action
            if self.action_no_effect:
                self.commit_remaining = 0

            if self.awaiting_post_fire_plan:
                action = self._take_ready(game, phase)
                self._submit(game, phase)
                if action is not None:
                    self.awaiting_post_fire_plan = False
                    return self._emit_action(
                        game, action, "post_fire_fresh_plan")
                self.async_wait_frames += 1
                self.post_fire_wait_frames += 1
                return self._emit_action(
                    game, (1, 1, 0), "post_fire_wait")

            if phase == "combat" and self._verified_hit(game):
                self.commit_remaining = 0
                # Do not launch another pre-shot snapshot.  The next accepted
                # movement plan must include the bullet created by this shot.
                self._ready = None
                self.awaiting_post_fire_plan = True
                return self._emit_action(game, (1, 1, 1), "forced_fire")

            if self.commit_remaining > 0 and not game.tanks[0].hit_something:
                self.commit_remaining -= 1
                self._submit(game, phase)
                action = self.committed_action
                if phase == "post_kill":
                    action = (action[0], action[1], 0)
                return self._emit_action(game, action, "hold")

            action = self._take_ready(game, phase)
            self._submit(game, phase)
            if action is not None:
                return self._emit_action(game, action, "async_plan")

            # Non-blocking fallback: preserve the last teacher locomotion while
            # a fresh exact plan is in flight.  Never invent a firing action.
            fallback = self.last_motion_action
            if phase == "post_kill" or fallback[2]:
                fallback = (fallback[0], fallback[1], 0)
            self.async_wait_frames += 1
            return self._emit_action(game, fallback, "async_wait")
        finally:
            self.main_act_seconds.append(time.perf_counter() - started)

    def telemetry(self):
        result = super().telemetry()
        main = np.asarray(self.main_act_seconds, dtype=np.float64)
        worker = np.asarray(self.worker_seconds, dtype=np.float64)
        ages = np.asarray(self.result_age_frames, dtype=np.float64)
        result.update({
            "opponent_profile": self.opponent_profile,
            "async_plans": self.async_plans,
            "async_results": self.async_results,
            "stale_results": self.stale_results,
            "deadline_results": self.deadline_results,
            "async_wait_frames": self.async_wait_frames,
            "post_fire_wait_frames": self.post_fire_wait_frames,
            "result_age_mean_frames": 0.0 if not len(ages) else ages.mean(),
            "result_age_p95_frames": 0.0 if not len(ages)
            else np.percentile(ages, 95),
            "main_act_mean_ms": 0.0 if not len(main) else 1000 * main.mean(),
            "main_act_p95_ms": 0.0 if not len(main)
            else 1000 * np.percentile(main, 95),
            "worker_mean_ms": 0.0 if not len(worker)
            else 1000 * worker.mean(),
            "worker_p95_ms": 0.0 if not len(worker)
            else 1000 * np.percentile(worker, 95),
        })
        return result
