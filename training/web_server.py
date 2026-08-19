#!/usr/bin/env python3
"""Local HTTP bridge for the Tank Trouble browser arena.

The browser is only a renderer and input surface.  The authoritative 25 FPS
simulation, scoring and AI decisions stay in the existing Python engine.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import random
import sys
import threading
import time
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tank_trouble_original.game import Game  # noqa: E402
from tank_trouble_original.laika import LaikaAI  # noqa: E402
from training.baselines import HunterPolicy, RandomPolicy  # noqa: E402
from training.p27_guided_search import P27GuidedSearchPolicy  # noqa: E402
from training.p27_risk_value import P27BRiskValuePolicy  # noqa: E402
from training.sparse_exact_safety_policy import (  # noqa: E402
    SparseExactSafetyPolicy,
)


BASE_NET = os.path.join(ROOT, "training/models/p26_amortized_mpc_iter05.pt")
VALUE_NET = os.path.join(ROOT, "training/models/p27b_risk_value_iter00.pt")
EMPTY_CONTROLS = {
    "forward": False,
    "backup": False,
    "turn_left": False,
    "turn_right": False,
    "fire": False,
}
POLICY_LABELS = {
    "p27-exact-shield": "P27b + Exact Shield（最强·慢）",
    "p27-hybrid": "P27b Hybrid（公平实验）",
    "p27b": "P27b（流畅）",
    "laika": "Laika",
    "hunter": "Hunter",
    "random": "Random",
    "human": "你",
}
# Keep deployment decisions at the 25 Hz physics rate. A paired 60-seed test
# found that holding P27b for two frames reduced true win rate from 93.3% to
# 83.3% on that sample, so action-repeat is not a valid latency shortcut.
POLICY_DECISION_INTERVALS = {}


def _controls(value=None):
    value = value or {}
    return {key: bool(value.get(key, False)) for key in EMPTY_CONTROLS}


def _apply(tank, controls):
    for key, value in _controls(controls).items():
        setattr(tank, key, value)


class LaikaPolicy:
    """Expose the original controller through the normal policy interface."""

    name = "laika"

    def __init__(self):
        self.reset()

    def reset(self):
        self.controller = None
        self.tank = None

    def act(self, game):
        tank = game.tanks[0]
        if not tank.alive:
            return {}
        if tank is not self.tank:
            self.tank = tank
            self.controller = LaikaAI(game, tank)
        if self.controller.make_decisions_and_update_goal():
            self.controller.decide_actions_to_achieve_goal()
        self.controller.set_input_to_do_actions()
        return {key: bool(getattr(tank, key)) for key in EMPTY_CONTROLS}


@contextmanager
def tank_perspective(game, side):
    """Temporarily expose ``side`` as tank0 to an existing policy.

    Training policies consistently use tank0 as self.  Swapping the two tank
    entries, their public cell records and temporary numbers gives the exact
    same observation convention for a right-side policy.  The swap only lasts
    for the read-only ``policy.act`` call and is restored before the live step.
    """
    if side == 0:
        yield game
        return
    tanks = game.tanks
    fields = game.tank_fields
    tanks[0], tanks[1] = tanks[1], tanks[0]
    fields[0], fields[1] = fields[1], fields[0]
    tanks[0].number, tanks[1].number = 0, 1
    try:
        yield game
    finally:
        tanks[0].number, tanks[1].number = 1, 0
        tanks[0], tanks[1] = tanks[1], tanks[0]
        fields[0], fields[1] = fields[1], fields[0]


class Arena:
    FPS = 25

    def __init__(self, seed=970000):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.seed = int(seed)
        self.mode = "watch"
        self.left_policy_name = "p27b"
        self.right_policy_name = "laika"
        self.human_controls = dict(EMPTY_CONTROLS)
        self.paused = False
        self.policy_cache = {}
        self.streak = 0
        self.best_streak = 0
        self.last_events = []
        self.last_step_ms = 0.0
        self.last_decision_ms = [0.0, 0.0]
        self.held_controls = [dict(EMPTY_CONTROLS), dict(EMPTY_CONTROLS)]
        self.step_timestamps = deque(maxlen=60)
        self._new_game(self.seed, keep_scores=False)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _new_game(self, seed, keep_scores=True):
        old_scores = list(self.game.scores) if keep_scores and hasattr(
            self, "game") else [0, 0]
        self.seed = int(seed)
        # Keep native Laika attached when selected so exact-state clones also
        # contain its RNG and hidden goal stack.  Other opponents are driven
        # externally by the arena.
        self.game = Game(seed=self.seed, ai_enabled=True)
        self.game.scores = old_scores
        self._sync_native_opponent()
        self._reset_policies()

    def _sync_native_opponent(self):
        native_laika = self.right_policy_name == "laika"
        self.game.ai_enabled = native_laika
        tank = self.game.tanks[1]
        if native_laika:
            if tank.ai is None:
                tank.ai = LaikaAI(self.game, tank)
        else:
            tank.ai = None

    def _reset_policies(self):
        self.held_controls = [dict(EMPTY_CONTROLS), dict(EMPTY_CONTROLS)]
        for policy in self.policy_cache.values():
            policy.reset()
            if hasattr(policy, "_web_latencies"):
                policy._web_latencies.clear()

    def _make_policy(self, name, side):
        key = (side, name)
        if key in self.policy_cache:
            return self.policy_cache[key]
        if name == "p27-exact-shield":
            policy = SparseExactSafetyPolicy(
                base_net=BASE_NET, value_net=VALUE_NET,
                top_k=12, search_horizon=72,
                search_max_death=0.0, search_max_dd=0.0,
                successor_shield=True, successor_horizon=72,
                successor_shield_max_safe_roots=2,
                suppress_secured_fire=True, min_unsecured_fire_gain=2.0,
                audit_interval=1, proactive_interval=48,
                behavior_full_search=True, search_hold_frames=12,
                search_on_fire=True, risk_search_threshold=0.18,
                long_tail_fire_horizon=375,
                deterministic_search_seeds=True,
                unsafe_fallback_mode="strip_fire")
            policy.set_round_seed(self.seed)
        elif name == "p27-hybrid":
            policy = P27GuidedSearchPolicy(
                base_net=BASE_NET, value_net=VALUE_NET,
                top_k=4, horizon=36, hold=8, deadline_ms=30.0,
                action_hold_frames=6, prior_refresh_frames=6,
                seed=self.seed + side * 7919)
        elif name == "p27b":
            policy = P27BRiskValuePolicy(
                base_net=BASE_NET, value_net=VALUE_NET, fire_margin=0.16)
        elif name == "hunter":
            policy = HunterPolicy()
        elif name == "random":
            policy = RandomPolicy(seed=self.seed + side * 7919)
        elif name == "laika":
            policy = LaikaPolicy()
        else:
            raise ValueError(f"unknown policy: {name}")
        policy._web_latencies = deque(maxlen=500)
        self.policy_cache[key] = policy
        return policy

    def _policy_controls(self, side, name):
        policy = self._make_policy(name, side)
        started = time.perf_counter()
        with tank_perspective(self.game, side):
            result = policy.act(self.game)
        self.last_decision_ms[side] = (time.perf_counter() - started) * 1000.0
        policy._web_latencies.append(self.last_decision_ms[side])
        return _controls(result)

    def _decide(self, side):
        if side == 0 and self.mode == "play":
            self.last_decision_ms[side] = 0.0
            return dict(self.human_controls)
        name = self.left_policy_name if side == 0 else self.right_policy_name
        if side == 1 and name == "laika":
            # Tank.update() owns native Laika exactly once per live frame.
            self.last_decision_ms[side] = 0.0
            return dict(EMPTY_CONTROLS)
        interval = POLICY_DECISION_INTERVALS.get(name, 1)
        if self.game.frame % interval:
            self.last_decision_ms[side] = 0.0
            return dict(self.held_controls[side])
        controls = self._policy_controls(side, name)
        self.held_controls[side] = dict(controls)
        return controls

    def _on_round_events(self, events):
        for event in events:
            if event[0] == "round_end":
                if event[1] == 0:
                    self.streak += 1
                    self.best_streak = max(self.best_streak, self.streak)
                else:
                    self.streak = 0
            elif event[0] == "new_round":
                self._sync_native_opponent()
                self._reset_policies()

    def _run(self):
        target = 1.0 / self.FPS
        deadline = time.perf_counter()
        while not self.stop_event.is_set():
            if self.paused:
                time.sleep(0.02)
                deadline = time.perf_counter()
                continue
            started = time.perf_counter()
            with self.lock:
                # No controls can affect a frozen scoring tail.  Skipping both
                # networks here removes hundreds of milliseconds of wasted
                # inference between rounds.
                if self.game.frozen:
                    left = right = dict(EMPTY_CONTROLS)
                    self.last_decision_ms = [0.0, 0.0]
                else:
                    left = self._decide(0)
                    right = self._decide(1)
                _apply(self.game.tanks[0], left)
                _apply(self.game.tanks[1], right)
                events = list(self.game.step())
                self.last_events = [list(event) for event in events]
                self._on_round_events(events)
                self.last_step_ms = (time.perf_counter() - started) * 1000.0
                self.step_timestamps.append(time.perf_counter())
            deadline += target
            time.sleep(max(0.001, deadline - time.perf_counter()))
            if time.perf_counter() - deadline > target * 4:
                deadline = time.perf_counter()

    def command(self, payload):
        action = payload.get("action")
        with self.lock:
            if action == "input":
                self.human_controls = _controls(payload.get("controls"))
            elif action == "mode":
                mode = payload.get("mode", self.mode)
                if mode not in ("watch", "play", "selfplay"):
                    raise ValueError("invalid mode")
                self.mode = mode
                left = payload.get("left_policy", self.left_policy_name)
                right = payload.get("right_policy", self.right_policy_name)
                if left not in POLICY_LABELS or right not in POLICY_LABELS:
                    raise ValueError("invalid policy")
                self.left_policy_name = (
                    "p27b" if left == "human" else left)
                self.right_policy_name = "laika" if right == "human" else right
                self._sync_native_opponent()
                self._reset_policies()
            elif action == "new_maze":
                value = payload.get("seed")
                new_seed = random.randrange(1, 2 ** 31) if value in (
                    None, "", "random") else int(value)
                self._new_game(new_seed, keep_scores=True)
            elif action == "reset_score":
                self.game.scores = [0, 0]
                self.streak = 0
                self.best_streak = 0
            elif action == "pause":
                self.paused = bool(payload.get("paused", not self.paused))
            else:
                raise ValueError("unknown action")

    def _policy_telemetry(self, side, name):
        policy = self.policy_cache.get((side, name))
        if policy is not None and hasattr(policy, "telemetry"):
            return policy.telemetry()
        if policy is not None and hasattr(policy, "policy_frames"):
            samples = sorted(getattr(policy, "_web_latencies", ()))

            def percentile(frac):
                if not samples:
                    return 0.0
                index = min(len(samples) - 1, int(frac * (len(samples) - 1)))
                return float(samples[index])

            frames = max(1, int(policy.policy_frames))
            searches = int(policy.exact_searches)
            unsafe = int(policy.unsafe_searches)
            return {
                "frames": frames,
                "searches": searches,
                "search_rate": searches / frames,
                "search_changes": unsafe,
                "change_rate": unsafe / max(1, searches),
                "deadline_hits": 0,
                "candidates_evaluated": int(policy.exact_candidates),
                "last_decision_ms": self.last_decision_ms[side],
                "p50_ms": percentile(0.50),
                "p95_ms": percentile(0.95),
                "last_reason": (
                    (policy.last_temporal_sample or {}).get(
                        "reason", "network")),
                "audits": int(policy.audit_frames),
                "unsafe_audits": int(policy.unsafe_audits),
                "fire_rejections": int(policy.long_tail_fire_rejections),
            }
        if policy is not None:
            samples = sorted(getattr(policy, "_web_latencies", ()))

            def percentile(frac):
                if not samples:
                    return 0.0
                index = min(len(samples) - 1, int(frac * (len(samples) - 1)))
                return float(samples[index])

            return {
                "frames": len(samples),
                "searches": 0,
                "search_rate": 0.0,
                "search_changes": 0,
                "change_rate": 0.0,
                "deadline_hits": 0,
                "last_decision_ms": self.last_decision_ms[side],
                "p50_ms": percentile(0.50),
                "p95_ms": percentile(0.95),
                "last_reason": "network",
            }
        return {
            "frames": 0,
            "searches": 0,
            "search_rate": 0.0,
            "search_changes": 0,
            "change_rate": 0.0,
            "deadline_hits": 0,
            "last_decision_ms": self.last_decision_ms[side],
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "last_reason": "network" if name == "p27b" else name,
        }

    def state(self):
        with self.lock:
            game = self.game
            effective_fps = 0.0
            if len(self.step_timestamps) >= 2:
                elapsed = self.step_timestamps[-1] - self.step_timestamps[0]
                effective_fps = ((len(self.step_timestamps) - 1) / elapsed
                                 if elapsed > 0 else 0.0)
            left_name = "human" if self.mode == "play" else self.left_policy_name
            right_name = self.right_policy_name
            return {
                "connected": True,
                "mode": self.mode,
                "paused": self.paused,
                "seed": self.seed,
                "frame": game.frame,
                "round": game.round_number,
                "fps": self.FPS,
                "effective_fps": effective_fps,
                "frozen": game.frozen,
                "world_width": len(game.maze) * game.scale,
                "world_height": len(game.maze[0]) * game.scale,
                "wall_width": game.wall_half_t * 2,
                "bullet_radius": 3.5 * (game.scale / 50.0),
                "walls": [list(wall) for wall in game.walls],
                "scores": list(game.scores),
                "streak": self.streak,
                "best_streak": self.best_streak,
                "left_policy": self.left_policy_name,
                "right_policy": self.right_policy_name,
                "left_label": POLICY_LABELS[left_name],
                "right_label": POLICY_LABELS[right_name],
                "tanks": [
                    {
                        "index": tank.number,
                        "x": tank.x,
                        "y": tank.y,
                        "rotation": tank.rotation,
                        "display_scale": tank.display_scale,
                        "alive": tank.alive,
                        "bullets_fired": tank.bullets_fired,
                    }
                    for tank in game.tanks
                ],
                "bullets": [
                    {
                        "x": bullet.x,
                        "y": bullet.y,
                        "owner": bullet.owner.number,
                        "lifetime": bullet.lifetime,
                    }
                    for bullet in game.bullets if not bullet.removed
                ],
                "events": self.last_events,
                "runtime": {
                    "step_ms": self.last_step_ms,
                    "left_decision_ms": self.last_decision_ms[0],
                    "right_decision_ms": self.last_decision_ms[1],
                },
                "telemetry": self._policy_telemetry(
                    0, self.left_policy_name),
                "available_policies": [
                    {"value": key, "label": value}
                    for key, value in POLICY_LABELS.items()
                    if key != "human"
                ],
            }


class Handler(BaseHTTPRequestHandler):
    arena = None

    def _headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _json(self, value, status=200):
        self._headers(status)
        try:
            self.wfile.write(json.dumps(
                value, ensure_ascii=False,
                separators=(",", ":")).encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            # A browser can cancel an old poll while a long exact-state audit
            # holds the arena lock.  This is a normal disconnected client,
            # not a backend failure worth printing a traceback for.
            return

    def do_OPTIONS(self):
        self._headers(204)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self._json(self.arena.state())
        elif path == "/api/health":
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/command":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8"))
            self.arena.command(payload)
            self._json({"ok": True, "state": self.arena.state()})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--seed", type=int, default=970000)
    args = parser.parse_args()
    # Small batch-1 MLPs are substantially faster and more predictable without
    # PyTorch's default intra-op thread-pool overhead on laptop CPUs.
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    arena = Arena(seed=args.seed)
    Handler.arena = arena
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Tank Trouble arena: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        arena.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
