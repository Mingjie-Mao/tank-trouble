"""Torch-free post-kill survival scoring shared by the runtime and training.

``post_kill_survival_scores`` is pure numpy plus exact sandbox rollouts, but it
used to live in ``killfield_fast_distill``.  Importing it therefore pulled the
whole distillation stack, and through ``killfield_student`` it pulled torch —
about 0.77s of the 0.91s needed to import ``killfield_realtime``.  Every spawned
realtime worker paid that cold start before it could run a plan that itself
costs roughly 14ms.

This module owns the function and the two constants it needs, and imports only
torch-free dependencies.  ``killfield_distill`` and ``killfield_fast_distill``
re-export from here, so existing import sites keep working unchanged.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original import constants as C
from training.mpc_agent import CANDIDATES, make_sandbox
from training.survival_expert_iter_530 import apply_action


SCORE_SCALE = 12_000.0
POST_KILL_FIRE_PENALTY = 3000.0


def post_kill_survival_scores(game, horizon=75):
    """Exact offline survival labels for the original live-bullet window.

    Each movement is held in a sandbox until scoring freeze or the requested
    horizon.  The real teacher replans every two frames, so this is a receding
    horizon controller rather than a claim that one command is globally best.
    """
    scores = np.full(len(CANDIDATES), -1e9, dtype=np.float32)
    remaining = max(
        1, game.end_count - C.NUMBEROFFRAMESFROZEN)
    rollout_frames = min(int(horizon), int(remaining))
    for move_index, (throttle, turn) in enumerate(
            (action[:2] for action in CANDIDATES[::2])):
        sandbox = make_sandbox(game, "L1", rng_seed=0)
        me = sandbox.tanks[0]
        start_x, start_y = me.x, me.y
        action = (throttle, turn, 0)
        apply_action(sandbox, action)
        min_clearance = 8.0
        survived = True
        elapsed = 0
        for elapsed in range(rollout_frames):
            events = sandbox.step()
            if not me.alive:
                survived = False
                break
            if sandbox.bullets:
                clearance = min(
                    math.hypot(b.x - me.x, b.y - me.y)
                    for b in sandbox.bullets) / max(sandbox.scale, 1e-6)
                min_clearance = min(min_clearance, clearance)
            if sandbox.frozen or any(
                    event[0] == "round_end" for event in events):
                break

        if survived:
            displacement = math.hypot(me.x - start_x, me.y - start_y) \
                / max(sandbox.scale, 1e-6)
            control_cost = (
                0.20 * float(throttle != 1)
                + 0.10 * float(turn != 1))
            score = (SCORE_SCALE + 40.0 * min(min_clearance, 8.0)
                     + 0.5 * min(displacement, 8.0) - control_cost)
        else:
            score = -SCORE_SCALE + 8.0 * elapsed
        no_fire_index = move_index * 2
        scores[no_fire_index] = score
        scores[no_fire_index + 1] = score - POST_KILL_FIRE_PENALTY
    return scores
