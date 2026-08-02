"""Exact cloning and fidelity checks for the local Tank Trouble engine.

Unlike ``training.mpc_agent.make_sandbox``, this module intentionally keeps
the live RNG and Laika's internal goal/action state.  It is therefore for the
privileged local-teacher track, not for an observation-only fair-play agent.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from dataclasses import dataclass
from typing import Callable, Mapping

from tank_trouble_original.game import Game


Controls = Mapping[str, bool]
ControlSchedule = Callable[[int], Controls]


def clone_exact_game(game: Game, verify: bool = False) -> Game:
    """Return an independent clone that preserves every future-relevant bit."""
    cloned = copy.deepcopy(game)
    if verify and state_fingerprint(game) != state_fingerprint(cloned):
        raise RuntimeError("exact game clone does not match the source state")
    return cloned


def state_bytes(game: Game) -> bytes:
    """Serialize the complete object graph for strict fidelity comparisons."""
    return pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL)


def state_fingerprint(game: Game) -> str:
    return hashlib.sha256(state_bytes(game)).hexdigest()


def apply_controls(game: Game, controls: Controls) -> None:
    tank = game.tanks[0]
    tank.forward = bool(controls.get("forward", False))
    tank.backup = bool(controls.get("backup", False))
    tank.turn_left = bool(controls.get("turn_left", False))
    tank.turn_right = bool(controls.get("turn_right", False))
    tank.fire = bool(controls.get("fire", False))


@dataclass(frozen=True)
class FidelityResult:
    matched: bool
    frames_checked: int
    mismatch_frame: int | None = None
    source_events: tuple | None = None
    clone_events: tuple | None = None
    source_fingerprint: str | None = None
    clone_fingerprint: str | None = None


def verify_clone_trajectory(
        source: Game,
        schedule: ControlSchedule,
        frames: int,
        *,
        clone: Game | None = None) -> FidelityResult:
    """Advance source and clone under identical controls and compare each frame."""
    cloned = clone if clone is not None else clone_exact_game(source, verify=True)
    for frame in range(int(frames)):
        controls = schedule(frame)
        apply_controls(source, controls)
        apply_controls(cloned, controls)
        source_events = tuple(source.step())
        clone_events = tuple(cloned.step())
        source_hash = state_fingerprint(source)
        clone_hash = state_fingerprint(cloned)
        if source_events != clone_events or source_hash != clone_hash:
            return FidelityResult(
                matched=False,
                frames_checked=frame + 1,
                mismatch_frame=frame,
                source_events=source_events,
                clone_events=clone_events,
                source_fingerprint=source_hash,
                clone_fingerprint=clone_hash,
            )
    return FidelityResult(matched=True, frames_checked=int(frames))
