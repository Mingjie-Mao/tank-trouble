#!/usr/bin/env python3
"""Record the Python port frame by frame for the JS engine parity test.

The JS engine (`web/lib/engine/`) is a translation of `tank_trouble_original/`.
This script plays fixed seeds with tank 1 under Laika and tank 0 under a
scripted input stream both languages reproduce exactly. Per frame it writes:

* a SHA-1 over every discrete part of the state — round and settlement
  counters, scores, crate timer, each tank's alive flag, ammo, trigger, wall
  contact and buttons, each bullet's name, owner, lifetime and owner-exit
  flag, and every event — which the JS engine must match exactly;
* every continuous part — tank poses, bullet positions and velocities —
  rounded to 1e-6, which the JS engine must match within 1e-6.

Floats are not compared bit for bit because Python's `math.sin`/`math.cos`
come from the platform libm and V8 ships its own: the two differ in the last
bit for some angles, so exact equality holds on some seeds and not others.
Measured over these seeds the difference never exceeds ~6e-14 and never
changes a discrete outcome.

    python3 tools/record_engine_parity.py   # writes web/tests/fixtures/engine-parity.json.gz
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tank_trouble_original import Game  # noqa: E402

SEEDS = [1, 7, 42, 970000, 970123, 990000, 2024, 31337, 123456789, 4000000, 2**33 + 5, 0]
FRAMES = 2500
OUT = ROOT / "web" / "tests" / "fixtures" / "engine-parity.json.gz"


class Driver:
    """xorshift32 inputs for tank 0, held for 3..10 frames, fire on ~1 frame in 6."""

    def __init__(self, seed):
        self.s = (seed * 2654435761 + 1) & 0xFFFFFFFF or 1
        self.hold = 0
        self.buttons = (False, False, False, False, False)

    def next32(self):
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s & 0xFFFFFFFF
        return self.s

    def step(self):
        if self.hold <= 0:
            r = self.next32()
            throttle, turn = r % 3, (r // 3) % 3
            self.buttons = (throttle == 2, throttle == 0, turn == 0, turn == 2, (r // 9) % 6 == 0)
            self.hold = 3 + (r // 54) % 8
        self.hold -= 1
        return self.buttons


def b(v):
    return "1" if v else "0"


def snapshot(g, events):
    """(SHA-1 of the discrete state, list of continuous values rounded to 1e-6)."""
    parts = [str(g.frame), str(g.round_number), str(g.alive_count), str(g.end_count),
             str(g.reset_count), b(g.frozen), f"{g.crate_timer:.9f}",
             ",".join(map(str, g.scores))]
    floats = []
    for t in g.tanks:
        parts.append("T" + ",".join([b(t.alive), str(t.bullets_fired), b(t.trigger_released),
                                     b(t.hit_something), b(t.forward), b(t.backup),
                                     b(t.turn_left), b(t.turn_right), b(t.fire)]))
        floats += [t.x, t.y, t.rotation]
    for bl in g.bullets:
        parts.append("B" + ",".join([bl.name, str(bl.owner.number), str(bl.lifetime),
                                     b(bl.has_exited_owner)]))
        floats += [bl.x, bl.y, bl.x_speed, bl.y_speed]
    for e in events:
        parts.append("E" + ",".join("null" if v is None else str(v) for v in e))
    return (hashlib.sha1("|".join(parts).encode()).hexdigest()[:16],
            [round(x, 6) for x in floats])


def record(seed):
    g = Game(seed=seed, ai_enabled=True)
    drive = Driver(seed)
    out = [snapshot(g, [])]
    for _ in range(FRAMES):
        if not g.frozen:
            t = g.tanks[0]
            t.forward, t.backup, t.turn_left, t.turn_right, t.fire = drive.step()
        events = g.step()
        out.append(snapshot(g, events))
    return {"seed": seed, "rounds": g.round_number, "frames": out}


def main():
    cases = [record(s) for s in SEEDS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(gzip.compress(json.dumps({"frames": FRAMES, "cases": cases}).encode(), 9))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB): "
          + ", ".join(f"{c['seed']}:{c['rounds']} rounds" for c in cases))


if __name__ == "__main__":
    main()
