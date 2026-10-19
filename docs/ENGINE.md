# Engine — the original game in Python and JavaScript, checked frame by frame

`tank_trouble_original/` is a Python port of the decompiled Flash game
(`swf_decompiled/`; mapping in [PORT_NOTES.md](PORT_NOTES.md)), checked by
`test_original_port.py`. `web/lib/engine/` translates it module by module to
JavaScript; the browser arena, the RL environment, Laika and Tactical all run on it.

## Parity

`tools/record_engine_parity.py` plays 12 seeds × 2,500 frames (156 rounds,
every reset included) in Python with Laika on tank 1 and a scripted input
stream on tank 0, recording per frame a hash of the discrete state (counters,
scores, flags, ammo, buttons, bullets, events) and the continuous state
rounded to 1e-6. `web/tests/engine-parity.test.mjs` replays it in JS:
**the discrete state matches on all 30,000 frames and every float is within
1e-6.**

This needed CPython's `random.Random` reproduced bit for bit (MT19937, its
seeding, 53-bit `random()`, rejection-sampled `randrange()`), so a seed gives
the same maze, spawns and Laika coin flips in both languages, and Python's
round-half-to-even in Laika. Floats agree to ~1e-14 rather than exactly
because Python's libm and V8 round some sines and cosines differently in the
last bit; it never changes a discrete outcome.

## What the original does

**Frame order.** AVM1 runs the most recently created clips first (Ruffle,
`core/src/avm1/runtime.rs`: `add_to_exec_list`, `run_frame`). Every frame:

1. every bullet, newest first — move, bounce, hit-test the tanks where the
   previous frame left them;
2. tank 1, with Laika deciding inside its update;
3. tank 0;
4. the root timeline's round logic — settlement countdown, freeze, reset.

**Geometry** (from the SWF). The bullet hit test covers the whole tank —
hull, turret and barrel — with no exemption for the shooter. The barrel is
x ∈ [−7.5, 7.5], y ∈ [−55, −21.25] in tank units; the muzzle sits inside it at
y = −51.14.

**Consequences.** A fresh bullet first moves on the next frame, ahead of its
shooter, so it has left the barrel before its first hit test: firing while
driving is safe without any exemption, and firing point-blank into a wall
kills the shooter (`test_original_port.py` checks both). Laika decides against
tank 0's position from the previous frame.
