# Own browser engine — the Python port, in JS, checked frame by frame

Checkpoint: 2026-10-09. Code: `web/lib/engine/` (constants, rng, maze, game,
laika), `tools/record_engine_parity.py`, `web/tests/engine-parity.test.mjs`.

## What changed

The browser arena, the RL environment, Laika, Tactical and the planner under it
used to run on a vendored JS engine snapshot. They now run on
`web/lib/engine/`, a module-by-module translation of this repository's Python
port `tank_trouble_original/` — itself ported from the decompiled Flash source
and checked by `test_original_port.py`. The vendored engine files are gone; only
the third-party planner remains vendored, now importing this engine.

## How it is checked

`tools/record_engine_parity.py` plays 12 seeds for 2500 frames each — 152
rounds, including every round reset — with Laika on tank 1 and tank 0 under a
scripted input stream both languages reproduce exactly. For every frame it
records:

- a SHA-1 of the **discrete state** — round and settlement counters, scores,
  crate timer, each tank's alive flag, ammo, trigger, wall contact and buttons
  (Laika's included), each bullet's name, owner, lifetime and owner-exit flag,
  and every event;
- the **continuous state** — tank poses, bullet positions and velocities —
  rounded to 1e-6.

`web/tests/engine-parity.test.mjs` replays the same seeds and inputs. Result:
**the discrete state matches on every one of the 30,000 frames, and every
float is within 1e-6.**

Two things were needed to get there:

- **The random stream.** The engine reimplements CPython's `random.Random`
  bit for bit — MT19937, CPython's integer seeding over the seed's 32-bit words,
  `random()`'s 53-bit construction and `randrange()`'s rejection sampling — so
  a seed produces the same maze, spawns, headings and Laika coin flips as the
  Python port.
- **Python's rounding.** Laika quantises headings with Python's `round()`,
  which rounds halves to even; `Math.round` rounds them up.

Floats are matched within a tolerance rather than bit for bit because Python's
`math.sin`/`math.cos` come from the platform libm and V8 ships its own: they
differ in the last bit for some angles. On three of the twelve seeds the runs
are bit-identical throughout; on the rest the largest difference is ~6e-14, it
resets with every round, and it never changes a discrete outcome. Bit-exact
agreement across the two languages is not available, and on another platform it
would not be available for the Python port against itself either.

## Rule differences from the previous engine

| | This engine | Previous engine |
|---|---|---|
| A tank's own bullet becomes able to hit it | once it has been **outside the tank's hull** at a frame's hit test | once it has **bounced** |
| Random stream | CPython MT19937: same maze per seed as the Python port | mulberry32: different mazes |
| Laika's heading quantisation | ties to even (Python) | ties up |
| Wall sliding | none — the original collision model only | optional frictional sliding |

The two self-hit rules disagree in exactly one situation: a shot that bounces
before it has left the shooter's hull, i.e. firing point-blank into a wall.
This engine lets that bullet pass back through the shooter; the previous one
kills the shooter on the spot. Driving straight after one's own shot is safe
under both, because a tank (4 px/frame) cannot catch its bullet (4.5 px/frame)
once the bullet has cleared the 15 px barrel.

Removing wall sliding means policies once promoted under it — Tactical Smooth —
now run under the original collision model, and their published figures no
longer describe this build.

## What the original actually does

Reading the shapes out of the original SWF settles the geometry:

- The bullet hit test (`DefineSprite_175_bullet`) is `hitTest(x, y, true)`
  against the whole tank clip — hull, turret and barrel — with **no exemption
  for the shooter**.
- The barrel is the rectangle x ∈ [−7.5, 7.5], y ∈ [−55, −21.25] in tank units
  (shapes 80 and 82 in turret sprite 134), with a 0.05 hairline. The Python
  port's measured hit shape uses a half-width of 8.5; the exact value is 7.5.
- The muzzle sits at y = −51.14, **inside** the barrel, about one pixel short
  of its tip.

So in the original a fresh bullet starts inside its shooter, and nothing in the
bullet script exempts the shooter. Whether that ever kills the shooter depends
on which runs first within a frame.

### Open question: update order

The Python port updates the round logic, then the tanks, then the bullets. Under
that order a tank driving forward and firing would hit itself on the next frame
— the bullet gains only 0.5 px a frame on it — which is why both ports added an
exemption. If AVM1 instead runs the most recently created clips first, bullets
update before tanks, a fresh bullet clears the barrel before it is ever tested,
no exemption is needed, and a point-blank wall shot kills the shooter. That
last behaviour is what the *bounce* rule produces and the *exit* rule does not.

This is to be settled empirically in the original running under Ruffle, by two
tests: fire while driving straight, and fire with the barrel against a wall.
The answer decides whether the reference port changes its update order and
drops the exemption; the JS engine would follow, and the parity test would
confirm it still matches. Retraining on this engine waits for it.

## Performance

`Tank.hitCheck` takes the heading's sine and cosine once per call rather than
once per probe point and allocates nothing, with the same arithmetic per point:
the parity test is unchanged by it. Walls are bucketed on a dense integer grid.
