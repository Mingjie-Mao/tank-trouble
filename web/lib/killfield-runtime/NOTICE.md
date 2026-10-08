# Third-party planner

The files under `src/killfield/` are vendored from
[`Cichlider/killfield`](https://github.com/Cichlider/killfield) at commit
[`67d6a836993bde1c406f91aa7d3e544203a3b4af`](https://github.com/Cichlider/killfield/commit/67d6a836993bde1c406f91aa7d3e544203a3b4af)
("Restore original in-page UI", 2026-08-12): the density-field MPC planner
that Tactical builds on. The upstream MIT license is preserved in `LICENSE`.

The upstream game engine that was vendored alongside it has been removed. The
planner now runs on this repository's own engine, `web/lib/engine/`, a
translation of the Python port `tank_trouble_original/` that is checked frame
by frame against it (`web/tests/engine-parity.test.mjs`).

## Local modifications

| File | Change |
|---|---|
| all of `src/killfield/` | Engine imports point at `web/lib/engine/` instead of the removed upstream engine. |
| `src/killfield/teacher.js` | Added `continuityMargin`, fire-opportunity and movement-switch telemetry used by this repository's promotion gates. |
| `src/killfield/sandbox.js` | Bullet ownership is resolved by position in the supplied view rather than by `tank.number`, so mirrored right-side rollouts attribute live bullets correctly. |

Upstream has moved on since this snapshot; any KillField score quoted by this
repository describes the snapshot above, run on this repository's engine.
