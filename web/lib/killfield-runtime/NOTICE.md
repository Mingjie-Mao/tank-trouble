# Third-party JavaScript runtime baseline

The files under `src/` and `test/suite.js` are vendored from
[`Cichlider/killfield`](https://github.com/Cichlider/killfield) at commit
[`67d6a836993bde1c406f91aa7d3e544203a3b4af`](https://github.com/Cichlider/killfield/commit/67d6a836993bde1c406f91aa7d3e544203a3b4af)
("Restore original in-page UI", 2026-08-12).

They are included as a browser-native physics/search **comparison baseline**.
They are not presented as the P27b model or as original work from this
repository. The upstream MIT license is preserved in `LICENSE`.

## Local modifications

The engine files (`constants.js`, `rng.js`, `game.js`, `maze.js`) are
byte-identical to upstream. Three files carry local changes and are therefore
**not** a faithful copy of the upstream commit:

| File | Change |
|---|---|
| `src/killfield/teacher.js` | Added `continuityMargin`, fire-opportunity and movement-switch telemetry used by this repository's promotion gates. |
| `src/killfield/sandbox.js` | Bullet ownership is resolved by position in the supplied view rather than by `tank.number`, so mirrored right-side rollouts attribute live bullets correctly. |
| `src/laika.js` | Dropped one unused import. |

## Upstream has moved on

Upstream `main` is at `40e94eb` (2026-08-14), **14 commits ahead** of the
vendored snapshot. Those commits include substantial agent and engine changes —
fire-continuation rollouts, frictional wall sliding, refined wall-contact
physics, hunt urgency, and ammo pressure — so any KillField score quoted by
this repository describes the 2026-08-12 snapshot above, not the current
upstream agent.

Local integration code lives outside this directory in `web/lib/browser-arena.js`.
