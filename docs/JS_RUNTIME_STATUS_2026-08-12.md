# JavaScript runtime checkpoint — 2026-08-12

This checkpoint is intentionally local and has not been pushed.

## What is deployed locally

- Browser-native fixed-step physics at 25 FPS.
- Browser-native Laika.
- Browser-native KillField search as an explicitly labelled third-party MIT
  comparison baseline.
- Tank Trouble Tactical: frozen H36 attack/search plus verified two-stage
  visible-bullet evasion and full settlement-window survival planning.
- Watch, human-vs-AI, and AI-vs-AI modes.
- A runtime switch that preserves the Python P27b / Hybrid / Exact Shield
  research stack.

## Runtime results

| Scenario | Frames | Mean | p95 | Result |
|---|---:|---:|---:|---|
| KillField JS vs Laika (matched checkpoint) | 5,000 | 2.594 ms | 9.620 ms | real-time |
| KillField JS self-play | 1,000 | 8.610 ms | 11.618 ms | real-time |
| H48 deeper search | 1,000 | 4.099 ms | 15.473 ms | real-time, weaker |
| Consensus safety candidate | 1,000 | 3.425 ms | 19.859 ms | real-time, weaker |
| Two-stage search M15 | 1,000 | 6.169 ms | 28.755 ms | real-time, weaker |
| Tank Trouble Tactical (matched checkpoint) | 5,000 | 2.843 ms | 10.696 ms | real-time, promoted |

The 40 ms physical-frame budget is met at p95. The matched maximums were
32.047 ms for KillField and 71.801 ms for Tactical. The entire browser arena,
including physics, Laika and both search layers, now runs in a dedicated Web
Worker. That does not shorten the rare search spike, but it prevents that work
from blocking Canvas controls and rendering on the main thread.

## Strength checks

All JavaScript comparisons below use the same JS engine, same seed block
`970000..970059`, and the original `round_end` scoring rule.

| Policy | Wins | Losses | Double deaths | Delta vs H36 | Decision |
|---|---:|---:|---:|---:|---|
| KillField H36 baseline | 58/60 (96.7%) | 2 | 0 | — | keep as baseline |
| Safety prototype | 55/60 (91.7%) | 2 | 3 | -3 wins | rejected |
| Consensus safety | 56/60 (93.3%) | 3 | 1 | -2 wins | rejected |
| H40 | 56/60 (93.3%) | 4 | 0 | -2 wins | rejected |
| H42 | 51/60 (85.0%) | 4 | 5 | -7 wins | rejected |
| H44 | 54/60 (90.0%) | 4 | 2 | -4 wins | rejected |
| H48 | 56/60 (93.3%) | 1 | 3 | -2 wins | rejected |
| Two-stage M15 | 54/60 (90.0%) | 4 | 2 | -4 wins | rejected |

The strict paired H48 matrix was: two baseline losses repaired, but one new
loss and three new double deaths. Deeper planning changes more correct actions
than it fixes on this block.

The two-stage candidate evaluated `a0 for 8 frames -> a1 for 28 frames` over
three root actions and ten continuations, gated by the H36 action margin. A
20-seed screen looked promising, but the required 60-seed comparison rejected
it. After fixing the live action commitment to match the planned eight-frame
first stage, it repaired both H36 losses but introduced four losses and two
double deaths. Failure traces also exposed repeated low-risk overrides with
small score differences. It remains test code, not a selectable web policy.

## Promoted Tactical result

Failure attribution on the separate `980000..980059` development block found
two concrete gaps rather than a general need for deeper attack search:

- H36 could continue a hold or forced-fire action while a visible bullet was
  already on a lethal path.
- After killing Laika, its one-action settlement policy could turn the win
  into a double death before the scoring freeze.

Tank Trouble Tactical keeps H36 unchanged in ordinary play. It opens a small
two-stage safety search only when an engine rollout proves the current action
dies to bullets that are already visible; after a kill it plans across the
entire deterministic settlement window. The deployable verifier uses no
Laika-private goal state and does not copy the live RNG.

| Paired block | KillField H36 | Tank Trouble Tactical | Delta | Regressions |
|---|---:|---:|---:|---:|
| Development `980000..980059` | 53/60 (88.3%) | 60/60 (100.0%) | +7 | 0 |
| Blind A `990000..990099` | 86/100 (86.0%) | 94/100 (94.0%) | +8 | 2 |
| Blind B `1000000..1000099` | 87/100 (87.0%) | 95/100 (95.0%) | +8 | 0 |
| Blind C `1010000..1010099` | 89/100 (89.0%) | 96/100 (96.0%) | +7 | 1 |
| Blind combined | 262/300 (87.3%) | 285/300 (95.0%) | +23 | 3 |

The first blind block changed 12 outcomes: ten non-wins became wins and two
wins became later losses. The second changed eight outcomes and all eight were
repairs. The third changed nine outcomes: eight repairs and one regression.
Across all three untouched blocks, 26 non-wins became wins and three wins
became losses. The exact paired McNemar two-sided p-value is 0.0000152. The
Wilson 95% intervals are 83.1–90.6% for KillField and 91.9–97.0% for Tactical.
This passes the product promotion gate and is now the default Browser JS
controller; KillField remains selectable as the comparison baseline.

## Web UI checkpoint

- The page now follows KillField's light, compact visual language: warm-white
  background, thin borders, teal primary controls, inline scoreboard and
  compact pill buttons.
- The arena remains a large-screen version rather than copying the upstream
  752 px layout. At a 1280×720 viewport the live canvas measured 1189×660 with
  no horizontal overflow and sustained 25.1 FPS.
- Browser simulation now has a dedicated Worker owner and a message-based
  command surface. A 400-frame deterministic trajectory test, including a
  round reset, is bit-for-bit identical to direct `BrowserArena` execution.
- Watch, human play, self-play, runtime switching, pause and fullscreen remain
  available. The Worker bundle is included in the production build.

## P27b action-repeat check

The Python P27b browser had temporarily held each network action for two
physical frames. A paired 60-seed evaluation rejected that shortcut:

| Deployment | True win | Loss | Double death |
|---|---:|---:|---:|
| P27b every frame | 93.3% | 5.0% | 1.7% |
| P27b hold 2 | 83.3% | 13.3% | 3.3% |

Outcome changed on 10/60 games and win rate fell by 10 points. The web bridge
therefore runs P27b every physical frame again.

Python and JavaScript percentages above are **not cross-runtime comparisons**:
their RNGs generate different mazes and trajectories. Only paired comparisons
within one runtime are treated as causal evidence.

## Next product gate

1. Freeze this 95.0% controller and keep the three blind blocks untouched.
2. Attribute the remaining 15 non-wins and the three paired regressions in
   offline shadow mode. Prioritise immediate new-shot danger and own-bullet
   self-kills; do not widen the live gate without a new independent block.
3. Add a fair, visible-state opponent fire envelope only when no current-bullet
   two-stage escape survives. This is a sparse fallback, not a deeper attack
   search on every frame.
4. Add Random, Hunter and Dodger opponent suites before claiming general
   superiority. Keep Laika as the product benchmark, not the only test.
5. Use P27b later only as a cheap search gate or action prior if it reduces CPU
   without losing paired wins. The promoted browser controller remains the
   deployment default until an identical benchmark proves otherwise.
