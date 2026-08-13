# Tank Trouble Tactical vs KillField — product comparison

Checkpoint: 2026-08-12. This comparison is local and has not been pushed.

## What the latest controller actually is

Tank Trouble Tactical is a new browser controller, not a merge that replaces
every P26–P41 implementation. It extends the frozen KillField H36 controller
with two narrowly gated safety planners:

1. During combat, it keeps the H36 action unless an engine rollout proves that
   already-visible bullets kill it. Only then does it search a two-stage,
   no-fire escape (`9 roots × 9 continuations`, split after four frames).
2. After Laika dies, it plans two no-fire stages across the full live settlement
   window so that an earned win is less likely to become a double death.

The P26/P27b pure-network line and the later P28–P41 research experiments remain
available as separate Python baselines, teachers and sources of ideas. The new
browser product inherits their main engineering lesson—freeze a strong base and
make sparse, verified corrections—but it does not execute all of their models.

## Complete comparison

| Dimension | KillField H36 | Tank Trouble Tactical (latest) | Verdict |
|---|---|---|---|
| Product role | Third-party MIT comparison baseline | Default browser controller in this project | Tactical promoted |
| Main decision method | Hand-scored receding-horizon search | Same frozen H36 attack controller plus verified safety planning | Evolution, not a wholesale replacement |
| Neural network | None | None in the browser policy | Both are search/rule/model based |
| Ordinary attack search | 10 selectable first actions, 36-frame rollout | Identical H36 search when no verified danger exists | No broad extra compute |
| Action representation | One rollout action, internally held for the opening segment; live commitment smooths motion | Same base action; dangerous states may use `a0 × 4 -> a1 × 32` | Tactical can express an evasive sequence |
| Fire logic | Engine-verified forced fire; moving+fire actions are masked | Same verified fire authority | Equal |
| Incoming-bullet safety | Risk score and base rollout; own-bullet guard | Adds an exact current-visible-bullet death check and 81 two-stage no-fire candidates | Tactical stronger on attributed failures |
| Post-kill safety | Replans one no-fire movement at a time | Plans two stages across the entire remaining settlement window | Tactical prevents more double deaths |
| Opponent model in attack search | L2: a fresh Laika script in rollouts | Same for inherited H36 attack search | Both are Laika-oriented here |
| Private live Laika state | Does not copy live goal/RNG into the sandbox | Safety verifier uses visible bullets and L1/frozen-opponent rollouts; does not copy live goal/RNG | Deployable layer stays visible-state based |
| World model | Browser-native deterministic physics sandbox | Same physics plus two extra verified safety rollouts | Equal foundation |
| Current web execution | Runs with the entire arena in the project's dedicated Web Worker | Runs in the same Worker; Canvas and controls remain on the main thread | Equal architecture; Tactical's extra spikes do not block UI |
| Mean decision time, matched 5,000 frames | 2.594 ms | 2.843 ms | +0.249 ms cost |
| p95 / p99, matched 5,000 frames | 9.620 / 16.158 ms | 10.696 / 20.693 ms | Both below the 40 ms frame budget at p99 |
| Maximum in that run | 32.047 ms | 71.801 ms | Tactical still has rare deadline spikes |
| Reported search-frame rate | 40.4% | 26.9% | Telemetry definitions reflect base replanning/holds; not a direct measure of total CPU |
| Laika blind wins | 262/300 = 87.33% | 285/300 = 95.00% | Tactical +23 wins / +7.67 points |
| Wilson 95% interval | 83.09–90.63% | 91.92–96.95% | Intervals barely separate |
| Paired changed outcomes | 26 losses/double deaths later repaired by Tactical | 3 KillField wins later regressed | Net +23; exact McNemar p=0.0000152 |
| Generalisation evidence | Not established by this benchmark | Not established yet | No claim beyond Laika/current JS runtime |
| Python dependency | None | None for Browser JS mode | Equal |
| Browser modes | Watch, human play, AI vs AI through project integration | Watch, human play and AI vs AI | Equal UI surface in this project |
| Licence/provenance | Upstream MIT code, retained attribution | Project code around the attributed KillField base | KillField is not claimed as original work |

## Strength methodology

The promotion result uses three untouched, contiguous, paired seed blocks:

| Blind block | KillField | Tactical | Net wins | KillField win -> Tactical loss |
|---|---:|---:|---:|---:|
| `990000..990099` | 86/100 | 94/100 | +8 | 2 |
| `1000000..1000099` | 87/100 | 95/100 | +8 | 0 |
| `1010000..1010099` | 89/100 | 96/100 | +7 | 1 |
| **Combined** | **262/300** | **285/300** | **+23** | **3** |

Both controllers run in the same JavaScript engine, on the same maze and spawn
for each seed, with the original `round_end` result. That supports a causal
comparison between these two controllers. It does not support direct numeric
comparison with Python P27b results because Python and JavaScript RNGs produce
different trajectories, nor with a screenshot from the upstream website that
does not publish the same seeds and test protocol.

## What remains weak

The 300-round result leaves 15 Tactical non-wins. Current traces point to three
product gaps:

- Laika can create a new bullet one frame before impact; the visible-bullet
  gate cannot react before a bullet exists.
- Some own-bullet self-kills pass through the inherited one-stage guard when
  the arena is saturated with ricochets.
- If all 81 two-stage escape candidates die, the controller returns to the
  baseline action because it has no stronger sparse fallback.

The Worker fixes page responsiveness, not these policy failures. It also does
not make a 71.8 ms search decision meet a 40 ms simulation deadline; it merely
keeps that work off the rendering thread.

## Product plan after this checkpoint

### Gate 1 — freeze and diagnose

- Keep the 95.0% policy and all current thresholds frozen.
- Replay the 15 non-wins and three regressions with the offline exact shadow
  sandbox, and label each as avoidable, no-safe-candidate or trajectory-shift.
- Do not tune on the three published blind blocks and then report them as blind
  again.

### Gate 2 — fix only demonstrated gaps

- Add a visible-state opponent fire envelope only when the existing 81-way
  escape search finds no survivor or an immediate firing line is present.
- Replace the inherited own-bullet one-stage fallback with the same verified
  two-stage safety representation.
- Cap the fallback by time/branch budget. Preserve H36 unchanged on ordinary
  frames.

Promotion requirement: a new 100-seed development block must show more repairs
than regressions, then a fresh 300-seed blind suite must beat this frozen 95.0%
checkpoint without materially worsening p95 or p99 latency.

### Gate 3 — generalise and reduce CPU

- Add fixed Random, Hunter and Dodger opponent suites plus search-vs-search
  self-play checks. Treat these as regression tests, not as the main Laika
  optimisation target.
- Only after collecting enough safety-audit labels, test P27b or a smaller
  network as a cheap gate/action prior. The network should skip useless search;
  it should not replace physics verification in high-risk states.
- Promote a network-guided version only if paired outcomes are preserved. ONNX,
  C++ and broader self-play training are optional optimisations, not the next
  blocker for a browser product that already averages under 3 ms per decision.

## Product decision

Use Tank Trouble Tactical as the browser default and keep KillField selectable
as the transparent baseline. Keep P27b/P26–P41 as a separate research stack.
The next milestone is not another P-number or a deeper global search; it is a
targeted repair of the remaining safety failures, followed by an untouched
generalisation and latency gate.
