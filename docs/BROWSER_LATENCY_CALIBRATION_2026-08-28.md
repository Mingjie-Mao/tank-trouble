# Browser latency calibration — the promotion gate's p95 clause does not hold

Checkpoint: 2026-08-28. Local only, not pushed.

Closes the open item at the end of
[`KILLFIELD_UPSTREAM_ABLATION_2026-08-16.md`](KILLFIELD_UPSTREAM_ABLATION_2026-08-16.md),
which ends:

> **Next step, once the promoted champion is deployed:** read the page's own p95
> for the champion and for the frozen predecessor, on the same machine and mode,
> and calibrate the Node figures against them.

That step had not been taken. Every latency figure in this repository is a Node
measurement; the 40 ms budget those figures are compared against is a browser
frame budget. This document measures both sides on one machine.

## Question

Two conclusions rest on Node latency and were flagged as needing revisiting:

- **The promotion gate.** `README.md` states p95 must remain below the 40 ms
  frame budget, and Tactical Smooth was promoted at 20.05 ms.
- **Phase 6.** `evasionRootBreadth` pruning was rejected partly because "p95
  already passes the gate", so its double-KO cost was judged not worth paying.

> Measured in a browser — the environment the 40 ms budget actually describes —
> does the champion clear the gate?

## Method

The quantity is fixed by what the page displays. `BrowserArena` records
`agent.drive(game)` for side 0 into a rolling 600-frame buffer and reports
`percentile(decisionSamples, 0.95)`; the page renders that value live. Node and
browser run the *same* `browser-arena.js`, so the two sides are the same code
path measuring the same thing.

The comparable statistic is the **distribution of that rolling reading**, not a
single pooled percentile. A person watching the page sees the reading move; the
published 20.05 ms is one window. Every prior report in this repository pooled
instead, which is the main reason the published figure is low.

| Side | Harness |
|---|---|
| Node | `BrowserArena` driven 25,000 frames, watch mode, seed 970000, rolling p95 sampled every 12 frames (2,034 samples) |
| Browser | the arena page, watch mode, seed 970000, rolling p95 polled from the telemetry DOM node twice a second |

**Runs were serialised.** Node and browser were never measured concurrently —
an early browser sample taken while a Node run was active read 13–21% of windows
over 40 ms and dropped to 0% once the machine was idle. This reproduces, on a
second occasion, the parallel-load effect the ablation document already warned
about, and it is why the deployed page's own historical 55.7 ms reading cannot
be interpreted without knowing that machine's load.

Host: Apple M4, 10 cores, 32 GB. Browser measurements were taken in the in-app
browser, not the author's Chrome.

## Results

### Node, exclusive, rolling-600 p95 distribution

| | v2 frozen (Tactical) | v3 champion (Smooth) |
|---|---:|---:|
| Rounds completed in 25,000 frames | 39 | 69 |
| Rolling p95 median | 14.96 ms | **28.14 ms** |
| Rolling p95 p90 | 25.48 ms | 40.34 ms |
| Rolling p95 max | 40.78 ms | **53.92 ms** |
| **Windows over 40 ms** | **0.2%** | **10.4%** |

The champion's *final* window reads 20.13 ms, reproducing the published
20.05 ms. The distribution behind that single number is where the champion
already spends 10.4% of its windows over budget — in Node, before any browser
penalty. The frozen predecessor spends 0.2%.

### Browser, same local build, same session

| | v2 frozen | v3 champion |
|---|---:|---:|
| Rolling p95 median | 42.8 ms | **75.3 ms** |
| Rolling p95 p90 | 48.3 ms | 77.5 ms |
| Rolling p95 max | 73.2 ms | 78.0 ms |
| **Windows over 40 ms** | **66.2%** | **100%** |
| Effective FPS (median / min) | 25.0 / 17.1 | 25.0 / 22.9 |
| Samples | 133 | 30 |

The deployed page, which still serves the pre-Smooth build, was measured
separately for v2: median 35.8 ms, max 54.1 ms, 9.7% of windows over 40 ms over
194 s. Its max reproduces the 52.8–55.7 ms the ablation document recorded, so
that historical observation was not an artefact — it is the tail of this
distribution, and short windows simply miss it.

### Calibration ratio

Browser ÷ Node on the rolling p95 median, matched policy:

```
v2, local build      42.8 / 14.96 = 2.86x
v3, local build      75.3 / 28.14 = 2.68x
v2, deployed build   35.8 / 14.96 = 2.39x
```

The ablation document estimated the gap at roughly 2.7x from a single
observation. Measured across three pairings it is **2.4–2.9x**, and it is stable
across policies. The estimate was correct.

## Findings

**1. The champion does not clear the gate in a browser.**

75.3 ms median against a 40 ms budget, with every sampled window over budget.
The extrapolation the ablation document proposed (apply the ratio to 20.05 ms)
predicted ~54 ms; the direct measurement is worse than the extrapolation,
because the extrapolation started from the published single-window figure rather
than from the distribution.

**2. The frozen predecessor does not clear it either.**

42.8 ms median locally, 35.8 ms on the deployed build, 66.2% and 9.7% of windows
over budget. This is a **pre-existing defect that Smooth roughly doubled**, not
one Smooth introduced — the same shape the ablation document found for p99 in
selfplay. The gate has never been satisfied in the environment it describes; it
was satisfied only in Node.

**3. It is a margin failure, not a liveness failure.**

Effective FPS — computed from real step timestamps, not a nominal constant —
held a median of 25.0 for both policies. The loop is still delivering 25 physics
frames per second. Individual frames exceed the budget often; average throughput
is maintained. The correct statement is that the safety margin is gone, not that
the arena fails to run.

**4. The promotion protocol's latency clause measured the wrong statistic.**

Strength was gated on a pre-registered, paired, 2000-round holdout. Latency was
gated on one isolated pooled number. The rolling-window distribution — the thing
the page shows and a viewer experiences — was never part of the gate, and on it
the champion is a far larger regression than the recorded "+7.5 ms at p95":
median 1.9x, and 52x on the share of windows over budget.

## Consequences

**Phase 6 must be re-decided.** `evasionRootBreadth` pruning was rejected partly
on the premise that p95 already passed the gate. That premise is false. Phase 6
measured breadth=3 cutting p99 by 40% at no measurable win-rate cost (paired
p=1.000 at every setting), against a monotone double-KO rise 9 → 13 → 15.

The better design named there is still the right one: **budget-triggered
narrowing** — expand fully by default, narrow only when the frame's elapsed time
already approaches the deadline — rather than paying the quality cost on all
frames to help the ones that need it. Attribution is already known: 77.8% of
frames above p99 are the `visible_bullet_two_stage` 9×9 evasion search.

**`README.md` currently contradicts itself** and should be reconciled:

| Line | Text | Status |
|---|---|---|
| 48 | Node figures and the browser budget "have not been cross-calibrated" | correct, now superseded by this document |
| 97 | "20.05 ms remains comfortably real-time" | **false in a browser** |
| 234 | "Latency is a promotion gate. p95 must remain below the 40 ms frame budget" | states a rule that was never evaluated in a browser, and that neither champion passes |

## Budget-triggered narrowing — implemented, and it does not close the gate

Implemented as `evasionBudgetMs` on `TacticalSafetyAgent`, exposed as policy
`p27-js-tactical-v3-b<ms>`. A deadline is anchored to the start of the whole
decision (`decisionStartedAt`, set in the innermost `act()`), and the two-stage
evasion search stops opening new roots once it is passed. Roots are already
sorted by achieved clearance, so the tail it drops is the worst part of the
search. The first root is always expanded, so a plan is always returned.

Default is `null` — disabled. **The budget is wall-clock, so an enabled policy is
non-deterministic and must never be used for paired seed evaluation.** This is
verified rather than asserted: `p27-js-tactical-v3` reproduces itself
bit-for-bit across runs, while `-b12` diverges from it at frame 654 of seed
970000 after narrowing twice.

### What it fixes

| Node, full `drive()`, rolling p95 | v3 | v3-b12 |
|---|---:|---:|
| Median | 28.14 ms | **16.72 ms** |
| Max | 53.92 ms | **22.34 ms** |
| Windows over 40 ms | 10.4% | **0%** |

Isolating the audit confirms the mechanism: audit-only p95 falls from 69.77 ms
to 12.75 ms, an 82% cut, while the main search is untouched at 17.10 → 16.40 ms.

### What it does not fix

| Browser, rolling p95 | v3 | v3-b12 |
|---|---:|---:|
| Median | 75.3 ms | **61.5 ms** |
| Max | 78.0 ms | 78.3 ms |
| Windows over 40 ms | 100% | **100%** |

An 18% improvement, and still every window over budget.

**The reason is that the Phase 6 attribution was for the wrong percentile and
the wrong mode.** That attribution — 77.8% of frames above p99 are the 9x9
evasion search — was measured in **selfplay**, where `oppModel` is `L1`. The
deployed page runs **watch**, where `oppModel` is `L2` and every rollout
executes Laika's real algorithm. And the audit fires on only ~0.6% of frames, so
it lives at p99, not p95: narrowing it removes a tail the gate does not measure.

What remains is arithmetic. The main search alone is 16.40 ms at p95 in Node;
at the calibrated 2.68x that is ~44 ms in a browser, already over budget before
the audit contributes anything. **The gate failure is the main 18-plan x 36-frame
L2 rollout, not the evasion search.**

Narrowing is kept because it is cheap, provably free when the frame can afford
the search, and it eliminates the Node overrun outright. It is not a fix for the
browser gate, and should not be reported as one. Closing that gap means changing
the main search — fewer plans, a shorter horizon, or a cheaper opponent model in
the rollout — each of which is a strength change requiring the full promotion
protocol.

## Limitations

- **Sampling was throttled unequally.** The browser's `setInterval` was
  throttled by the host: 30 samples for v3 over 224 s against 133 for v2 over
  227 s. The v3 distribution is tight (min 56.8, max 78.0) so the conclusion is
  not sensitive to this, but the two browser arms are not equally resolved.
- **The local dev server is not the deployed static build.** v2 reads 42.8 ms
  locally against 35.8 ms deployed. Cross-build comparisons are therefore
  invalid; only the local v2/v3 pair and the Node v2/v3 pair are matched.
- **Not paired by round.** Both sides run watch mode on seed 970000, but v3
  completes 69 rounds in 25,000 frames against v2's 39, so the position
  distributions differ. Latency is being compared across policies, not across
  identical positions.
- **One host, one browser.** Measured in the in-app browser on an M4. The ratio
  is expected to vary with host and browser; the direction and rough magnitude
  are what this establishes, not a portable constant.
- **`decision_p50_ms` from the Node harness is the final window's p50 only**,
  and is noisy — v2 reported 7.1 ms against a published 0.47 ms. It is excluded
  from every conclusion here; only the 2,034-sample rolling p95 distribution is
  used.
- **An agent's own telemetry counters are per-round, not cumulative.** The arena
  calls `_configureControllers()` on every `new_round` event, which rebuilds
  both agents. Any figure read from `agent.telemetry()` after a multi-round run
  therefore describes the last round only — the audit-only p95 and audit counts
  above are of that kind. The rolling p95 distribution is not affected: it comes
  from `decisionSamples`, which lives on the arena and survives the rebuild.

## Reproduction

Node side, `web/scripts/calibrate-latency.mjs`, run serially and exclusively:

```bash
npm run calibrate:latency -- p27-js-tactical-v3 25000 970000
npm run calibrate:latency -- p27-js-tactical-v2 25000 970000
```

Browser side: `npm run build && npm run start`, open the arena, select the
policy, and poll the `决策 p95` telemetry node twice a second for ~200 s with
nothing else running on the machine.
