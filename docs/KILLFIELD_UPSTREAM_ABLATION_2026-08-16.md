# Upstream KillField ablation — what the 2026-08-14 commits actually changed

Checkpoint: 2026-08-16. Local only, not pushed.

Continues [`TACTICAL_VS_KILLFIELD_2026-08-12.md`](TACTICAL_VS_KILLFIELD_2026-08-12.md),
which compared Tactical against the vendored KillField snapshot. That snapshot
is now 14 commits behind upstream. This document measures what those 14 commits
did, because the answer changes which of them are worth adopting.

## Question

Upstream added, on 2026-08-14, a batch of agent and engine changes: fire
continuation rollouts, frictional wall sliding, refined wall-contact physics,
wall recovery, faster predicted-hit preference, hunt urgency, and ammo
pressure. Upstream reports 897/1000 = 89.7% true wins against Laika.

The vendored snapshot scores 262/300 = 87.3% on this repository's blind seeds.
Those two numbers come from different seed bases, different round counts and
different physics, so their difference is not an effect size. The question this
ablation answers is narrower and comparable:

> Of the upstream improvement, how much is **strategy** and how much is
> **engine physics** — and which one produces the smoother movement?

## Method

Three configurations, all built inside a clean upstream clone so that no local
Tactical modification can leak in:

| Config | Agent (`src/killfield/`) | Engine (`src/game.js` etc.) |
|---|---|---|
| **A** | upstream HEAD `40e94eb` | upstream HEAD `40e94eb` |
| **B** | `67d6a83` (the vendored snapshot) | upstream HEAD `40e94eb` |
| **C** | `67d6a83` | `67d6a83` |

Attribution follows directly:

- **A − B** isolates the strategy bundle (fire continuation, hunt urgency, ammo
  pressure, flight-time penalty).
- **B − C** isolates the physics bundle — referred to below as **K4**, which is
  a small bundle rather than a single change: `b04d8cf` frictional sliding,
  `b917682` contact refinement, and `d61408e` wall recovery / shallow-overlap
  separation.

A is an unmodified clone, not a hand-assembled tree. Its correctness is
confirmed by upstream's own deterministic checkpoint: the first 100 rounds of
seed `20260814` reproduce **88 W / 8 L / 4 D** exactly, and the full 1000-round
run reproduces **897 W / 52 L / 51 D (50 mutual kills + 1 timeout)** — every
figure matching the upstream README.

### The upstream benchmark cannot be used for paired statistics

`test/killfield_vs_laika.js` reuses a single `Game` across all rounds, and the
maze for each round is drawn from that shared rng. Two configs whose rounds take
different numbers of frames therefore consume the rng differently and their maze
sequences diverge. Measured directly: of 60 rounds, **only round 0 shares a maze
fingerprint**; divergence begins at round 1, where the two configs' opening
rounds lasted 236 and 191 frames.

Any McNemar test, flip count or paired interval computed over that benchmark's
round index is therefore invalid. A second harness (`paired-bench.mjs`) rebuilds
`new Game({seed: base + i})` per round — the protocol this repository already
uses in `playLeagueGame` — which makes round *i* the same maze in every config.
Pairing validity was then verified rather than assumed: **300/300 maze
fingerprints identical**.

Because that harness also rebuilds the agent each round, its absolute win rates
are not comparable to upstream's continuous-run figures. The two protocols are
reported separately and never mixed.

## Results

### 1000 rounds, upstream protocol (unpaired), seed 20260814

| Config | AI | Physics | True wins | Losses | Double-KO | True win % |
|---|---|---|---:|---:|---:|---:|
| A | latest | K4 | 897 | 52 | 50 | **89.7%** |
| B | old | K4 | 865 | 76 | 59 | 86.5% |
| C | old | old | 884 | 66 | 50 | 88.4% |

```
A − B = +3.2pp    strategy bundle
B − C = −1.9pp    K4 physics bundle

88.4 − 1.9 + 3.2 = 89.7    the three configurations are self-consistent
```

Unpaired at n=1000 the standard error of a difference of proportions is about
1.45pp, so B−C at 1.3 SE is **not** significant on its own. This is what
motivated the paired run.

### 300 rounds, paired protocol, seeds 20260814+i

| | C (old physics) | B (K4) | Δ |
|---|---:|---:|---:|
| True wins | 258 | 242 | −16 |
| Losses | 25 | 30 | +5 |
| Double-KO | 16 | 27 | **+11** |
| True win % | 86.00% | 80.67% | −5.33pp |
| `hitSomething` frames | 20.91% | **2.75%** | −87% |
| Zero motion under command | 12.15% | **0.08%** | −99% |
| Mean round frames | 317.5 | 290.7 | −8.4% |

Paired transitions (C → B):

```
win → win        209
win → non-win     49
non-win → win     33
non-win → non-win  9

McNemar exact, two-sided   p = 0.097        not significant
Paired 95% CI (analytic)   [−11.2pp, +0.6pp]   crosses zero
Paired 95% CI (bootstrap)  [−11.3pp, +0.7pp]
```

## Findings

**1. K4 is the movement mechanism, and it is not a strength mechanism.**

With the agent held completely fixed, wall-contact physics removes essentially
all wall grinding: 20.91% → 2.75% contact frames, and commanded-but-zero motion
falls from 12.15% to 0.08%. A separate diagnostic batch measured direction
switching at 6.87/s → 3.64/s.

Its effect on strength is negative in direction and not significant in
magnitude: −1.9pp unpaired at n=1000, −5.33pp paired at n=300 with p=0.097 and
an interval crossing zero.

For this project the useful phrasing is that **motion-quality improvement and
policy-strength improvement are separate axes**, and K4 moves only the first.

**2. The strategy bundle is the strength mechanism, and it is worth +3.2pp.**

The prior working assumption — that fire continuation was responsible for the
smoother movement — is refuted. Forced fire accounts for only 1.90–1.99% of
combat frames in the frozen Tactical champion, while wall contact accounts for
21–22%. Fire continuation and the rest of the strategy bundle are worth
adopting, but for win rate, not for movement.

**3. K4's cost is predominantly double-KO, not defeat.**

Of the 49 rounds K4 converted from wins, 23 became mutual kills and 25 became
losses; overall double-KOs rose 16 → 27 (+69%) while losses rose only 25 → 30.
Rounds also finished 8.4% faster. The mechanism is consistent: smoother movement
advances the engagement more quickly and pushes the tank deeper, which makes the
post-kill settlement window more dangerous rather than making the tank easier to
beat.

This matters because the failure mode K4 introduces is precisely the one this
repository's controller already defends against — `settlement_two_stage`,
`unsafe_settlement_suppressed` and `visible_bullet_two_stage` have no equivalent
in the bare upstream agent, which carries only `postKillSurvivalScores`.

## Consequence for the adoption plan

K4 alone is negative. Upstream's net gain exists because the strategy bundle was
adopted alongside it:

```
K4 alone            smoother movement, −1.9 to −5.3pp
K4 + strategy       upstream's 89.7%, net +1.3pp over C
```

A candidate that ships K4 without compensating strategy or scoring work should
therefore be expected to lose win rate, and should not be judged a failure for
doing so if the movement metrics move as measured here.

### Falsifiable prediction for Phase 1

Adding K4 to Tactical should raise double-KOs by **substantially less than the
+69% seen on the bare upstream agent**, because Tactical's settlement layers
target exactly that failure mode.

- If the increase is far below +69%, K4's cost here is much smaller than
  upstream's and it can proceed on its own.
- If the increase approaches +69%, the settlement layers are not covering the
  new physics, and they must be repaired before K4 is promoted.

This is the gate for Phase 1, and it is a stronger signal than the win rate
itself, which at n=300 cannot resolve an effect of this size.

## Phase 1 result — K4 ported onto Tactical

K4 is implemented behind `Game({ wallSliding })`, default off. The original
collision branch is retained unchanged in the `else`, and `makeSandbox`
propagates the flag so the planner's world model matches the real engine.

Verification that the default path is untouched: the vendored suite passes
46/46, and frozen Tactical reproduces its published figures exactly — 291/300
on base `2900000`, 285/300 on base `3200000`, pooled **576/600 = 96.00%**.

Pooled over both promotion seed bases, 600 rounds:

| | Frozen | + K4 | Δ |
|---|---:|---:|---:|
| True wins | 576 (96.00%) | 563 (93.83%) | −2.17pp |
| Losses | 15 | 28 | **+13** |
| Double-KO | 9 | **9** | **0** |
| `hitSomething` frames | 20.0–21.0% | **3.0–3.2%** | −85% |
| Zero motion under command | 11.6–12.5% | **0.10–0.19%** | −99% |
| Decision p95 | 12.3–12.7 ms | 14.8–14.9 ms | +2.4 ms |

```
paired: win → non-win 36, non-win → win 23, net −13
McNemar exact p = 0.1175        not significant
paired 95% CI = [−4.67pp, +0.34pp]
```

**The prediction held.** Double-KOs went 9 → 9 (+0%) against +69% on the bare
upstream agent: Tactical's settlement layers fully absorb the failure mode K4
introduces there. The cost instead appeared as ordinary defeats, 15 → 28.

Note that the two seed bases disagreed in sign — base `2900000` gave −4.67pp
(McNemar p=0.016) while base `3200000` gave +0.33pp (p=1.0). A single 300-round
base can flip the conclusion, which is why only the pooled figure is reported.

### Root cause of the +13 losses

Not the opponent. Laika barely touches walls to begin with (2.32% of combat
frames versus Tactical's 22.98%) because it follows cell-centre shortest paths,
so K4 has little to give it.

The mechanism is inside the planner. `hitSomething` is one of the conditions
that breaks action commitment:

```js
if (this.commitRemaining > 0 && !game.tanks[0].hitSomething) {
  this.commitRemaining -= 1;
  return this.emitAction(game, this.committedAction, "hold");
}
```

It used to fire on 22% of frames, forcing frequent replanning. Under K4 it fires
on 3.2%, so commitment almost always runs its full four frames:

| Decision | Frozen | + K4 |
|---|---:|---:|
| `hold` | 53.60% | 71.48% |
| `plan` | 41.58% | **23.87%** |

Replanning frequency falls by 43%. The planner is not paying a cost for a
problem that no longer exists — it is **losing a trigger it had been relying on**.

This makes Phase 2 (commitment and stale-penalty cleanup) a prerequisite for
K4 rather than an optional follow-up, and the latency budget allows it: p95 is
14.9 ms against a 40 ms frame.

## Phase 2A — commitment sweep

`commitMoveFrames` became a constructor option on `KillFieldAgent`, defaulting
to the original `COMMIT_MOVE_FRAMES` (4), exposed as policies
`p27-js-tactical-v2-c<n>`. Every arm below is 600 rounds: 300 on each promotion
seed base, paired against frozen on identical seeds.

| Arm | True wins | Losses | Double-KO | `plan`% | flips/s | Wall % | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | **576 (96.00%)** | 15 | 9 | 39.80% | 7.15 | 20.50% | 0.64 ms | 15.87 ms |
| K4-C4 | 563 (93.83%) | 28 | 9 | 23.86% | 3.81 | 3.08% | 0.84 ms | 18.72 ms |
| K4-C3 | 565 (94.17%) | 24 | 11 | 28.34% | 4.12 | 2.97% | 0.73 ms | 15.98 ms |
| **K4-C2** | **569 (94.83%)** | 26 | 5 | 35.49% | 4.57 | 2.78% | 1.02 ms | 18.94 ms |
| K4-C1 | 565 (94.17%) | 29 | 6 | 50.06% | 5.41 | 2.66% | 3.06 ms | 17.13 ms |

Paired against frozen:

```
K4-C4   Δ=−2.17pp  p=0.118  CI[−4.67,+0.34]   36 harmed / 23 saved
K4-C3   Δ=−1.83pp  p=0.177  CI[−4.25,+0.59]   33 / 22
K4-C2   Δ=−1.17pp  p=0.410  CI[−3.54,+1.21]   30 / 23
K4-C1   Δ=−1.83pp  p=0.185  CI[−4.30,+0.63]   34 / 23
```

C2 is the optimum: it recovers roughly half of K4's cost. The curve is not
monotone — C1 overshoots into thrashing, with `plan` at 50.06% (above frozen's
39.80%), flips rising to 5.41/s, and median decision time tripling from 1.02 ms
to 3.06 ms. C1 also inverts across seed bases (286/279 versus C2's 283/286),
ranking best on one base and worst on the other, which is the variance signature
that rules it out independently of its mean.

**Selection rule, fixed before the data was read:** among candidates at pooled
≥96.0%, take the lowest flips/s; if none reach 96.0%, take the highest pooled
whose flips/s does not exceed frozen's. C2 wins under the second clause.

C2 still sits 1.17pp below the 96.0% promotion gate, and at n=600 that gap is
not statistically resolvable (p=0.41).

## Phase 2B — selective wall-contact replanning (negative result)

C2 compensates globally: it raises search frequency on every frame, while the
signal K4 removed was selective. The hypothesis was that reinstating a targeted
trigger would be more precise than shortening commitment everywhere.

K4 already sets `tank.wallSliding` whenever a contact was resolved as a slide,
which is semantically the right event. Measured first, before building on it:

```
under K4:  hitSomething 3.20%   wallSliding 32.13%   overlap 3.00%
original:  hitSomething 22.98%
```

`wallSliding` is not a scarce signal needing restoration — it is **denser than
the trigger it would replace** (32% vs 23%). Using it therefore requires a
*longer* nominal commitment to reach the same effective replanning interval,
which is why the sweep runs C4–C6 rather than C2–C3.

| Arm | True wins | Losses | Double-KO | `plan`% | flips/s | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | **576 (96.00%)** | 15 | 9 | 39.80% | 7.15 | 0.64 ms | 15.87 ms |
| K4-C2 | **569 (94.83%)** | 26 | 5 | 35.49% | 4.57 | 1.02 ms | 18.94 ms |
| K4-C5-WR | **569 (94.83%)** | 22 | 9 | 38.60% | 4.18 | 1.50 ms | 27.97 ms |
| K4-C4-WR | 561 (93.50%) | 30 | 9 | 40.84% | 4.32 | 1.70 ms | 27.88 ms |
| K4-C6-WR | 559 (93.17%) | 31 | 10 | 37.46% | 4.09 | 1.45 ms | 28.06 ms |

```
vs frozen (paired, n=600)
K4-C2      Δ=−1.17pp  p=0.410
K4-C5-WR   Δ=−1.17pp  p=0.419
K4-C4-WR   Δ=−2.50pp  p=0.067
K4-C6-WR   Δ=−2.83pp  p=0.043   only significant result in the sweep, and negative

K4-C2 vs K4-C5-WR   Δ=0.00pp  p=1.000   29 harmed / 29 saved
```

**The direction is refuted.** No WR arm beats C2; the best merely ties it, and
two are worse. C2 and C5-WR are strength-indistinguishable yet disagree on 58
individual rounds, so they are two distinct policies of equal strength rather
than one policy expressed twice. The tiebreak is latency: every WR arm sits at
27.9–28.1 ms p95 against 15.9–18.9 ms for the non-WR arms — a consistent +9 ms
that buys nothing.

Two conclusions worth keeping:

1. **`plan%` is not the causal variable.** C4-WR matches frozen's replanning
   rate most closely (40.84% vs 39.80%) and finishes second from last; C2 plans
   least among the serious candidates and wins. Calibrating to `plan%` was the
   wrong target — it is an intermediate quantity, not a lever.
2. **The lost signal was the event, not the frequency.** `hitSomething` under
   the original physics meant *the plan failed and must change*. `wallSliding`
   means *the plan is executing and grazed a wall*. Interrupting on the latter
   spends search on frames that did not need a new decision. What K4 removed is
   not recoverable by replacing the trigger, because under K4 the informative
   event genuinely no longer occurs.

### Note on reading per-base splits

The net per-base change is misleading on its own. Frozen → K4-C2 nets −8 on base
`2900000` and +1 on base `3200000`, which suggests the damage is concentrated on
one base. The flip counts show otherwise: **17 rounds flipped from win on
`2900000` and 13 on `3200000`** — the second base merely happened to save 14 in
the other direction. Any hypothesis that K4 fails on a specific class of
position is unsupported by these splits; the netting hid comparable two-way
churn on both bases.

## Phase 2B — selective wall-contact replanning (rejected)

C2 compensates globally: it raises the search rate on every frame, whereas the
trigger K4 removed was selective, firing only on wall contact. The hypothesis
was that restoring a selective trigger would beat the blunt instrument.

K4 already exposes `tank.wallSliding`, set whenever a contact was resolved as a
slide or a recovery — semantically "wall contact materially altered this frame's
motion". It was wired to break commitment behind `wallContactReplan`, exposed as
the `-wr` policy suffix.

Measuring the signal first changed the framing. Under K4, `wallSliding` fires on
**32.13%** of combat frames against `hitSomething`'s 22.98% under the original
model, and it subsumes `hitSomething` almost entirely (3.20% fires, 3.00%
overlap). It is therefore *denser* than the signal it replaces, not sparser: it
needs a **longer** nominal commitment to reach the same effective replan
interval, which is why the initially favoured C3-WR overshot to `plan` 46.27%.

| Arm | True wins | Losses | Double-KO | `plan`% | flips/s | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | **576 (96.00%)** | 15 | 9 | 39.80% | 7.15 | 0.64 ms | 15.87 ms |
| K4-C2 | **569 (94.83%)** | 26 | 5 | 35.49% | 4.57 | 1.02 ms | **18.94 ms** |
| K4-C4-WR | 561 (93.50%) | 30 | 9 | 40.84% | 4.32 | 1.70 ms | 27.88 ms |
| K4-C5-WR | **569 (94.83%)** | 22 | 9 | 38.60% | 4.18 | 1.50 ms | 27.97 ms |
| K4-C6-WR | 559 (93.17%) | 31 | 10 | 37.46% | 4.09 | 1.45 ms | 28.06 ms |

```
frozen → K4-C4-WR   Δ=−2.50pp  p=0.067
frozen → K4-C5-WR   Δ=−1.17pp  p=0.419
frozen → K4-C6-WR   Δ=−2.83pp  p=0.043
K4-C2  → K4-C5-WR   Δ= 0.00pp  p=1.000   29 harmed / 29 saved
```

The best WR arm exactly ties C2 at 569/600, but costs 9 ms more at p95. The two
are not the same policy — they disagree on 58 of 600 rounds and merely trade an
equal number of wins — so the tie is genuine rather than degenerate.

Two conclusions:

1. **Selective replanning is not better than global commitment shortening
   here.** The hypothesis is rejected on its own terms, and rejected again on
   cost.
2. **Matching `plan`% does not match behaviour.** C4-WR lands at 40.84%, nearest
   to frozen's 39.80%, yet is among the weakest arms. Frozen replans when motion
   *stops*; WR replans when motion *slides* — different populations of position.
   `plan`% is a calibration aid, not an objective.

`NO_EFFECT_REPEAT_PENALTY` remains untouched and now looks even lower priority:
zero-motion frames are down to 0.13%, so it can barely fire in either direction.

**Best candidate after Phase 2: K4-C2 at 569/600 = 94.83%**, 1.17pp below the
promotion gate and not statistically separable from frozen at n=600 (p=0.41).

## Phase 3 — K1 fire continuation

K1 replaces the ten-action search with eighteen plans (nine persistent no-fire
controls, plus nine "fire this frame, then move" continuations sharing one
stationary-fire first action), and demotes verified fire from an override to a
replan trigger. Behind `fireContinuation`, exposed as the `-fc` policy suffix.
`densityRollout` is shared, so the regression check matters here: the suite
passes 46/46 and frozen reproduces its movement figures bit-for-bit.

| Arm | True wins | Losses | Double-KO | Wall % | `plan`% | flips/s |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 576 (96.00%) | 15 | 9 | 20.50% | 39.80% | 7.15 |
| **frozen + K1** | **579 (96.50%)** | 16 | 5 | 20.98% | 41.79% | 7.04 |
| K4-C4 | 563 (93.83%) | 28 | 9 | 3.08% | 23.86% | 3.81 |
| **K4-C4 + K1** | **571 (95.17%)** | 20 | 9 | **3.30%** | 26.71% | 3.66 |
| K4-C2 | 569 (94.83%) | 26 | 5 | 2.78% | 35.49% | 4.57 |
| K4-C2 + K1 | 566 (94.33%) | 27 | 7 | 2.63% | 37.82% | 4.40 |

```
K1 marginal effect (paired)
  on frozen    +0.50pp  p=0.711   13 harmed / 16 saved
  on K4-C4     +1.33pp  p=0.332   22 / 30
  on K4-C2     −0.50pp  p=0.761   23 / 20

versus frozen
  frozen + K1  +0.50pp  p=0.711
  K4-C4 + K1   −0.83pp  p=0.568
  K4-C2        −1.17pp  p=0.410
```

**No K1 effect is significant at n=600.** Every interval spans zero. The
findings below are directional and need confirmation at larger n.

**K1 and commitment shortening are substitutes, not complements.** K1 adds
+1.33pp on top of the original four-frame commitment but −0.50pp on top of C2.
Both raise effective search quality, and stacking them overshoots the same way
C1 did. The best K4 arm is therefore C4+K1, *not* the accumulation of every
improvement found so far.

**K1's marginal value here is far below upstream's.** The upstream ablation
credits the strategy bundle with +3.2pp on the bare agent; on frozen Tactical
K1 is worth +0.50pp. The plausible reason is overlap: `unsafe_settlement_*`
already suppresses shots whose settlement is unsurvivable, which is part of what
scoring a shot against its own follow-up movement buys. This is the second
instance of an upstream improvement being partly pre-absorbed by this
repository's safety layers — the first being K4's double-KO cost.

### Latency

Arms measured under three-way parallel load report p95 up to 38.9 ms. Re-measured
alone, **K4-C4+K1 is 20.2 ms** against the 40 ms frame budget. Parallel-load p95
figures in this document are therefore not admissible for the promotion gate;
only isolated runs are.

### Status

Best movement-quality candidate: **K4-C4+K1, 571/600 = 95.17%**, wall contact
3.30%, p95 20.2 ms isolated — 0.83pp below the gate, not separable from frozen.

Best overall: **frozen+K1 at 579/600 = 96.50%**, the only arm exceeding frozen,
but it carries no movement improvement (wall contact 20.98%). It is *not*
pursued as a separate candidate: +0.50pp at p=0.711 is a three-round difference
indistinguishable from noise, and buying it costs an 80% larger search, a higher
p95, and a change to the search core — with no movement benefit, which is the
objective. K1 earns its place only as a component of the K4 stack.

## Phase 4 — pre-registered fresh holdout

**Written before the holdout was run.**

Eleven candidates were evaluated across Phases 2 and 3 and **none produced a
significant effect** (p from 0.33 to 1.00). Selecting the best point estimate
from eleven arms inside a ±2pp noise band is a multiple-comparison trap, and
adding rounds to bases `2900000`/`3200000` only sits the exam on the paper the
candidate was selected with. Sweeping stops here.

**Candidate, frozen: K4 + C4 + K1** — `p27-js-tactical-v2-c4-fc` with
`wallSliding: true`. No further tuning of commitment, penalties or search.

**Baseline:** frozen Tactical, `p27-js-tactical-v2`, `wallSliding: false`.

**Holdout:** seed base `3500000`, never used for any tuning decision, 2000
paired rounds (round *i* is the same maze in both arms).

**Sample size.** From the observed paired variance at n=600 (SE 1.166pp), a
1000–1200 round holdout gives a CI half-width of 1.62–1.78pp — wider than the
margin below, so it could not establish non-inferiority *even at exactly zero
difference*. 2000 rounds gives 1.25pp, which can conclude when the observed
difference is within about 0.25pp of zero.

**Primary endpoint:** paired true-win difference, candidate − frozen.

**Pre-registered non-inferiority margin: −1.5pp.**
`H0: Smooth − Frozen ≤ −1.5pp`; reject, and declare non-inferiority, only if the
lower bound of the paired 95% CI exceeds −1.5pp. "Not significant" will not be
reported as "equivalent".

**Secondary:** losses, double-KOs, paired harm/save flips, wall-contact %,
zero-motion %, `plan`%, flips/s, and isolated (non-parallel) p50/p95.

**Decision table, fixed in advance:**

| Holdout outcome | Conclusion |
|---|---|
| CI lower bound > −1.5pp | Non-inferior within the pre-registered margin; promote Tactical Smooth |
| Point estimate ≥ 0 | Promote, and run a final confirmation |
| −1.5pp ≲ Δ < 0, CI lower bound below −1.5pp | Report as an explicit product trade-off, not as an equivalence claim |
| CI entirely below −1.5pp | Do not promote as default champion |

**Holdout discipline:** once these results are read, base `3500000` is spent. No
subsequent parameter change may be validated on it.

### Holdout result

2000 paired rounds on base `3500000`. Pairing verified: 2000/2000 identical
seeds.

| | Frozen | Tactical Smooth | Δ |
|---|---:|---:|---:|
| True wins | 1914 (95.70%) | 1912 (95.60%) | **−0.10pp** |
| Losses | 57 | 66 | +9 |
| Double-KO | 29 | 22 | **−7** |
| `hitSomething` frames | 21.35% | **3.00%** | −86% |
| Zero motion under command | 12.52% | **0.12%** | −99% |
| flips/s | 7.23 | **3.60** | −50% |
| `plan`% | 39.89% | 26.31% | — |
| p50 / p95 (isolated, 150 rounds) | 0.47 / 12.57 ms | 0.66 / **20.05 ms** | +7.5 ms |

```
Paired transitions   win→win 1833, win→non-win 81, non-win→win 79, non-win→non-win 7
Difference           −0.100pp   SE 0.633pp
95% CI               [−1.340pp, +1.140pp]
McNemar exact        p = 0.937

Pre-registered test  CI lower bound −1.340pp > margin −1.5pp
                     H0 rejected — NON-INFERIOR within the registered margin
```

**Decision: promote.** The first row of the pre-registered decision table
applies. The result is also the most favourable reading of that table: the point
estimate is −0.10pp, a two-round difference across 2000 rounds, with harm and
save flips almost perfectly balanced (81 vs 79).

Note that the power calculation was load-bearing. At the originally proposed
1000–1200 rounds the CI half-width would have been 1.62–1.78pp, putting the
lower bound outside the margin **even at this near-zero observed difference**;
the conclusion would have been "not established" regardless of the true effect.

The earlier −0.83pp measured on the tuning bases did not reproduce on fresh
seeds. The correct reading is not that the earlier result was wrong, but that
the effect is small enough to be dominated by seed-block variation — which is
precisely why the holdout was run on a base that had never informed a decision.

Secondary endpoints: double-KOs fall 29 → 22, every motion-quality metric
improves by 50–99%, and p95 rises from 12.57 ms to 20.05 ms — a 60% increase,
but still half the 40 ms frame budget. Both latency figures here are isolated
single-process runs; the parallel-load numbers reported earlier in this document
inflate p95 by up to 1.9× and are not admissible for the gate.

## Summary

Tactical Smooth = frozen Tactical + K4 wall-contact physics + K1 fire
continuation, with commitment left at its original four frames.

It obtains upstream KillField's movement quality — the property this
investigation set out to explain and acquire — at no measurable cost in true win
rate, verified on a pre-registered fresh holdout against a margin fixed before
the data was seen.

Two upstream improvements were also shown to be partly redundant against this
repository's existing safety layers: K4's double-KO cost (+69% on the bare
upstream agent, 0% here) and K1's win-rate contribution (+3.2pp upstream as part
of the strategy bundle, +0.50pp on frozen Tactical).

## Reproduction

```bash
git clone https://github.com/Cichlider/killfield.git
cd killfield

# Config A: unmodified HEAD. Must print 88 W / 8 L / 4 D.
node test/killfield_vs_laika.js 100 60 20260814 2048

# Config B: vendored agent on current engine
git checkout 67d6a83 -- src/killfield/

# Config C: vendored agent and engine
git checkout 67d6a83 -- src/
```

Paired runs use a per-round `new Game({seed: base + i})` harness; the upstream
benchmark's continuous-`Game` protocol must not be used for paired statistics.

## Appendix — what the privileged teacher's 99% actually measures

The Python exact-state teacher scores 99.0% and is described in the README as a
label oracle rather than a stronger agent. `exact_state.py` says so directly: it
"intentionally keeps the live RNG and Laika's internal goal/action state".
`clone_exact_game` is a `deepcopy`, so a rollout knows every future random draw
and reads the opponent's actual plan instead of inferring it — exactly the two
things `makeSandbox` scrubs (reseeded rng, rebuilt opponent controller).

To measure how much of the gap is information rather than search quality, an
`L3` sandbox mode grants the browser agent the same two privileges, restoring
only those specific fields rather than deep-copying the object graph:

```js
sb.rng.state = game.rng.state;            // every future random draw
ai.myGoal    = { ...live.myGoal };        // the opponent's real current goal
ai.myActions = live.myActions.map(...);   // and its queued actions
```

600 rounds, both promotion seed bases:

| Agent | Information | Physics | True wins | Wilson 95% CI |
|---|---|---|---:|---|
| Tactical | fair | original | 576/600 = 96.00% | [94.12, 97.30] |
| Tactical | **privileged** | original | 586/600 = 97.67% | [96.12, 98.61] |
| Tactical Smooth | **privileged** | K4 + K1 | **596/600 = 99.33%** | [98.30, 99.74] |
| Python exact-state MPC | privileged | original | 297/300 = 99.00% | [97.10, 99.66] |

Privileged Smooth versus the Python teacher: Δ=+0.33pp, z=0.537, **p=0.592 —
statistically indistinguishable**.

**The 99% is an information advantage, not an algorithmic one.** Given the same
hidden state, the browser search reaches the same level as the privileged
offline teacher. The two figures were never comparable as agent strength, and
the README is right to label the teacher an oracle.

### A second, less obvious result

K4+K1 is worth **+1.67pp under privileged information** (97.67% → 99.33%) while
being worth **−0.10pp under fair information** (the pre-registered holdout).
The same code, the same opponent, opposite verdicts.

The reading: **world-model fidelity only pays once the other uncertainty is
gone.** When the planner cannot know what the opponent will do, the error from
that dominates, and a more accurate collision model changes little — it mostly
contributes side effects, which is why K4 alone measured negative upstream. Once
the opponent's plan is known, physics accuracy converts directly into win rate.

This also bounds the remaining headroom honestly. Fair-play Tactical at 96.00%
is not 3pp away from a better algorithm; it is 3pp away from information it is
not allowed to have.

## Phase 6 — the latency tail (pruning rejected)

Promotion cleared the p95 gate, but measuring per-frame cost in **selfplay**
mode — the only mode where two agents plan on the same frame — exposed a tail
that had never been measured, because every prior latency report used watch
mode:

| | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| frozen Tactical | 1.21 ms | 9.31 ms | 26.57 ms | 159.8 ms |
| Tactical Smooth | 0.95 ms | 25.82 ms | 55.60 ms | 277.5 ms |

The p99 exceeds the 40 ms frame budget in both, so **this is a pre-existing
defect, not one Smooth introduced** — Smooth roughly doubles it.

Attribution of frames above p99 (Smooth, 32k frames):

```
visible_bullet_two_stage   253/325 = 77.8%    the 9x9 evasion search
topology_chase              31
plan (K1's 18 plans)        12/325 =  3.7%
density field rebuild       10/325 =  3.1%
```

Neither K1 nor field rebuilding explains the tail. It is the evasion search:
nine roots, each expanded with nine continuations, 81 rollouts worst case.

`evasionRootBreadth` was added to rank surviving roots by the clearance already
achieved and expand only the widest N. Three settings, 600 paired rounds each:

| Breadth | True wins | Double-KO | p95 | p99 | Paired vs b9 |
|---|---:|---:|---:|---:|---|
| **9 (default)** | 571 | **9** | 24.93 ms | 52.39 ms | — |
| 5 | 570 | 13 | 24.18 ms | 36.66 ms | win p=1.000, dbl p=0.289 |
| 3 | 570 | 15 | 22.19 ms | 31.07 ms | win p=1.000, dbl p=0.109 |

Win rate is untouched at every setting — the paired harm/save counts are
symmetric (7/6, 9/8, 6/6) and every p is 1.000. Double-KOs rise monotonically,
9 → 13 → 15. No individual comparison is significant, but three ordered points
with three same-signed comparisons are enough to treat the increase as
systematic rather than noise.

**Rejected, and the default is unchanged.** The trade is bad in this project's
terms: the problem is confined to the selfplay demonstration mode, p95 already
passes the promotion gate, and the cost lands on double-KOs — precisely the
metric where this repository's settlement layers are its advantage over
upstream (K4 costs the bare upstream agent +69% double-KOs and costs Tactical
0%). Spending that advantage to improve a non-blocking performance figure is
the wrong direction.

The option remains in the code, defaulted to full expansion.

If the tail is attacked later, the better design is a **budget-triggered**
narrowing rather than an unconditional one: expand fully by default and narrow
only when the frame's elapsed time already approaches the deadline. The version
measured here pays the quality cost on all frames to help the 1% that need it.

Note also that `max` is not a usable metric here: repeated runs of the same
configuration produced 779.9 ms and 247.1 ms while p99 stayed at 32.0 and 31.1.
Single-frame extremes track GC, not policy.

## Phase 7 — opponent modelling (no value found)

The privileged ablation put the remaining headroom in opponent prediction:
K4+K1 is worth +1.67pp when the opponent's plan is known and −0.10pp when it is
not. The obvious follow-up was to pair the champion with this repository's
existing `VisibleOpponentModel`, which learns action run-lengths and transitions
from visible controls only.

Reading the code first changed what the experiment could possibly show.
`opponentBehavior` feeds only the **shot-settlement audit** and the **intercept
predictor**; the main 36-frame rollout uses `oppModel` and is untouched. And in
watch mode `oppModel` is already `L2`, which runs Laika's real algorithm. So
against Laika the model can only replace real-algorithm prediction with a weaker
statistical one, in two narrow places. The +1.67pp is not reachable this way at
all: that gap is Laika's internal goal stack plus future RNG, both hidden by
definition.

The only setting where the model has room is selfplay, where `oppModel` is `L1`
(freeze the opponent's current buttons). That is the four-opponent pool, so the
test ran there — 320 paired rounds, same seeds and protocol as the promotion
gate.

| Opponent | Champion | + opponent model | Paired flips |
|---|---:|---:|---|
| laika-js | 71/80 | 71/80 | 0 harmed / 0 saved |
| hunter-js | 75/80 | 75/80 | 0 / 0 |
| dodger-js | 77/80 | 75/80 | **2 / 0** |
| random-js | 79/80 | 79/80 | 0 / 0 |
| **Total** | **302/320** | **300/320** | Δ=−0.63pp, CI [−1.49, +0.24], p=0.500 |

**318 of 320 rounds are identical round-for-round.** The model changed nothing
against three of the four opponents, and cost two rounds against Dodger, both
converted into timeout draws — consistent with hesitating while chasing an
evasive opponent whose dodges the model predicts.

**Not adopted.** The result is worth recording as a bound rather than a failure:
opponent modelling has little room in this project as currently wired, because
the component that would benefit — the main rollout — does not consume it, and
where it is consumed the incumbent predictor is already the opponent's real
algorithm.

Making the main rollout consume a learned model is a much larger change, and the
privileged ablation caps what it could return: against Laika the ceiling is `L2`,
which the search already uses. Against a searching opponent in selfplay the
model would have to predict another Tactical, which is harder than predicting a
scripted one.

## Where this leaves the project

| Axis | Status |
|---|---|
| Movement quality | Solved. Wall contact 21% → 3%, zero-motion 12.5% → 0.12% |
| Policy strength | K1 adopted; +0.50pp on frozen, not significant alone |
| Latency tail | Understood and attributed; pruning rejected as a bad trade |
| Opponent modelling | Bounded and rejected — no measurable room in the current wiring |
| Remaining gap to the oracle | ~3pp, and it is information the agent may not have |
