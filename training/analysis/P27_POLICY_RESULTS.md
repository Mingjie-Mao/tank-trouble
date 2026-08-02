# P27 Policy Results

Date: 2026-08-01

## Baseline

Current deployment champion remains:

- Model: `training/models/p26_amortized_mpc_iter05.pt`
- Deployment fire margin: `0.16`
- Official: `1000@970000 = 87.1%`, `500@990000 = 86.4%`
- Behavior gate reference observed earlier:
  - `40@970000 ~= 90.0%`
  - `blind_fire = 4`, `dead_end_stall = 4`
  - `missed_fire_window = 328`, `stutter_stall = 246`
  - `shots/game = 2.0`, `hit = 26.3%`, `DD = 2.5%`

## P27 Macro Probe

Macro probe across `970000`, `990000`, and `973000` produced 532 hard states.

- Escape best: `272 / 532 = 51.1%`
- Hold/reposition best: `214 / 532 = 40.2%`
- Fan-fire best: `31 / 532 = 5.8%`
- Single-fire best: `15 / 532 = 2.8%`
- Positive macro over hold: `177 / 532 = 33.3%`
- Positive fan-fire over hold: `22 / 532 = 4.1%`

Interpretation: broad sweep fire is rarely the best hard-state macro. Most useful macro information is about escape/stuck recovery, but direct macro control is risky because it can interrupt good base-policy actions.

## P27 Macro Head Iter00

Files:

- Policy/training code: `training/p27_macro_policy.py`
- Model: `training/models/p27_macro_iter00.pt`
- Observer integration: `training/p26_behavior_observer.py`
- First run summary: `training/analysis/runs/p27_macro_iter00_observer_40_970000_summary.json`
- Conservative run summary: `training/analysis/runs/p27_macro_iter00_conservative_observer_40_970000_summary.json`

Training:

- Data: 3 macro-probe files, 532 samples
- Positive macro rate: `33.3%`
- Validation was unstable and showed overfitting after mid training.

Real behavior gate:

- Raw macro wrapper, `40@970000`: `72.5%`
  - `loss = 25.0%`, `DD = 2.5%`
  - `blind_fire = 1`
  - `missed_fire_window = 267`
  - `stutter_stall = 267`
  - `dead_end_stall = 7`
- Conservative category-gated wrapper, `40@970000`: `70.0%`
  - `loss = 17.5%`, `DD = 12.5%`
  - `blind_fire = 10`
  - `missed_fire_window = 354`
  - `stutter_stall = 289`
  - `dead_end_stall = 11`
  - Macro triggers in 40 games: 138

Decision: reject P27 macro action override. It takes control too often and disrupts the champion policy.

## Fire Assist Probe

Config:

- Base champion: `p26_amortized_mpc_iter05.pt`
- `fire_margin = 0.16`
- `fire_assist_line = 0.72`
- `fire_assist_max_risk = 0.25`
- `fire_assist_min_delta = -0.03`
- `suppress_blind_fire_line = 0.35`
- Summary: `training/analysis/runs/p27_fire_assist_observer_40_970000_summary.json`

Result:

- `40@970000 = 85.0%`
- `loss = 10.0%`, `DD = 5.0%`
- `blind_fire = 0`
- `missed_fire_window = 366`
- `stutter_stall = 274`
- `dead_end_stall = 7`
- `shots/game = 3.625`, `hit = 17.9%`

Decision: reject this direct fire-assist config. It fires more, but lowers hit quality and does not solve missed windows.

## P27 Value Assist

Files:

- Code: `training/p27_macro_policy.py`
- Summary: `training/analysis/runs/p27_value_assist_iter00_observer_40_970000_summary.json`

Config:

- Base champion: `p26_amortized_mpc_iter05.pt`
- Macro/value source: `p27_macro_iter00.pt`
- `macro_mode = value`
- `suppress_blind_fire_line = 0.35`
- Score assists only, no fixed action sequence.

Result:

- `40@970000 = 87.5%`
- `loss = 10.0%`, `DD = 2.5%`
- `blind_fire = 0`
- `missed_fire_window = 401`
- `stutter_stall = 259`
- `dead_end_stall = 6`
- `shots/game = 2.175`, `hit = 21.8%`
- Assist triggers:
  - `fire_score_bonus = 110`
  - `escape_score_bonus = 101`
  - `blind_fire_penalty = 2`

Decision: reject this value-assist version. It is less destructive than fixed macro sequences, but still triggers too often and does not reduce missed-fire windows.

## Suppress-Only Safety Gate

Files:

- Summary: `training/analysis/runs/p26_champion_suppress_blind_observer_40_970000_summary.json`

Config:

- Base champion: `p26_amortized_mpc_iter05.pt`
- `fire_margin = 0.16`
- `suppress_blind_fire_line = 0.35`
- No P27 macro/value net.

Result:

- `40@970000 = 90.0%`
- `loss = 7.5%`, `DD = 2.5%`
- `blind_fire = 0`
- `missed_fire_window = 309`
- `stutter_stall = 239`
- `dead_end_stall = 4`
- `shots/game = 1.875`, `hit = 28.0%`

Decision: this is the safest candidate found in this pass. It preserves the behavior-gate win rate and removes visible blind-fire without adding instability. It should get 300-game medium validation before any promotion.

Medium validation:

- Summary: `training/analysis/runs/p27_suppress_medium_observer_300_970000_summary.json`
- `300@970000 = 84.0%`
- `loss = 10.3%`, `DD = 5.0%`, `draw = 0.7%`
- `blind_fire = 0`
- `missed_fire_window = 3732`
- `stutter_stall = 2236`
- `dead_end_stall = 52`
- `shots/game = 2.393`, `hit = 24.8%`

Final decision: reject suppress-only as a promotion candidate. The 40-game pass was not stable under 300-game validation.

## P27b Risk/Value Head Iter00

Files:

- Code: `training/p27_risk_value.py`
- Model: `training/models/p27b_risk_value_iter00.pt`
- Observer integration: `training/p26_behavior_observer.py`
- 300-game summaries:
  - `training/analysis/runs/p27b_risk_value_iter00_observer_300_970000_summary.json`
  - `training/analysis/runs/p27b_risk_value_iter00_observer_300_990000_summary.json`

Training data:

- Strong-teacher hard-state phases:
  - `p26v9_strong_h72s3_970000`
  - `p26v9_strong_h72s3_990000`
- 17 shards, 306 hard-state samples.
- Main categories:
  - `unsafe_movement = 166`
  - `movement_value_gap = 67`
  - `missed_fire_window = 46`
  - `missed_kill_line = 19`
  - `unsafe_fire_death = 5`
  - `double_death_risk = 2`

Config:

- Base champion: `training/models/p26_amortized_mpc_iter05.pt`
- P27b value/risk head: `training/models/p27b_risk_value_iter00.pt`
- `fire_margin = 0.16`
- `p27b_assist_margin = 0.08`
- `p27b_assist_weight = 0.35`
- `p27b_max_bonus = 0.10`
- `p27b_death_weight = 0.12`
- `p27b_double_death_weight = 0.18`
- `p27b_fire_delta_margin = 0.14`

Small gates:

- `40@970000 = 95.0%`
  - `loss = 5.0%`, `DD = 0.0%`
  - `shots/game = 2.025`, `hit = 32.1%`
- `40@990000 = 92.5%`
  - `loss = 7.5%`, `DD = 0.0%`
  - `shots/game = 2.975`, `hit = 20.2%`
- `40@973000 = 95.0%`
  - `loss = 5.0%`, `DD = 0.0%`
  - `shots/game = 2.225`, `hit = 30.3%`

Medium gates:

- `300@970000 = 89.3%`
  - `loss = 8.0%`, `DD = 2.7%`
  - `shots/game = 2.43`, `hit = 24.6%`
  - issues: `missed_fire_window = 3424`, `stutter_stall = 2122`, `dead_end_stall = 36`
- `300@990000 = 90.3%`
  - `loss = 5.0%`, `DD = 4.7%`
  - `shots/game = 2.68`, `hit = 24.1%`
  - issues: `missed_fire_window = 3334`, `stutter_stall = 2115`, `dead_end_stall = 29`

Decision: P27b deserves official validation. It beats the current champion's official band (`87.1%`, `86.4%`) on both medium seeds, but double-death risk is still high enough that it should not replace the champion until the full `1000@970000` and `500@990000` results are known.

Official validation:

- Log: `training/logs/p27b_iter00_official_grade.log`
- `1000@970000 = 88.2%`
  - `loss = 8.5%`, `DD = 3.2%`, `draw = 0.1%`
  - `shots/game = 2.5`, `hit = 23.5%`, `avg length = 17.6s`
- `500@990000 = 89.6%`
  - `loss = 5.8%`, `DD = 4.2%`, `draw = 0.4%`
  - `shots/game = 2.7`, `hit = 23.3%`, `avg length = 18.4s`

Decision: P27b iter00 is the strongest local candidate so far and beats P26v5 iter05 official (`87.1%`, `86.4%`). It should be treated as the current candidate champion, but not the endpoint: the remaining gap is dominated by double-death risk, missed fire windows, and visible stutter/dead-end behavior.

## P27c Rule-Gate Attempts

Goal: directly reduce DD, missed-fire, and stutter/dead-end issues without retraining.

Strong gate attempt:

- Summary: `training/analysis/runs/p27c_gated_iter00_observer_40_970000_summary.json`
- `40@970000 = 87.5%`
- `loss = 5.0%`, `DD = 5.0%`, `draw = 2.5%`
- `shots/game = 1.55`, `hit = 29.0%`, `avg length = 24.3s`
- issues: `missed_fire_window = 741`, `stutter_stall = 393`, `dead_end_stall = 6`

Lite gate attempt:

- Summary: `training/analysis/runs/p27c_lite_iter00_observer_40_970000_summary.json`
- `40@970000 = 92.5%`
- `loss = 5.0%`, `DD = 2.5%`
- `shots/game = 2.05`, `hit = 26.8%`, `avg length = 18.4s`
- issues: `missed_fire_window = 427`, `stutter_stall = 280`, `dead_end_stall = 1`

Decision: reject rule-only P27c. Both variants underperform P27b's small-gate `40@970000 = 95.0%` and do not reliably reduce the visible issue counts. Simple rule gates make the policy more hesitant and can lengthen games. The next direction must be hard-case retraining from P27b's actual failure and visible issue states.

## P27c Hard-Case Pipeline

Files:

- Pipeline: `training/run_p27c_hardcase_pipeline.sh`
- Candidate output: `training/models/p27c_risk_value_iter01.pt`
- Log: `training/logs/p27c_hardcase_pipeline.log`

Plan:

1. Collect P27b loss/double-death final windows with `score_horizon = 96`, `score_samples = 3`.
2. Collect P27b visible missed-fire/stutter/dead-end states with observer labels.
3. Train a new P27c risk/value head with higher category weights for `double_death_risk`, `fire_into_double_death`, `unsafe_fire_death`, `missed_fire_window`, `stutter_stall`, and `dead_end_stall`.
4. Gate with observer runs at `80@970000`, `80@990000`, and `80@973000`.

## Observed Failure Categories

The user's visual observations match the instrumented metrics:

- Good: bullet dodging is relatively strong.
- Bad: clear shots are still declined.
- Bad: policy often stutters in place.
- Bad: policy sometimes waits in corners or dead ends.
- Bad: blind/post-kill firing appears when fire control is made more aggressive.
- Bad: action-level macro override can make the model look much worse than the official aggregate score.

## Next P27 Direction

Do not make P27 a macro action controller yet.

Build P27 as a small value/risk head attached to P26, trained on better hard-state labels rather than the current macro labels:

- `danger_next_1s`
- `danger_next_2s`
- `dead_end_escape_value`
- `stuck_escape_value`
- `kill_window_value`
- `shot_quality`
- `double_death_risk`

Use these heads only as score adjustments/gates:

- Penalize actions that keep the tank stalled in dead ends.
- Penalize firing when `shot_quality` is low or `double_death_risk` is high.
- Slightly lower fire margin only when `kill_window_value` is high and `shot_quality` is high.
- Add movement value bonus only when stuck/dead-end risk is detected.

The important change is that P27 should alter action scores, not run fixed action sequences. That keeps P26's strong base behavior and targets only the remaining failure states.

Immediate next run:

1. Collect hard states from the current champion without extra gates, with the behavior observer categories enabled.
2. Label those states with stronger teacher rollouts that produce action-level risk/value labels, not macro-sequence labels.
3. Train a true P27 risk/value head and use it as small action-score adjustments.
4. Reject any candidate that does not pass `40@970000`, `40@990000`, then `300@970000` before official grading.

## P28 Hybrid Search Progress

File:

- Policy/eval harness: `training/p28_hybrid_fallback.py`

Baseline:

- Current pure-network champion remains P27b iter00:
  `1000@970000 = 88.2%`, `500@990000 = 89.6%`.

P28 fallback-trigger search was rejected:

- Safe fallback config: `40@970000 = 92.5%`, `DD = 5.0%`,
  `wall = 5.34s/game`.
- Problem: sparse triggers are faster, but they do not raise the ceiling enough
  and still leave double-death risk.

P28 prior-search, horizon 48, top-K 12:

- `40@970000 = 100.0%`, `loss = 0.0%`, `DD = 0.0%`,
  `wall = 15.46s/game`.
- `40@990000 = 97.5%`, `loss = 2.5%`, `DD = 0.0%`,
  `wall = 19.27s/game`.
- `40@973000 = 92.5%`, `loss = 7.5%`, `DD = 0.0%`,
  `wall = 16.26s/game`.
- Failure seeds on `973000`: `973016` and `973018` died to Laika direct
  shots; `973027` died to self shot.

P28 prior-search, safer horizon 72:

- Slice `20@973010 = 100.0%`, `loss = 0.0%`, `DD = 0.0%`.
- Full `40@973000 = 95.0%`, `loss = 5.0%`, `DD = 0.0%`,
  `wall = 17.69s/game`.
- Remaining failure seeds: `973018` died to Laika direct shot; `973026`
  died to Laika bounce shot.

P28 prior-search, stronger horizon 96:

- Slice `12@973016 = 100.0%`, `loss = 0.0%`, `DD = 0.0%`,
  `wall = 27.02s/game`.
- Full `40@973000 = 95.0%`, `loss = 2.5%`, `DD = 0.0%`,
  `draw = 2.5%`, `wall = 28.87s/game`.
- Remaining failures: `973006` draw after 2500 frames with 47 shots;
  `973010` died to Laika bounce shot.

P28 prior-search, deterministic horizon 72, samples 2:

- Added `--deterministic-search-seeds` so rollout RNG is derived from
  round seed, frame, candidate, and sample rather than worker scheduling.
- Slice `30@973000 = 100.0%`, `loss = 0.0%`, `DD = 0.0%`,
  `wall = 24.14s/game`.
- Full `40@973000 = 97.5%`, `loss = 2.5%`, `DD = 0.0%`,
  `draw = 0.0%`, `wall = 25.88s/game`.
- Remaining failure: `973034` died to Laika direct shot after 990 frames.
- Back-check `40@970000 = 95.0%`, `loss = 5.0%`, `DD = 0.0%`,
  `draw = 0.0%`, `wall = 31.61s/game`.
- Failures on `970000`: `970017` died to Laika bounce shot; `970031`
  died to Laika direct shot.
- Back-check `40@990000 = 92.5%`, `loss = 7.5%`, `DD = 0.0%`,
  `draw = 0.0%`, `wall = 38.72s/game`.
- Failures on `990000`: `990011` and `990037` died to self shots;
  `990024` died to Laika direct shot.

Decision:

- P28 proves the 96%+ target is reachable with a stronger teacher/hybrid
  search path on the hard `973000` band, but the same best config does not
  hold 96% across `970000` and `990000`.
- Every-frame search is also too slow for final deployment: `970000` and
  `990000` took 21-26 minutes for only 40 games.
- The best current P28 teacher config is deterministic horizon 72 with
  `search_samples = 2`, not horizon 96. It clears the hard `973000` band at
  `97.5%` while avoiding h96's draw/over-conservative behavior.
- The next target is to compress useful P28 decisions into a faster P29
  network while fixing the remaining direct-shot/self-shot losses and the
  long-game tendency.

Next:

1. Collect P28 teacher decisions on failure/uncertain frames for P29 distill.
2. Add targeted hard-case traces for `973034`, `970031`, and `990024`
   direct-shot losses, plus `990011` and `990037` self-shot losses.
3. Train P29 from P28-adjusted action values, not raw MPC score alone.
4. Gate P29 at `80@970000`, `80@990000`, and `80@973000` before any medium or
   official validation.

## P29 P28 Distill Iter00

Files:

- Collector/training pipeline: `training/p29_p28_distill.py`
- Pipeline runner: `training/run_p29_p28_distill_pipeline.sh`
- Log: `training/logs/p29_p28_distill_pipeline.log`
- Model: `training/models/p29_p28_distill_iter00.pt`

Goal:

- Distill P28 deterministic horizon-72, samples-2 teacher decisions into a
  fast network.
- Focus on the known P28/P27b hard cases: direct-shot losses, self-shot
  losses, active pursuit, missed finish windows, stutter, dead-end stalls, and
  long-game behavior.

P28 teacher back-check before distill:

- `40@970000 = 95.0%`, failures `970017` bounce shot and `970031` direct shot.
- `40@990000 = 92.5%`, failures `990011` self shot, `990024` direct shot, and
  `990037` self shot.
- Existing hard-band result: `40@973000 = 97.5%`, remaining failure `973034`
  direct shot.

Collected P29 hard labels:

- Targeted P28 failure seeds:
  - `973034`: 86 labels, dominated by `direct_shot_loss`, `blind_fire`,
    `missed_fire_window`, and `stutter_stall`.
  - `970017`: 53 labels, dominated by `bounce_shot_loss`, `blind_fire`, and
    `dead_end_stall`.
  - `970031`: 57 labels, dominated by `direct_shot_loss`, `blind_fire`, and
    `stutter_stall`.
  - `990011`: 77 labels, dominated by `self_shot_loss`, `blind_fire`,
    `dead_end_stall`, and `missed_fire_window`.
  - `990024`: 58 labels, dominated by `direct_shot_loss`, `finish_window`, and
    `blind_fire`.
  - `990037`: 66 labels, dominated by `self_shot_loss`, `blind_fire`, and
    `dead_end_stall`.
- Broad P27b hard-frame phases:
  - `40@970000`: 373 labels from a `95.0%` rollout.
  - `40@990000`: 352 labels from a `92.5%` rollout.
  - `40@973000`: 432 labels from a `95.0%` rollout.

Training:

- 43 shards, 2938 samples.
- Largest categories:
  - `missed_fire_window = 643`
  - `stutter_stall = 657`
  - `dead_end_stall = 314`
  - `unsafe_movement = 303`
  - `bounce_shot_loss = 210`
  - `self_shot_loss = 134`
  - `direct_shot_loss = 126`
- Final epoch:
  - train `3.7209`
  - val `15.5953`
  - score `8.2677`
  - aux `46.5331`
  - rank `1.6261`
  - top1 `19.8%`
  - top3 `44.0%`
  - aux accuracy `84.7%`

Gate results with `fire_margin = 0.16`:

- `80@970000 = 90.0%`
  - `loss = 3.75%`, `DD = 6.25%`
  - `shots/game = 2.59`, `hit = 22.2%`
  - issues: `missed_fire_window = 901`, `stutter_stall = 612`
- `80@990000 = 87.5%`
  - `loss = 10.0%`, `DD = 2.5%`
  - `shots/game = 2.46`, `hit = 20.3%`
  - issues: `missed_fire_window = 815`, `stutter_stall = 598`
- `80@973000 = 88.75%`
  - `loss = 8.75%`, `DD = 2.5%`
  - `shots/game = 2.21`, `hit = 25.4%`
  - issues: `missed_fire_window = 876`, `stutter_stall = 589`

Decision:

- Reject P29 iter00 as a deployment or medium-validation candidate.
- It does not beat P27b official champion-level results
  (`1000@970000 = 88.2%`, `500@990000 = 89.6%`) with enough safety margin, and
  it regresses double-death risk on `970000`.
- It also does not inherit P28's hard-band `97.5%`, so current distillation is
  losing important teacher information.

Diagnosis:

- The collected hard labels are useful: they confirm the user's visual issues
  are real and frequent, especially missed fire windows, stutter, dead-end
  stalls, and terminal direct/self-shot deaths.
- The training mix is likely inconsistent: older raw-score hard labels and new
  P28 adjusted-value labels were trained together, which can confuse score and
  rank targets.
- The policy-side usage was too conservative for finishing behavior: the gate
  used no opportunity bonus, no escape bonus, and no stall-fire penalty, so the
  network still declined many clear windows and kept stuttering.
- The final checkpoint appears worse than earlier validation points; future
  runs should save and evaluate the best validation checkpoint rather than only
  the last epoch.

Next:

1. Do a fast P29b policy calibration sweep before retraining: add explicit
   opportunity fire bonus, escape bonus, stall-fire penalty, and stronger
   double-death fire risk penalty around `p29_p28_distill_iter00.pt`.
2. If the sweep improves `80@970000`, `80@990000`, and `80@973000`, run
   300-game medium validation.
3. If it does not improve, retrain P29b using only P28-adjusted labels, or
   regenerate older hard phases with the same P28 objective so all targets mean
   the same thing.
4. Add early stopping / best-checkpoint selection before the next official
   attempt.

## P29b/P29c outcome and P30 correction plan

P29b policy calibration found `finish_safe` as the best P29 deployment
configuration, but its three-seed confirmation was only
`91.25% / 88.75% / 93.75%` (pooled `91.25%`, DD `2.92%`). It reduced loss and
DD versus the uncalibrated P29 policy, but did not fix missed fire, stutter, or
dead-end behavior consistently enough for medium validation.

P29c removed mixed objectives, added 13 temporal/context features, split
validation by complete round, deduplicated states, and saved the best
checkpoint. The data pipeline was technically sound, but the teacher labels
were not:

- 1382 unique states from 97 rounds.
- Short-teacher disagreement was `30.8%` to `41.6%` on broad phases.
- Validation top1 was `10.5%`, top3 `29.2%`.
- Gates were `86.25% / 87.5% / 90.0%` (pooled `87.92%`, DD `2.92%`).

The P29c collector averaged conflicting h48/h72 labels with an h96 review.
That produces a smooth numeric target but not a decisive action target. P30
therefore changes both label semantics and the learning task:

1. h48 and h72 must agree with sufficient margin, otherwise h96 is sampled
   four times and needs at least three matching votes.
2. States still ambiguous after review train risk heads only; they cannot train
   movement/fire action targets.
3. P30 keeps the P27b action by default and predicts only `keep/override`, a
   replacement movement, a replacement fire decision, value gain, and
   base/replacement outcome risk.
4. Normal background frames are collected in addition to failure and visible
   issue frames, preventing an override network trained only on failures from
   triggering everywhere.
5. Every gate uses the behavior observer and records missed fire, blind fire,
   stutter, dead-end stall, passive map control, post-kill fire, and bullet
   dodging together with win/loss/DD.

P30 iter00 was started from the unchanged P27b champion. It uses 15 known
failure seeds plus 64 broad games for each of `970000`, `990000`, and `973000`.
The candidate is saved separately as
`training/models/p30_consensus_correction_iter00.pt`; P27b remains deployment
champion unless all three behavior gates justify further validation.

## Progressive Risk-Constrained MPC teacher

Progressive Risk-Constrained MPC begins the search-teacher improvement track
without replacing the existing search teacher or deployment policy. Its first
implementation changes the search procedure:

- All 18 root actions share the same freshly sampled Laika RNG stream.
- Search advances through `h24 -> h48 -> h72 -> h96` while reusing the same
  sandbox branches instead of restarting every horizon.
- Candidate counts shrink `18 -> 6 -> 3 -> 2`; the P27b action remains in the
  final comparison.
- Only the final two candidates receive four stochastic samples.
- The root action is committed for 24 frames, then a cheap closed-loop macro
  controller replans every 16 frames using the maze firing-position distance,
  32-heading firing line, alignment, and incoming risk.
- Selection first requires zero observed death and zero observed double death.
  Only safe candidates are ranked by value; if none are safe, the minimum-risk
  action is used.
- The objective includes mean score, kill value, death/DD penalties, and a
  standard-deviation tail penalty.

The deterministic probe at initial state `973034` used exactly 1272 simulated
frames and took about 0.62 seconds on the current machine. The equivalent P28
`h72`, top-12, two-sample decision took about 0.73 seconds. This is an initial
single-state timing result, not yet evidence that this is a stronger teacher.

The teacher evaluation waits for P30 to release the machine. It then runs three
12-game behavior screens. It enters three 40-game confirmations only if every
screen seed is at least 80% and pooled win rate is at least 90%. Every round
records the standard visible issue categories as well as search decisions and
simulated-frame cost.
