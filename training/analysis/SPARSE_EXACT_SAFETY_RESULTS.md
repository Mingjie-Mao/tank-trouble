# Sparse Exact-State Safety Policy

## Decision

The fixed-scheduler configuration passes the pre-registered 120-game gate and
is the current high-reliability hybrid policy. It is not the pure-network
champion and it is not a proof of 100% performance.

- Gate: at least 116/120 wins and at most one double death
- Result: **119/120 wins = 99.17%**, zero ordinary losses, one double death
- 95% Wilson interval for the observed win rate: approximately 95.4%-99.9%
- Report: `training/analysis/runs/sparse_exact_safety_3x40_fixed_scheduler.json`

The remaining non-win is deterministic seed `990008`. The full frozen exact
teacher wins this seed, so the residual error belongs to sparse intervention
scheduling rather than to the exact engine clone or the teacher's action set.

## Frozen Configuration

- Base action network: `training/models/p26_amortized_mpc_iter05.pt`
- Risk/value network: `training/models/p27b_risk_value_iter00.pt`
- Exact audit: one proposed action every frame, horizon 72
- Full-search triggers: unsafe audit, behavior issue, successor follow-up, or
  48 frames since the previous proactive search
- Full-search action commitment: 12 frames, with continued one-action audits
- Exact full search: top 12 prior actions, widen to all 18 when necessary
- Critical safe-root burst: disabled
- Fire margin: 0.16

The proactive scheduler is deadline-based. It uses elapsed frames since the
last proactive search instead of `frame % 48 == 0`, because a 12-frame action
commitment could skip a modulo deadline and create a 96-frame planning gap.

## Results

| Seed band | Wins | Losses | Double deaths | Full-search frames |
| --- | ---: | ---: | ---: | ---: |
| 970000-970039 | 40/40 | 0 | 0 | 8.25% |
| 990000-990039 | 39/40 | 0 | 1 | 11.57% |
| 973000-973039 | 40/40 | 0 | 0 | 7.55% |
| Pooled | 119/120 | 0 | 1 | 9.33% |

Cost and behavior telemetry:

- 38,796 policy frames
- 36,564 one-action exact audits, or 94.25% of policy frames
- 3,619 full searches, or 9.33% weighted by policy frames
- Mean per-round full-search rate: 8.66%
- Median / P90 / P95 / maximum rate: 5.56% / 19.48% / 25.00% / 39.62%
- 7,304,904 simulated frames
- Behavior detections: 928 missed-fire-window, 330 dead-end-stall,
  220 stutter-stall, 62 blind-fire, 32 passive-map-control

Comparison on the same three 40-game seed bands:

- Previous sparse scheduler: 117/120 = 97.5%, two losses and one double death
- Fixed scheduler: 119/120 = 99.17%, zero losses and one double death
- Full Exact-State Safety-Shielded MPC: 120/120 = 100.0%

The full teacher separately scored 297/300 = 99.0% on unseen seed bands with
three ordinary losses and zero double deaths. Neither sample proves a true
100% win probability.

## Remaining Failure

Seed `990008` ended in a Laika-bounce double death. The sparse policy made 74
full searches in the round. Starting at frame 202 it produced 65 consecutive
`no_safe_search_action` events; incoming projectile risk became visible only
at frame 257 and rose to 0.909 by frame 265.

This is not primarily a late bullet-dodge failure. The policy entered a state
where the 72-frame search could no longer find a viable root before the
observable projectile threat appeared. Per-frame action safety detects local
death risk, but it does not guarantee long-term recoverability or map control.

A static critical-safe-root burst can win `990008`, but it also regressed
`970004` and used up to 68% full-search frames in the targeted test. That rule
is rejected because its causal trigger is too broad.

## Rejected Network Shortcuts

The exact teacher's hidden-state safety decision could not yet be compressed
reliably into a standalone classifier:

- Observation-only residual: 15% out-of-fold accuracy, 0% held-out accuracy
- Initial privileged classifier: 40% out-of-fold, 12.5% held-out
- Expanded 640-dimensional full-state classifier: no high-confidence operating
  point passed the calibration gate

No online A/B was run for these classifiers. This prevented an uncalibrated
network from replacing exact safety checks.

## Next Gate

Do not start a broad parameter sweep or a 900-game validation yet. First:

1. Replay `990008` from saved exact states and locate the earliest divergence
   from the full teacher before the safe-root set collapses.
2. Measure a recoverability signal, such as safe-root trend plus consecutive
   unsafe audits, instead of using a single safe-root threshold.
3. Run an A/B gate on `990008`, `970004`, `990022`, and at least nine controls.
   Require all three regressions to win, no new control failure, and a pooled
   full-search rate no higher than 12%.
4. Only then run fresh unseen 3x100 validation. Promotion requires at least
   297 wins, at most one double death, and no seed-band collapse.

This sequence targets the known causal gap while preserving the 119/120
baseline. It does not assume that more search or a larger network must help.

## Narrow-Replan Follow-up

The causal handoff trace for `990008` found a single decisive frame. Full
teacher takeover at frames 169-173 won, while takeover at frame 174 or later
double-died. At frame 173 the 12-frame commitment had just expired: the sparse
policy audited action 4 as locally safe, while the full teacher reranked to
action 0. A one-shot replan after a search with at most three safe roots fixed
all three historical regressions:

- `970004`: win
- `990008`: win, with 14 full searches instead of 74
- `990022`: win
- Nine control seeds: 9/9 wins
- Fresh screen: 36/36 wins, zero double deaths, 8.06% pooled search rate

The candidate then failed the pre-registered fresh 3x100 gate. It was stopped
as soon as the fourth non-win made 297/300 impossible:

- Completed: 152/300
- Result: 148 wins, 4 losses, 0 double deaths = 97.37% on completed games
- Pooled full-search rate: 7.97%
- Failures: `983043`, `983064`, `993007`, `993042`
- Partial data: `training/analysis/runs/sparse_exact_safety_narrow_replan_unseen_3x100.json.partial.jsonl`
- Summary: `training/analysis/runs/sparse_exact_safety_narrow_replan_unseen_3x100_stopped_summary.json`

The full frozen teacher won all four failed seeds. Every sparse failure entered
a long `no_safe_search_action` interval; three died to Laika bounce shots and
one to a self-shot. Handoff experiments on `983043` were non-monotonic: frame
260 takeover lost, while frame 270 takeover won. Therefore takeover time is
not a scalar safety threshold, and binary-searching a universal trigger is not
valid.

The narrow-replan candidate is rejected for promotion. The stable hybrid
champion remains the fixed scheduler with critical and narrow modes disabled,
at 119/120 on its formal benchmark. Further local trigger tuning is paused;
the next high-leverage work is improving the full teacher from constant-action
72-frame rollouts to short action-sequence planning, then validating that
teacher before any additional distillation or sparse scheduling work.

The teacher-side hard gate starts with seven seeds:

- Existing full-teacher losses: `975062`, `981086`, `991011`
- Sparse-only losses that the full teacher wins: `983043`, `983064`,
  `993007`, `993042`

The next method should search short action sequences rather than only repeating
one root action for all 72 rollout frames. A conservative first version is a
two-stage exact beam: execute the first action for a short chunk, retain the
best safe states, then branch to a second action. It should activate only when
the constant-action search is constrained or low-value, so normal teacher
latency remains unchanged. The method must win all seven hard seeds and a
12-seed control gate before any broad teacher benchmark is justified.
