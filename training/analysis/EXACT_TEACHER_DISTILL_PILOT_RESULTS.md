# Exact-State Safety-Shielded MPC Distillation Pilot

## Frozen teacher

The frozen teacher is `exact_state_safety_shielded_mpc_v1`.

Compared with the previous prior-guided MPC teacher, it keeps the same learned
candidate prior and 72-frame fixed-action rollouts, but changes the decision
contract in four important ways:

1. It clones the local engine's RNG and Laika internal state exactly instead of
   resetting or sampling the opponent inside each sandbox.
2. It rejects any root action whose rollout predicts death or double death.
3. It widens beyond the prior top 12 when those candidates contain no safe
   action, and checks one-step successor viability when only one or two safe
   roots remain.
4. It suppresses redundant fire after a kill is already secured and suppresses
   no-kill fire whose value gain is below 2 score points.

The fixed local benchmark result is 120 wins from 120 games across
`970000:40`, `990000:40`, and `973000:40`, with no loss and no double death.
This is a privileged local-engine result, not proof of 100% on arbitrary unseen
observation-only states.

## Label contract

The new collector preserves the official `top_k=12` teacher decision. After
the teacher acts, it evaluates only the missing candidates to produce complete
18-action labels without changing the selected action.

Each state records:

- the action actually executed and the prior network action;
- exact per-action score and safety facts (kill, death, double death, survival);
- the final fire/no-fire target and a mask for unambiguous fire labels;
- safety widening, successor shield, and fire-suppression interventions;
- visible behavior issues for every played round.

Consistency gates require zero selected-unsafe actions, zero mismatch between
the executed action and policy target, and zero selected fire-label mismatch.

## Pilot data

The 12-game pilot at seeds `980000..980011` produced:

- 12 wins, 0 losses, 0 double deaths;
- 1,120 labelled states, 1,026 with valid safe action targets;
- 169 low-gain fire suppressions;
- 94 safety widenings;
- 96 successor-shield checks and 3 successor-shield overrides;
- 26 secured-kill fire suppressions.

Visible behavior issues were still present despite the 12/12 score:

- 60 missed-fire-window events;
- 28 stutter/stall events;
- 18 dead-end stalls;
- 9 blind-fire events;
- 4 passive-map-control events.

## Training and rejection

Risk-positive weighting fixed the first offline regression. On the held-out
round groups, danger recall rose from 90.4% to 99.3%, fire-label accuracy rose
from 48.7% to 99.1%, and action agreement stayed at 22.5%.

The online gates nevertheless rejected the candidates:

- A new hard safety gate scored 2/12 at `970000` and 2/12 with two double
  deaths at `973000`. It masked too many actions and frequently stopped moving.
- Full-network fine-tuning under the unchanged P27b runtime scored 7/12 at
  `970000`. Absolute score calibration shifted by about 0.606 per action and
  firing became too rare.
- A trunk-frozen, score-anchored candidate limited average score change to
  0.0033, but triggered a severe long-game/stall regression and was stopped.

No candidate replaces the current P27b champion. No larger data collection is
justified by this pilot.

## Next supported strategy

Keep the current champion outputs frozen. Train a bounded residual correction
network from `(observation, prior action)` to:

1. predict whether the exact teacher has a high-confidence reason to override;
2. predict one replacement movement/fire action only on those states;
3. predict exact risk for the prior and replacement actions;
4. abstain by default, preserving the champion's score and fire calibration.

The online gate should measure override precision first. Expansion is warranted
only if a small three-seed screen preserves champion win rate, firing frequency,
and average round length while reducing the recorded failure categories.
