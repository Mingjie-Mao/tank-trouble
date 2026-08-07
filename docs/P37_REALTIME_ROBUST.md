# P37 Realtime Robust Teacher

## Final policy split

The behavior-preserving Laika playback policy is
`p37-killfield-realtime --opponent-profile laika`.

- The inverse field and the original P37 score function are unchanged.
- Firing is a stationary atomic action: the eight move/turn-and-fire columns
  remain for checkpoint compatibility, but cannot win selection and are
  sanitized again before execution.
- A shot invalidates every pre-shot asynchronous plan.  The teacher stays
  still until a fresh plan containing the new own bullet is available, and a
  24-frame exact execution guard replaces any motion that would intersect an
  existing own bullet.
- Planning runs on a snapshot in a persistent worker.
- The game thread never waits for field construction or MPC.
- Results older than the configured wall-clock deadline, round, target cell,
  or six-frame freshness window are discarded.
- One persistent worker keeps the target-cell field cache hot and avoids
  duplicate cache misses across workers.
- While a plan is in flight, the last P37 locomotion macro continues without
  inventing a firing action.
- An engine-verified hit still overrides immediately.

At the actual 25 FPS playback cadence, a 100-frame run with 512 rays, two
bounces and a 36-frame horizon measured 2.26 ms game-thread p95 and 96.1 ms
background-plan p95. Accepted plans were 1.09 frames old on average and two
frames old at p95. No moving-fire action was emitted. A normal desktop uses
process mode by preference; restricted runners fall back to a thread worker.

## Post-kill survival

The old branch that blindly repeated the last movement after Laika died was
removed. During the original 75 live frames, the teacher evaluates nine no-fire
movements against every existing bullet and replans every one or two frames.
An explicit lethal-line test makes idle die while a moving action survives.

## Fairness and cheating boundary

The inverse field, current poses, visible bullets, maze, and deterministic
physics are model-based planning, not future-state leakage.

The original `L2` rollout does instantiate the Laika source algorithm. It does
not copy Laika's current goal stack and does not read the real game RNG, but it
is still opponent-specific white-box knowledge. It is acceptable for a
"best response to bundled Laika" benchmark, but it should not be called a
policy-agnostic or screen-only agent.

Profiles make that distinction explicit:

- `laika`: exact original white-box P37 behavior;
- `mixed`: 70% original behavior plus 30% robust opponent hypotheses;
- `human`: never calls the Laika algorithm and never reads the human's current
  private button flags. It evaluates fixed plausible movement/fire hypotheses
  from visible state and combines mean and worst-case scores.

## Search plus network fallback

`HybridPolicy` now supports the same opponent profiles. The network orders the
candidate actions; forward search verifies candidates until the time budget is
about to expire. With a 35 ms budget and a 48-frame horizon, sampled first-step
latencies were approximately 16 ms for `laika`, 23--29 ms for `human`, and
occasionally about 40 ms for `mixed` as one indivisible candidate can overrun
the remaining soft budget.

This improves robustness to non-Laika controls but does not prove superiority
against skilled humans. A credible human claim requires games against recorded
human input traces or live players; synthetic action hypotheses only remove the
known Laika-model dependency.

## Commands

Behavior-preserving realtime Laika teacher:

```bash
python3 training/watch.py \
  --policy p37-killfield-realtime \
  --opponent-profile laika \
  --plan-deadline 2.0 \
  --seed 37500001 \
  --field-rays 512 \
  --field-bounces 2 \
  --field-flight-frames 75
```

Opponent-agnostic teacher profile:

```bash
python3 training/watch.py \
  --policy p37-killfield-realtime \
  --opponent-profile human \
  --plan-deadline 2.0 \
  --seed 37500001
```

Play directly against the realtime teacher.  The policy controls red tank0;
the human controls black tank1 with arrow keys and `M` to fire:

```bash
python3 training/watch.py \
  --policy p37-killfield-realtime \
  --opponent-profile human \
  --human-opponent \
  --plan-deadline 2.0 \
  --seed 37500001 \
  --field-rays 512 \
  --field-bounces 2 \
  --field-flight-frames 75
```

Time-budgeted search plus network policy:

```bash
python3 training/watch.py \
  --policy hybrid \
  --opponent-profile human \
  --search-budget-ms 35 \
  --k 5 \
  --seed 37500001
```
