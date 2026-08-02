# P26 Human Observations

## 2026-08-01 00:07 AEST - P26v5 Champion Watch

Observed on `p26_amortized_mpc_iter05.pt` with `fire-margin=0.16`.

### Problems

- `post_kill_fire`: Keeps firing after Laika is already dead.
  - Impact: Mostly visual / control hygiene after the round is decided, but it
    can indicate that the policy lacks an explicit terminal-state gate.
  - Strategy: Add an action guard for dead enemy / frozen round in watch and
    deployment wrappers. Do not train this as a scoring improvement unless it
    can still affect double death before `round_end`.

- `passive_map_control`: Does not actively move through the map or seek better
  positions.
  - Impact: Misses proactive kill creation and can look much weaker than the
    official win rate in human inspection.
  - Strategy: Add map-control / position-value labels on hard states, using a
    stronger teacher horizon. Avoid rewarding random movement.

- `stutter_stall`: Sits in place or jitters in place instead of committing to a
  path.
  - Impact: Creates dead-end and corner stalls; likely overlaps with
    `unsafe_movement` and `movement_value_gap`.
  - Strategy: Detect low displacement over a short window and record it as a
    separate failure tag. Upweight only when the teacher chooses a clearly
    different escape/movement action.

- `dead_end_stall`: Remains still in corners or dead-end corridors.
  - Impact: Converts otherwise survivable states into delayed losses.
  - Strategy: Add local topology features or labels for escape availability,
    then train with category weight lower than hard fire errors but higher than
    generic movement gaps.

- `blind_fire`: Fires without being aligned with Laika or a plausible rebound
  line.
  - Impact: Wastes shots, lowers hit rate, and can create unsafe bullets.
  - Strategy: Add a fire-line quality / no-line penalty label. Keep fire loss
    small; prefer gating and ranking corrections to avoid the P26v8 over-fire
    regression.

### Strengths

- `bullet_dodge_good`: Dodges bullets reasonably well.
  - Impact: This is a core strength and should be protected.
  - Strategy: Keep a holdout metric for double death, death rate, and dodge
    survival. Do not increase aggression unless hit rate and loss/DD stay stable.

### Current Implication

P26v8 already showed that simply increasing hard-case pressure can damage fire
quality: `120@970000 = 79.2%`, `shots/game = 3.7`, `hit = 15.0%`. For the next
iteration after P26v9, movement/topology and fire-line quality should be modeled
explicitly instead of pushing generic aux/fire losses.
