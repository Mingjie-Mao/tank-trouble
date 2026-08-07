# P27 Macro Probe Results

P27a started with an offline macro-action probe instead of immediate training.
The probe runs the current champion (`p26_amortized_mpc_iter05.pt`,
`fire_margin=0.16`) against Laika, captures real hard states, and evaluates
short macro actions in the sandbox teacher.

Macro actions tested:

- `hold_fire_reposition`
- `single_fire_policy`
- `fan_left_3`
- `fan_right_3`
- `fan_center_3`
- `escape_back_left`
- `escape_back_right`
- `escape_forward_left`
- `escape_forward_right`
- `escape_forward`

## Probe Runs

All runs used `horizon=72`, `samples=1`, and at most 5 hard states per round.

`40@970000`

- Records: 182
- Positive macro rate: 30.8%
- Best kind: escape 54.4%, fan-fire 4.4%, hold 39.0%, single-fire 2.2%
- Fan positive rate: 3.8%
- Categories: blind_fire 2, dead_end_stall 68, missed_fire_window 60,
  passive_map_control 11, stutter_stall 41

`40@990000`

- Records: 176
- Positive macro rate: 29.0%
- Best kind: escape 46.0%, fan-fire 6.8%, hold 45.5%, single-fire 1.7%
- Fan positive rate: 4.5%
- Categories: dead_end_stall 73, missed_fire_window 55,
  passive_map_control 13, stutter_stall 35

`40@973000`

- Records: 174
- Positive macro rate: 40.2%
- Best kind: escape 52.9%, fan-fire 6.3%, hold 36.2%, single-fire 4.6%
- Fan positive rate: 4.0%
- Categories: blind_fire 2, dead_end_stall 56, missed_fire_window 75,
  passive_map_control 7, stutter_stall 34

## Combined Result

Across 532 hard states:

- Escape best: 272 / 532 = 51.1%
- Hold/reposition best: 214 / 532 = 40.2%
- Fan-fire best: 31 / 532 = 5.8%
- Single-fire best: 15 / 532 = 2.8%
- Any positive macro over hold: 177 / 532 = 33.3%
- Positive fan-fire over hold: 22 / 532 = 4.1%

## Interpretation

Fast fan-fire exists and is sometimes useful, but it is not the main missing
skill. It should be a high-confidence kill-window macro only. If used broadly,
it will likely repeat the iter10 failure mode: more shots, lower hit rate, and
more blind fire.

The dominant improvement signal is escape/reposition. This matches the human
observations: the champion often stutters, stays in dead ends, and fails to
actively take better map positions.

## P27 Decision

P27 should not start as a sweep-fire model.

P27 should start as:

1. A macro-score head trained on macro probe data.
2. Explicit value/risk heads:
   - `escape_value`
   - `stuck_escape_value`
   - `fire_quality`
   - `kill_window`
   - `danger_next_1s`
   - `danger_next_2s`
   - `double_death_risk`
3. A deployment wrapper that only executes macro actions when confidence is
   high, otherwise falls back to the current champion policy.

First target:

- Reduce `stutter_stall` and `dead_end_stall` without increasing blind fire.
- Keep fan-fire as a rare macro for confirmed high-quality kill windows.

Promotion rule:

- Do not promote on macro probe alone.
- A P27 candidate must beat current champion behavior gates on
  `40@970000`, `40@990000`, and `40@973000`, with no increase in blind fire or
  double death.
