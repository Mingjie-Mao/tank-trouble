# P26v10 Behavior Iteration Notes

Current champion remains:

- Model: `training/models/p26_amortized_mpc_iter05.pt`
- Deployment config: `fire_margin=0.16`
- Official baseline: `1000@970000 = 87.1%`, `500@990000 = 86.4%`

## Behavior Baseline

Observer runs use real P26-vs-Laika games and record per-round issues.

`iter05, margin=0.16`

- `40@970000`: win 90.0%, loss 7.5%, DD 2.5%, shots/game 2.00, hit 26.3%
- Issues: missed_fire_window 328, stutter_stall 246, blind_fire 4, dead_end_stall 4
- `40@990000`: win 90.0%, loss 10.0%, DD 0.0%, shots/game 2.25, hit 24.4%
- Issues: missed_fire_window 377, stutter_stall 311, blind_fire 2, dead_end_stall 3

Observed strengths:

- Bullet dodging is comparatively good and should be protected.

Observed problems:

- Clear-looking fire windows are often not used.
- Movement still stutters or stalls.
- Some dead-end/corner stalls remain.
- Blind fire exists but is low in the champion compared with later variants.

## Rejected Attempts

### P26v9 / iter10

Raw iter10 looked stronger on `970000`, but it over-fired and generalized worse:

- `40@970000`: win 95.0%, shots/game 3.77, hit 13.9%, blind_fire 32
- `40@990000`: win 87.5%, shots/game 3.35, hit 18.7%, blind_fire 60

Decision: reject as champion. It increases firing but not firing quality.

### Simple Fire Gate / Margin Changes

`iter05 + fire_assist_line=0.85 + suppress_blind_fire_line=0.25`

- Six-game check did not reduce missed fire and increased shots.

`iter05 + margin=0.12`

- Six-game check looked promising, but `40@970000` dropped to 85.0%.
- Blind fire rose to 18 and DD/draw increased.

`iter05 + margin=0.10`

- Six-game check increased blind fire and stutter.

Decision: do not lower the global fire margin.

### Behavior Hard-Case Training

Generated behavior hard shards from real observer issues:

- `p26v10_behavior_iter05_m016_970000`: 207 rows
- `p26v10_behavior_iter05_m016_990000`: 203 rows

Direct training on these rows produced `iter11`, which failed:

- `40@970000`: win 80.0%, loss 12.5%, DD 7.5%
- Issues worsened: missed_fire_window 439, stutter_stall 289, blind_fire 18

Root cause:

- The first hard-case dataset was too broad.
- Most `missed_fire_window` frames were not true teacher-confirmed fire frames.
- Many `stutter_stall` rows had low regret and weak corrective signal.

Filtered hard shard:

- `p26v10_behavior_filtered_iter05_m016`: 27 rows
- Categories: stutter_stall 18, missed_fire_window 8, blind_fire 1
- Teacher-fire rate: 40.7%

Training this into `iter12` still failed:

- `40@970000`: win 87.5%, below champion 90.0%
- Stutter slightly improved, but blind_fire rose to 19.

Decision: do not continue behavior hard-case micro-training in the current head setup.

### iter10 + Blind-Fire Suppression

`iter10 + suppress_blind_fire_line=0.35`

- `40@970000`: win 92.5%, blind_fire 0, but hit 18.1%
- `40@990000`: win 87.5%, blind_fire 0, below champion 90.0%

Decision: useful diagnostic, not a deployable champion.

## Current Conclusion

The champion stays `iter05 + margin=0.16`.

The main remaining issue is not a single fire threshold. Attempts to make the current heads more aggressive tend to:

- increase shots,
- reduce hit rate,
- increase blind fire,
- or degrade cross-seed stability.

The behavior observer is useful and should remain mandatory after each candidate.

## Next Strategy

Do not keep micro-tuning the same score/fire heads. The next real improvement should add explicit lightweight value/risk heads:

- `danger_next_1s`
- `danger_next_2s`
- `kill_window`
- `escape_value`
- `stuck_escape_value`
- `fire_quality`

Training data should use observer labels only as selectors, then keep only teacher-confirmed high-value frames. The model needs to know why a move is safe, why a fire line is good, and why staying in place is bad. The current scalar score head is not separating those concepts reliably.
