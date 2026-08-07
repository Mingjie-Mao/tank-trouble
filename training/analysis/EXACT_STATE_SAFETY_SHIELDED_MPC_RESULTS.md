# Exact-State Safety-Shielded MPC

## Status

This is the current **search-teacher champion** for the local fixed benchmark.
It is not the fast deployment network.

- `40@970000`: 40/40 wins, 0 losses, 0 double deaths
- `40@990000`: 40/40 wins, 0 losses, 0 double deaths
- `40@973000`: 40/40 wins, 0 losses, 0 double deaths
- Pooled: **120/120 = 100.0%**
- Wall time: 1334.35 seconds with 12 workers
- Result: `training/analysis/runs/exact_state_fire_cost_official_3x40.json`

The previous Deterministic Prior-Guided MPC backcheck was 38/40, 37/40,
and 39/40 (114/120 = 95.0%).

## Method

The teacher combines:

1. Exact cloning of the local engine, including RNG and Laika internal state.
2. Network-prior top-12 action ordering with exact deterministic rollouts.
3. Strict rejection of predicted death and double death.
4. Expansion to all 18 actions when prior pruning finds no safe action.
5. A one-frame successor viability shield near the safe-set boundary.
6. Suppression of redundant fire after an exact kill is already secured.
7. A 2.0-point minimum gain for fire that predicts no kill, accounting for
   projectile risk beyond the 72-frame MPC horizon.

Exact clone fidelity was verified for 20,000 frames across 20 seeds with no
event or full-state fingerprint mismatch.

## Reproduce

```bash
python3 training/exact_state_mpc_teacher.py \
  --seed-list 970000:40,990000:40,973000:40 \
  --workers 12 \
  --out training/analysis/runs/exact_state_fire_cost_official_3x40.json
```

## Scope

This is a privileged local teacher: it reads state that exists inside the
local original-code engine, including Laika's hidden state and RNG. Therefore
the 100.0% result is valid for this fixed local benchmark, but it is not proof
of 100% observation-only performance on arbitrary unseen seeds.

The current fast network champion remains:

- Base: `training/models/p26_amortized_mpc_iter05.pt`
- Risk/value head: `training/models/p27b_risk_value_iter00.pt`
- Official: 88.2% at `1000@970000`, 89.6% at `500@990000`

The next stage is to collect exact-teacher trajectories and distill the
safety, movement, and fire decisions into a new fast network, while keeping
this teacher as the labeling and regression oracle.
