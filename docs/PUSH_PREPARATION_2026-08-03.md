# Push Preparation (2026-08-03)

## Repository state

- Local `main` and `origin/main` both point to `f31e4d5` after `git fetch`.
- The current work is local only. Nothing has been staged, committed, or pushed.
- The tracked change is the P26 viewer support in `training/watch.py`.

## Include in Git

- Reproducible training and evaluation source for P26-P30.
- Exact-state and sparse exact-safety teacher implementations.
- Unit tests and verification scripts.
- Portable `training/run_*.sh` orchestration scripts.
- Strategy and result summaries under `docs/` and `training/analysis/`.
- Champion manifests under `training/champions/`.
- These compact, decision-relevant run reports:
  - `exact_state_clone_fidelity.json`
  - `exact_state_fire_cost_official_3x40.json`
  - `exact_teacher_unseen_3x100.json`
  - `sparse_exact_safety_3x40_fixed_scheduler.json`
  - `sparse_exact_safety_narrow_replan_unseen_3x100_stopped_summary.json`

## Keep local

- Model checkpoints in `training/models/`.
- Raw P26 amortized datasets in `training/p26_amortized_data/`.
- Bulk analysis output under `training/analysis/runs/`, except the five reports above.
- Runtime logs and PID files under `training/logs/`.
- `local_backups/` and `screenlog.*`.

The ignored model checkpoints are required to reproduce the current champion exactly. If they need to be shared, publish only the promoted checkpoints with checksums through Git LFS or a release artifact; do not add the whole model directory to normal Git history.

Required champion artifacts:

- `training/models/p26_amortized_mpc_iter05.pt` (about 15 MB)
- `training/models/p27b_risk_value_iter00.pt` (about 5.7 MB)

Their SHA-256 values are already recorded in the exact-state and sparse-safety champion manifests and were rechecked during this preparation.

## Validation before commit

1. Run Python syntax compilation for all new source and test modules.
2. Run the exact-state, teacher, fire-decoupled, and sparse-safety unit tests.
3. Run `zsh -n` on every new orchestration script.
4. Validate champion manifests and selected reports as JSON.
5. Run `git diff --check` and inspect `git add --dry-run .`.
6. Scan the final candidate set for credentials and machine-specific absolute paths.

## Suggested commits

1. `training: add amortized policy and behavior evaluation pipeline`
2. `training: add exact-state safety teachers and distillation`
3. `docs: record strategy, champions, and validated results`

Keep model publication separate from source commits so source review and repository cloning stay lightweight.
