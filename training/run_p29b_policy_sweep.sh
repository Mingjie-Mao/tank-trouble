#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p29b_policy_sweep.log"
pid_path="training/logs/p29b_policy_sweep.pid"

mkdir -p training/logs training/analysis/runs
printf "%s start P29b tiered policy sweep\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"

caffeinate -dims python3 -u training/p29b_policy_sweep.py \
  --base-net training/models/p26_amortized_mpc_iter05.pt \
  --value-net training/models/p29_p28_distill_iter00.pt \
  --workers "${P29B_WORKERS:-10}" \
  --screen-n "${P29B_SCREEN_N:-40}" \
  --confirm-n "${P29B_CONFIRM_N:-80}" \
  --medium-n "${P29B_MEDIUM_N:-300}" \
  --report training/analysis/runs/p29b_policy_sweep_report.json \
  >> "$log_path" 2>&1
code=$?
printf "%s final exit %s\n" "$(date)" "$code" >> "$log_path"
exit "$code"
