#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p25v3_firegate_sweep.log"
pid_path="training/logs/p25v3_firegate_sweep.pid"

mkdir -p training/logs
printf "%s start P25v3 firegate sweep\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"

caffeinate -dims python3 training/opportunity_distill_v3.py gate-sweep \
  --net training/models/p25v2_opportunity_best.pt \
  --n 200 \
  --seed 973000 \
  --workers 10 \
  --margins 0 0.02 0.05 0.08 0.12 0.16 \
  >> "$log_path" 2>&1

code=$?
printf "%s exit %s\n" "$(date)" "$code" >> "$log_path"
exit "$code"
