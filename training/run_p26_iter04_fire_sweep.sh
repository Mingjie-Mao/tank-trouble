#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26_iter04_fire_sweep.log"
pid_path="training/logs/p26_iter04_fire_sweep.pid"
model_path="training/models/p26_amortized_mpc_iter04.pt"

mkdir -p training/logs
printf "%s start P26 iter04 fire-margin sweep\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"

run_eval() {
  local margin="$1"
  printf "%s margin=%s\n" "$(date)" "$margin" >> "$log_path"
  caffeinate -dims python3 training/p26_amortized_mpc.py eval \
    --net "$model_path" \
    --n 120 \
    --seed 973000 \
    --workers 10 \
    --kill-weight 0 \
    --death-weight 0 \
    --double-death-weight 0 \
    --survive-weight 0 \
    --fire-prob-weight 0 \
    --fire-margin "$margin" \
    --fire-threshold 0 \
    >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for margin=%s\n" "$(date)" "$code" "$margin" >> "$log_path"
  return "$code"
}

for margin in 0 0.02 0.05 0.08 0.12 0.16; do
  run_eval "$margin" || exit $?
done

printf "%s final exit 0\n" "$(date)" >> "$log_path"
