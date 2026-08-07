#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26v7_failure_attribution.log"
pid_path="training/logs/p26v7_failure_attribution.pid"
net_path="training/models/p26_amortized_mpc_iter05.pt"
margin="0.16"
workers="10"

mkdir -p training/logs training/analysis/runs
printf "%s start P26v7 failure attribution\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "net=%s margin=%s workers=%s\n" "$net_path" "$margin" "$workers" \
  >> "$log_path"

run_attr() {
  local n="$1"
  local seed="$2"
  local phase="$3"
  local out="$4"
  printf "%s start attribution %s@%s phase=%s\n" \
    "$(date)" "$n" "$seed" "$phase" >> "$log_path"
  caffeinate -dims python3 training/p26_failure_attribution.py \
    --net "$net_path" \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin "$margin" \
    --window-frames 100 \
    --stride 4 \
    --score-horizon 48 \
    --fire-target-margin "$margin" \
    --hard-phase "$phase" \
    --max-cases 40 \
    --out "$out" \
    >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for attribution %s@%s\n" \
    "$(date)" "$code" "$n" "$seed" >> "$log_path"
  return "$code"
}

run_attr 300 970000 \
  p26v7_fail_iter05_m016_970000 \
  training/analysis/runs/p26v7_failure_iter05_m016_300_970000.json
code=$?
if [ "$code" -eq 0 ]; then
  run_attr 300 990000 \
    p26v7_fail_iter05_m016_990000 \
    training/analysis/runs/p26v7_failure_iter05_m016_300_990000.json
  code=$?
fi

printf "%s final exit %s\n" "$(date)" "$code" >> "$log_path"
exit "$code"
