#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26v6_iter08_medium_eval.log"
pid_path="training/logs/p26v6_iter08_medium_eval.pid"
net_path="training/models/p26_amortized_mpc_iter08.pt"
margin="0.16"
workers="10"

mkdir -p training/logs
printf "%s start P26v6 iter08 medium eval\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "net=%s margin=%s workers=%s\n" "$net_path" "$margin" "$workers" \
  >> "$log_path"

run_eval() {
  local n="$1"
  local seed="$2"
  printf "%s start %s@%s\n" "$(date)" "$n" "$seed" >> "$log_path"
  caffeinate -dims python3 training/p26_amortized_mpc.py eval \
    --net "$net_path" \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
    --kill-weight 0 \
    --death-weight 0 \
    --double-death-weight 0 \
    --survive-weight 0 \
    --fire-prob-weight 0 \
    --fire-margin "$margin" \
    --fire-threshold 0 \
    >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for %s@%s\n" "$(date)" "$code" "$n" "$seed" \
    >> "$log_path"
  return "$code"
}

run_eval 300 970000
code=$?
if [ "$code" -eq 0 ]; then
  run_eval 300 990000
  code=$?
fi
if [ "$code" -eq 0 ]; then
  run_eval 300 973000
  code=$?
fi

printf "%s final exit %s\n" "$(date)" "$code" >> "$log_path"
exit "$code"
