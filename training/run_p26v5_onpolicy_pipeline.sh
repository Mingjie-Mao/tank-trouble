#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26v5_onpolicy_pipeline.log"
pid_path="training/logs/p26v5_onpolicy_pipeline.pid"
champion="training/models/p26_amortized_mpc_iter04.pt"
workers="10"
margin="0.16"

mkdir -p training/logs
printf "%s start P26v5 on-policy pipeline\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "champion=%s margin=%s workers=%s\n" "$champion" "$margin" "$workers" \
  >> "$log_path"

run_step() {
  printf "%s $*\n" "$(date)" >> "$log_path"
  caffeinate -dims "$@" >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for $*\n" "$(date)" "$code" >> "$log_path"
  return "$code"
}

run_eval() {
  local net="$1"
  local n="$2"
  local seed="$3"
  run_step python3 training/p26_amortized_mpc.py eval \
    --net "$net" \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
    --kill-weight 0 \
    --death-weight 0 \
    --double-death-weight 0 \
    --survive-weight 0 \
    --fire-prob-weight 0 \
    --fire-margin "$margin" \
    --fire-threshold 0
}

run_step python3 training/p26_amortized_mpc.py collect \
  --phase p26v5_onpolicy_iter04_m016 \
  --rounds 160 \
  --workers "$workers" \
  --actor-kind p26 \
  --actor-net "$champion" \
  --epsilon 0.04 \
  --frame-stack 4 \
  --fire-margin "$margin" \
  --fire-target-margin "$margin" \
  --kill-weight 0 \
  --death-weight 0 \
  --double-death-weight 0 \
  --survive-weight 0 \
  --fire-prob-weight 0 \
  || exit $?

run_step python3 training/p26_amortized_mpc.py train \
  --index 5 \
  --epochs 28 \
  --batch 4096 \
  --width 1024 \
  --frame-stack 4 \
  --aux-weight 0 \
  --fire-weight 0 \
  --rank-weight 0.55 \
  --rank-margin 0.03 \
  --fire-target-margin "$margin" \
  || exit $?

run_step python3 training/p26_amortized_mpc.py train \
  --index 6 \
  --epochs 28 \
  --batch 4096 \
  --width 1024 \
  --frame-stack 4 \
  --aux-weight 0 \
  --fire-weight 0.05 \
  --rank-weight 0.55 \
  --rank-margin 0.03 \
  --fire-target-margin "$margin" \
  --fire-pos-weight-cap 35 \
  || exit $?

run_eval training/models/p26_amortized_mpc_iter05.pt 80 9700000 \
  || exit $?
run_eval training/models/p26_amortized_mpc_iter05.pt 120 973000 \
  || exit $?
run_eval training/models/p26_amortized_mpc_iter06.pt 80 9700000 \
  || exit $?
run_eval training/models/p26_amortized_mpc_iter06.pt 120 973000

code=$?
printf "%s final exit %s\n" "$(date)" "$code" >> "$log_path"
exit "$code"
