#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26v6_failure_weighted_pipeline.log"
pid_path="training/logs/p26v6_failure_weighted_pipeline.pid"
champion="training/models/p26_amortized_mpc_iter05.pt"
workers="10"
margin="0.16"

mkdir -p training/logs
printf "%s start P26v6 failure-weighted pipeline\n" "$(date)" > "$log_path"
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
  --phase p26v6_iter05_m016_failure_weighted \
  --rounds 240 \
  --workers "$workers" \
  --actor-kind p26 \
  --actor-net "$champion" \
  --epsilon 0.05 \
  --frame-stack 4 \
  --fire-margin "$margin" \
  --fire-target-margin "$margin" \
  --kill-weight 0 \
  --death-weight 0 \
  --double-death-weight 0 \
  --survive-weight 0 \
  --fire-prob-weight 0 \
  --result-win-weight 0.75 \
  --result-loss-weight 3.0 \
  --result-double-death-weight 5.0 \
  --result-draw-weight 1.5 \
  || exit $?

run_step python3 training/p26_amortized_mpc.py train \
  --index 7 \
  --init-net "$champion" \
  --epochs 18 \
  --batch 4096 \
  --width 1024 \
  --lr 0.0001 \
  --frame-stack 4 \
  --aux-weight 0 \
  --fire-weight 0 \
  --rank-weight 0.65 \
  --rank-margin 0.035 \
  --fire-target-margin "$margin" \
  || exit $?

run_step python3 training/p26_amortized_mpc.py train \
  --index 8 \
  --init-net "$champion" \
  --epochs 18 \
  --batch 4096 \
  --width 1024 \
  --lr 0.0001 \
  --frame-stack 4 \
  --aux-weight 0 \
  --fire-weight 0 \
  --rank-weight 0.80 \
  --rank-margin 0.05 \
  --fire-target-margin "$margin" \
  || exit $?

for net in training/models/p26_amortized_mpc_iter07.pt \
           training/models/p26_amortized_mpc_iter08.pt; do
  run_eval "$net" 120 970000 || exit $?
  run_eval "$net" 120 990000 || exit $?
  run_eval "$net" 120 973000 || exit $?
done

printf "%s final exit 0\n" "$(date)" >> "$log_path"
