#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26v8_hardcase_pipeline.log"
pid_path="training/logs/p26v8_hardcase_pipeline.pid"
champion="training/models/p26_amortized_mpc_iter05.pt"
candidate="training/models/p26_amortized_mpc_iter09.pt"
workers="10"
margin="0.16"
include_phases="teacher,bootstrap_p25v3,bootstrap_p22,score_rank_teacher,score_rank_p25v3,score_rank_p22,p26v5_onpolicy_iter04_m016,p26v7_fail_iter05_m016_970000,p26v7_fail_iter05_m016_990000"
phase_weights="p26v7_fail_iter05_m016_970000=2.0,p26v7_fail_iter05_m016_990000=2.0"

mkdir -p training/logs
printf "%s start P26v8 hard-case pipeline\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "champion=%s candidate=%s margin=%s workers=%s\n" \
  "$champion" "$candidate" "$margin" "$workers" >> "$log_path"
printf "include_phases=%s\nphase_weights=%s\n" \
  "$include_phases" "$phase_weights" >> "$log_path"

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

run_step python3 training/p26_amortized_mpc.py train \
  --index 9 \
  --init-net "$champion" \
  --epochs 4 \
  --batch 4096 \
  --width 1024 \
  --lr 0.00005 \
  --frame-stack 4 \
  --aux-weight 0.35 \
  --fire-weight 0.30 \
  --rank-weight 0.30 \
  --rank-margin 0.03 \
  --fire-target-margin "$margin" \
  --fire-pos-weight-cap 35 \
  --include-phases "$include_phases" \
  --exclude-phase-prefixes "smoke,p26v6_" \
  --phase-weights "$phase_weights" \
  || exit $?

run_eval "$candidate" 120 970000 || exit $?
run_eval "$candidate" 120 990000 || exit $?
run_eval "$candidate" 120 973000 || exit $?

printf "%s final exit 0\n" "$(date)" >> "$log_path"
