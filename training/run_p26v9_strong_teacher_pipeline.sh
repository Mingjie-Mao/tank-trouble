#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26v9_strong_teacher_pipeline.log"
pid_path="training/logs/p26v9_strong_teacher_pipeline.pid"
base_net="${1:-training/models/p26_amortized_mpc_iter05.pt}"
candidate="training/models/p26_amortized_mpc_iter10.pt"
workers="${P26_WORKERS:-10}"
margin="0.16"
games="${P26_STRONG_GAMES:-160}"
horizon="${P26_STRONG_HORIZON:-72}"
samples="${P26_STRONG_SAMPLES:-3}"
include_base="teacher,bootstrap_p25v3,bootstrap_p22,score_rank_teacher,score_rank_p25v3,score_rank_p22,p26v5_onpolicy_iter04_m016,p26v7_fail_iter05_m016_970000,p26v7_fail_iter05_m016_990000"
strong_a="p26v9_strong_h${horizon}s${samples}_970000"
strong_b="p26v9_strong_h${horizon}s${samples}_990000"
include_phases="${include_base},${strong_a},${strong_b}"
phase_weights="p26v7_fail_iter05_m016_970000=1.0,p26v7_fail_iter05_m016_990000=1.0,${strong_a}=1.5,${strong_b}=1.5"
category_weights="teacher_close=0.5,unsafe_movement=2.5,movement_value_gap=2.0,missed_fire_window=1.8,unsafe_fire_death=2.0,double_death_risk=2.2,fire_into_double_death=2.5,waste_or_unsafe_fire=1.4,missed_kill_line=1.8"

mkdir -p training/logs training/analysis/runs
printf "%s start P26v9 strong-teacher pipeline\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "base_net=%s candidate=%s margin=%s workers=%s games=%s horizon=%s samples=%s\n" \
  "$base_net" "$candidate" "$margin" "$workers" "$games" "$horizon" \
  "$samples" >> "$log_path"

run_step() {
  printf "%s $*\n" "$(date)" >> "$log_path"
  caffeinate -dims "$@" >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for $*\n" "$(date)" "$code" >> "$log_path"
  return "$code"
}

run_attr() {
  local seed="$1"
  local phase="$2"
  local out="$3"
  run_step python3 training/p26_failure_attribution.py \
    --net "$base_net" \
    --n "$games" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin "$margin" \
    --window-frames 120 \
    --stride 4 \
    --score-horizon "$horizon" \
    --score-samples "$samples" \
    --fire-target-margin "$margin" \
    --hard-phase "$phase" \
    --max-cases 50 \
    --out "$out"
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

run_attr 970000 "$strong_a" \
  "training/analysis/runs/p26v9_strong_h${horizon}s${samples}_${games}_970000.json" \
  || exit $?
run_attr 990000 "$strong_b" \
  "training/analysis/runs/p26v9_strong_h${horizon}s${samples}_${games}_990000.json" \
  || exit $?

run_step python3 training/p26_amortized_mpc.py train \
  --index 10 \
  --init-net "$base_net" \
  --epochs 3 \
  --batch 4096 \
  --width 1024 \
  --lr 0.00002 \
  --frame-stack 4 \
  --aux-weight 0.05 \
  --fire-weight 0.03 \
  --rank-weight 0.25 \
  --rank-margin 0.035 \
  --fire-target-margin "$margin" \
  --fire-pos-weight-cap 35 \
  --include-phases "$include_phases" \
  --exclude-phase-prefixes "smoke,p26v6_" \
  --phase-weights "$phase_weights" \
  --category-weights "$category_weights" \
  || exit $?

run_eval "$candidate" 120 970000 || exit $?
run_eval "$candidate" 120 990000 || exit $?
run_eval "$candidate" 120 973000 || exit $?

printf "%s final exit 0\n" "$(date)" >> "$log_path"
