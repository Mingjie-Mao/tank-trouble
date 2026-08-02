#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p27c_hardcase_pipeline.log"
pid_path="training/logs/p27c_hardcase_pipeline.pid"
base_net="${P27C_BASE_NET:-training/models/p26_amortized_mpc_iter05.pt}"
p27b_net="${P27C_VALUE_NET:-training/models/p27b_risk_value_iter00.pt}"
candidate="${P27C_OUT_NET:-training/models/p27c_risk_value_iter01.pt}"
workers="${P27C_WORKERS:-10}"
margin="0.16"
fail_games="${P27C_FAIL_GAMES:-180}"
visible_games="${P27C_VISIBLE_GAMES:-120}"
fail_horizon="${P27C_FAIL_HORIZON:-96}"
fail_samples="${P27C_FAIL_SAMPLES:-3}"
visible_horizon="${P27C_VISIBLE_HORIZON:-72}"
visible_samples="${P27C_VISIBLE_SAMPLES:-2}"
gate_n="${P27C_GATE_N:-80}"

fail_a="p27c_p27b_fail_h${fail_horizon}s${fail_samples}_970000"
fail_b="p27c_p27b_fail_h${fail_horizon}s${fail_samples}_990000"
visible_c="p27c_p27b_visible_h${visible_horizon}s${visible_samples}_973000"
base_a="p26v9_strong_h72s3_970000"
base_b="p26v9_strong_h72s3_990000"
include_phases="${base_a},${base_b},${fail_a},${fail_b},${visible_c}"
phase_weights="${base_a}=1.0,${base_b}=1.0,${fail_a}=2.4,${fail_b}=2.4,${visible_c}=1.8"
category_weights="teacher_close=0.25,unsafe_movement=2.8,movement_value_gap=2.0,missed_fire_window=2.4,missed_kill_line=2.2,unsafe_fire_death=3.2,double_death_risk=4.2,fire_into_double_death=4.2,waste_or_unsafe_fire=2.2,blind_fire=2.0,stutter_stall=2.0,dead_end_stall=2.4,passive_map_control=1.8,post_kill_fire=2.0"

mkdir -p training/logs training/analysis/runs
printf "%s start P27c hardcase pipeline\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "base_net=%s p27b_net=%s candidate=%s margin=%s workers=%s\n" \
  "$base_net" "$p27b_net" "$candidate" "$margin" "$workers" >> "$log_path"
printf "fail_games=%s fail_horizon=%s fail_samples=%s visible_games=%s visible_horizon=%s visible_samples=%s gate_n=%s\n" \
  "$fail_games" "$fail_horizon" "$fail_samples" "$visible_games" \
  "$visible_horizon" "$visible_samples" "$gate_n" >> "$log_path"

run_step() {
  printf "%s $*\n" "$(date)" >> "$log_path"
  caffeinate -dims "$@" >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for $*\n" "$(date)" "$code" >> "$log_path"
  return "$code"
}

run_failure_attr() {
  local seed="$1"
  local phase="$2"
  local out="$3"
  run_step python3 training/p26_failure_attribution.py \
    --net "$base_net" \
    --p27b-net "$p27b_net" \
    --n "$fail_games" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin "$margin" \
    --window-frames 140 \
    --stride 4 \
    --score-horizon "$fail_horizon" \
    --score-samples "$fail_samples" \
    --fire-target-margin "$margin" \
    --hard-phase "$phase" \
    --max-cases 80 \
    --out "$out"
}

run_visible_attr() {
  local seed="$1"
  local phase="$2"
  run_step python3 training/p26_behavior_observer.py \
    --net "$base_net" \
    --p27b-net "$p27b_net" \
    --n "$visible_games" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin "$margin" \
    --hard-phase "$phase" \
    --hard-max-per-round 8 \
    --hard-min-gap-frames 24 \
    --score-horizon "$visible_horizon" \
    --score-samples "$visible_samples" \
    --fire-target-margin "$margin" \
    --out "training/analysis/runs/${phase}.jsonl" \
    --summary "training/analysis/runs/${phase}_summary.json"
}

run_observer_gate() {
  local seed="$1"
  run_step python3 training/p26_behavior_observer.py \
    --net "$base_net" \
    --p27b-net "$candidate" \
    --n "$gate_n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin "$margin" \
    --out "training/analysis/runs/p27c_risk_value_iter01_observer_${gate_n}_${seed}.jsonl" \
    --summary "training/analysis/runs/p27c_risk_value_iter01_observer_${gate_n}_${seed}_summary.json"
}

run_failure_attr 970000 "$fail_a" \
  "training/analysis/runs/p27c_p27b_fail_h${fail_horizon}s${fail_samples}_${fail_games}_970000.json" \
  || exit $?
run_failure_attr 990000 "$fail_b" \
  "training/analysis/runs/p27c_p27b_fail_h${fail_horizon}s${fail_samples}_${fail_games}_990000.json" \
  || exit $?
run_visible_attr 973000 "$visible_c" || exit $?

run_step python3 training/p27_risk_value.py train \
  --include-phases "$include_phases" \
  --phase-weights "$phase_weights" \
  --category-weights "$category_weights" \
  --out "$candidate" \
  --epochs 100 \
  --batch 128 \
  --width 512 \
  --lr 0.00015 \
  --aux-weight 0.14 \
  --rank-weight 0.45 \
  --rank-margin 0.04 \
  || exit $?

run_observer_gate 970000 || exit $?
run_observer_gate 990000 || exit $?
run_observer_gate 973000 || exit $?

printf "%s final exit 0\n" "$(date)" >> "$log_path"
