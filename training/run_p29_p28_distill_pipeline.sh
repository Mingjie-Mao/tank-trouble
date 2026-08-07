#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p29_p28_distill_pipeline.log"
pid_path="training/logs/p29_p28_distill_pipeline.pid"
base_net="${P29_BASE_NET:-training/models/p26_amortized_mpc_iter05.pt}"
value_net="${P29_VALUE_NET:-training/models/p27b_risk_value_iter00.pt}"
candidate="${P29_OUT_NET:-training/models/p29_p28_distill_iter00.pt}"
workers="${P29_WORKERS:-10}"
margin="0.16"
games="${P29_GAMES:-40}"
gate_n="${P29_GATE_N:-80}"
horizon="${P29_HORIZON:-72}"
samples="${P29_SAMPLES:-2}"

target_a="p29_p28_h${horizon}s${samples}_target_973034"
target_b="p29_p28_h${horizon}s${samples}_target_970017"
target_c="p29_p28_h${horizon}s${samples}_target_970031"
target_d="p29_p28_h${horizon}s${samples}_target_990011"
target_e="p29_p28_h${horizon}s${samples}_target_990024"
target_f="p29_p28_h${horizon}s${samples}_target_990037"
phase_a="p29_p28_h${horizon}s${samples}_p27b_970000"
phase_b="p29_p28_h${horizon}s${samples}_p27b_990000"
phase_c="p29_p28_h${horizon}s${samples}_p27b_973000"
base_a="p26v9_strong_h72s3_970000"
base_b="p26v9_strong_h72s3_990000"
legacy_fail_a="p27c_p27b_fail_h96s3_970000"
legacy_fail_b="p27c_p27b_fail_h96s3_990000"
legacy_visible="p27c_p27b_visible_h72s2_973000"
target_phases="${target_a},${target_b},${target_c},${target_d},${target_e},${target_f}"
include_phases="${base_a},${base_b},${legacy_fail_a},${legacy_fail_b},${legacy_visible},${target_phases},${phase_a},${phase_b},${phase_c}"
phase_weights="${base_a}=0.8,${base_b}=0.8,${legacy_fail_a}=1.0,${legacy_fail_b}=1.0,${legacy_visible}=1.0,${target_a}=4.0,${target_b}=3.2,${target_c}=4.0,${target_d}=3.8,${target_e}=4.0,${target_f}=3.8,${phase_a}=2.5,${phase_b}=2.5,${phase_c}=2.8"
category_weights="teacher_close=0.25,unsafe_movement=2.4,movement_value_gap=1.8,missed_fire_window=2.6,missed_kill_line=2.2,unsafe_fire_death=3.0,double_death_risk=4.0,fire_into_double_death=4.0,waste_or_unsafe_fire=2.2,blind_fire=2.2,stutter_stall=2.4,dead_end_stall=3.0,passive_map_control=2.4,post_kill_fire=1.4,direct_shot_loss=4.5,bounce_shot_loss=3.5,self_shot_loss=4.0,finish_window=3.0,active_pursuit_gap=3.0,long_game=2.2"

mkdir -p training/logs training/analysis/runs
printf "%s start P29 P28-distill pipeline\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "base_net=%s value_net=%s candidate=%s margin=%s workers=%s games=%s gate_n=%s horizon=%s samples=%s\n" \
  "$base_net" "$value_net" "$candidate" "$margin" "$workers" "$games" \
  "$gate_n" "$horizon" "$samples" >> "$log_path"
printf "include_phases=%s\nphase_weights=%s\ncategory_weights=%s\n" \
  "$include_phases" "$phase_weights" "$category_weights" >> "$log_path"

run_step() {
  printf "%s $*\n" "$(date)" >> "$log_path"
  caffeinate -dims "$@" >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for $*\n" "$(date)" "$code" >> "$log_path"
  return "$code"
}

run_collect() {
  local policy="$1"
  local seed="$2"
  local n="$3"
  local phase="$4"
  local max_records="$5"
  run_step python3 training/p29_p28_distill.py \
    --phase "$phase" \
    --rollout-policy "$policy" \
    --base-net "$base_net" \
    --value-net "$value_net" \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin "$margin" \
    --teacher-horizon "$horizon" \
    --teacher-samples "$samples" \
    --search-horizon "$horizon" \
    --search-samples "$samples" \
    --top-k 12 \
    --search-death-penalty 0.35 \
    --search-dd-penalty 0.75 \
    --search-kill-bonus 0.04 \
    --deterministic-search-seeds \
    --max-records-per-round "$max_records" \
    --fire-target-margin "$margin" \
    --out "training/analysis/runs/${phase}.jsonl" \
    --summary "training/analysis/runs/${phase}_summary.json"
}

run_gate() {
  local seed="$1"
  run_step python3 training/p26_behavior_observer.py \
    --net "$base_net" \
    --p27b-net "$candidate" \
    --n "$gate_n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin "$margin" \
    --out "training/analysis/runs/p29_p28_distill_iter00_observer_${gate_n}_${seed}.jsonl" \
    --summary "training/analysis/runs/p29_p28_distill_iter00_observer_${gate_n}_${seed}_summary.json"
}

run_collect p28 973034 1 "$target_a" 80 || exit $?
run_collect p28 970017 1 "$target_b" 80 || exit $?
run_collect p28 970031 1 "$target_c" 80 || exit $?
run_collect p28 990011 1 "$target_d" 80 || exit $?
run_collect p28 990024 1 "$target_e" 80 || exit $?
run_collect p28 990037 1 "$target_f" 80 || exit $?
run_collect p27b 970000 "$games" "$phase_a" 10 || exit $?
run_collect p27b 990000 "$games" "$phase_b" 10 || exit $?
run_collect p27b 973000 "$games" "$phase_c" 12 || exit $?

run_step python3 training/p27_risk_value.py train \
  --include-phases "$include_phases" \
  --phase-weights "$phase_weights" \
  --category-weights "$category_weights" \
  --out "$candidate" \
  --epochs 120 \
  --batch 128 \
  --width 512 \
  --lr 0.00012 \
  --aux-weight 0.14 \
  --rank-weight 0.50 \
  --rank-margin 0.04 \
  || exit $?

run_gate 970000 || exit $?
run_gate 990000 || exit $?
run_gate 973000 || exit $?

printf "%s final exit 0\n" "$(date)" >> "$log_path"
