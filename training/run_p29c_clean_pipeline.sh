#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p29c_clean_pipeline.log"
pid_path="training/logs/p29c_clean_pipeline.pid"
base_net="${P29C_BASE_NET:-training/models/p26_amortized_mpc_iter05.pt}"
rollout_value_net="${P29C_ROLLOUT_VALUE_NET:-training/models/p27b_risk_value_iter00.pt}"
candidate="${P29C_OUT_NET:-training/models/p29c_clean_distill_iter00.pt}"
workers="${P29C_WORKERS:-10}"
games="${P29C_GAMES:-32}"
gate_n="${P29C_GATE_N:-80}"
objective="p29c_p28_ensemble_v1"

target_seeds=(973034 970017 970031 990011 990024 990037)
target_phases=()
for seed in "${target_seeds[@]}"; do
  target_phases+=("p29c_clean_target_${seed}")
done
broad_phases=(
  p29c_clean_p27b_970000
  p29c_clean_p27b_990000
  p29c_clean_p27b_973000
)
include_phases="${(j:,:)target_phases},${(j:,:)broad_phases}"

mkdir -p training/logs training/analysis/runs
printf "%s start P29c clean P28-ensemble pipeline\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "base=%s rollout_value=%s candidate=%s workers=%s games=%s gate=%s objective=%s\n" \
  "$base_net" "$rollout_value_net" "$candidate" "$workers" "$games" \
  "$gate_n" "$objective" >> "$log_path"
printf "phases=%s\n" "$include_phases" >> "$log_path"

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
    --value-net "$rollout_value_net" \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin 0.16 \
    --teacher-horizon 72 \
    --teacher-samples 2 \
    --teacher-secondary-horizon 48 \
    --teacher-secondary-samples 1 \
    --teacher-review-horizon 96 \
    --teacher-review-samples 2 \
    --teacher-review-gap 0.0 \
    --advantage-target \
    --advantage-min-scale 0.05 \
    --advantage-clip 6.0 \
    --include-context \
    --objective-version "$objective" \
    --search-horizon 72 \
    --search-samples 2 \
    --top-k 12 \
    --search-death-penalty 0.35 \
    --search-dd-penalty 0.75 \
    --search-kill-bonus 0.04 \
    --deterministic-search-seeds \
    --max-records-per-round "$max_records" \
    --min-gap-frames 10 \
    --out "training/analysis/runs/${phase}.jsonl" \
    --summary "training/analysis/runs/${phase}_summary.json"
}

for (( index=1; index<=${#target_seeds[@]}; index++ )); do
  run_collect p28 "${target_seeds[$index]}" 1 \
    "${target_phases[$index]}" 80 || exit $?
done

run_collect p27b 970000 "$games" "${broad_phases[1]}" 10 || exit $?
run_collect p27b 990000 "$games" "${broad_phases[2]}" 10 || exit $?
run_collect p27b 973000 "$games" "${broad_phases[3]}" 12 || exit $?

run_step python3 training/p29c_clean_train.py \
  --include-phases "$include_phases" \
  --objective-version "$objective" \
  --out "$candidate" \
  --report training/analysis/runs/p29c_clean_train_report.json \
  --epochs 120 \
  --min-epochs 25 \
  --patience 16 \
  --batch 128 \
  --width 512 \
  --lr 0.00012 \
  --val-fraction 0.22 \
  --rank-weight 0.40 \
  --policy-weight 0.28 \
  || exit $?

run_gate() {
  local seed="$1"
  run_step python3 training/p26_behavior_observer.py \
    --net "$base_net" \
    --p27b-net "$candidate" \
    --n "$gate_n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin 0.16 \
    --out "training/analysis/runs/p29c_clean_iter00_observer_${gate_n}_${seed}.jsonl" \
    --summary "training/analysis/runs/p29c_clean_iter00_observer_${gate_n}_${seed}_summary.json"
}

run_gate 970000 || exit $?
run_gate 990000 || exit $?
run_gate 973000 || exit $?

printf "%s final exit 0\n" "$(date)" >> "$log_path"
