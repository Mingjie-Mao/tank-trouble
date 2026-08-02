#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p30_consensus_pipeline.log"
pid_path="training/logs/p30_consensus_pipeline.pid"
base_net="${P30_BASE_NET:-training/models/p26_amortized_mpc_iter05.pt}"
value_net="${P30_VALUE_NET:-training/models/p27b_risk_value_iter00.pt}"
candidate="${P30_OUT_NET:-training/models/p30_consensus_correction_iter00.pt}"
workers="${P30_WORKERS:-10}"
games="${P30_GAMES:-64}"
gate_n="${P30_GATE_N:-80}"
objective="p30_consensus_correction_v1"

failure_seeds=(
  970014 970020 970054 970057 970068 970089
  990003 990029 990039 990047 990086 990099
  973002 973010 973034
)
target_phases=()
for seed in "${failure_seeds[@]}"; do
  target_phases+=("p30_consensus_target_${seed}")
done
broad_phases=(
  p30_consensus_p27b_970000
  p30_consensus_p27b_990000
  p30_consensus_p27b_973000
)
include_phases="${(j:,:)target_phases},${(j:,:)broad_phases}"

mkdir -p training/logs training/analysis/runs
printf "%s start P30 consensus-correction pipeline\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "base=%s value=%s candidate=%s workers=%s games=%s gate=%s\n" \
  "$base_net" "$value_net" "$candidate" "$workers" "$games" \
  "$gate_n" >> "$log_path"
printf "phases=%s\n" "$include_phases" >> "$log_path"

run_step() {
  printf "%s $*\n" "$(date)" >> "$log_path"
  caffeinate -dims "$@" >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for $*\n" "$(date)" "$code" >> "$log_path"
  return "$code"
}

run_collect() {
  local seed="$1"
  local n="$2"
  local phase="$3"
  local max_records="$4"
  local terminal_stride="$5"
  run_step python3 training/p29_p28_distill.py \
    --phase "$phase" \
    --rollout-policy p27b \
    --base-net "$base_net" \
    --value-net "$value_net" \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin 0.16 \
    --teacher-label-mode consensus \
    --teacher-horizon 48 \
    --teacher-samples 1 \
    --teacher-secondary-horizon 72 \
    --teacher-secondary-samples 1 \
    --teacher-consensus-gap 0.01 \
    --teacher-review-horizon 96 \
    --teacher-review-samples 1 \
    --teacher-review-votes 4 \
    --teacher-review-min-votes 3 \
    --teacher-review-action-gap 0.005 \
    --override-min-gain 0.02 \
    --include-context \
    --objective-version "$objective" \
    --search-death-penalty 0.35 \
    --search-dd-penalty 0.75 \
    --search-kill-bonus 0.04 \
    --max-records-per-round "$max_records" \
    --min-gap-frames 12 \
    --background-stride 75 \
    --terminal-window 160 \
    --terminal-stride "$terminal_stride" \
    --out "training/analysis/runs/${phase}.jsonl" \
    --summary "training/analysis/runs/${phase}_summary.json"
}

for (( index=1; index<=${#failure_seeds[@]}; index++ )); do
  run_collect "${failure_seeds[$index]}" 1 \
    "${target_phases[$index]}" 28 6 || exit $?
done

run_collect 970000 "$games" "${broad_phases[1]}" 20 8 || exit $?
run_collect 990000 "$games" "${broad_phases[2]}" 20 8 || exit $?
run_collect 973000 "$games" "${broad_phases[3]}" 20 8 || exit $?

run_step python3 training/p30_consensus_correction.py train \
  --include-phases "$include_phases" \
  --objective-version "$objective" \
  --out "$candidate" \
  --report training/analysis/runs/p30_consensus_train_report.json \
  --epochs 120 \
  --min-epochs 25 \
  --patience 16 \
  --batch 128 \
  --width 512 \
  --lr 0.00012 \
  --val-fraction 0.22 \
  || exit $?

run_gate() {
  local seed="$1"
  run_step python3 training/p26_behavior_observer.py \
    --net "$base_net" \
    --p27b-net "$value_net" \
    --p30-net "$candidate" \
    --n "$gate_n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin 0.16 \
    --p30-override-threshold 0.72 \
    --p30-background-override-threshold 0.84 \
    --p30-min-predicted-gain 0.03 \
    --p30-max-override-death 0.55 \
    --p30-max-override-dd 0.50 \
    --out "training/analysis/runs/p30_iter00_observer_${gate_n}_${seed}.jsonl" \
    --summary "training/analysis/runs/p30_iter00_observer_${gate_n}_${seed}_summary.json"
}

run_gate 970000 || exit $?
run_gate 990000 || exit $?
run_gate 973000 || exit $?

printf "%s final exit 0\n" "$(date)" >> "$log_path"
