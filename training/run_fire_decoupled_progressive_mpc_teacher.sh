#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

method="fire_decoupled_progressive_mpc"
log_path="training/logs/${method}_teacher.log"
pid_path="training/logs/${method}_teacher.pid"
base_net="${FIRE_DECOUPLED_MPC_BASE_NET:-training/models/p26_amortized_mpc_iter05.pt}"
value_net="${FIRE_DECOUPLED_MPC_VALUE_NET:-training/models/p27b_risk_value_iter00.pt}"
workers="${FIRE_DECOUPLED_MPC_WORKERS:-10}"
screen_n="${FIRE_DECOUPLED_MPC_SCREEN_N:-12}"
confirm_n="${FIRE_DECOUPLED_MPC_CONFIRM_N:-40}"

mkdir -p training/logs training/analysis/runs
printf "%s start Fire-Decoupled Progressive MPC teacher\n" "$(date)" \
  > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "base=%s value=%s workers=%s screen=%s confirm=%s\n" \
  "$base_net" "$value_net" "$workers" "$screen_n" "$confirm_n" \
  >> "$log_path"

run_step() {
  printf "%s $*\n" "$(date)" >> "$log_path"
  caffeinate -dims "$@" >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for $*\n" "$(date)" "$code" >> "$log_path"
  return "$code"
}

run_eval() {
  local label="$1"
  local n="$2"
  local seed="$3"
  local run_workers="$workers"
  if (( n < run_workers )); then
    run_workers="$n"
  fi
  run_step python3 training/p26_behavior_observer.py \
    --net "$base_net" \
    --p27b-net "$value_net" \
    --progressive-risk-mpc \
    --n "$n" \
    --seed "$seed" \
    --workers "$run_workers" \
    --fire-margin 0.16 \
    --risk-mpc-horizons 24,48,72,96 \
    --risk-mpc-widths 6,3,2,2 \
    --risk-mpc-final-samples 4 \
    --risk-mpc-commit-frames 24 \
    --risk-mpc-replan-interval 16 \
    --risk-mpc-death-penalty 0.55 \
    --risk-mpc-dd-penalty 1.0 \
    --risk-mpc-kill-bonus 0.04 \
    --risk-mpc-tail-penalty 0.15 \
    --risk-mpc-max-death 0.0 \
    --risk-mpc-max-dd 0.0 \
    --risk-mpc-fire-min-gain 0.015 \
    --risk-mpc-fire-max-extra-death 0.0 \
    --risk-mpc-fire-max-extra-dd 0.0 \
    --risk-mpc-root-fire-min-line 0.35 \
    --risk-mpc-root-fire-max-alignment 0.30 \
    --risk-mpc-root-fire-pressure-radius 0.75 \
    --out "training/analysis/runs/${method}_${label}_${n}_${seed}.jsonl" \
    --summary "training/analysis/runs/${method}_${label}_${n}_${seed}_summary.json"
}

# These five seeds were wins for the 95% prior-search baseline and losses for
# the rejected progressive search. They are the cheapest paired regression set.
target_seeds=(970002 970007 990002 990010 973008)
target_files=()
for seed in "${target_seeds[@]}"; do
  run_eval target 1 "$seed" || exit $?
  target_files+=(
    "training/analysis/runs/${method}_target_1_${seed}_summary.json")
done

target_pass="$(jq -s \
  '(map(.results.win // 0) | add) >= 4 and
   (map(.results.double_death // 0) | add) == 0 and
   (map(.issues.blind_fire // 0) | add) <= 10' \
  "${target_files[@]}")"
printf "%s target_pass=%s\n" "$(date)" "$target_pass" >> "$log_path"
if [[ "$target_pass" != "true" ]]; then
  printf "%s final status=target_rejected exit=0\n" "$(date)" >> "$log_path"
  exit 0
fi

screen_files=()
for seed in 970000 990000 973000; do
  run_eval screen "$screen_n" "$seed" || exit $?
  screen_files+=(
    "training/analysis/runs/${method}_screen_${screen_n}_${seed}_summary.json")
done

screen_pass="$(jq -s \
  '(map(.results.win // 0) | add) >= 34 and
   all(.[]; .win_rate >= 0.8333333333) and
   (map(.results.double_death // 0) | add) == 0 and
   (map(.issues.blind_fire // 0) | add) <= 36' \
  "${screen_files[@]}")"
printf "%s screen_pass=%s\n" "$(date)" "$screen_pass" >> "$log_path"
if [[ "$screen_pass" != "true" ]]; then
  printf "%s final status=screen_rejected exit=0\n" "$(date)" >> "$log_path"
  exit 0
fi

for seed in 970000 990000 973000; do
  run_eval confirm "$confirm_n" "$seed" || exit $?
done

printf "%s final status=confirm_complete exit=0\n" "$(date)" >> "$log_path"
