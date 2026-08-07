#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/progressive_risk_mpc_teacher.log"
pid_path="training/logs/progressive_risk_mpc_teacher.pid"
p30_pid_path="training/logs/p30_consensus_pipeline.pid"
p30_log_path="training/logs/p30_consensus_pipeline.log"
base_net="${PROGRESSIVE_RISK_MPC_BASE_NET:-training/models/p26_amortized_mpc_iter05.pt}"
value_net="${PROGRESSIVE_RISK_MPC_VALUE_NET:-training/models/p27b_risk_value_iter00.pt}"
workers="${PROGRESSIVE_RISK_MPC_WORKERS:-10}"
screen_n="${PROGRESSIVE_RISK_MPC_SCREEN_N:-12}"
confirm_n="${PROGRESSIVE_RISK_MPC_CONFIRM_N:-40}"

mkdir -p training/logs training/analysis/runs
printf "%s start Progressive Risk-Constrained MPC teacher waiter\n" \
  "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "base=%s value=%s workers=%s screen=%s confirm=%s\n" \
  "$base_net" "$value_net" "$workers" "$screen_n" "$confirm_n" \
  >> "$log_path"

if [[ -f "$p30_pid_path" ]]; then
  p30_pid="$(cat "$p30_pid_path")"
  while kill -0 "$p30_pid" 2>/dev/null; do
    printf "%s waiting for P30 pid=%s\n" "$(date)" "$p30_pid" \
      >> "$log_path"
    sleep 60
  done
  if ! grep -q "final exit 0" "$p30_log_path"; then
    printf "%s final status=p30_failed_or_incomplete exit=1\n" "$(date)" \
      >> "$log_path"
    exit 1
  fi
fi

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
  run_step python3 training/p26_behavior_observer.py \
    --net "$base_net" \
    --p27b-net "$value_net" \
    --progressive-risk-mpc \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
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
    --out "training/analysis/runs/progressive_risk_mpc_${label}_${n}_${seed}.jsonl" \
    --summary "training/analysis/runs/progressive_risk_mpc_${label}_${n}_${seed}_summary.json"
}

for seed in 970000 990000 973000; do
  run_eval screen "$screen_n" "$seed" || exit $?
done

screen_files=(
  "training/analysis/runs/progressive_risk_mpc_screen_${screen_n}_970000_summary.json"
  "training/analysis/runs/progressive_risk_mpc_screen_${screen_n}_990000_summary.json"
  "training/analysis/runs/progressive_risk_mpc_screen_${screen_n}_973000_summary.json"
)
screen_pass="$(jq -s \
  '((map(.win_rate) | add / length) >= 0.90) and all(.[]; .win_rate >= 0.80)' \
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
