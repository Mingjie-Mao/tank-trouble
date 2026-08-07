#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p27b_iter00_official_grade.log"
pid_path="training/logs/p27b_iter00_official_grade.pid"
base_net="training/models/p26_amortized_mpc_iter05.pt"
value_net="training/models/p27b_risk_value_iter00.pt"
margin="0.16"
assist_margin="0.08"
assist_weight="0.35"
max_bonus="0.10"
kill_weight="0.04"
death_weight="0.12"
double_death_weight="0.18"
survive_weight="0.02"
risk_threshold="0.55"
fire_delta_margin="0.14"
workers="10"

mkdir -p training/logs
printf "%s start P27b iter00 official grade\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "base_net=%s value_net=%s margin=%s workers=%s\n" \
  "$base_net" "$value_net" "$margin" "$workers" >> "$log_path"
printf "assist_margin=%s assist_weight=%s max_bonus=%s risk_threshold=%s fire_delta_margin=%s\n" \
  "$assist_margin" "$assist_weight" "$max_bonus" "$risk_threshold" \
  "$fire_delta_margin" >> "$log_path"

run_eval() {
  local n="$1"
  local seed="$2"
  printf "%s start %s@%s\n" "$(date)" "$n" "$seed" >> "$log_path"
  caffeinate -dims python3 training/p27_risk_value.py eval \
    --base-net "$base_net" \
    --value-net "$value_net" \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
    --fire-margin "$margin" \
    --assist-margin "$assist_margin" \
    --assist-weight "$assist_weight" \
    --max-bonus "$max_bonus" \
    --kill-weight "$kill_weight" \
    --death-weight "$death_weight" \
    --double-death-weight "$double_death_weight" \
    --survive-weight "$survive_weight" \
    --risk-threshold "$risk_threshold" \
    --fire-delta-margin "$fire_delta_margin" \
    >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for %s@%s\n" "$(date)" "$code" "$n" "$seed" \
    >> "$log_path"
  return "$code"
}

run_eval 1000 970000
code=$?
if [ "$code" -eq 0 ]; then
  run_eval 500 990000
  code=$?
fi

printf "%s final exit %s\n" "$(date)" "$code" >> "$log_path"
exit "$code"
