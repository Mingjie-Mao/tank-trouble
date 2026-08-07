#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p25v3_firegate_official_grade.log"
pid_path="training/logs/p25v3_firegate_official_grade.pid"
net_path="training/models/p25v2_opportunity_best.pt"
margin="0.08"
workers="10"

mkdir -p training/logs
printf "%s start P25v3 firegate official grade\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"
printf "net=%s margin=%s workers=%s\n" "$net_path" "$margin" "$workers" \
  >> "$log_path"

run_eval() {
  local n="$1"
  local seed="$2"
  printf "%s start %s@%s\n" "$(date)" "$n" "$seed" >> "$log_path"
  caffeinate -dims python3 training/opportunity_distill_v3.py eval \
    --net "$net_path" \
    --n "$n" \
    --seed "$seed" \
    --workers "$workers" \
    --margin "$margin" \
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
