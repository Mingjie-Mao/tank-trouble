#!/usr/bin/env bash
set -u

log_path="training/logs/p27_macro_probe.log"
mkdir -p training/logs training/analysis/runs
printf "%s start P27 macro probe\n" "$(date)" > "$log_path"

run_probe() {
  local seed="$1"
  printf "%s probe 40@%s\n" "$(date)" "$seed" | tee -a "$log_path"
  python3 training/p27_macro_probe.py \
    --n 40 \
    --workers 10 \
    --seed "$seed" \
    --net training/models/p26_amortized_mpc_iter05.pt \
    --fire-margin 0.16 \
    --horizon 72 \
    --samples 1 \
    --max-states-per-round 5 \
    --out "training/analysis/runs/p27_macro_probe_40_${seed}.jsonl" \
    --summary "training/analysis/runs/p27_macro_probe_40_${seed}_summary.json" \
    --macro-data "training/analysis/runs/p27_macro_probe_40_${seed}.npz" \
    2>&1 | tee -a "$log_path"
  local code=${PIPESTATUS[0]}
  printf "%s exit %s for probe 40@%s\n" "$(date)" "$code" "$seed" \
    | tee -a "$log_path"
  return "$code"
}

run_probe 970000 || exit $?
run_probe 990000 || exit $?
run_probe 973000 || exit $?

printf "%s final exit 0\n" "$(date)" | tee -a "$log_path"
