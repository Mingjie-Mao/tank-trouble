#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26_score_rank_pipeline.log"
pid_path="training/logs/p26_score_rank_pipeline.pid"

mkdir -p training/logs
printf "%s start P26 score+rank augmented pipeline\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"

run_step() {
  printf "%s $*\n" "$(date)" >> "$log_path"
  caffeinate -dims "$@" >> "$log_path" 2>&1
  local code=$?
  printf "%s exit %s for $*\n" "$(date)" "$code" >> "$log_path"
  return "$code"
}

run_step python3 training/p26_amortized_mpc.py collect \
  --phase score_rank_teacher \
  --rounds 64 \
  --workers 10 \
  --actor-kind teacher \
  --epsilon 0.03 \
  --frame-stack 4 \
  || exit $?

run_step python3 training/p26_amortized_mpc.py collect \
  --phase score_rank_p25v3 \
  --rounds 128 \
  --workers 10 \
  --actor-kind p25v3 \
  --epsilon 0.03 \
  --frame-stack 4 \
  || exit $?

run_step python3 training/p26_amortized_mpc.py collect \
  --phase score_rank_p22 \
  --rounds 64 \
  --workers 10 \
  --actor-kind p22 \
  --epsilon 0.03 \
  --frame-stack 4 \
  || exit $?

run_step python3 training/p26_amortized_mpc.py train \
  --index 4 \
  --epochs 24 \
  --batch 4096 \
  --width 1024 \
  --frame-stack 4 \
  --aux-weight 0 \
  --fire-weight 0 \
  --rank-weight 0.5 \
  --rank-margin 0.03 \
  || exit $?

run_step python3 training/p26_amortized_mpc.py eval \
  --net training/models/p26_amortized_mpc_iter04.pt \
  --n 80 \
  --seed 9700000 \
  --workers 10 \
  --kill-weight 0 \
  --death-weight 0 \
  --double-death-weight 0 \
  --survive-weight 0 \
  --fire-prob-weight 0 \
  --fire-margin 0 \
  --fire-threshold 0 \
  || exit $?

run_step python3 training/p26_amortized_mpc.py eval \
  --net training/models/p26_amortized_mpc_iter04.pt \
  --n 120 \
  --seed 973000 \
  --workers 10 \
  --kill-weight 0 \
  --death-weight 0 \
  --double-death-weight 0 \
  --survive-weight 0 \
  --fire-prob-weight 0 \
  --fire-margin 0 \
  --fire-threshold 0

code=$?
printf "%s final exit %s\n" "$(date)" "$code" >> "$log_path"
exit "$code"
