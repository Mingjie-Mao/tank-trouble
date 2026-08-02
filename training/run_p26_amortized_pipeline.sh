#!/bin/zsh
set -u

root_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root_dir" || exit 1

log_path="training/logs/p26_amortized_pipeline.log"
pid_path="training/logs/p26_amortized_pipeline.pid"

mkdir -p training/logs
printf "%s start P26 amortized MPC pipeline runner\n" "$(date)" > "$log_path"
printf "%s\n" "$$" > "$pid_path"

while screen -ls | grep -q "p25v3_firegate_official"; do
  printf "%s waiting for P25v3 official grade to finish\n" "$(date)" \
    >> "$log_path"
  sleep 60
done

printf "%s start P26A first real run\n" "$(date)" >> "$log_path"
caffeinate -dims python3 training/p26_amortized_mpc.py pipeline \
  --teacher-rounds 32 \
  --p25v3-rounds 64 \
  --p22-rounds 32 \
  --workers 10 \
  --epochs 8 \
  --batch 4096 \
  --width 768 \
  --frame-stack 4 \
  --epsilon 0.03 \
  --gate-n 80 \
  --gate-seed 9700000 \
  --eval-n 120 \
  --eval-seed 973000 \
  >> "$log_path" 2>&1

code=$?
printf "%s final exit %s\n" "$(date)" "$code" >> "$log_path"
exit "$code"
