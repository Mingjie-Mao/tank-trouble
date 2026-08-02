#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/.."

LOG="training/logs/p27_suppress_medium_eval.log"
mkdir -p training/logs training/analysis/runs

{
  echo "===== P27 suppress-only medium eval started $(date) ====="
  echo "Base: training/models/p26_amortized_mpc_iter05.pt"
  echo "Config: fire_margin=0.16 suppress_blind_fire_line=0.35"

  python3 training/p26_behavior_observer.py \
    --n 300 --workers 10 --seed 970000 \
    --net training/models/p26_amortized_mpc_iter05.pt \
    --fire-margin 0.16 \
    --suppress-blind-fire-line 0.35 \
    --out training/analysis/runs/p27_suppress_medium_observer_300_970000.jsonl \
    --summary training/analysis/runs/p27_suppress_medium_observer_300_970000_summary.json

  python3 training/p26_behavior_observer.py \
    --n 300 --workers 10 --seed 990000 \
    --net training/models/p26_amortized_mpc_iter05.pt \
    --fire-margin 0.16 \
    --suppress-blind-fire-line 0.35 \
    --out training/analysis/runs/p27_suppress_medium_observer_300_990000.jsonl \
    --summary training/analysis/runs/p27_suppress_medium_observer_300_990000_summary.json

  python3 - <<'PY'
import json
from pathlib import Path

paths = [
    Path("training/analysis/runs/p27_suppress_medium_observer_300_970000_summary.json"),
    Path("training/analysis/runs/p27_suppress_medium_observer_300_990000_summary.json"),
]
print("===== P27 suppress-only medium eval final summary =====", flush=True)
for path in paths:
    data = json.loads(path.read_text())
    print(
        f"{data['n']}@{data['seed']} win={data['win_rate']:.1%} "
        f"loss={data['loss_rate']:.1%} dd={data['double_death_rate']:.1%} "
        f"shots/game={data['shots_per_game']:.2f} hit={data['hit_rate']:.1%} "
        f"issues={data['issues']}",
        flush=True,
    )
print("Recommended gate: promote only if both seeds stay >= champion medium/official band and blind_fire does not return.", flush=True)
PY

  echo "===== P27 suppress-only medium eval finished $(date) ====="
} 2>&1 | tee "$LOG"
