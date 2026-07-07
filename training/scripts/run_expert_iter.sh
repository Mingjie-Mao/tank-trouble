#!/bin/bash
# 专家迭代后台启动器 (macOS / Linux 通用)
# 用法: bash training/scripts/run_expert_iter.sh [连跑轮数] [每轮局数] [进程数]
#   例: bash training/scripts/run_expert_iter.sh 3 3000 60    # 租的 64 核机
#       bash training/scripts/run_expert_iter.sh 1 1500 8     # 本机过夜
set -e
cd "$(dirname "$0")/../.."
mkdir -p training/logs

COUNT=${1:-1}
ROUNDS=${2:-3000}
WORKERS=${3:-$(($(getconf _NPROCESSORS_ONLN) - 2))}
LOG="training/logs/run_expert_iter_$(date +%m%d_%H%M).log"

# macOS 防休眠; Linux 无 caffeinate 则直接跑
if command -v caffeinate >/dev/null 2>&1; then KEEP="caffeinate -i"; else KEEP=""; fi

nohup $KEEP python3 training/expert_iter.py run \
    --count "$COUNT" --rounds "$ROUNDS" --workers "$WORKERS" \
    > "$LOG" 2>&1 &

echo "已后台启动 (PID $!): $COUNT 轮 x $ROUNDS 局, $WORKERS 进程"
echo "看进度:  tail -f $LOG"
echo "看台账:  python3 training/expert_iter.py history"
