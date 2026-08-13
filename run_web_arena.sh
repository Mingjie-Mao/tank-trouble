#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir"

python3 -u training/web_server.py --port 8766 &
arena_pid=$!
trap 'kill "$arena_pid" 2>/dev/null || true' EXIT INT TERM

cd "$project_dir/web"
NEXT_PUBLIC_ENABLE_PYTHON_RESEARCH=1 npm run dev
