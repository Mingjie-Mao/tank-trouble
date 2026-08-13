#!/bin/zsh
set -e

ROOT="${0:A:h}"
cd "$ROOT/web"

echo "Tank Trouble Browser JS arena: http://127.0.0.1:3000"
echo "Physics, Laika and Tank Trouble Tactical search run in the browser."
exec npm run dev
