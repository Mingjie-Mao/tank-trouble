#!/bin/zsh
set -e

ROOT="${0:A:h}"
cd "$ROOT/web"

echo "Tank Trouble arena: http://127.0.0.1:3000"
echo "The engine, Laika and our AI run in the browser (Rust compiled to WebAssembly)."
exec npm run dev
