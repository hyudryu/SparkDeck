#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

frontend_index="frontend/dist/index.html"
frontend_needs_build=0
if [ ! -f "$frontend_index" ]; then
  frontend_needs_build=1
elif find frontend \
     \( -path frontend/node_modules -o -path frontend/dist \) -prune -o \
     -type f \
     \( -name 'package.json' -o -name 'package-lock.json' -o -name 'index.html' -o \
        -name 'vite.config.ts' -o -name 'tsconfig*.json' -o -path 'frontend/src/*' \) \
     -newer "$frontend_index" -print -quit | grep -q .; then
  frontend_needs_build=1
fi

if [ "$frontend_needs_build" -eq 1 ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "SparkDeck's web app needs Node.js and npm for the first build." >&2
    exit 1
  fi
  npm --prefix frontend ci --no-audit --no-fund
  npm --prefix frontend run build
fi

exec python server.py
