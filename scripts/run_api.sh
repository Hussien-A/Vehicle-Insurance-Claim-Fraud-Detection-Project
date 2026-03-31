#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/api"
if [ -d "$ROOT/.venv" ]; then
  "$ROOT/.venv/bin/uvicorn" main:app --reload --host 0.0.0.0 --port 8000
else
  echo "Create a venv first: python3 -m venv .venv && .venv/bin/pip install -r api/requirements.txt"
  exit 1
fi
