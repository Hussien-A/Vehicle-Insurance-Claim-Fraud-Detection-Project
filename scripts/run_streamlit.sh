#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -d "$ROOT/.venv" ]; then
  "$ROOT/.venv/bin/streamlit" run streamlit_app/app.py
else
  echo "Create a venv first: python3 -m venv .venv && .venv/bin/pip install -r streamlit_app/requirements.txt"
  exit 1
fi
