#!/usr/bin/env bash
set -euo pipefail

# Run Streamlit using the project's .venv Python interpreter.
# This does not require sourcing/activating the venv in the caller shell.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo ".venv Python not found or not executable at $VENV_PY"
  echo "Create the virtualenv and install requirements first:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

# Execute streamlit module with the venv python so that Streamlit runs
# inside the correct environment and picks up the right packages.
exec "$VENV_PY" -m streamlit run "$ROOT_DIR/app.py" "$@"
