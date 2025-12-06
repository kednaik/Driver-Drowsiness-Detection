#!/usr/bin/env bash
set -euo pipefail

# Run the project's pytest suite using the project's virtual environment (if present).
# Usage: ./scripts/run_tests.sh [pytest-args]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Ensure project root is on PYTHONPATH so tests can import package modules
export PYTHONPATH="$ROOT_DIR"

# Run pytest with any provided arguments (defaults to -q)
if [ "$#" -eq 0 ]; then
  pytest -q
else
  pytest "$@"
fi
