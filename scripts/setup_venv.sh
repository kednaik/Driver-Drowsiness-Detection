#!/usr/bin/env bash
# Create and activate a Python virtual environment, then install requirements.
# Usage: bash scripts/setup_venv.sh

set -euo pipefail

VENV_DIR=".venv"
PYTHON_CMD="python3"

# Optional: install system build dependencies (Debian/Ubuntu).
# Enable by passing `--system-deps` as the first argument or setting
# `INSTALL_SYSTEM_DEPS=true` in the environment.
if [ "${1:-}" = "--system-deps" ] || [ "${INSTALL_SYSTEM_DEPS:-}" = "true" ]; then
  if command -v apt >/dev/null 2>&1; then
    echo "Installing system build dependencies (requires sudo)..."
    sudo apt update
    sudo apt install -y build-essential cmake python3-dev libgtk-3-dev libboost-all-dev pkg-config
  else
    echo "apt not found; skipping system dependency installation."
  fi
fi

# Create venv if not present
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  $PYTHON_CMD -m venv "$VENV_DIR"
else
  echo "Virtual environment $VENV_DIR already exists."
fi

# Activate and install
echo "Activating virtual environment and installing dependencies..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found in project root. Please add it before running this script."
fi

echo "Virtual environment is ready. Activate it with: source $VENV_DIR/bin/activate"
