#!/usr/bin/env bash
# Wrapper that sets up the environment before calling evaluate_task.py
# This ensures PYTHONPATH includes AlgoTune regardless of how evolve-sdk calls it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/algotune:${PYTHONPATH:-}"
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/evaluate_task.py" "$@"
