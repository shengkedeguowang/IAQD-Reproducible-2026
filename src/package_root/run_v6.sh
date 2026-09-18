#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_EXE=${PYTHON_EXE:-python3}
PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON_EXE" "$SCRIPT_DIR/run_all_v6.py" "$@"

