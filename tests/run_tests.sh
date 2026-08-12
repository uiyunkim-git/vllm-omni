#!/usr/bin/env bash
# Bootstraps a virtualenv and runs the test suite.
#
#   ./run_tests.sh                 # unit + integration (integration skips w/o docker)
#   ./run_tests.sh unit            # unit layer only  (<10s, no docker needed)
#   ./run_tests.sh integration     # router-in-docker layer
#   ./run_tests.sh unit -k window  # extra args pass straight to pytest
#
# The venv lives at tests/.venv and is reused across runs.
set -euo pipefail
cd "$(dirname "$0")"

# Pick an interpreter that can actually create a venv (ensurepip present).
pick_python() {
    local candidates=("${PYTHON:-}" python3 python3.12 python3.11 \
                      "$HOME/anaconda3/bin/python" "$HOME/miniconda3/bin/python")
    for c in "${candidates[@]}"; do
        [ -n "$c" ] || continue
        command -v "$c" >/dev/null 2>&1 || continue
        "$c" -c "import ensurepip" >/dev/null 2>&1 || continue
        echo "$c"
        return 0
    done
    echo "error: no python with venv support found (apt install python3-venv, or set PYTHON=...)" >&2
    return 1
}

if [ ! -d .venv ]; then
    PYTHON="$(pick_python)"
    echo "[run_tests] creating venv at tests/.venv (using $PYTHON)"
    "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --quiet --disable-pip-version-check -r requirements.txt

exec python -m pytest "$@"
