#!/usr/bin/env bash
# Bootstraps a virtualenv and runs the test suite.
#
#   ./run_tests.sh                  # unit layer: offline, no docker, no network
#   ./run_tests.sh unit -k namespace   # extra args pass straight to pytest
#   ./run_tests.sh integration      # central vs. mock worker agent (loopback only)
#   ./run_tests.sh all              # unit + integration
#   RUN_LIVE=1 ./run_tests.sh e2e   # opt-in smoke against a RUNNING stack
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

# Expand the convenience target "all"; everything else goes to pytest verbatim.
args=("$@")
if [ "${1:-}" = "all" ]; then
    shift
    args=(unit integration "$@")
fi

exec python -m pytest "${args[@]}"
