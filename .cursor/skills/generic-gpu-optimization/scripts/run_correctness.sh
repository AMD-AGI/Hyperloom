#!/usr/bin/env bash
# =============================================================================
# run_correctness.sh — runs the detected TEST_COMMAND
# =============================================================================
set -euo pipefail
[ -n "${RESULT_DIR:-}" ] || { echo "RESULT_DIR not set"; exit 1; }
source "$RESULT_DIR/state.env"
source "$RESULT_DIR/detected.env"
[ -f "$RESULT_DIR/kept_env.sh" ] && source "$RESULT_DIR/kept_env.sh"

if [ -z "${TEST_COMMAND:-}" ] || [ "$CORRECTNESS_MODE" = "none" ]; then
    echo "[correctness] no tests configured; skipping"
    exit 0
fi

cd "$REPO_ROOT"
echo "[correctness] $TEST_COMMAND"
exec bash -c "$TEST_COMMAND"
