#!/usr/bin/env bash
# =============================================================================
# run_bench.sh — dispatcher that runs the detected/overridden BENCH_COMMAND
# Sources $RESULT_DIR/state.env, $RESULT_DIR/detected.env, $RESULT_DIR/kept_env.sh
# =============================================================================
set -euo pipefail

[ -n "${RESULT_DIR:-}" ] || { echo "RESULT_DIR not set"; exit 1; }

source "$RESULT_DIR/state.env"
source "$RESULT_DIR/detected.env"
[ -f "$RESULT_DIR/kept_env.sh" ] && source "$RESULT_DIR/kept_env.sh"

[ -n "${BENCH_COMMAND:-}" ] || {
    echo "ERROR: BENCH_COMMAND empty. Run detect again, or set BENCH_COMMAND before invoking."
    exit 2
}

cd "$REPO_ROOT"
echo "[bench] $BENCH_COMMAND"
echo "[bench] kept env: $(env | grep -E '^(HSA_|GPU_MAX|HIP_|HSA_FORCE|MIOPEN|TORCH|PYTORCH)' | tr '\n' ' ')"
exec bash -c "$BENCH_COMMAND"
