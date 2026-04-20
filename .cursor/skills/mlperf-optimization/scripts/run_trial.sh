#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# MLPerf Optimization — Standalone Trial Runner
#
# CLI wrapper for run_mlperf_trial. Allows the agent to run individual trials.
#
# Usage:
#   ./run_trial.sh --label test --tier 1
#   ./run_trial.sh --label gbs64 --tier 2 --iters 500 --env "PRIMUS_GLOBAL_BATCH_SIZE=64"
#   ./run_trial.sh --label cfg_A --tier 3 --env "PRIMUS_GLOBAL_BATCH_SIZE=32"
#   ./run_trial.sh --label final --tier 4
#
# Required env vars: MLPERF_DIR, CONFIG_SH
# Optional: RESULT_DIR
# =============================================================================

: "${MLPERF_DIR:?MLPERF_DIR env var required}"
: "${CONFIG_SH:?CONFIG_SH env var required}"

LABEL=""
TIER=1
ITERS=""
EXTRA_ENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)  LABEL="$2"; shift 2 ;;
        --tier)   TIER="$2"; shift 2 ;;
        --iters)  ITERS="$2"; shift 2 ;;
        --env)    EXTRA_ENV="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --label NAME --tier 1|2|3|4 [--iters N] [--env 'KEY=VAL ...']"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$LABEL" ]; then
    echo "ERROR: --label is required" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
RESULT_DIR="${RESULT_DIR:-/root/mlperf_results/${TIMESTAMP}}"
mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "MLPerf Trial: label=$LABEL tier=$TIER iters=${ITERS:-auto}"
echo "Results: $RESULT_DIR"
echo "============================================================"

run_mlperf_trial "$LABEL" "$TIER" "$ITERS" "$EXTRA_ENV"

echo ""
echo "--- Done ---"
echo "Filtered log: $RESULT_DIR/attempt_${LABEL}.log"
echo "Raw log: $RESULT_DIR/attempt_${LABEL}_raw.log"
grep "^TRIAL_RESULT" "$RESULT_DIR/attempt_${LABEL}.log" 2>/dev/null || echo "WARNING: No TRIAL_RESULT found"
