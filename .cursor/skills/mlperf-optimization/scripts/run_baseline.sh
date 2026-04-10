#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# MLPerf Optimization — Baseline Run
#
# Runs a Tier 1 trial to establish baseline ms/iter.
# Uses run_mlperf_trial for log filtering, MLLOG_TRAIN_LOSS_LOG_FREQ override,
# and structured TRIAL_RESULT output.
#
# Required env vars: MLPERF_DIR, CONFIG_SH
# Optional: RESULT_DIR, TRAIN_ITERS
# =============================================================================

: "${MLPERF_DIR:?MLPERF_DIR env var required}"
: "${CONFIG_SH:?CONFIG_SH env var required}"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
RESULT_DIR="${RESULT_DIR:-/root/mlperf_results/${TIMESTAMP}}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"

mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "MLPerf Optimization — Baseline"
echo "Config: $CONFIG_SH"
echo "Iters: $TRAIN_ITERS (Tier 1 trial)"
echo "Results: $RESULT_DIR"
echo "============================================================"

# --- Baseline run (Tier 1) ---
run_mlperf_trial "baseline" 1 "$TRAIN_ITERS"

# --- Parse TRIAL_RESULT ---
RESULT_LINE=$(grep "^TRIAL_RESULT" "$RESULT_DIR/attempt_baseline.log" || echo "")
if [ -z "$RESULT_LINE" ]; then
    echo "ERROR: No TRIAL_RESULT found in baseline log" >&2
    exit 1
fi

eval "$(parse_trial_result "$RESULT_LINE")"
BASELINE_MS="$TRIAL_MS_PER_ITER"
BASELINE_GBS="$TRIAL_GBS"
BASELINE_LOSS="$TRIAL_LAST_LOSS"
BASELINE_STATUS="$TRIAL_STATUS"

echo ""
echo "=== Baseline: ${BASELINE_MS} ms/iter (GBS=${BASELINE_GBS}, status=${BASELINE_STATUS}) ==="

if [ "$BASELINE_STATUS" = "nan" ]; then
    echo "ERROR: NaN detected in baseline — FP8 instability?" >&2
    exit 1
fi

if [ "$BASELINE_STATUS" = "no_data" ]; then
    echo "ERROR: No data from baseline — check primus_mllog installation" >&2
    exit 1
fi

# --- Extract time-to-train from raw log ---
TTT_INFO=$(extract_time_to_train "$RESULT_DIR/attempt_baseline_raw.log")
echo "Time-to-train info: $TTT_INFO"

# --- Initialize results.tsv ---
cat > "$RESULT_DIR/results.tsv" <<EOF
attempt	ms_per_iter	speedup_pct	status	description
0	${BASELINE_MS}	0.0	baseline	Baseline (8 GPU, GBS=${BASELINE_GBS}, FP8=hybrid)
EOF

# --- Extract losses from raw log ---
echo ""
echo "Loss trajectory:"
extract_losses "$RESULT_DIR/attempt_baseline_raw.log"

# --- Write run context ---
cat > "$RESULT_DIR/run_context.env" <<EOF
MLPERF_DIR=$MLPERF_DIR
CONFIG_SH=$CONFIG_SH
RESULT_DIR=$RESULT_DIR
BASELINE_MS=$BASELINE_MS
BASELINE_GBS=$BASELINE_GBS
BASELINE_LOSS=$BASELINE_LOSS
MASTER_PORT=$_CURRENT_PORT
EOF

echo ""
echo "============================================================"
echo "Baseline complete: ${BASELINE_MS} ms/iter (GBS=${BASELINE_GBS})"
echo "Results: $RESULT_DIR"
echo "============================================================"
