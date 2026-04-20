#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# MLPerf Optimization — Baseline Run (Full Convergence)
#
# Runs a Tier 4 full convergence trial to establish the real baseline TTT.
# The baseline MUST run to convergence (eval_loss ≤ 3.34) or exhaust all iters.
# Do NOT use Tier 1/2/3 for baseline — only Tier 4 provides a valid TTT reference.
#
# Reference: current best known TTT ~206 min (must be re-verified).
#
# Required env vars: MLPERF_DIR, CONFIG_SH
# Optional: RESULT_DIR
# =============================================================================

: "${MLPERF_DIR:?MLPERF_DIR env var required}"
: "${CONFIG_SH:?CONFIG_SH env var required}"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
RESULT_DIR="${RESULT_DIR:-/root/mlperf_results/${TIMESTAMP}}"

mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "MLPerf Optimization — Baseline (Tier 4 Full Convergence)"
echo "Config: $CONFIG_SH"
echo "Reference TTT: ~206 min"
echo "Results: $RESULT_DIR"
echo "============================================================"
echo ""
echo "WARNING: This is a full convergence run. Do NOT interrupt."
echo ""

# --- Baseline run (Tier 4 — full convergence, no timeout) ---
run_mlperf_trial "baseline" 4

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
BASELINE_RUN_STATUS="${TRIAL_RUN_STATUS:-unknown}"

# --- Extract TTT (primary baseline metric) ---
TTT_INFO=$(extract_time_to_train "$RESULT_DIR/attempt_baseline_raw.log")
BASELINE_TTT_SECONDS=$(echo "$TTT_INFO" | cut -f1)
BASELINE_TTT_STATUS=$(echo "$TTT_INFO" | cut -f2)
BASELINE_TTT_MINUTES=$(python3 -c "print(f'{float(\"$BASELINE_TTT_SECONDS\") / 60:.1f}')")

echo ""
echo "============================================================"
echo "Baseline Results:"
echo "  TTT:        ${BASELINE_TTT_MINUTES} min (${BASELINE_TTT_SECONDS} s)"
echo "  Status:     ${BASELINE_TTT_STATUS}"
echo "  ms/iter:    ${BASELINE_MS}"
echo "  GBS:        ${BASELINE_GBS}"
echo "  Reference:  206 min"
echo "============================================================"

if [ "$BASELINE_TTT_STATUS" = "aborted" ]; then
    echo "CRITICAL: Baseline did NOT converge (status=aborted)." >&2
    echo "Final loss: ${BASELINE_LOSS}. Investigate before proceeding." >&2
fi

if [ "$BASELINE_STATUS" = "nan" ]; then
    echo "ERROR: NaN detected in baseline — FP8 instability?" >&2
    exit 1
fi

if [ "$BASELINE_STATUS" = "no_data" ]; then
    echo "ERROR: No data from baseline — check primus_mllog installation" >&2
    exit 1
fi

# --- Initialize results.tsv with TTT ---
cat > "$RESULT_DIR/results.tsv" <<EOF
attempt	ms_per_iter	ttt_seconds	ttt_minutes	status	description
0	${BASELINE_MS}	${BASELINE_TTT_SECONDS}	${BASELINE_TTT_MINUTES}	baseline	Baseline full run (8 GPU, GBS=${BASELINE_GBS}, FP8=hybrid, TTT=${BASELINE_TTT_MINUTES}min)
EOF

# --- Write run context ---
cat > "$RESULT_DIR/run_context.env" <<EOF
MLPERF_DIR=$MLPERF_DIR
CONFIG_SH=$CONFIG_SH
RESULT_DIR=$RESULT_DIR
BASELINE_MS=$BASELINE_MS
BASELINE_GBS=$BASELINE_GBS
BASELINE_LOSS=$BASELINE_LOSS
BASELINE_TTT_SECONDS=$BASELINE_TTT_SECONDS
BASELINE_TTT_MINUTES=$BASELINE_TTT_MINUTES
BASELINE_TTT_STATUS=$BASELINE_TTT_STATUS
MASTER_PORT=$_CURRENT_PORT
EOF

echo ""
echo "Baseline complete. TTT=${BASELINE_TTT_MINUTES} min (ref: 206 min)"
echo "Results: $RESULT_DIR"
