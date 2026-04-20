#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# MLPerf Optimization — MBS / Eval Interval Sweep
#
# Runs Tier 2 trials with different MBS/GA combinations within the winning
# GBS/LR/parallelism config to find the optimal per-iteration cost.
#
# NOTE: For GBS/LR/TP/EP/DP config selection, use actions/config-selection.md
# (three-stage workflow with Tier 1 → 2L → 3).
#
# Required env vars: MLPERF_DIR, CONFIG_SH, RESULT_DIR
# Optional: SWEEP_ITERS (default 500), WINNING_GBS (default from config)
# =============================================================================

: "${MLPERF_DIR:?MLPERF_DIR env var required}"
: "${CONFIG_SH:?CONFIG_SH env var required}"
: "${RESULT_DIR:?RESULT_DIR env var required}"

SWEEP_ITERS="${SWEEP_ITERS:-500}"

source "$CONFIG_SH"
WINNING_GBS="${WINNING_GBS:-$PRIMUS_GLOBAL_BATCH_SIZE}"
WINNING_LR="${WINNING_LR:-$PRIMUS_LR}"
DP="${DP:-8}"

echo "============================================================"
echo "MLPerf Optimization — MBS Sweep (Tier 2)"
echo "Winning config: GBS=$WINNING_GBS LR=$WINNING_LR DP=$DP"
echo "Iters per trial: $SWEEP_ITERS"
echo "Results: $RESULT_DIR"
echo "============================================================"

cat > "$RESULT_DIR/sweep_results.tsv" <<EOF
mbs	ga_steps	ms_per_iter	loss_at_end	status
EOF

for mbs in 1 2 4; do
    ga=$((WINNING_GBS / (mbs * DP)))
    if [ "$ga" -lt 1 ]; then
        echo "--- Skipping MBS=$mbs: GA would be $ga (< 1) ---"
        continue
    fi
    check=$((ga * mbs * DP))
    if [ "$check" -ne "$WINNING_GBS" ]; then
        echo "--- Skipping MBS=$mbs: GBS mismatch ($check != $WINNING_GBS) ---"
        continue
    fi

    label="mbs${mbs}_ga${ga}"

    echo ""
    echo "--- Sweep: MBS=$mbs GA=$ga (GBS=$WINNING_GBS) ---"

    run_mlperf_trial "sweep_${label}" 2 "$SWEEP_ITERS" \
        "PRIMUS_MICRO_BATCH_SIZE=$mbs"

    RESULT_LINE=$(grep "^TRIAL_RESULT" "$RESULT_DIR/attempt_sweep_${label}.log" 2>/dev/null || echo "")

    if [ -n "$RESULT_LINE" ]; then
        eval "$(parse_trial_result "$RESULT_LINE")"
        ms_per_iter="$TRIAL_MS_PER_ITER"
        last_loss="$TRIAL_LAST_LOSS"
        status="$TRIAL_STATUS"
    else
        ms_per_iter="N/A"
        last_loss="N/A"
        status="error"
    fi

    echo -e "${mbs}\t${ga}\t${ms_per_iter}\t${last_loss}\t${status}" >> "$RESULT_DIR/sweep_results.tsv"
    echo "  Result: ${ms_per_iter} ms/iter, last_loss=${last_loss}, status=${status}"
done

echo ""
echo "============================================================"
echo "Sweep complete. Results: $RESULT_DIR/sweep_results.tsv"
cat "$RESULT_DIR/sweep_results.tsv"
echo "============================================================"
