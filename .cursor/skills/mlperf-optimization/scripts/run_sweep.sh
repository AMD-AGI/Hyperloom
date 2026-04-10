#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# MLPerf Optimization — GBS × LR Sweep
#
# Runs Tier 2 trials with different GBS/LR/MBS combinations to find the
# optimal operating point for time-to-target.
#
# Required env vars: MLPERF_DIR, CONFIG_SH, RESULT_DIR
# =============================================================================

: "${MLPERF_DIR:?MLPERF_DIR env var required}"
: "${CONFIG_SH:?CONFIG_SH env var required}"
: "${RESULT_DIR:?RESULT_DIR env var required}"

SWEEP_ITERS="${SWEEP_ITERS:-500}"

echo "============================================================"
echo "MLPerf Optimization — GBS × LR Sweep (Tier 2)"
echo "Iters per trial: $SWEEP_ITERS"
echo "Results: $RESULT_DIR"
echo "============================================================"

cat > "$RESULT_DIR/sweep_results.tsv" <<EOF
gbs	lr	mbs	ga_steps	ms_per_iter	loss_at_end	status
EOF

sweep_configs=(
    "16:2.0e-4:2:1"
    "32:4.0e-4:2:2"
    "32:4.0e-4:4:1"
    "64:5.6e-4:2:4"
    "64:5.6e-4:4:2"
)

for config_str in "${sweep_configs[@]}"; do
    IFS=: read -r gbs lr mbs ga <<< "$config_str"
    label="gbs${gbs}_lr${lr}_mbs${mbs}"

    echo ""
    echo "--- Sweep: GBS=$gbs LR=$lr MBS=$mbs GA=$ga ---"

    run_mlperf_trial "sweep_${label}" 2 "$SWEEP_ITERS" \
        "PRIMUS_GLOBAL_BATCH_SIZE=$gbs PRIMUS_LR=$lr PRIMUS_MICRO_BATCH_SIZE=$mbs PRIMUS_MIN_LR=$(python3 -c "print(f'{float(\"$lr\") * 0.1:.1e}')")"

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

    echo -e "${gbs}\t${lr}\t${mbs}\t${ga}\t${ms_per_iter}\t${last_loss}\t${status}" >> "$RESULT_DIR/sweep_results.tsv"
    echo "  Result: ${ms_per_iter} ms/iter, last_loss=${last_loss}, status=${status}"
done

echo ""
echo "============================================================"
echo "Sweep complete. Results: $RESULT_DIR/sweep_results.tsv"
cat "$RESULT_DIR/sweep_results.tsv"
echo "============================================================"
