#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# Training Optimization — Parameter Sweep
#
# Sweeps micro_batch_size while maintaining constant GBS.
# Each point: kill → launch → measure → record.
#
# Required env vars: CONFIG_YAML, NUM_GPUS, PRIMUS_ROOT, BASELINE_GBS
# Optional: RESULT_DIR, KEPT_OVERRIDES, MBS_VALUES
# =============================================================================

: "${CONFIG_YAML:?CONFIG_YAML env var required}"
: "${NUM_GPUS:?NUM_GPUS env var required}"
: "${PRIMUS_ROOT:?PRIMUS_ROOT env var required}"
: "${BASELINE_GBS:?BASELINE_GBS env var required}"

RESULT_DIR="${RESULT_DIR:-.}"
KEPT_OVERRIDES="${KEPT_OVERRIDES:-}"
MBS_VALUES="${MBS_VALUES:-1 2 4 8 16}"

# Compute DP (assume TP and PP from config, default TP=1 PP=1)
TP="${TP:-1}"
PP="${PP:-1}"
DP=$((NUM_GPUS / (TP * PP)))

SWEEP_FILE="$RESULT_DIR/sweep_results.tsv"
echo -e "mbs\tga_steps\tms_per_iter\tstatus" > "$SWEEP_FILE"

echo "============================================================"
echo "Training Sweep: MBS = [$MBS_VALUES]"
echo "GBS=$BASELINE_GBS  DP=$DP  TP=$TP  PP=$PP"
echo "============================================================"

for MBS in $MBS_VALUES; do
    GA=$((BASELINE_GBS / (MBS * DP)))

    # Skip invalid configs
    if [ $((MBS * DP * GA)) -ne "$BASELINE_GBS" ]; then
        echo "SKIP: mbs=$MBS (GBS not divisible)"
        echo -e "${MBS}\t${GA}\t-\tskip_indivisible" >> "$SWEEP_FILE"
        continue
    fi

    echo ""
    echo "--- MBS=$MBS  GA=$GA ---"

    kill_training
    PORT=$(next_port)

    cd "$PRIMUS_ROOT"
    LOG="/tmp/sweep_mbs${MBS}.log"

    set +e
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$PORT" \
        -m primus.cli.main train pretrain \
        --config "$CONFIG_YAML" \
        $KEPT_OVERRIDES \
        micro_batch_size=$MBS \
        profile=false use_pytorch_profiler=false \
        2>&1 | tee "$LOG"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${MBS}\t${GA}\t-\tcrash" >> "$SWEEP_FILE"
        continue
    fi

    MS=$(extract_ms_per_iter "$LOG" 5 5 2>/dev/null || echo "-")
    if [ "$MS" = "-" ] || [ -z "$MS" ]; then
        echo -e "${MBS}\t${GA}\t-\tno_data" >> "$SWEEP_FILE"
        continue
    fi

    # Verify GBS
    if ! verify_gbs "$LOG" "$BASELINE_GBS" 2>/dev/null; then
        echo -e "${MBS}\t${GA}\t${MS}\tinvalid_gbs" >> "$SWEEP_FILE"
        continue
    fi

    echo -e "${MBS}\t${GA}\t${MS}\tok" >> "$SWEEP_FILE"
    echo "  → $MS ms/iter"
done

echo ""
echo "============================================================"
echo "Sweep complete. Results: $SWEEP_FILE"
cat "$SWEEP_FILE"
echo "============================================================"
