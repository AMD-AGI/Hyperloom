#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# Training Optimization — Profile an existing config
#
# Runs training with profiling enabled and collects trace for TraceLens.
# Use after applying optimizations to get a comparison trace.
#
# Required env vars: CONFIG_YAML, NUM_GPUS, PRIMUS_ROOT
# Optional: RESULT_DIR, KEPT_OVERRIDES, PROFILE_LABEL
# =============================================================================

: "${CONFIG_YAML:?CONFIG_YAML env var required}"
: "${NUM_GPUS:?NUM_GPUS env var required}"
: "${PRIMUS_ROOT:?PRIMUS_ROOT env var required}"

RESULT_DIR="${RESULT_DIR:-.}"
PROFILE_LABEL="${PROFILE_LABEL:-optimized}"
KEPT_OVERRIDES="${KEPT_OVERRIDES:-}"

echo "============================================================"
echo "Training Profile: $PROFILE_LABEL"
echo "Config: $CONFIG_YAML"
echo "Overrides: $KEPT_OVERRIDES"
echo "============================================================"

kill_training
PORT=$(next_port)

cd "$PRIMUS_ROOT"
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$PORT" \
    -m primus.cli.main train pretrain \
    --config "$CONFIG_YAML" \
    $KEPT_OVERRIDES \
    profile=true use_pytorch_profiler=true \
    profile_step_start=6 profile_step_end=7 \
    2>&1 | tee "$RESULT_DIR/${PROFILE_LABEL}_profile.log"

# Find and copy trace
TRACE_FILE=$(find "$PRIMUS_ROOT" /tmp /shared_nfs -name "*.pt.trace.json" \
    -newer "$RESULT_DIR/${PROFILE_LABEL}_profile.log" 2>/dev/null | head -1)

if [ -n "$TRACE_FILE" ]; then
    cp "$TRACE_FILE" "$RESULT_DIR/${PROFILE_LABEL}_trace.json"
    filter_trace "$RESULT_DIR/${PROFILE_LABEL}_trace.json" \
                 "$RESULT_DIR/${PROFILE_LABEL}_filtered_trace.json"
    echo "Trace: $RESULT_DIR/${PROFILE_LABEL}_trace.json"
else
    echo "WARNING: No trace file found"
fi

# Also extract ms/iter from this run
MS=$(extract_ms_per_iter "$RESULT_DIR/${PROFILE_LABEL}_profile.log" 5 5 2>/dev/null || echo "N/A")
echo "ms/iter (profiling run): $MS"
