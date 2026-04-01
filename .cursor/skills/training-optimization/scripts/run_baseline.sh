#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# Training Optimization — Baseline + Profile (two passes)
#
# Pass 1: Clean training run (no profiling overhead) for baseline ms/iter
# Pass 2: Profiling run to collect torch.profiler Chrome trace
#
# Required env vars: CONFIG_YAML, NUM_GPUS, PRIMUS_ROOT
# Optional: RESULT_DIR, MASTER_PORT, WARMUP_ITERS, MEASURE_ITERS
# =============================================================================

: "${CONFIG_YAML:?CONFIG_YAML env var required}"
: "${NUM_GPUS:?NUM_GPUS env var required}"
: "${PRIMUS_ROOT:?PRIMUS_ROOT env var required}"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
RESULT_DIR="${RESULT_DIR:-/shared_nfs/training-optimization/results/${TIMESTAMP}}"
WARMUP_ITERS="${WARMUP_ITERS:-5}"
MEASURE_ITERS="${MEASURE_ITERS:-5}"

mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "Training Optimization — Baseline + Profile"
echo "Config: $CONFIG_YAML"
echo "GPUs: $NUM_GPUS"
echo "Results: $RESULT_DIR"
echo "============================================================"

# --- Pass 1: Baseline (no profiling) ---
echo ""
echo "[1/2] Running baseline training (clean, no profiling)..."

kill_training
PORT=$(next_port)

cd "$PRIMUS_ROOT"
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$PORT" \
    -m primus.cli.main train pretrain \
    --config "$CONFIG_YAML" \
    profile=false use_pytorch_profiler=false \
    2>&1 | tee "$RESULT_DIR/baseline.log"

BASELINE_MS=$(extract_ms_per_iter "$RESULT_DIR/baseline.log" "$WARMUP_ITERS" "$MEASURE_ITERS")
echo ""
echo "=== Baseline: ${BASELINE_MS} ms/iter ==="

# Extract GBS
BASELINE_GBS=$(python3 -c "
import re
with open('$RESULT_DIR/baseline.log') as f:
    content = f.read()
for pat in [r'global.batch.size[:\s=]+(\d+)', r'global_batch_size[:\s=]+(\d+)']:
    m = re.search(pat, content, re.IGNORECASE)
    if m:
        print(m.group(1))
        break
")
echo "GBS: $BASELINE_GBS"

# Initialize results.tsv
cat > "$RESULT_DIR/results.tsv" <<EOF
attempt	ms_per_iter	speedup_pct	status	description
0	${BASELINE_MS}	0.0	baseline	Baseline (${NUM_GPUS} GPU, $(basename "$CONFIG_YAML"))
EOF

# --- Pass 2: Profiling ---
echo ""
echo "[2/2] Running profiling pass..."

kill_training
PORT=$(next_port)

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$PORT" \
    -m primus.cli.main train pretrain \
    --config "$CONFIG_YAML" \
    profile=true use_pytorch_profiler=true \
    profile_step_start=6 profile_step_end=7 \
    2>&1 | tee "$RESULT_DIR/profile.log"

# Find and copy trace file
TRACE_FILE=$(find "$PRIMUS_ROOT" /tmp /shared_nfs -name "*.pt.trace.json" \
    -newer "$RESULT_DIR/profile.log" 2>/dev/null | head -1)

if [ -n "$TRACE_FILE" ]; then
    cp "$TRACE_FILE" "$RESULT_DIR/baseline_trace.json"
    echo "Trace saved to $RESULT_DIR/baseline_trace.json"

    # Filter for TraceLens
    filter_trace "$RESULT_DIR/baseline_trace.json" "$RESULT_DIR/filtered_trace.json"
else
    echo "WARNING: No trace file found"
fi

# Write run context
cat > "$RESULT_DIR/run_context.env" <<EOF
CONFIG_YAML=$CONFIG_YAML
NUM_GPUS=$NUM_GPUS
PRIMUS_ROOT=$PRIMUS_ROOT
RESULT_DIR=$RESULT_DIR
BASELINE_MS=$BASELINE_MS
BASELINE_GBS=$BASELINE_GBS
MASTER_PORT=$_CURRENT_PORT
EOF

echo ""
echo "============================================================"
echo "Baseline complete: ${BASELINE_MS} ms/iter (GBS=${BASELINE_GBS})"
echo "Results: $RESULT_DIR"
echo "============================================================"
