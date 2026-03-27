#!/bin/bash
# Usage: bash run_attempt.sh <attempt_number> <description> [override1=val1 override2=val2 ...]
# Runs Qwen3 8B full finetune with given overrides and logs results to results.tsv

ATTEMPT=$1
DESC=$2
shift 2
OVERRIDES="$@"

RESULTS_DIR="/shared_nfs/nehaprakriya/results/qwen3_8b_optimization_20260323"
CONFIG="$RESULTS_DIR/configs/baseline.yaml"
LOG_DIR="$RESULTS_DIR/logs"
TSV="$RESULTS_DIR/results.tsv"

if [ ! -f "$TSV" ]; then
    echo -e "attempt\ttok_per_sec_per_gpu\tspeedup_pct\tstatus\tdescription" > "$TSV"
fi

echo "=== Attempt $ATTEMPT: $DESC ==="
echo "Overrides: $OVERRIDES"

LOG_FILE="$RESULTS_DIR/logs/attempt_${ATTEMPT}.log"

/opt/venv/bin/tune run --nnodes 1 --nproc_per_node 8 full_finetune_distributed \
    --config "$CONFIG" \
    max_steps_per_epoch=15 \
    profiler.enabled=False \
    log_peak_memory_stats=False \
    $OVERRIDES \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

rm -rf "$RESULTS_DIR/checkpoints/epoch_0/" 2>/dev/null

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${ATTEMPT}\t-1\t0.0\tcrash\t${DESC}" >> "$TSV"
    echo "CRASH (exit=$EXIT_CODE)"
    exit 1
fi

# Extract tokens_per_second_per_gpu from the disk log (most recent log file)
LATEST_LOG=$(ls -t "$LOG_DIR"/log_*.txt 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo -e "${ATTEMPT}\t-1\t0.0\terror\t${DESC} (no log found)" >> "$TSV"
    exit 1
fi

# Parse steps 6-15 tokens_per_second_per_gpu
TPS=$(/opt/venv/bin/python3 -c "
import re, sys
lines = open('$LATEST_LOG').readlines()
tps_vals = []
for line in lines:
    m = re.match(r'Step (\d+) \|.*tokens_per_second_per_gpu:(\S+)', line)
    if m:
        step = int(m.group(1))
        if step >= 6:
            tps_vals.append(float(m.group(2)))
if tps_vals:
    avg = sum(tps_vals) / len(tps_vals)
    print(f'{avg:.2f}')
else:
    print('-1')
")

echo "Avg tokens/sec/gpu (steps 6-15): $TPS"
echo -e "${ATTEMPT}\t${TPS}\tPENDING\t\t${DESC}" >> "$TSV"
