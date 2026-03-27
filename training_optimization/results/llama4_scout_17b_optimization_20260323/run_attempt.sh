#!/bin/bash
set -o pipefail

ATTEMPT=$1; DESC=$2; shift 2; OVERRIDES="$@"
RESULTS_DIR="/shared_nfs/nehaprakriya/results/llama4_scout_17b_optimization_20260323"
CONFIG="$RESULTS_DIR/configs/baseline.yaml"
LOG_DIR="$RESULTS_DIR/logs"
TSV="$RESULTS_DIR/results.tsv"
BASELINE_TPS=19.90

if [ ! -f "$TSV" ]; then
    echo -e "attempt\ttok_per_sec_per_gpu\tspeedup_pct\tstatus\tdescription" > "$TSV"
fi

echo "=== Attempt $ATTEMPT: $DESC ==="
echo "Overrides: $OVERRIDES"

LOG_FILE="$RESULTS_DIR/logs/attempt_${ATTEMPT}.log"
timeout 3600 /opt/venv/bin/tune run --nnodes 1 --nproc_per_node 8 full_finetune_distributed \
    --config "$CONFIG" \
    max_steps_per_epoch=15 \
    profiler.enabled=False \
    log_peak_memory_stats=False \
    fsdp_cpu_offload=False \
    $OVERRIDES \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
rm -rf "$RESULTS_DIR/checkpoints/epoch_0/" 2>/dev/null

LATEST_LOG=$(ls -t "$LOG_DIR"/log_*.txt 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo -e "${ATTEMPT}\t-1\t0.0\tcrash\t${DESC}" >> "$TSV"
    echo "RESULT: CRASH — no log file found"
    exit 1
fi

STEP_COUNT=$(/opt/venv/bin/python3 -c "
import re
lines = open('$LATEST_LOG').readlines()
count = sum(1 for l in lines if re.match(r'Step \d+ \|', l))
print(count)
")

if [ "$STEP_COUNT" -lt 10 ]; then
    echo -e "${ATTEMPT}\t-1\t0.0\tcrash\t${DESC}" >> "$TSV"
    echo "RESULT: CRASH — only $STEP_COUNT steps completed"
    exit 1
fi

TPS=$(/opt/venv/bin/python3 -c "
import re
lines = open('$LATEST_LOG').readlines()
tps_vals = []
for line in lines:
    m = re.match(r'Step (\d+) \|.*tokens_per_second_per_gpu:(\S+)', line)
    if m:
        step = int(m.group(1))
        if step >= 6:
            tps_vals.append(float(m.group(2)))
if tps_vals:
    print(f'{sum(tps_vals)/len(tps_vals):.2f}')
else:
    print('-1')
")

SPEEDUP=$(/opt/venv/bin/python3 -c "
tps = float('$TPS')
baseline = $BASELINE_TPS
if tps > 0:
    print(f'{((tps - baseline) / baseline) * 100:.1f}')
else:
    print('0.0')
")

echo "RESULT: avg tok/s/gpu (steps 6-15) = $TPS  speedup = ${SPEEDUP}%"
echo -e "${ATTEMPT}\t${TPS}\t${SPEEDUP}\tok\t${DESC}" >> "$TSV"
