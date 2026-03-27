#!/bin/bash
set -e
cd /shared_nfs/nehaprakriya/agentic-rc/yanyuan_runs/glm5_optimization

SUMMARY="batch_combined_summary.txt"
echo "=== Combined Winner Sweep $(date) ===" > "$SUMMARY"

run_one() {
    local NAME="$1"
    local DS="$2"
    local EXTRA_SARGS="$3"
    local EXTRA_ENVS="$4"

    echo ">>> Starting $NAME (ds=$DS) at $(date)" | tee -a "$SUMMARY"
    EXPERIMENT="$NAME" \
    EXTRA_SERVER_ARGS="--nsa-decode-backend aiter --enable-mixed-chunk --num-continuous-decode-steps $DS $EXTRA_SARGS" \
    EXTRA_ENV="$EXTRA_ENVS" \
    bash rapid_experiment.sh 2>&1 | tail -10
    
    if [ -f "results_v3/${NAME}_tp4_isl1024_osl1024_conc64.json" ]; then
        python3 -c "
import json
d = json.load(open('results_v3/${NAME}_tp4_isl1024_osl1024_conc64.json'))
total = d['total_token_throughput']
per_gpu = total / 4
output = d['output_throughput']
tpot = d['mean_tpot_ms']
delta = (total - 1403.43) / 1403.43 * 100
line = f'$NAME: {total:.1f} tok/s ({delta:+.1f}%) per-GPU={per_gpu:.1f} TPOT={tpot:.1f}ms'
print(line)
with open('$SUMMARY', 'a') as f:
    f.write(line + '\n')
"
    fi
    echo "" >> "$SUMMARY"
}

# 1. ds=8 (lower decode steps with combined wins)
run_one "combined_ds8" 8 "" ""

# 2. ds=64 (higher decode steps with combined wins)
run_one "combined_ds64" 64 "" ""

# 3. Combined + NCCL Ring with more channels
run_one "combined_nccl_ring32" 16 "" "export NCCL_MIN_NCHANNELS=32; export NCCL_ALGO=Ring"

# 4. Combined + allreduce fusion
run_one "combined_allreduce_fusion" 16 "--enable-aiter-allreduce-fusion" ""

# 5. Combined + higher concurrency (conc=128)
CONC=128 run_one "combined_conc128" 16 "--cuda-graph-max-bs 128" ""

echo "=== All done $(date) ===" >> "$SUMMARY"
cat "$SUMMARY"
