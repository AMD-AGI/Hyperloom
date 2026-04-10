#!/bin/bash

set -e

# Create results directory
mkdir -p /results

# Profiler setup
if [[ $PROFILER == "rpd" ]]; then
    export NVTE_NVTX_ENABLED=1
    echo "Profiler is set to RPD"
    bash loadTracer.sh
else
    echo "No profiler enabled (production mode)"
fi

cd /workspace/code

echo "============================================"
echo "MLPerf GPT-OSS-20B Training"
echo "============================================"
echo "Config: ${EXP}"
echo "Data:   ${DATA_PATH}"
echo "GPUs:   ${GPUS_PER_NODE}"
echo "Nodes:  ${NNODES}"
echo "============================================"

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Launch distributed training
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py

ret_code=$?

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result
result=$(( end - start ))
result_name="GPT_OSS_20B"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

if [[ $PROFILER == "rpd" ]]; then
   # Following step is very expensive for full E2E run and unnecessary data burden. Commenting out
   # python3 /workspace/deps/rocmProfileData/tools/rpd2tracing.py --start 10% --end 90% trace.rpd trace.json
    sqlite3 -header -csv trace.rpd "select * from top;" > trace.csv
    python3 /workspace/utilities/metrics/profiling/pie_chart_plot.py trace.csv "LLAMA2 E2E" "${IMAGE_NAME}" 2.0
    # pip install /workspace/utilities/mint/dist/mint-0.1.0-py3-none-any.whl
    # python3 -m mint.trace.kernels .
    mkdir -p /results/artifacts
    cp *.png *.csv *.rpd /results/artifacts
fi

if [[ $ret_code != 0 ]]; then
    echo "Training failed with exit code: $ret_code"
    exit $ret_code
fi

exit 0
