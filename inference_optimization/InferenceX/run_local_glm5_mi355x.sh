#!/usr/bin/env bash
# =============================================================================
# Run GLM5 benchmark directly inside a MI355X Docker container.
# Usage (inside container, from InferenceX repo root):
#   bash run_local_mi355x.sh
# =============================================================================

set -x
set -euo pipefail

CURDIR=$(cd $(dirname $0); pwd)

export HF_TOKEN="${HF_TOKEN:-***REMOVED***}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"

# Sequence lengths (options: 1024/1024, 1024/8192, 8192/1024)
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"

# Concurrency (suggested values: 4, 8, 16, 32, 64)
export CONC="${CONC:-64}"

export TP="${TP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-1}"

# export MODEL="amd/GLM-5-MXFP4"
export MODEL="zai-org/GLM-5-FP8"
export MAX_MODEL_LEN=$(( ISL + OSL + 200 ))
export RANDOM_RANGE_RATIO=0.8
export PORT=8888

# Metadata required by process_result.py

export PRECISION=fp8
export FRAMEWORK=sglang
export MODEL_PREFIX=glm5
export RUNNER_TYPE=mi355x
export DP_ATTENTION=false
export SPEC_DECODING=none
export DISAGG=false
export IMAGE="rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260324"

# Result filename (unique identifier, no .json suffix)
export RESULT_FILENAME="glm5_isl${ISL}_osl${OSL}_${PRECISION}_mi355x_sglang_tp${TP}-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}"


if [[ -z "$HF_TOKEN" ]]; then
    echo "[ERROR] HF_TOKEN is not set. Please run: export HF_TOKEN=hf_xxxxxxxxxxxxxxxx"
    exit 1
fi

echo "======================================================="
echo " GLM5 Benchmark - MI355X"
echo "======================================================="
echo "  MODEL          : $MODEL"
echo "  TP             : $TP"
echo "  ISL/OSL        : ${ISL}/${OSL}"
echo "  MAX_MODEL_LEN  : $MAX_MODEL_LEN"
echo "  CONC           : $CONC"
echo "  RESULT_FILENAME: $RESULT_FILENAME"
echo "======================================================="
echo ""

# -----------------------------------------------------------------------------
# Run benchmark (start sglang server + client load test)
# -----------------------------------------------------------------------------
echo "[Step 1/2] Running benchmark..."
bash benchmarks/single_node/glm5_${PRECISION}_mi355x.sh

# -----------------------------------------------------------------------------
# Post-process: generate aggregated result
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2/2] Processing results..."
cd /workspace
python3 ${CURDIR}/utils/process_result.py

# -----------------------------------------------------------------------------
# Print summary
# -----------------------------------------------------------------------------
echo ""
echo "======================================================="
echo " Done! Output files:"
echo "   Raw result : ${RESULT_FILENAME}.json"
echo "   Agg result : agg_${RESULT_FILENAME}.json"
echo "   GPU metrics: gpu_metrics.csv"
echo "   Server log : server.log"
echo "======================================================="
echo ""

python3 - <<PYEOF
import json
with open("agg_${RESULT_FILENAME}.json") as f:
    d = json.load(f)
print("--- Key Metrics ---")
print(f"  Token Throughput per GPU (tok/s/gpu) : {d.get('tput_per_gpu', 0):.3f}")
print(f"  Input Token Throughput per GPU       : {d.get('input_tput_per_gpu', 0):.3f}")
print(f"  Output Token Throughput per GPU      : {d.get('output_tput_per_gpu', 0):.3f}")
print(f"  Interactivity (tok/s/user)           : {d.get('median_intvty', 0):.3f}")
if d.get('median_ttft'): print(f"  TTFT (ms)                            : {d['median_ttft']*1000:.2f}")
if d.get('median_tpot'): print(f"  TPOT (ms)                            : {d['median_tpot']*1000:.2f}")
if d.get('median_e2el'): print(f"  End-to-end Latency (s)               : {d['median_e2el']:.3f}")
PYEOF
