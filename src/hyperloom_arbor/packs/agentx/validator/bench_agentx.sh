#!/usr/bin/env bash
# AgentX agentic-coding bench — model-agnostic.
#
# Arbor injects: PORT, MODEL_PATH, MODEL_NAME, CONCURRENCY, NUM_PROMPTS, BENCH_OUTPUT
# Campaign config server_env provides: FRAMEWORK, MODEL_PREFIX, PRECISION, SPEC_DECODING,
#   TP, EP_SIZE, DP_ATTENTION, KV_OFFLOADING, DURATION, IMAGE, RUNNER_TYPE, etc.
#
# Server is already running — this script only runs the aiperf trace replay.

set -euo pipefail

: "${PORT:?Arbor must inject PORT}"
: "${MODEL_PATH:?Arbor must inject MODEL_PATH}"
: "${MODEL_NAME:?Arbor must inject MODEL_NAME}"
: "${CONCURRENCY:?Arbor must inject CONCURRENCY}"
: "${NUM_PROMPTS:?Arbor must inject NUM_PROMPTS}"
: "${BENCH_OUTPUT:?Arbor must inject BENCH_OUTPUT}"

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX}"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

# ---- Map Arbor env vars to InferenceX conventions ----
# MODEL must be the full local path so aiperf can load the tokenizer.
# SERVED_MODEL_NAME is the wire name the server registered (used for --model in requests).
export MODEL="${MODEL_PATH}"
export SERVED_MODEL_NAME="${MODEL_NAME}"
export MODEL_PREFIX="${MODEL_PREFIX:?campaign config must set MODEL_PREFIX}"
export FRAMEWORK="${FRAMEWORK:?campaign config must set FRAMEWORK}"
export PRECISION="${PRECISION:-fp4}"
export TP="${TP:-8}"
export EP_SIZE="${EP_SIZE:-1}"
export DP_ATTENTION="${DP_ATTENTION:-false}"
export CONC="$CONCURRENCY"
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}"
export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-0}"
export SPEC_DECODING="${SPEC_DECODING:-none}"
export IS_MULTINODE="false"
export PP_SIZE=1
export DCP_SIZE=1
export PCP_SIZE=1
export DISAGG="false"

# Fast mode for profiling runs
if [ "${NUM_PROMPTS:-320}" -le 50 ]; then
    export AIPERF_EXPERIMENTAL_FAST=1
    echo "[bench_agentx] FAST MODE (NUM_PROMPTS=${NUM_PROMPTS})"
else
    export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-0}"
fi
export DURATION="${DURATION:-3600}"

RESULT_DIR="$(dirname "$BENCH_OUTPUT")/agentic_run"
export RESULT_DIR
mkdir -p "$RESULT_DIR"

export RESULT_FILENAME="agg_${MODEL_PREFIX}_${FRAMEWORK}_agentx_conc${CONC}"
export AGENTIC_OUTPUT_DIR="$(dirname "$BENCH_OUTPUT")"
export RUNNER_TYPE="${RUNNER_TYPE:-mi355x}"
export IMAGE="${IMAGE:-unknown}"

export AIPERF_SERVER_URL="http://localhost:${PORT}"
METRIC_PREFIX="${FRAMEWORK}:"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="${AIPERF_REQUIRED_SERVER_METRIC_PREFIX:-$METRIC_PREFIX}"
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"

export WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126}"
export SERVED_MODEL_NAME="$MODEL_NAME"
export AIPERF_EXTRA_INPUTS="${AIPERF_EXTRA_INPUTS:-}"

echo "[bench_agentx] PORT=$PORT MODEL=$MODEL CONC=$CONC TP=$TP FRAMEWORK=$FRAMEWORK DURATION=$DURATION"
echo "[bench_agentx] RESULT_DIR=$RESULT_DIR RESULT_FILENAME=$RESULT_FILENAME"

# ---- Install aiperf, download traces, build command, run ----
install_agentic_deps
resolve_trace_source

build_replay_cmd "$RESULT_DIR"

# SGLang needs explicit server-metrics endpoint
if [ "$FRAMEWORK" = "sglang" ]; then
    REPLAY_CMD+=" --server-metrics http://localhost:${PORT}/metrics"
fi

run_agentic_replay_and_write_outputs "$RESULT_DIR"

# ---- Convert aiperf aggregate → Arbor bench_output.json ----
AGG_JSON="${AGENTIC_OUTPUT_DIR}/${RESULT_FILENAME}.json"

if [ ! -f "$AGG_JSON" ]; then
    echo "[bench_agentx] ERROR: aggregated result not found at $AGG_JSON" >&2
    exit 1
fi

python3 - "$AGG_JSON" "$BENCH_OUTPUT" <<'PYEOF'
import json, sys

agg_path, out_path = sys.argv[1], sys.argv[2]
with open(agg_path) as f:
    agg = json.load(f)

req = agg.get("request_metrics", {})
per_gpu = req.get("throughput", {}).get("per_gpu", {})

output_throughput = per_gpu.get("output_tput_tps", 0.0)
total_throughput  = per_gpu.get("total_tput_tps",  0.0)

result = {
    "output_throughput":           output_throughput,
    "total_throughput_per_gpu":    total_throughput,
    "conc":                        agg.get("conc", 0),
    "tp":                          agg.get("tp",   0),
    "ep":                          agg.get("ep",   0),
    "scenario_type":               agg.get("scenario_type", "agentic-coding"),
    "spec_decoding":               agg.get("spec_decoding", ""),
    "kv_offloading":               agg.get("kv_offloading", ""),
    "num_requests_successful":     agg.get("num_requests_successful", 0),
    "num_requests_total":          agg.get("num_requests_total", 0),
    "kv_cache_hit_rate":           agg.get("server_metrics", {}).get("cache", {}).get("gpu_cache_hit_rate"),
    "agg_json_path":               agg_path,
    "status": "ok",
}

with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"[bench_agentx] output_throughput={output_throughput:.1f} tok/s/GPU  "
      f"total={total_throughput:.1f} tok/s/GPU  "
      f"requests={result['num_requests_successful']}/{result['num_requests_total']}")
PYEOF
