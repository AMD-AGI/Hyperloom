#!/usr/bin/env bash
# =============================================================================
# inference-optimization/scripts/common.sh
#
# Shared helpers for run_baseline.sh, run_profile.sh, and run_sweep.sh:
#   - kill_server          - tear down vLLM or SGLang serving processes
#   - wait_for_health      - HTTP /health wait with PID and log checks
#   - check_benchmark_lib  - verify and source InferenceX benchmark_lib.sh
#   - filter_trace         - shrink Chrome trace JSON for TraceLens
#   - check_gpu_memory     - optional free-memory gate before starting server
#
# Requires: FRAMEWORK (sglang|vllm) for kill_server and wait_for_health.
# =============================================================================

# Mode detection
MODE="${MODE:-local}"
if [ "$MODE" = "remote" ]; then
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization}"
else
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/opt/hyperloom}"
fi

# Constants
SERVER_KILL_WAIT_S="${SERVER_KILL_WAIT_S:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kill serving stack. Uses global FRAMEWORK (sglang|vllm).
kill_server() {
    if [ "$FRAMEWORK" = "vllm" ]; then
        # vLLM V1 uses multiprocessing.spawn - must kill the entire process tree
        # First, find vLLM main process and kill its process group
        local vllm_pids
        vllm_pids=$(ps aux | grep "[v]llm.entrypoints" | awk '{print $2}' || true)
        if [ -n "$vllm_pids" ]; then
            for pid in $vllm_pids; do
                # Kill entire process group (catches all child workers)
                kill -TERM -- -$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') 2>/dev/null || true
            done
        fi
        # Also catch any stray vllm processes
        ps aux | grep "[v]llm" | awk '{print $2}' | xargs -r kill -TERM 2>/dev/null || true
    else
        ps aux | grep "[p]ython3 -m sglang" | awk '{print $2}' | xargs -r kill -TERM 2>/dev/null || true
    fi
    sleep "$SERVER_KILL_WAIT_S"
    # Force kill anything still alive
    if [ "$FRAMEWORK" = "vllm" ]; then
        ps aux | grep "[v]llm" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    else
        ps aux | grep "[p]ython3 -m sglang" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    fi
    # Kill lingering multiprocessing workers (both vLLM and SGLang)
    ps aux | grep "[m]ultiprocessing.spawn" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    # Kill Ray workers spawned by vLLM TP/EP (use [bracket] pattern to avoid matching self)
    ps aux | grep "[r]ay::RayWorkerWrapper" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    ps aux | grep "[v]llm.worker" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    # Kill any orphaned torch/python GPU processes
    ps aux | grep "[p]ython3.*torch" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    sleep "$SERVER_KILL_WAIT_S"
}

# Wait until http://0.0.0.0:<port>/health responds. Args: port log_file pid [timeout_sec]
wait_for_health() {
    local port=$1 log_file=$2 pid=$3 timeout=${4:-600}
    local start_ts elapsed
    start_ts=$(date +%s)
    echo "Waiting for $FRAMEWORK server on port $port (timeout=${timeout}s)..."
    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: Server process $pid died. Last log lines:"
            tail -20 "$log_file"
            return 1
        fi
        if curl -s --max-time 5 "http://0.0.0.0:$port/health" > /dev/null 2>&1; then
            echo "Server ready on port $port"
            return 0
        fi
        elapsed=$(( $(date +%s) - start_ts ))
        if [ "$elapsed" -gt "$timeout" ]; then
            echo "ERROR: Server startup timed out after ${timeout}s. Last log lines:"
            tail -20 "$log_file"
            return 1
        fi
        sleep 5
    done
}

# Ensure benchmarks/benchmark_lib.sh exists under the InferenceX repo, then source it.
check_benchmark_lib() {
    local inference_root="${1:?InferenceX repo path required}"
    local lib="${inference_root}/benchmarks/benchmark_lib.sh"
    if [ ! -f "$lib" ]; then
        echo "ERROR: benchmark library not found: $lib" >&2
        echo "Set INFERENCEX_PATH to the InferenceX repository root (must contain benchmarks/benchmark_lib.sh)." >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "$lib"
}

# Filter a Chrome trace JSON.gz for TraceLens (drop noisy python_function events).
# Args: source_trace_path destination_trace_path
filter_trace() {
    local src="${1:?source trace path required}"
    local dst="${2:?destination trace path required}"
    FILTER_TRACE_SRC="$src" FILTER_TRACE_DST="$dst" python3 -c '
import gzip, json, os
src = os.environ["FILTER_TRACE_SRC"]
dst = os.environ["FILTER_TRACE_DST"]
with gzip.open(src) as f:
    trace = json.load(f)
keep = {"kernel", "gpu_memcpy", "gpu_memset", "cpu_op", "cuda_runtime", "ac2g", "user_annotation", "gpu_user_annotation"}
orig = len(trace["traceEvents"])
trace["traceEvents"] = [e for e in trace["traceEvents"] if e.get("cat", "") in keep]
filt = len(trace["traceEvents"])
with gzip.open(dst, "wt") as f:
    json.dump(trace, f)
size_mb = os.path.getsize(dst) / 1024 / 1024
print(f"Filtered: {orig} -> {filt} events ({size_mb:.1f}MB)")
print("Trace integrity: OK")
' 2>&1 || return 1
}

# Require sufficient free GPU memory on device 0, retrying with kill_server between attempts.
# Uses globals: FRAMEWORK, GPU_FREE_THRESHOLD (default 0.85), GPU_CHECK_RETRIES (default 2).
check_gpu_memory() {
    local thresh="${GPU_FREE_THRESHOLD:-0.85}"
    local retries="${GPU_CHECK_RETRIES:-2}"
    local try
    for try in $(seq 1 "$retries"); do
        if GPU_FREE_THRESHOLD="$thresh" python3 -c "
import torch
import os
thresh = float(os.environ.get('GPU_FREE_THRESHOLD', '0.85'))
f, t = torch.cuda.mem_get_info(0)
ratio = f / t
print(f'GPU memory: {f/1024**3:.1f}GB free / {t/1024**3:.1f}GB total ({ratio*100:.1f}% free)')
assert ratio > thresh, f'GPU not free enough: {ratio*100:.1f}% <= {thresh*100:.1f}% required'
" 2>&1; then
            return 0
        fi
        if [ "$try" -lt "$retries" ]; then
            echo "WARNING: GPU memory not free, attempting cleanup (try $try/$retries)..."
            kill_server
        else
            echo "ERROR: GPU memory still not free after $retries attempts."
            echo "Set GPU_FREE_THRESHOLD=0.5 to lower the threshold, or manually free GPU memory."
            return 1
        fi
    done
    return 1
}
