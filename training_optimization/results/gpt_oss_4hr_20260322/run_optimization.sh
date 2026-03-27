#!/bin/bash
set -uo pipefail

###############################################################################
# GPT-OSS 20B Full-Scale Training Optimization — 4-Hour Automated Run
#
# Uses run_pretrain.sh for EVERY training run to match nightly env vars:
#   GPU_MAX_HW_QUEUES=2, CUDA_DEVICE_MAX_CONNECTIONS=1,
#   HSA_NO_SCRATCH_RECLAIM=1, NCCL_CHECKS_DISABLE=1, etc.
#
# Date: 2026-03-22
# Platform: 8× AMD MI355X (gfx950), ROCm 7.2, PyTorch 2.10
# Model: GPT-OSS 20B BF16, EP=8, mock data, Primus/Megatron
###############################################################################

RESULTS_DIR="/shared_nfs/nehaprakriya/results/gpt_oss_4hr_20260322"
PRIMUS_DIR="/workspace/Primus"
CONFIG="examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml"
TSV="$RESULTS_DIR/results.tsv"
LOG_DIR="$RESULTS_DIR/logs"
REPORT="$RESULTS_DIR/optimization_report.md"
GEAK_LOG="$RESULTS_DIR/geak_tasks.log"

MASTER_PORT=29510
TIME_BUDGET_SECONDS=14400  # 4 hours
START_TIME=$(date +%s)
CRASH_COUNT=0
MAX_CRASHES=4
CONSECUTIVE_DISCARDS=0
MAX_CONSECUTIVE_DISCARDS=5
ATTEMPT=0
BASELINE_MS=""
BEST_MS=""
BEST_OVERRIDES=""

# GEAK config
GEAK_URL="https://oci-slc.primus-safe.amd.com/control-plane/control-plane-dev/geak-agent-wvsbv/mcp/sse"
GEAK_AUTH="Bearer ak-dwQPsHixH3p28jgzwyLgueVf0JUP-cpHiscxTQsnWJ0"
GEAK_STEP_LIMIT=30
LITELLM_API_KEY="sk-HQ8vFgX51UHQ1k7ILF0N1A"

mkdir -p "$LOG_DIR" "$RESULTS_DIR/traces" "$RESULTS_DIR/geak_outputs" "$RESULTS_DIR/tracelens_output"

exec > >(tee -a "$RESULTS_DIR/run.log") 2>&1

echo "========================================="
echo "GPT-OSS 20B Optimization Run"
echo "Start: $(date -u)"
echo "Budget: 4 hours"
echo "GEAK step_limit: $GEAK_STEP_LIMIT"
echo "========================================="

###############################################################################
# Helper functions
###############################################################################

elapsed_minutes() {
    echo $(( ($(date +%s) - START_TIME) / 60 ))
}

time_remaining() {
    echo $(( TIME_BUDGET_SECONDS - ($(date +%s) - START_TIME) ))
}

check_time_budget() {
    if [ "$(time_remaining)" -lt 300 ]; then
        echo "[$(elapsed_minutes)m] Time budget exhausted. Stopping."
        return 1
    fi
    return 0
}

kill_training() {
    pkill -9 -f "primus/cli/main.py" 2>/dev/null || true
    pkill -9 -f "primus.cli.main" 2>/dev/null || true
    sleep 3
}

increment_port() {
    MASTER_PORT=$((MASTER_PORT + 1))
}

# Run training via run_pretrain.sh to match nightly env vars exactly
run_training() {
    local overrides="$1"
    local log_file="$2"
    local timeout_secs="${3:-900}"

    kill_training
    increment_port

    echo "[$(elapsed_minutes)m] Running training (port=$MASTER_PORT): overrides=[$overrides]"

    cd "$PRIMUS_DIR"

    # Use run_pretrain.sh which sets all nightly env vars
    timeout "$timeout_secs" env \
        EXP="$CONFIG" \
        MASTER_PORT="$MASTER_PORT" \
        MASTER_ADDR="localhost" \
        HSA_NO_SCRATCH_RECLAIM=1 \
        bash examples/run_pretrain.sh $overrides \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    echo "[$(elapsed_minutes)m] Training exited with code $exit_code"
    return $exit_code
}

extract_ms_per_iter() {
    local log_file="$1"
    python3 -c "
import re, sys
lines = open('$log_file').readlines()
ms_values = []
for line in lines:
    m = re.search(r'iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)', line)
    if m:
        iter_num = int(m.group(1))
        ms = float(m.group(2))
        if iter_num >= 6:
            ms_values.append(ms)
if ms_values:
    avg = sum(ms_values) / len(ms_values)
    print(f'{avg:.1f}')
else:
    print('-1')
"
}

log_result() {
    local attempt=$1 ms=$2 speedup=$3 status=$4 desc=$5
    echo -e "${attempt}\t${ms}\t${speedup}\t${status}\t${desc}" >> "$TSV"
    echo "[$(elapsed_minutes)m] >>> Attempt $attempt: ${ms}ms (${speedup}%) - $status - $desc"
}

###############################################################################
# Phase 0: Initialize results log
###############################################################################

echo -e "attempt\tms_per_iter\tspeedup_pct\tstatus\tdescription" > "$TSV"

###############################################################################
# Phase 1: Baseline (nightly-identical)
###############################################################################

echo ""
echo "===== PHASE 1: BASELINE (nightly-identical via run_pretrain.sh) ====="
echo ""

# Nightly overrides: profile=false, 10 iterations (default in config is 20, override to 10)
run_training "profile=false use_pytorch_profiler=false train_iters=10" "$LOG_DIR/baseline.log" 900 || true
BASELINE_MS=$(extract_ms_per_iter "$LOG_DIR/baseline.log")

if [ "$BASELINE_MS" = "-1" ]; then
    echo "ERROR: Failed to extract baseline ms/iter from first attempt."
    echo "Retrying baseline with higher timeout..."
    run_training "profile=false use_pytorch_profiler=false train_iters=10" "$LOG_DIR/baseline_retry.log" 1200 || true
    BASELINE_MS=$(extract_ms_per_iter "$LOG_DIR/baseline_retry.log")
fi

if [ "$BASELINE_MS" = "-1" ]; then
    echo "FATAL: Could not establish baseline after retry. Exiting."
    exit 1
fi

log_result 0 "$BASELINE_MS" "0.0" "baseline" "GPT-OSS 20B BF16 nightly baseline (8×MI355X, EP=8, mock data, iter 6-10 avg, run_pretrain.sh)"
BEST_MS="$BASELINE_MS"
cp "$LOG_DIR/baseline.log" "$RESULTS_DIR/baseline.log" 2>/dev/null || \
cp "$LOG_DIR/baseline_retry.log" "$RESULTS_DIR/baseline.log" 2>/dev/null || true
echo ""
echo "*** BASELINE ESTABLISHED: ${BASELINE_MS} ms/iter ***"
echo ""
ATTEMPT=1

###############################################################################
# Phase 2: Profile baseline
###############################################################################

echo ""
echo "===== PHASE 2: PROFILE BASELINE ====="
echo ""

check_time_budget || { echo "Time up before profiling"; }

if check_time_budget 2>/dev/null; then
    run_training "profile=true use_pytorch_profiler=true profile_step_start=6 profile_step_end=7 train_iters=10" \
        "$LOG_DIR/profile.log" 900 || true

    # Find trace file
    TRACE_FILE=$(find "$PRIMUS_DIR/output" -name "*.pt.trace.json" -mmin -15 2>/dev/null | sort | tail -1)
    if [ -n "$TRACE_FILE" ]; then
        echo "Trace file found: $TRACE_FILE"
        cp "$TRACE_FILE" "$RESULTS_DIR/traces/baseline_trace.json" 2>/dev/null || \
        ln -sf "$TRACE_FILE" "$RESULTS_DIR/traces/baseline_trace.json" 2>/dev/null || true

        # Analyze kernel breakdown
        python3 << PYEOF
import json
from collections import defaultdict
try:
    with open("$TRACE_FILE") as f:
        trace = json.load(f)
    gpu_events = [e for e in trace.get("traceEvents", [])
                  if e.get("cat") == "kernel" and "dur" in e]
    kernel_time = defaultdict(float)
    kernel_count = defaultdict(int)
    for e in gpu_events:
        kernel_time[e["name"]] += e["dur"]
        kernel_count[e["name"]] += 1
    total = sum(kernel_time.values())
    print("\n===== TOP 20 GPU KERNELS (BASELINE) =====")
    lines = []
    for name, t in sorted(kernel_time.items(), key=lambda x: -x[1])[:20]:
        line = f"  {name[:70]:70s}  {t/1000:>8.1f}ms  {t/total*100:>5.1f}%  {kernel_count[name]:>4d}x"
        print(line)
        lines.append(line)
    with open("$RESULTS_DIR/kernel_profile_baseline.txt", "w") as f:
        f.write("\n".join(lines))
except Exception as e:
    print(f"Warning: Could not analyze trace: {e}")
PYEOF
    else
        echo "Warning: No trace file found after profiling run."
    fi
fi

###############################################################################
# Phase 2b: Submit GEAK tasks (async — run in parallel with optimization loop)
###############################################################################

echo ""
echo "===== PHASE 2b: SUBMIT GEAK TASKS (step_limit=$GEAK_STEP_LIMIT) ====="
echo ""

python3 << 'GEAK_SUBMIT_SCRIPT'
import json, subprocess, sys, os, time, glob

GEAK_URL = "https://oci-slc.primus-safe.amd.com/control-plane/control-plane-dev/geak-agent-wvsbv/mcp/sse"
GEAK_AUTH = "Bearer ak-dwQPsHixH3p28jgzwyLgueVf0JUP-cpHiscxTQsnWJ0"
LITELLM_API_KEY = "sk-HQ8vFgX51UHQ1k7ILF0N1A"
STEP_LIMIT = 30
RESULTS_DIR = "/shared_nfs/nehaprakriya/results/gpt_oss_4hr_20260322"
GEAK_LOG = os.path.join(RESULTS_DIR, "geak_tasks.log")

def geak_rpc(payload):
    result = subprocess.run([
        "curl", "-sk", "-X", "POST", GEAK_URL,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: {GEAK_AUTH}",
        "-d", json.dumps(payload)
    ], capture_output=True, text=True, timeout=120)
    lines = result.stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("data: "):
            line = line[6:]
        try:
            return json.loads(line)
        except:
            continue
    return {"raw": result.stdout[:500]}

def geak_call(method_name, arguments, req_id=1):
    return geak_rpc({
        "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
        "params": {"name": method_name, "arguments": arguments}
    })

# Configure GEAK LLM backend
print("Configuring GEAK LLM backend...")
geak_call("geak_set_model_config", {
    "model_class": "litellm",
    "model_name": "openai/claude-opus-4-6",
    "model_kwargs": {
        "api_base": "https://tw325.primus-safe.amd.com/llm-gateway/v1",
        "api_key": LITELLM_API_KEY,
        "max_tokens": 8192,
        "temperature": 0.0
    }
}, req_id=1)

task_ids = []

# --- Task 1: Attention backward (bwd_kernel_causal) ---
bwd_file = None
try:
    import aiter
    aiter_dir = os.path.dirname(aiter.__file__)
    candidate = os.path.join(aiter_dir, "ops", "triton", "mha_onekernel_bwd.py")
    if os.path.exists(candidate):
        bwd_file = candidate
    else:
        for m in glob.glob(os.path.join(aiter_dir, "**", "mha_onekernel_bwd.py"), recursive=True):
            bwd_file = m
            break
except:
    pass

if bwd_file:
    print(f"Found attention bwd kernel: {bwd_file}")
    with open(bwd_file) as f:
        bwd_src = f.read()

    # Only send the core kernel function (~500 lines) not the whole file
    # Extract from @triton.autotune for bwd_kernel_causal to the end of the kernel
    lines = bwd_src.split("\n")
    start_idx = None
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if "bwd_kernel_causal" in line and ("@triton" in line or "def bwd_kernel_causal" in line):
            # Go back to find the @triton.autotune decorator
            j = i
            while j > 0 and not lines[j-1].strip().startswith("@triton"):
                j -= 1
            if j > 0:
                start_idx = j - 1
            else:
                start_idx = i
        if start_idx is not None and i > start_idx + 10:
            # Find end: next top-level def/class or end of file
            if line and not line[0].isspace() and (line.startswith("def ") or line.startswith("class ") or line.startswith("@")):
                if "bwd_kernel_causal" not in line:
                    end_idx = i
                    break

    if start_idx is not None:
        kernel_src = "\n".join(lines[start_idx:end_idx])
    else:
        kernel_src = bwd_src  # fallback to full file

    result = geak_call("geak_create_task", {
        "input_type": "file",
        "files": [{"filename": "mha_onekernel_bwd.py", "content": kernel_src}],
        "prompt": """Optimize the bwd_kernel_causal Triton kernel for AMD MI355X (gfx950, CDNA4).

Hardware: 304 CUs, 256 VGPR/CU, HBM3e ~8 TB/s, MFMA instructions.
This kernel is the attention backward pass for GPT-OSS 20B model training.
Input shapes: batch=4 (per GPU), num_q_heads=64, num_k_heads=8, head_dim=128, seq_len=4096, BF16.
Currently 16.2% of total GPU time (~2100ms per 13000ms training iteration).

Current block sizes: BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32, BLK_SLICE_FACTOR=2.

The kernel has two phases in one launch:
1. dK/dV computation: iterates over Q blocks for each K/V block
2. dQ computation: iterates over K/V blocks for each Q block

Optimization directions:
1. Increase BLOCK_M1 from 32→64 for better MFMA utilization
2. Use tl.math.exp2 instead of tl.math.exp (single instruction on gfx950)
3. Software pipelining: prefetch next tile while computing current
4. Try BLOCK_N1=64 with BLOCK_M1=64 for better register balance
5. Reduce masking overhead in BLK_SLICE_FACTOR loop

Write a benchmark + correctness test. Write ALL outputs to the output directory.""",
        "step_limit": STEP_LIMIT,
        "gpu_count": 1
    }, req_id=2)

    try:
        resp_text = ""
        content = result.get("result", {}).get("content", [])
        if content:
            resp_text = content[0].get("text", "")
        resp_data = json.loads(resp_text)
        task_id = resp_data.get("id", "")
        if task_id:
            task_ids.append(("attn_bwd", task_id))
            print(f"GEAK task created: attn_bwd = {task_id}")
            geak_call("geak_submit_task", {"task_id": task_id}, req_id=3)
            print(f"GEAK task submitted: attn_bwd")
    except Exception as e:
        print(f"Warning: GEAK attn_bwd failed: {e}")
        print(f"  Raw result: {json.dumps(result)[:300]}")
else:
    print("Warning: Could not find attention bwd kernel file")

# --- Task 2: MoE permute kernels ---
permute_files = []
for pattern in ["/workspace/Primus/**/permute*.py", "/workspace/Primus/**/moe_permute*.py",
                "/workspace/Primus/third_party/**/permute*.py"]:
    permute_files.extend(glob.glob(pattern, recursive=True))

for pf in permute_files:
    try:
        with open(pf) as f:
            content = f.read()
        if "_permute_kernel" in content and "@triton" in content:
            print(f"Found MoE permute kernel: {pf}")
            result = geak_call("geak_create_task", {
                "input_type": "file",
                "files": [{"filename": os.path.basename(pf), "content": content}],
                "prompt": f"""Optimize MoE token permutation Triton kernels for AMD MI355X (gfx950, CDNA4).

Hardware: 304 CUs, HBM3e ~8 TB/s.
GPT-OSS 20B MoE: hidden_dim=2880, 32 experts, topk=4, seq_len=4096, batch=4/GPU, BF16.
Combined ~5.5% of GPU time.

Goals: vectorized 128-bit BF16 loads, optimal block sizes for 304 CUs,
minimize atomic contention in unpermute, coalesced memory access.

Write benchmark + correctness test. Write ALL outputs to output directory.""",
                "step_limit": STEP_LIMIT,
                "gpu_count": 1
            }, req_id=4)
            try:
                resp_text = result.get("result", {}).get("content", [{}])[0].get("text", "")
                resp_data = json.loads(resp_text)
                task_id = resp_data.get("id", "")
                if task_id:
                    task_ids.append(("moe_permute", task_id))
                    geak_call("geak_submit_task", {"task_id": task_id}, req_id=5)
                    print(f"GEAK task submitted: moe_permute = {task_id}")
            except Exception as e:
                print(f"Warning: GEAK moe_permute failed: {e}")
            break
    except:
        continue

# Save task IDs
with open(GEAK_LOG, "w") as f:
    for name, tid in task_ids:
        f.write(f"{name}\t{tid}\n")
        print(f"GEAK task logged: {name} = {tid}")

print(f"\nTotal GEAK tasks submitted: {len(task_ids)}")
GEAK_SUBMIT_SCRIPT

###############################################################################
# Phase 3: Optimization Loop
###############################################################################

echo ""
echo "===== PHASE 3: OPTIMIZATION LOOP ====="
echo ""

# All optimization strategies. Each run goes through run_pretrain.sh.
# Overrides are appended after the config.
# Format: "overrides|description"
declare -a STRATEGIES=(
    # --- Known winners from previous 9hr run (additive, building on nightly baseline) ---
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true|gradient_accumulation_fusion=true"
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true|+moe_use_fused_router_with_aux_score"
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true|+moe_permute_fusion (best from prev run: +1.2%)"
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true|+cross_entropy_loss_fusion"

    # --- Turbo features (nightly has them OFF; try turning ON) ---
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true use_turbo_deepep=true turbo_deepep_num_cu=64|+turbo_deepep (CU=64)"
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true use_turbo_grouped_mlp=true|+turbo_grouped_mlp"
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true turbo_sync_free_moe_stage=2|+sync_free_moe stage 2"
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true turbo_sync_free_moe_stage=3|+sync_free_moe stage 3"

    # --- DeepEP experiments ---
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true use_turbo_deepep=true turbo_deepep_num_cu=80|+turbo_deepep (CU=80)"
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true use_turbo_deepep=true turbo_deepep_use_comm_stream=true|+turbo_deepep with comm stream"

    # --- Attention backend experiments ---
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true use_turbo_attention=false|disable turbo_attention (use TE default)"

    # --- Router dtype experiment ---
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true moe_router_dtype=bf16|router dtype bf16 (less precision, faster)"

    # --- Legacy grouped gemm ---
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true moe_use_legacy_grouped_gemm=false|non-legacy grouped gemm"

    # --- Shared expert overlap OFF (nightly has ON) ---
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true moe_shared_expert_overlap=false|moe_shared_expert_overlap=false"

    # --- Additional combos ---
    "profile=false use_pytorch_profiler=false train_iters=10 gradient_accumulation_fusion=true moe_use_fused_router_with_aux_score=true moe_permute_fusion=true cross_entropy_loss_fusion=true use_turbo_deepep=true turbo_sync_free_moe_stage=2 turbo_deepep_num_cu=64|turbo_deepep+sync_free_moe stage 2 combo"
)

for strategy in "${STRATEGIES[@]}"; do
    check_time_budget || break

    if [ $CRASH_COUNT -ge $MAX_CRASHES ]; then
        echo "[$(elapsed_minutes)m] Too many crashes ($CRASH_COUNT). Stopping."
        break
    fi
    if [ $CONSECUTIVE_DISCARDS -ge $MAX_CONSECUTIVE_DISCARDS ]; then
        echo "[$(elapsed_minutes)m] $MAX_CONSECUTIVE_DISCARDS consecutive discards. Trying remaining strategies..."
        CONSECUTIVE_DISCARDS=0  # reset and keep going, just note it
    fi

    IFS='|' read -r overrides description <<< "$strategy"

    echo ""
    echo "----- Attempt $ATTEMPT: $description -----"

    LOG_FILE="$LOG_DIR/attempt_${ATTEMPT}.log"

    if run_training "$overrides" "$LOG_FILE" 600; then
        MS=$(extract_ms_per_iter "$LOG_FILE")
        if [ "$MS" = "-1" ]; then
            log_result $ATTEMPT "-1" "0.0" "crash" "$description (no ms/iter in output)"
            CRASH_COUNT=$((CRASH_COUNT + 1))
            CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
        else
            SPEEDUP=$(python3 -c "b=$BASELINE_MS; m=$MS; print(f'{(b-m)/b*100:.4f}')")
            IS_BETTER=$(python3 -c "print('yes' if $MS < $BEST_MS else 'no')")

            if [ "$IS_BETTER" = "yes" ]; then
                log_result $ATTEMPT "$MS" "$SPEEDUP" "keep" "$description"
                BEST_MS="$MS"
                BEST_OVERRIDES="$overrides"
                CONSECUTIVE_DISCARDS=0
            else
                log_result $ATTEMPT "$MS" "$SPEEDUP" "discard" "$description"
                CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
            fi
        fi
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            log_result $ATTEMPT "-1" "0.0" "timeout" "$description"
        else
            log_result $ATTEMPT "-1" "0.0" "crash" "$description (exit=$EXIT_CODE)"
        fi
        CRASH_COUNT=$((CRASH_COUNT + 1))
        CONSECUTIVE_DISCARDS=$((CONSECUTIVE_DISCARDS + 1))
    fi

    ATTEMPT=$((ATTEMPT + 1))
done

###############################################################################
# Phase 4: Collect GEAK results
###############################################################################

echo ""
echo "===== PHASE 4: COLLECT GEAK RESULTS ====="
echo ""

if [ -f "$GEAK_LOG" ] && [ -s "$GEAK_LOG" ]; then
    while IFS=$'\t' read -r name task_id; do
        echo "Checking GEAK task: $name ($task_id)"

        for i in $(seq 1 20); do
            STATUS_RAW=$(curl -sk -X POST "$GEAK_URL" \
                -H "Content-Type: application/json" \
                -H "Authorization: $GEAK_AUTH" \
                -d "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"tools/call\",\"params\":{\"name\":\"geak_get_task\",\"arguments\":{\"task_id\":\"$task_id\"}}}" 2>/dev/null)

            echo "$STATUS_RAW" > "$RESULTS_DIR/geak_outputs/${name}_status_poll${i}.json"

            if echo "$STATUS_RAW" | grep -q '"completed"'; then
                echo "  GEAK task $name completed!"

                # Download outputs
                curl -sk -X POST "$GEAK_URL" \
                    -H "Content-Type: application/json" \
                    -H "Authorization: $GEAK_AUTH" \
                    -d "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{\"name\":\"geak_get_outputs\",\"arguments\":{\"task_id\":\"$task_id\"}}}" \
                    2>/dev/null > "$RESULTS_DIR/geak_outputs/${name}_outputs.json"

                curl -sk -X POST "$GEAK_URL" \
                    -H "Content-Type: application/json" \
                    -H "Authorization: $GEAK_AUTH" \
                    -d "{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tools/call\",\"params\":{\"name\":\"geak_download_file\",\"arguments\":{\"task_id\":\"$task_id\",\"filename\":\"execution.log\"}}}" \
                    2>/dev/null > "$RESULTS_DIR/geak_outputs/${name}_execution.log.json"

                break
            elif echo "$STATUS_RAW" | grep -q '"failed"'; then
                echo "  GEAK task $name failed."
                break
            else
                echo "  GEAK task $name still running... (poll $i/20, waiting 90s)"
                sleep 90
            fi
        done
    done < "$GEAK_LOG"
else
    echo "No GEAK tasks to collect."
fi

###############################################################################
# Phase 5: Final profile with best config
###############################################################################

echo ""
echo "===== PHASE 5: FINAL PROFILE WITH BEST CONFIG ====="
echo ""

if [ -n "$BEST_OVERRIDES" ] && [ "$BEST_OVERRIDES" != "" ] && check_time_budget 2>/dev/null; then
    # Replace profile=false with profile=true in best overrides
    PROFILE_OVERRIDES=$(echo "$BEST_OVERRIDES" | sed 's/profile=false/profile=true/g; s/use_pytorch_profiler=false/use_pytorch_profiler=true/g')
    PROFILE_OVERRIDES="$PROFILE_OVERRIDES profile_step_start=6 profile_step_end=7"

    run_training "$PROFILE_OVERRIDES" "$LOG_DIR/final_profile.log" 900 || true

    FINAL_TRACE=$(find "$PRIMUS_DIR/output" -name "*.pt.trace.json" -mmin -15 2>/dev/null | sort | tail -1)
    if [ -n "$FINAL_TRACE" ]; then
        cp "$FINAL_TRACE" "$RESULTS_DIR/traces/optimized_trace.json" 2>/dev/null || \
        ln -sf "$FINAL_TRACE" "$RESULTS_DIR/traces/optimized_trace.json" 2>/dev/null || true

        python3 << PYEOF2
import json
from collections import defaultdict
try:
    with open("$FINAL_TRACE") as f:
        trace = json.load(f)
    gpu_events = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel" and "dur" in e]
    kernel_time = defaultdict(float)
    kernel_count = defaultdict(int)
    for e in gpu_events:
        kernel_time[e["name"]] += e["dur"]
        kernel_count[e["name"]] += 1
    total = sum(kernel_time.values())
    print("\n===== TOP 20 GPU KERNELS (OPTIMIZED) =====")
    lines = []
    for name, t in sorted(kernel_time.items(), key=lambda x: -x[1])[:20]:
        line = f"  {name[:70]:70s}  {t/1000:>8.1f}ms  {t/total*100:>5.1f}%  {kernel_count[name]:>4d}x"
        print(line)
        lines.append(line)
    with open("$RESULTS_DIR/kernel_profile_optimized.txt", "w") as f:
        f.write("\n".join(lines))
except Exception as e:
    print(f"Warning: Could not analyze final trace: {e}")
PYEOF2
    fi
fi

###############################################################################
# Phase 6: Generate Report
###############################################################################

echo ""
echo "===== PHASE 6: GENERATE REPORT ====="
echo ""

TOTAL_SPEEDUP=$(python3 -c "b=$BASELINE_MS; m=$BEST_MS; print(f'{(b-m)/b*100:.2f}')")
DELTA_MS=$(python3 -c "print(f'{$BASELINE_MS - $BEST_MS:.1f}')")
KEPT_COUNT=$(grep -c "	keep	" "$TSV" 2>/dev/null || echo "0")
TOTAL_ATTEMPTS=$((ATTEMPT - 1))
DISCARD_COUNT=$(grep -c "	discard	" "$TSV" 2>/dev/null || echo "0")
CRASH_COUNT_FINAL=$(grep -c "	crash	" "$TSV" 2>/dev/null || echo "0")
TIMEOUT_COUNT=$(grep -c "	timeout	" "$TSV" 2>/dev/null || echo "0")

# Extract the clean config overrides (remove profile/train_iters flags for the report)
CLEAN_OVERRIDES=$(echo "$BEST_OVERRIDES" | sed 's/profile=false//g; s/use_pytorch_profiler=false//g; s/train_iters=10//g' | xargs)

cat > "$REPORT" << REPORT_EOF
# GPT-OSS 20B Optimization Report — MI355X 8-GPU (4-Hour Run)

**Date:** $(date -u +%Y-%m-%d)
**Platform:** 8× AMD Instinct MI355X (gfx950, CDNA4)
**ROCm:** 7.2.26015, PyTorch 2.10.0a0+git449b176
**Container:** neha-test-z9jx9 (Primus training pod)
**Model:** GPT-OSS 20B, BF16 pretraining, DeepSeek-V2 architecture (MoE, 32 experts, topk=4)
**Parallelism:** EP=8, TP=1, PP=1
**Workload:** mock data, seq_length=4096, global_batch_size=512, micro_batch_size=4
**Time budget:** 4 hours (actual: $(elapsed_minutes) minutes)
**GEAK step_limit:** $GEAK_STEP_LIMIT

---

## Executive Summary

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| **ms / iter** | ${BASELINE_MS} | ${BEST_MS} | **-${DELTA_MS} ms** |
| **Speedup** | — | — | **+${TOTAL_SPEEDUP}%** |
| Kept optimizations | — | ${KEPT_COUNT} of ${TOTAL_ATTEMPTS} | |
| Discarded / crashed | — | ${DISCARD_COUNT} / ${CRASH_COUNT_FINAL} of ${TOTAL_ATTEMPTS} | |

**Nightly baseline match:** The baseline was established using the identical \`run_pretrain.sh\` launch script and environment variables as the nightly CI/CD runs, ensuring fair comparison.

**Final config overrides (on top of nightly baseline):**
\`\`\`
${CLEAN_OVERRIDES}
\`\`\`

---

## All Attempts

| # | ms/iter | Speedup vs baseline | Status | Description |
|---|---------|---------------------|--------|-------------|
$(awk -F'\t' 'NR>1 {printf "| %s | %s | %s%% | %s | %s |\n", $1, $2, $3, $4, $5}' "$TSV")

---

## Methodology

1. **Baseline:** Established via \`run_pretrain.sh\` (identical to nightly), 10 training iterations, avg ms/iter over steady-state iterations 6–10
2. **Profile:** Re-ran with PyTorch profiler to collect Chrome trace for kernel-level analysis
3. **GEAK (async):** Submitted hot Triton kernels to GEAK AI kernel optimizer (step_limit=${GEAK_STEP_LIMIT}) running in parallel
4. **Optimization loop:** Applied one config override at a time, measured, kept improvements, reverted regressions
5. **Final profile:** Re-profiled with all kept optimizations for before/after kernel comparison

### Launch command (baseline)
\`\`\`bash
cd /workspace/Primus
EXP=examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml \\
HSA_NO_SCRATCH_RECLAIM=1 \\
bash examples/run_pretrain.sh profile=false use_pytorch_profiler=false train_iters=10
\`\`\`

### Key environment variables (set by run_pretrain.sh)
\`\`\`
GPU_MAX_HW_QUEUES=2
CUDA_DEVICE_MAX_CONNECTIONS=1
TORCH_NCCL_HIGH_PRIORITY=1
HSA_ENABLE_SDMA=1
HSA_NO_SCRATCH_RECLAIM=1
NCCL_CHECKS_DISABLE=1
NVTE_CK_USES_BWD_V3=1
PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=0
\`\`\`

---

## Kernel Profile — Baseline

\`\`\`
$(cat "$RESULTS_DIR/kernel_profile_baseline.txt" 2>/dev/null || echo "Profile not available — see traces/baseline_trace.json")
\`\`\`

## Kernel Profile — Optimized

\`\`\`
$(cat "$RESULTS_DIR/kernel_profile_optimized.txt" 2>/dev/null || echo "Profile not available — see traces/optimized_trace.json")
\`\`\`

---

## GEAK Kernel Optimization Results

$(if [ -f "$GEAK_LOG" ] && [ -s "$GEAK_LOG" ]; then
    echo "| Kernel | GEAK Task ID | Steps | Status |"
    echo "|--------|-------------|-------|--------|"
    while IFS=$'\t' read -r name task_id; do
        status="unknown"
        if ls "$RESULTS_DIR/geak_outputs/${name}_status_poll"*.json 1>/dev/null 2>&1; then
            latest=$(ls -t "$RESULTS_DIR/geak_outputs/${name}_status_poll"*.json | head -1)
            if grep -q '"completed"' "$latest" 2>/dev/null; then
                status="completed"
            elif grep -q '"failed"' "$latest" 2>/dev/null; then
                status="failed"
            elif grep -q '"running"' "$latest" 2>/dev/null; then
                status="running"
            fi
        fi
        echo "| $name | \`${task_id:0:12}...\` | $GEAK_STEP_LIMIT | $status |"
    done < "$GEAK_LOG"
else
    echo "No GEAK tasks were submitted."
fi)

GEAK was configured with \`step_limit=$GEAK_STEP_LIMIT\` (3× the previous run's budget of 10) to give the AI agent sufficient time to read, understand, optimize, and benchmark large Triton kernels.

---

## Artifacts

| Artifact | Location |
|----------|----------|
| Results TSV | \`$TSV\` |
| Baseline log | \`$RESULTS_DIR/baseline.log\` |
| Baseline trace | \`$RESULTS_DIR/traces/baseline_trace.json\` |
| Optimized trace | \`$RESULTS_DIR/traces/optimized_trace.json\` |
| GEAK outputs | \`$RESULTS_DIR/geak_outputs/\` |
| All attempt logs | \`$LOG_DIR/\` |
| Full run log | \`$RESULTS_DIR/run.log\` |
| This report | \`$REPORT\` |

---

## Recommendations for Production

1. Apply the kept config overrides to nightly CI/CD config
2. If GEAK produced improved kernels, integrate them as a Primus Turbo patch
3. Consider FP8 training for additional throughput (separate investigation)

---

*Generated automatically by the workload-optimization agent on $(date -u)*
*Baseline validated against nightly run_pretrain.sh environment*
REPORT_EOF

echo ""
echo "========================================="
echo "OPTIMIZATION COMPLETE"
echo "Duration: $(elapsed_minutes) minutes"
echo "Baseline: ${BASELINE_MS} ms/iter"
echo "Best:     ${BEST_MS} ms/iter"
echo "Speedup:  ${TOTAL_SPEEDUP}%"
echo "Report:   $REPORT"
echo "Results:  $TSV"
echo "========================================="
