---
name: inference-optimization
description: Closed-loop LLM inference optimization on AMD Instinct GPUs. Profiles the serving engine with torch.profiler, diagnoses bottlenecks with TraceLens MCP, optimizes hot kernels with GEAK MCP, patches them back into the framework, and re-benchmarks to verify improvement. Iterates until no more gains, then sweeps parameters for full Pareto curves. Use when the user asks to optimize inference throughput/latency, find serving bottlenecks, or improve tok/s on a model.
---

# LLM Inference Optimization — TraceLens + GEAK Closed Loop

## Overview

This is an **agent-driven inference optimization loop**. You (the agent) will:

1. Set up the environment and run a baseline benchmark
2. Profile and extract hot kernels (Inductor-generated or framework source)
3. **Enter a multi-round optimization loop**: identify hot kernels → GEAK optimize (with iterative prompt refinement) → patch → re-benchmark → keep/revert → re-profile → repeat
4. After the loop, sweep parameters (CONC, ISL/OSL) with the optimized version
5. Write a report with optimization history, Pareto curves, and InferenceX comparison

**CRITICAL LESSON (validated 2026-03-20/21):** When torch.compile works (most models), use it — Inductor Triton kernels are guaranteed on the hot path. But torch.compile is **NOT universally compatible** — DSR1's MLA + FP8 and gpt-oss's SWA both cause CUDA graph capture failure. When torch.compile fails, fall back to Strategy B (direct source edit on aiter/framework Triton kernels), but expect limited GEAK gains since vendor kernels are already optimized.

**CRITICAL LESSON (validated 2026-03-22):** When GEAK kernel optimization fails AND torch.compile is unavailable, **CUDA graph coverage + server parameter tuning** can still deliver significant E2E improvement. Always check `--cuda-graph-max-bs` matches the actual decode batch size — the default is often too low, causing most decode steps to skip CUDA graph and re-launch every kernel individually.

## Autonomy Rules

**This skill runs end-to-end without human confirmation.** Do NOT ask the user before:
- Running baseline/profiling scripts
- Submitting GEAK tasks
- Killing/restarting servers
- Patching kernels (Inductor cache or source files)
- Reverting failed patches

Execute the full pipeline autonomously. Present only the **final optimization report** to the user. If a step fails, fix it and retry (up to 3 times per step) before moving on.

**Default parameters** (auto-detect when user does not specify):

| Parameter | Default | Auto-detection |
|-----------|---------|----------------|
| MODEL | First model in `/shared_nfs/*/models/` | `ls /shared_nfs/*/models/ &#124; head -1` |
| TP | GPU count (capped at model requirement) | `amd-smi list &#124; grep "^GPU:" &#124; wc -l` |
| CONC | 4 (TP=1), 32 (TP=4), 64 (TP=8) | Based on TP |
| ISL | 1024 | — |
| OSL | 256 | — |
| FRAMEWORK | `sglang` | User-specified; `sglang` or `vllm` |
| INFERENCEX_PATH | `/shared_nfs/xiaofei/InferenceX` | `ls /shared_nfs/*/InferenceX/ &#124; head -1` |
| torch.compile | SGLang: try with 0.6 mem-fraction; vLLM: enabled by default (level=3) | Framework-dependent |

## Phase 0: Model Classification & Strategy Selection (NEW — validated 2026-03-22)

**Run this BEFORE any benchmarks.** Determines whether GEAK kernel optimization is worth attempting, and which optimization strategy to prioritize.

### Step 0 (MANDATORY): Search for official/CI test configurations FIRST

**CRITICAL LESSON (validated 2026-03-23 on Kimi-K2.5):** Do NOT blindly guess server launch parameters. Many models have non-obvious compatibility constraints (e.g., MLA head count divisibility, split prefill/decode backends, env var overrides from Docker). Guessing wastes 30+ minutes on failed launches.

**BEFORE launching any server, search for existing test configurations:**

```bash
# Search SGLang's test suite for this model's official config
MODEL_TYPE=$(python3 -c "import json; c=json.load(open('$MODEL/config.json')); print(c.get('model_type',''))")
find /sgl-workspace/sglang/test -name "*.py" | xargs grep -il "$MODEL_TYPE\|$(basename $MODEL)" 2>/dev/null

# Extract launch args and env vars from test files
for f in $(find /sgl-workspace/sglang/test -name "*.py" -exec grep -il "$MODEL_TYPE\|$(basename $MODEL)" {} \;); do
    echo "=== $f ==="
    grep -A 5 "other_args\|env\[" "$f" | head -30
done
```

**What to extract from test configs:**
- `--decode-attention-backend` / `--prefill-attention-backend` (may be DIFFERENT backends!)
- `--attention-backend` (unified backend, simpler but may not work for all models)
- Environment variables: `SGLANG_ROCM_FUSED_DECODE_MLA`, `SGLANG_USE_AITER`, etc.
- `--kv-cache-dtype` (some models crash with fp8 KV cache)
- `--trust-remote-code` (required for custom model code)
- Any model-specific workarounds

**Kimi-K2.5 / Kimi-K2 validated config (2026-03-23, MI35x):**
```bash
# MUST use split prefill/decode backends — unified --attention-backend fails
# Reason: MLA with TP=8 gives 8 heads/partition; aiter decode requires ≥16 heads,
# but aiter prefill works fine. Triton decode handles 8 heads correctly.
--decode-attention-backend triton --prefill-attention-backend aiter
# MUST disable fused decode MLA (Docker sets SGLANG_ROCM_FUSED_DECODE_MLA=1 by default)
export SGLANG_ROCM_FUSED_DECODE_MLA=0
# Do NOT use --kv-cache-dtype fp8_e4m3 (MLA ASM kernel assertion failure)
# Do NOT use unified --attention-backend aiter (num_head_qo % 16 assert)
# Do NOT use unified --attention-backend triton (MLA_FUSED_ROPE dispatch bug when FUSED_MLA=1)
```

**For vLLM, search for model-specific configurations:**

```bash
# Search vLLM's tests and examples for this model's config
VLLM_PKG=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
find "$VLLM_PKG/../.." -path "*/tests/*" -name "*.py" | xargs grep -il "$MODEL_TYPE\|$(basename $MODEL)" 2>/dev/null | head -10

# Search for model-specific args in vLLM source (e.g., block_size constraints for MLA)
grep -r "block.size\|block_size" "$VLLM_PKG/model_executor/models/" 2>/dev/null | grep -i "$MODEL_TYPE" | head -5

# Check vLLM's supported model list and any special flags
python3 -c "from vllm.config import ModelConfig; help(ModelConfig)" 2>/dev/null | grep -i "block\|mla\|eager" | head -10
```

**What to extract from vLLM test configs:**
- `--block-size` (MLA models may require `--block-size 1`)
- `--enforce-eager` (if torch.compile/CUDA graph fails)
- `--max-model-len` (may need capping for large models)
- `--gpu-memory-utilization` (model-dependent optimal value)
- `--max-num-seqs` (concurrency cap)
- Environment variables: `VLLM_ROCM_USE_AITER`, `AITER_ENABLE_VSKIP`, `NCCL_MIN_NCHANNELS`

**If no test config found:** Check the model's README, HuggingFace page, or framework documentation for recommended serving parameters before attempting manual configuration.

### Step 1: Classify model architecture

```bash
# Check model config for architecture hints (handles nested configs like VLMs)
python3 -c "
import json, sys
config = json.load(open('$MODEL/config.json'))
# For VLMs (e.g., Kimi-K2.5), the text backbone config is nested under 'text_config'
text_cfg = config.get('text_config', config)
arch = text_cfg.get('architectures', config.get('architectures', ['']))[0]
has_mla = text_cfg.get('kv_lora_rank', 0) > 0 or text_cfg.get('q_lora_rank', 0) > 0
has_moe = text_cfg.get('n_routed_experts', 0) > 0 or text_cfg.get('num_local_experts', 0) > 0 or text_cfg.get('num_experts', 0) > 0
has_swa = 'sliding_attention' in str(text_cfg.get('layer_types', [])) or text_cfg.get('sliding_window_size', 0) > 0
hidden = text_cfg.get('hidden_size', 0)
n_heads = text_cfg.get('num_attention_heads', 0)
print(f'Architecture: {arch}')
print(f'MoE: {has_moe} | MLA: {has_mla} | SWA: {has_swa} | Hidden: {hidden} | Heads: {n_heads}')
if has_mla and n_heads > 0:
    # MLA models need heads_per_tp % 16 == 0 for aiter decode; check TP compatibility
    for tp in [1,2,4,8]:
        hpt = n_heads // tp
        ok = 'OK' if hpt % 16 == 0 else 'NEED split backend (decode=triton, prefill=aiter)'
        print(f'  TP={tp}: {hpt} heads/partition -> {ok}')
if has_swa:
    print('WARNING: SWA models are torch.compile/FP8-KV/aiter-attn INCOMPATIBLE')
    print('STRATEGY: CUDA graph coverage + server params + attn kernel tuning (see gpt-oss-120b)')
elif has_mla:
    print('WARNING: MLA models are likely torch.compile INCOMPATIBLE (MLA+FP8 CUDA graph failure)')
    print('STRATEGY: Skip torch.compile, minimize GEAK attempts, focus on server parameter tuning')
    print('NOTE: Check official test config FIRST (Step 0) — MLA models often need split attention backends')
"
```

### Step 2: Strategy decision tree

| Model Type | torch.compile | GEAK Expected | Primary Strategy |
|-----------|--------------|--------------|-----------------|
| Dense | Try first | High (Inductor kernels) | torch.compile + GEAK (Strategy A) |
| MoE without MLA, no SWA | Try first | Medium | torch.compile + GEAK if Inductor works |
| **MoE + SWA** | **Incompatible** | **Low** | **CUDA graph coverage + backend exploration + server params** |
| **MoE + MLA** | **Skip** | **Low (~0-2%)** | **Backend exploration → server param tuning + 1 GEAK round** |
| **MoE + MLA + custom attention (e.g. NSA)** | **Skip** | **Low** | **Backend exploration → kernel tuning → combined testing** |
| Any model after vendor kernel >50% | — | Skip GEAK | **Backend exploration → server parameter tuning** |

**CRITICAL LESSON:** For models where vendor kernels dominate (>50% GPU time), the biggest gains come from **switching kernel backends and enabling scheduling modes**, NOT from parameter sweeps. Backend switches can change which GPU kernels are actually executed, while parameter sweeps only adjust batch sizes and scheduling thresholds. Individual backend switches may each give small gains (+2-5%), but **combining multiple winners often produces super-linear synergy (+10-20%)** because they affect different pipeline stages that amplify each other. **Always explore backends and scheduling modes BEFORE parameter sweeps.** See [`KNOWLEDGE-BASE.md`](KNOWLEDGE-BASE.md) for model-specific validated results.

**If model has SWA (Sliding Window Attention)**: torch.compile, FP8 KV cache, and aiter attention backend are all incompatible. Focus on: (1) CUDA graph max-bs expansion, (2) backend exploration, (3) server param tuning. See [`KNOWLEDGE-BASE.md`](KNOWLEDGE-BASE.md) for details.

**If model is MoE+MLA or has custom attention mechanisms**: Skip Phase 2's torch.compile attempt entirely. Run baseline without it, then immediately proceed to **Phase 5.5: Backend & Code Exploration** — this is where the major gains are. **Still attempt at least 1 round of GEAK** on the top Triton kernel candidate after backend exploration.

### Step 3: CUDA graph coverage check (NEW — always run)

**After baseline launch, ALWAYS verify CUDA graph coverage.** This is the most impactful non-kernel optimization (+35% validated on gpt-oss-120b):

```bash
# Check what batch sizes are captured
grep "cuda_graph_bs\|Capture cuda graph" $RESULT_DIR/server_baseline.log
# Compare with actual decode batch sizes from benchmark
# If max captured bs < typical decode bs → increase --cuda-graph-max-bs
```

**Rule of thumb**: Set `--cuda-graph-max-bs` to at least `CONC` (the max concurrency being benchmarked). Default is often 4-8, but benchmarks at CONC=16+ need `--cuda-graph-max-bs 16` or higher.

## Phase 1: Setup

### Step 1: Auto-detect environment

Run these commands automatically (do NOT ask the user):

```bash
# Auto-detect all parameters
MODEL=$(ls -d /shared_nfs/*/models/*/ 2>/dev/null | head -1)
GPU_COUNT=$(amd-smi list 2>/dev/null | grep "^GPU:" | wc -l)
GPU_TYPE=$(rocm-smi --showproductname 2>/dev/null | grep "GFX Version" | head -1 | grep -o "gfx[0-9]*")
INFERENCEX_PATH=$(ls -d /shared_nfs/*/InferenceX 2>/dev/null | head -1)

# Framework: sglang (default) or vllm
FRAMEWORK="${FRAMEWORK:-sglang}"
if [ "$FRAMEWORK" = "vllm" ]; then
    FRAMEWORK_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
else
    FRAMEWORK_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null)
fi

# Set TP based on model size
# Qwen3-8B: TP=1, Qwen3-30B: TP=1, Qwen3-235B: TP=8, DeepSeek-R1: TP=8
# Default: use all GPUs
TP=$GPU_COUNT

# Set CONC based on TP
if [ "$TP" -le 1 ]; then CONC=4; elif [ "$TP" -le 4 ]; then CONC=32; else CONC=64; fi
```

If the user specifies a model or parameters, use those instead of auto-detected values.

### Step 2: Set paths and env vars

All scripts and outputs use fixed paths on shared NFS:

```bash
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/inference-optimization}"
SCRIPTS_DIR="$SKILL_ROOT/scripts"

# If the user specified MODEL/TP/CONC/ISL/OSL/FRAMEWORK explicitly, those values override auto-detection.
export MODEL="$MODEL"
export TP="$TP"
export CONC="$CONC"
export ISL="${ISL:-1024}"
export OSL="${OSL:-256}"
export FRAMEWORK="${FRAMEWORK:-sglang}"
export INFERENCEX_PATH="$INFERENCEX_PATH"

# Framework-specific extra args (user can override)
# SGLang: SGLANG_EXTRA_ARGS="--attention-backend aiter --kv-cache-dtype fp8_e4m3"
# vLLM:   VLLM_EXTRA_ARGS="--max-model-len 4096 --kv-cache-dtype fp8_e4m3"
```

`run_baseline.sh` writes `run_context.env` into the current `RESULT_DIR`. Reuse that file whenever a later step needs to attach to the same server instance or trace directory.

## Phase 2: Baseline Benchmark

**Try torch.compile first, then fall back if incompatible.**

### Step 1: Try with torch.compile

**For SGLang:**
```bash
export FRAMEWORK=sglang
export SGLANG_EXTRA_ARGS="--enable-torch-compile --mem-fraction-static 0.6 --chunked-prefill-size 32768 --max-prefill-tokens 32768"
bash "$SCRIPTS_DIR/run_baseline.sh"
```

**NOTE on `--mem-fraction-static`:** The optimal value is model-dependent. Check the framework's official test configs (Phase 0 Step 0) for the recommended value. As a rule of thumb, torch.compile needs extra memory during compilation — use a lower value (e.g. 0.6) than without torch.compile (e.g. 0.8). The `run_baseline.sh` script auto-selects 0.6 when `--enable-torch-compile` is in `SGLANG_EXTRA_ARGS`, override via `MEM_FRACTION` env var.

**For vLLM (torch.compile enabled by default at level=3):**
```bash
export FRAMEWORK=vllm
export VLLM_EXTRA_ARGS="--max-model-len 4096"
bash "$SCRIPTS_DIR/run_baseline.sh"
```

vLLM v0.9+ enables torch.compile + CUDA graph by default. Use `VLLM_EXTRA_ARGS` for model-specific overrides:
- `--max-model-len N` — cap context length (reduces memory)
- `--gpu-memory-utilization 0.85` — GPU memory fraction (default in script: 0.85)
- `--kv-cache-dtype fp8_e4m3` — FP8 KV cache
- `--enforce-eager` — disable torch.compile + CUDA graph (for debugging)

### Step 2: Check for torch.compile failure

Monitor the server log for these errors during CUDA graph capture:

| Error pattern | Cause | Action |
|---------------|-------|--------|
| `get_heuristic_kernel_mla: cannot get heuristic kernel! q_type:fp8` | MLA + FP8 incompatible with torch.compile | Fall back |
| `CUDA error: out of memory` during Triton compilation | Model too large for 0.6 mem fraction | Try 0.5, then fall back |
| `Triton compilation failed` / `inductor error` | Unsupported op in compute graph | Fall back |

**If torch.compile succeeds:** Continue normally. First run takes ~5-8 min (Inductor compilation). Kernels are cached in `/tmp/torchinductor_root/`. **Do NOT delete this cache** — GEAK patches go there.

**If torch.compile fails (SGLang):** Remove `--enable-torch-compile` and rerun:

```bash
export SGLANG_EXTRA_ARGS="--chunked-prefill-size 196608 --max-prefill-tokens 196608 --mem-fraction-static 0.8"
bash "$SCRIPTS_DIR/run_baseline.sh"
```

**If torch.compile fails (vLLM):** Disable compilation:

```bash
export VLLM_EXTRA_ARGS="--max-model-len 4096 --enforce-eager"
bash "$SCRIPTS_DIR/run_baseline.sh"
```

Log in results.tsv: `torch_compile=failed, reason=<error>`. Proceed to Phase 3. In Phase 5, use Strategy B (framework source kernels) instead of Strategy A (Inductor cache). **GEAK gains will be limited** — without torch.compile, most compute is in vendor C++ kernels that GEAK cannot optimize.

The script outputs:
- `$RESULT_DIR/baseline_sglang_tp8_conc64_isl1024_osl1024.json` — benchmark results
- `$RESULT_DIR/server_baseline.log` — server log
- Server stays running (for Phase 3 profiling)

Record baseline metrics to `results.tsv`:
```
0	2768.59	output_tok/s	0.0	baseline	CONC=64 ISL=1024 OSL=1024
```

## Phase 3: Profile with torch.profiler

### ⚠️ vLLM V1 Profiling Caveat (validated 2026-03-23)

**vLLM V1 uses `multiprocessing.spawn` for GPU workers.** `torch.profiler` in the main process does NOT capture GPU kernel activity from child worker processes — the resulting trace will be near-empty (~11KB). This affects both `VLLM_TORCH_PROFILER_DIR` and the `/start_profile` HTTP endpoint.

**Workaround options for vLLM:**
1. **Use `/start_profile` + `/stop_profile` HTTP endpoints** — vLLM V1 routes these to the correct worker process internally. This is the recommended approach and should work if `VLLM_TORCH_PROFILER_DIR` is set before server launch.
2. **If HTTP profiling also fails**, skip to **Phase 5 Architecture-Based Kernel Identification** (no trace needed). Identify GEAK candidates based on model architecture knowledge instead.
3. **Do NOT use `rocprofv3` as fallback** — while `rocprofv3` can trace HIP API calls, it does NOT trace child processes by default, so vLLM worker GPU kernels are still invisible. Additionally, TraceLens cannot parse `rocprofv3` JSON format (see TraceLens Tips in Knowledge Base).

**rocprofv3 reference (for advanced users only):**
```bash
# rocprofv3 syntax (does NOT trace child processes — limited value for vLLM/SGLang):
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace -o /tmp/trace/output -- python3 script.py
# Common syntax mistakes:
#   ✗ rocprof --hip-trace -- python3 ...     (rocprof v1 syntax, deprecated)
#   ✗ rocprofv3 --roctx-trace ...            (use --marker-trace instead)
#   ✗ rocprofv3 --hip-trace python3 ...      (missing -- before command)
# Output is rocprofv3 JSON format, NOT PyTorch Kineto — TraceLens CANNOT parse it.
```

**NOTE: `run_baseline.sh` already handles Phase 2 AND Phase 3 in one run.** It pre-sets `SGLANG_TORCH_PROFILER_DIR` (or `VLLM_TORCH_PROFILER_DIR`) at server launch, runs the clean baseline (profiler inactive), then activates profiling via `/start_profile` HTTP endpoint — all without restarting.

When `run_profile.sh` is used separately against an already running server, it must reuse the original context. Pass either:
- `RUN_CONTEXT_FILE="$RESULT_DIR/run_context.env"`
- or the exact same `RESULT_DIR` / `TRACE_DIR` used by `run_baseline.sh`

Do **NOT** let `run_profile.sh` invent a fresh timestamped `TRACE_DIR` for an existing server — the profiler writes to the directory configured at server launch time.

If running manually (not using the script), `SGLANG_TORCH_PROFILER_DIR` must be set BEFORE launching the server.

### Manual profiling (if not using run_baseline.sh)

```bash
# Kill baseline server
ps aux | grep "python3 -m sglang" | grep -v grep | grep -v bash | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 8

# Set profiler output dir (on shared NFS)
TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
export SGLANG_TORCH_PROFILER_DIR="/shared_nfs/inference-optimization/traces/${TIMESTAMP}"
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

# Relaunch server with profiler env var active (same params as baseline)
python3 -m sglang.launch_server \
    --attention-backend aiter --model-path "$MODEL" \
    --host=0.0.0.0 --port 8888 --tensor-parallel-size $TP \
    --trust-remote-code --chunked-prefill-size 196608 \
    --mem-fraction-static 0.8 --disable-radix-cache \
    --num-continuous-decode-steps 4 --max-prefill-tokens 196608 \
    --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs "$CONC" \
    > /shared_nfs/inference-optimization/results/${TIMESTAMP}/server_profile.log 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port 8888 --server-log ... --server-pid $SERVER_PID
```

### Step 2: Run profiling benchmark

```bash
RUN_CONTEXT_FILE="$RESULT_DIR/run_context.env" bash "$SCRIPTS_DIR/run_profile.sh"
```

Or manually:
```bash
export PROFILE=1
run_benchmark_serving ... --num-prompts $CONC ...
unset PROFILE SGLANG_TORCH_PROFILER_DIR
```

**Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR` after profiling.** Leaking `PROFILE=1` causes 30x slowdown and caps `num_prompts` to CONC in all subsequent benchmarks.

### Trace output

SGLang generates per-TP, per-phase traces directly to `/shared_nfs/inference-optimization/traces/<timestamp>/`:

```
/shared_nfs/inference-optimization/traces/2026-03-19-17-38/
├── {id}-TP-0-EXTEND.trace.json.gz    (~6.4MB, prefill)
├── {id}-TP-0-DECODE.trace.json.gz    (~80KB, decode)
├── ... (per TP rank)
└── merged-{id}.trace.json.gz         (~51MB, all ranks merged)
```

Traces are already on shared NFS — no copy needed for TraceLens.

## Phase 4: TraceLens Analysis

Traces are already on shared NFS (output of Phase 3). **Use a single TP-0 trace for TraceLens analysis** — merged traces can exceed TraceLens memory limits even with 64GB (331MB compressed → 44GB peak with duplicate loads). For most models, a single TP trace captures the same kernel breakdown.

Call TraceLens MCP on the **filtered** trace (generated by `run_baseline.sh`):

```
Tool: check_trace_file
Args: { "trace_path": "/shared_nfs/inference-optimization/traces/<timestamp>/filtered-TP-0.trace.json.gz" }

Tool: run_full_standalone_analysis
Args: {
    "trace_path": "/shared_nfs/inference-optimization/traces/<timestamp>/filtered-TP-0.trace.json.gz",
    "platform": "MI355X",
    "trace_type": "pytorch",
    "cleanup": false,
    "output_dir": "/shared_nfs/inference-optimization/traces/<timestamp>/tracelens_output"
}
```

### Interpreting TraceLens output

The output contains `categories` with `gpu_kernel_time_ms` for each group. Example from DeepSeek-R1:

| Category | GPU time (ms) | % | GEAK candidate? |
|----------|-------------|---|-----------------|
| record_param_comms | 51.72 | 60.7% | No (NCCL comms) |
| MoE Fused | 21.25 | 24.9% | No (aiter vendor) |
| elementwise | 4.85 | 5.7% | Maybe |
| other | 5.21 | 6.1% | Check individually |
| GEMM | 1.78 | 2.1% | No (hipBLASLt) |
| triton | 0.007 | 0.01% | Yes but negligible |

Also check `gpu_utilization`:
- `computation_time_percent` — higher is better (85%+ is good)
- `idle_time_percent` — scheduling overhead
- `exposed_comm_time_percent` — communication not overlapped with compute

## Phase 5: Identify GEAK Candidates

Scan the kernel profile for GEAK candidates. If TraceLens was used, check its categories. Otherwise, parse the trace directly with Python (as fallback when TraceLens is unavailable or trace is too large):

```python
# Direct trace parsing with error recovery (handles truncated gzip files)
import gzip, json, time, os

# Wait for trace file to stabilize (SGLang profiler may still be writing)
for _ in range(6):
    size1 = os.path.getsize(trace_path) if os.path.exists(trace_path) else 0
    time.sleep(10)
    size2 = os.path.getsize(trace_path) if os.path.exists(trace_path) else 0
    if size1 == size2 and size2 > 0:
        break

# Try filtered trace first (much smaller), fall back to raw trace
filtered = trace_path.replace('.trace.json.gz', '-filtered.trace.json.gz')
if not os.path.exists(filtered):
    filtered = os.path.join(os.path.dirname(trace_path), 'filtered-TP-0.trace.json.gz')
actual_path = filtered if os.path.exists(filtered) else trace_path

try:
    with gzip.open(actual_path) as f:
        trace = json.load(f)
except EOFError:
    # Truncated gzip — read what we can
    print("WARNING: Trace file truncated, attempting partial read...")
    with gzip.open(actual_path, 'rb') as f:
        raw = f.read(500*1024*1024)
    text = raw.decode('utf-8', errors='replace')
    last = text.rfind('}')
    if last > 0:
        text = text[:last+1] + ']}'
    trace = json.loads(text)

kernels = {}
for e in trace.get('traceEvents', []):
    if e.get('cat') == 'kernel':
        name = e.get('name', '')
        kernels.setdefault(name, {'count': 0, 'total_us': 0})
        kernels[name]['count'] += 1
        kernels[name]['total_us'] += e.get('dur', 0)
total = sum(v['total_us'] for v in kernels.values())
for name, v in sorted(kernels.items(), key=lambda x: -x[1]['total_us']):
    pct = v['total_us'] / total * 100
    if pct < 3: break
    is_vendor = any(x in name for x in ['Cijk_', 'aiter::', 'hipModule', 'ck::kernel'])
    if not is_vendor:
        print(f"GEAK candidate: {name} ({pct:.1f}%)")
```

### Fallback: Architecture-based GPU breakdown (when profiling/TraceLens fails)

**If both torch.profiler AND TraceLens fail** (e.g., vLLM multiprocessing trace empty, rocprofv3 format incompatible), use model architecture to estimate GPU breakdown and identify GEAK candidates:

```python
# Architecture-based GPU breakdown estimation
import json
config = json.load(open(f'{MODEL}/config.json'))
text_cfg = config.get('text_config', config)
has_moe = text_cfg.get('n_routed_experts', 0) > 0 or text_cfg.get('num_local_experts', 0) > 0
has_mla = text_cfg.get('kv_lora_rank', 0) > 0
quant = config.get('quantization_config', {}).get('quant_method', 'none')

if has_moe and quant in ('compressed-tensors', 'gptq', 'awq'):
    print("Estimated GPU breakdown (MoE + INT4 quant):")
    print("  MoE GEMM (INT4 dequant + BF16 matmul): ~75-85%  → GEAK: limited (compute-bound)")
    print("  Attention (MLA decode):                 ~5-10%   → GEAK: limited (vendor kernel)")
    print("  Normalization + Routing:                ~3-5%    → GEAK: maybe")
    print("  TP Communication:                       ~3-5%    → GEAK: no")
    print("  GEAK target: fused_moe_kernel (Triton, in vLLM/SGLang source)")
elif has_moe:
    print("Estimated GPU breakdown (MoE + FP8/BF16):")
    print("  MoE GEMM:          ~50-60%  → GEAK: no (CK/aiter vendor)")
    print("  Attention:          ~10-15%  → GEAK: limited")
    print("  Elementwise/Norm:   ~10-20%  → GEAK: maybe (if Triton)")
    print("  Communication:      ~10-15%  → GEAK: no")
else:
    print("Estimated GPU breakdown (Dense model):")
    print("  GEMM (QKV/O/FFN):  ~60-70%  → GEAK: no (hipBLASLt)")
    print("  Attention:          ~10-15%  → GEAK: limited")
    print("  RMSNorm/Act:        ~5-10%   → GEAK: yes (if Inductor Triton)")
    print("  Communication:      ~5-10%   → GEAK: no")
```

Use the estimated breakdown to select GEAK candidates, then proceed to Phase 6 (server tuning) before Phase 7 (GEAK optimization). **Always verify with at least 1 A/B benchmark** — architecture-based estimates can be off by 2-3x.

**Decision table:**

| Kernel pattern | GEAK? | Why |
|----------------|-------|-----|
| `Cijk_*` (hipBLASLt GEMM) | No | Vendor BLAS, hand-tuned MFMA |
| `aiter::fmha_v3_*` | No | Vendor MLA attention |
| `moe_ck2stages_gemm*` | No | aiter fused MoE |
| `ck::kernel_moe_gemm` | **Maybe** | CK library MoE — check if source modifiable |
| `triton_*` / `_permute_kernel` | **Yes** | Triton kernels with Python source |
| `paged_attention_ll4mi_*` | **Maybe** | Custom paged attention — check source |
| `topkGatingSoftmax` | **Yes** | MoE routing kernel |
| `vectorized_elementwise_kernel` | **Maybe** | If >5% GPU time |
| Custom scheduling/routing kernels | **Yes** | Token dispatch, KV cache ops |

**If torch.compile fails** (e.g., DSR1 MLA FP8 incompatibility): Fall back to Strategy B. Search for aiter Triton kernels (`find /sgl-workspace/aiter -name "*.py" -exec grep -l "@triton.jit" {} \;`). Even with 30+ parameter signatures, the kernel BODY may be simple enough for GEAK — submit with explicit "DO NOT change function signature" constraint. But be aware: aiter kernels are already AMD-engineer-optimized, GEAK gains are unlikely (validated -0.55% on DSR1).

**EARLY EXIT (validated 2026-03-22 on DSR1):** If TraceLens shows >50% GPU time in vendor C++/ASM kernels (CK, aiter, hipModule) AND torch.compile is unavailable, **ensure Phase 6 (Server Parameter Tuning) is completed before GEAK**. On DSR1, GEAK optimization produced zero E2E improvement (GEMM -19.9% regression, fused_rms 0%) because vendor kernels are already highly optimized. Server parameter tuning (decode-steps, scheduling) is more productive. After server tuning, re-profile and then attempt GEAK on the new baseline — kernel bottleneck rankings may shift.

**If NO candidates found** (all vendor kernels): skip to Phase 6 (server parameter tuning), then Phase 9 (parameter sweep). Log:
```
-	-	-	-	skipped	No GEAK candidates: 90%+ GPU time in vendor kernels
```

**EXHAUSTIVE SEARCH REQUIRED before declaring "no candidates":**
Do NOT dismiss a kernel just because it has many parameters (e.g., 37 params). Parameter count ≠ optimization difficulty — what matters is:
1. Is it a `@triton.jit` kernel with Python source? → Potentially optimizable
2. What % of GPU time does it consume? → >3% is worth trying
3. Is the kernel body doing redundant memory ops? → GEAK can fix this

**Minimum search checklist before declaring "no candidates":**
- [ ] Checked TraceLens categories for non-vendor kernels
- [ ] Searched Inductor cache: `find /tmp/torchinductor_root -name "*.py" | xargs grep -l "@triton"` (torch.compile mode)
- [ ] Searched framework source: `find /opt/venv -path "*/sglang/*" -name "*.py" -exec grep -l "@triton.jit" {} \;`
- [ ] Searched aiter source: `find /sgl-workspace/aiter -name "*.py" -exec grep -l "@triton.jit" {} \;`
- [ ] Verified that ALL kernels >3% GPU time are vendor C++ (not Triton)
- [ ] **Searched aiter for FUSED kernels** that could replace multi-step pipelines: `find /sgl-workspace/aiter -name "*.py" -path "*/triton/*" | xargs grep -l "fused\|routing"`. Even if a fused kernel is NOT currently on the hot path, it may be a valuable GEAK target if it can replace a multi-kernel pipeline (e.g., separate gate_linear + sigmoid + topk → single `_routing_sigmoid_top1_kernel`). Validated: this approach produced +10.1% E2E on DSR1.

Only after ALL checkboxes pass can you skip to Phase 6 (server tuning) → Phase 9 (sweep).

**If candidates found**: rank by **patch safety AND GPU time %**:
1. **Fused pipeline kernels** (routing, activation+quant) — HIGHEST priority. Replace multi-kernel pipelines with single optimized kernel. Validated +10.1% E2E on DSR1.
2. **Reduction kernels** (`triton_red_*`, `triton_per_*`) — HIGH priority. Grid does NOT change, safest to patch.
3. **Pointwise kernels** (`triton_poi_*`) — Medium priority. Grid doesn't change.
4. **Template/GEMM kernels** (`triton_tem_*`) — LOWEST priority. Grid changes with block sizes, risky. Inductor autotuner already picks near-optimal block sizes; GEAK block tuning rarely helps E2E. **WARNING (validated 2026-03-22)**: GEMM software pipelining (load next tile while computing current) can show 1.4-2.3x micro-benchmark speedup but **-19.9% E2E regression** due to increased register pressure → lower occupancy. Prefer optimizations that REDUCE register usage over those that increase instruction-level parallelism.

Select **top 5 candidates** (by GPU time %) with reduction kernels first. Proceed to Phase 6 (server tuning), then Phase 7 (GEAK optimization). For each candidate, submit to GEAK with the appropriate prompt template (RMSNorm single-pass for dual-loop reduction kernels, general template for others).

## Phase 5.5: Backend & Code Exploration (CRITICAL — do before param sweeps)

**When to run**: AFTER baseline benchmark (Phase 2) and profiling (Phase 3-5). BEFORE Phase 6 parameter tuning. This phase typically delivers **much larger improvements** than parameter sweeps on vendor-kernel-dominated models.

**Rationale:** Parameter sweeps adjust numerical thresholds (batch sizes, memory fractions, channel counts) — these typically yield <1-3% individually. Backend switches change which actual GPU kernels and scheduling algorithms are used — these can yield 3-10%+ individually, and **combining multiple winners often produces super-linear synergy** because they affect different pipeline stages (kernel speed × scheduling efficiency). The agent should systematically discover and test ALL available backends and scheduling modes before touching numerical parameters.

### Step 1: Discover all backend and scheduling flags (MANDATORY)

Read the server args to find every backend switch, scheduling mode, and feature flag relevant to the model:

```bash
# Extract ALL backend-related flags from server_args.py
python3 -c "
import ast, sys
source = open('/sgl-workspace/sglang/python/sglang/srt/server_args.py').read()
tree = ast.parse(source)
backend_keywords = ['backend', 'enable_', 'disable_', 'fused', 'mixed', 'overlap', 'schedule', 'allreduce', 'fusion']
for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and any(kw in node.attr for kw in backend_keywords):
        print(f'  --{node.attr.replace(\"_\", \"-\")}')
" 2>/dev/null | sort -u | head -40

# Also check the argparse definitions directly for help text
grep -E "add_argument.*backend|add_argument.*enable-|add_argument.*fused|add_argument.*mixed|add_argument.*overlap" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -30
```

### Step 2: Identify model-specific backends

Based on the model architecture detected in Phase 0, find which backends have alternatives:

```bash
# For attention backends
grep -r "attention_backend\|decode_attention_backend\|prefill_attention_backend" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -10

# For NSA (Native Sparse Attention) models
grep -r "nsa_prefill_backend\|nsa_decode_backend" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -5

# For MoE models
grep -r "moe_runner_backend\|moe_a2a_backend" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -5

# For GEMM runners
grep -r "fp8_gemm_runner_backend\|fp4_gemm_runner_backend" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -5
```

### Step 3: Read the model forward pass (MANDATORY for TP>1)

Understanding the model's actual compute/communication pattern is essential. This takes 10 minutes but saves hours of blind sweeping:

```bash
# Find the model implementation
MODEL_TYPE=$(python3 -c "import json; c=json.load(open('$MODEL/config.json')); print(c.get('architectures',[''])[0])")
grep -rl "$MODEL_TYPE" /sgl-workspace/sglang/python/sglang/srt/models/ | head -3

# Count all-reduces per layer (determines communication bottleneck)
# Look for: tensor_model_parallel_all_reduce, attention_tensor_model_parallel_all_reduce
grep -n "all_reduce\|reduce_scatter" /sgl-workspace/sglang/python/sglang/srt/models/<model_file>.py

# Check which communication path is used (custom AR, quick AR, NCCL)
grep -i "allreduce\|custom_ar\|quick\|AiterCustom\|NCCL" $SERVER_LOG | head -20

# Check if piecewise CUDA graphs are enabled (compute/comm overlap)
grep "disable_piecewise_cuda_graph\|piecewise" $SERVER_LOG | head -5
```

### Step 4: Build the backend test matrix

Based on Steps 1-3, create a prioritized list of backend switches to test. **Test backends BEFORE numerical parameters.** Organize by tier:

```python
BACKEND_TESTS = []

# TIER 1: Attention/decode backend switches (change actual GPU kernels used)
# These have the highest per-switch impact because they replace entire kernel implementations.
# For each attention-related backend flag discovered in Step 2, test all available options.
# Examples: --attention-backend, --decode-attention-backend, --prefill-attention-backend,
#           --nsa-decode-backend, --nsa-prefill-backend, --linear-attn-backend

# TIER 2: Scheduling modes (change batching/overlap behavior)
# These affect how work is organized across forward passes.
# Examples: --enable-mixed-chunk, --enable-two-batch-overlap, --enable-single-batch-overlap,
#           --disable-overlap-schedule

# TIER 3: Compute fusion flags (fuse adjacent operations)
# These merge multiple small kernels into fewer larger ones.
# Examples: --enable-aiter-allreduce-fusion, --enable-flashinfer-allreduce-fusion,
#           --enable-fused-moe-sum-all-reduce, --enable-fused-qk-norm-rope

# TIER 4: MoE/GEMM backend switches
# Examples: --moe-runner-backend, --fp8-gemm-runner-backend, --mamba-backend

# TIER 5: Communication optimizations
# Examples: --enable-mscclpp, --disable-custom-all-reduce, --enable-torch-symm-mem
```

**Key principle:** Tier 1-2 switches change fundamentally different code paths. Tier 3-5 are usually incremental. Always test Tier 1-2 first.

### Step 5: Test backends individually, then combine winners

**This is the key step.** Run each backend switch as a separate experiment against the baseline. Then combine ALL winners in a single run:

```bash
for test in "${BACKEND_TESTS[@]}"; do
    NAME="${test[0]}"
    ARGS="${test[1]}"
    # Launch server with baseline + this one backend change
    # Run benchmark, compare against baseline
    # If > +1%, mark as WINNER
done

# CRITICAL: Combine ALL winners in a single experiment
COMBINED_ARGS="$WINNER1_ARGS $WINNER2_ARGS $WINNER3_ARGS"
# Launch server with ALL winners, benchmark
```

**Why combining is essential:** Individual gains do NOT predict combined gains. Switches that affect different pipeline stages (e.g., a kernel backend + a scheduling mode) can produce super-linear synergy because one change amplifies the benefit of the other. For example, a scheduling change that feeds more tokens per forward pass amplifies a kernel switch that processes each token faster. Always test the full combination — do not assume gains are simply additive.

**After combining backend winners, re-profile to identify new GEAK candidates.** Backend switches that replace vendor C++ kernels (aiter, CK, hipBLASLt) with Triton implementations create new optimization surface for GEAK that did not exist in the original baseline. For example, switching `--attention-backend` from `aiter` to `triton` puts Triton attention kernels on the hot path — these are now GEAK-optimizable, whereas the original aiter C++ kernels were not. Always re-run Phase 3-5 (profile → TraceLens → identify candidates) after backend exploration settles, before proceeding to Phase 7 (GEAK).

### Step 6: Kernel tuning for the model's specific shapes

After backend exploration, tune the vendor kernels for the model's exact tensor shapes. This is especially important when the model has unusual dimensions:

```bash
# Extract model dimensions for tuning
python3 -c "
import json
c = json.load(open('$MODEL/config.json'))
tc = c.get('text_config', c)
print(f'hidden_size: {tc.get(\"hidden_size\", 0)}')
print(f'intermediate_size: {tc.get(\"intermediate_size\", 0)}')
print(f'moe_intermediate_size: {tc.get(\"moe_intermediate_size\", 0)}')
print(f'num_experts: {tc.get(\"n_routed_experts\", 0) or tc.get(\"num_local_experts\", 0)}')
print(f'num_experts_per_tok: {tc.get(\"num_experts_per_tok\", 0)}')
print(f'num_attention_heads: {tc.get(\"num_attention_heads\", 0)}')
print(f'kv_lora_rank: {tc.get(\"kv_lora_rank\", 0)}')
"

# Check if aiter tuning tools exist
ls /sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py 2>/dev/null
ls /sgl-workspace/aiter/csrc/ck_fused_moe/fmoe_tune.py 2>/dev/null

# Run GEMM tuner for model-specific shapes (on a spare GPU)
# Run FMoE tuner for model-specific MoE shapes
# Merge results into aiter configs
```

Kernel tuning impact varies — for MoE-dominated models, the MoE kernels consume most compute and are already vendor-optimized, so tuning dense GEMM shapes may give <1%. But tuned kernels ensure the combined backend config runs at full speed.

### Step 7: Check for code-level bypasses and fast-path blockers

Read the framework and vendor library source to find cases where optimized paths are being bypassed or features are disabled for the current platform:

```bash
# Check for conditional bypasses in vendor kernel libraries
grep -rn "bypass\|skip\|fallback\|disabled" /sgl-workspace/aiter/aiter/*.py | head -20

# Check if the current platform (ROCm/CUDA) disables features
grep -n "is_hip\|_is_hip\|is_cuda\|disable.*cuda_graph\|disable.*piecewise" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -10

# Check what communication backends are actually active in the server log
grep -i "Using.*AllReduce\|allreduce.*path\|NCCL\|custom AR\|Quick" $SERVER_LOG | head -10

# Check if any flags are auto-set based on platform detection
grep -n "if is_hip\|if _is_hip\|if is_cuda" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -10
```

**What to look for:**
- **Conditional bypasses**: Code that skips tuned/optimized kernels for certain input sizes, dtypes, or quantization types. These are often conservative defaults that can be safely removed after verifying correctness.
- **Platform-gated features**: Features disabled for ROCm that work on CUDA (or vice versa). Some of these are stale restrictions from older hardware — the current GPU may support them.
- **Inactive communication fast-paths**: Custom all-reduce, quick all-reduce, or MSCCL++ that are available but not activating due to world-size restrictions or missing initialization.

Knowing what's disabled and why avoids wasting time testing inapplicable flags, and may reveal quick code-level wins.

## Phase 6: Server Parameter Tuning

**When to run**: AFTER Phase 5.5 (backend exploration). Backend switches should already be determined. This phase fine-tunes numerical parameters on top of the best backend configuration.

**Rationale**: Parameter tuning produces incremental gains (typically 0.5-3% per parameter) compared to backend switches. But parameters can still compound, and some parameters interact with backend choices. Always test parameters with the winning backend configuration active.

### SGLang parameter grid

Test each parameter independently on top of the winning backend config. Use CONC=64, ISL=1024, OSL=1024 as the benchmark config.

```bash
# Start with the winning backend args from Phase 5.5
BASE_ARGS="$WINNING_BACKEND_ARGS"

PARAM_GRID=(
    "--cuda-graph-max-bs $CONC"                  # ALWAYS test FIRST (MUST match CONC)
    "--num-continuous-decode-steps 8"             # model-dependent, test multiple values
    "--num-continuous-decode-steps 16"
    "--num-continuous-decode-steps 32"
    "--mem-fraction-static 0.90"                  # more KV cache → larger batches
    "--schedule-conservativeness 0.5"
    "--chunked-prefill-size 65536"
)

# NCCL/RCCL environment variables (test via EXTRA_ENV)
NCCL_GRID=(
    "export NCCL_MIN_NCHANNELS=32"
    "export NCCL_ALGO=Ring"
    "export NCCL_ALGO=Tree"
)
```

### vLLM parameter grid

```bash
VLLM_PARAM_GRID=(
    "--gpu-memory-utilization 0.90"     # more KV cache → larger batches
    "--gpu-memory-utilization 0.92"
    "--max-num-seqs 256"
    "--max-num-seqs 512"
    "--max-num-batched-tokens 16384"
    "--max-num-batched-tokens 32768"
    "--compilation-config level=0"      # test if eager is faster for MoE+MLA
)
```

### Procedure

For each parameter:
1. Kill server → restart with winning backends + this param → warmup (1x CONC prompts) → benchmark (3x CONC prompts)
2. Compare output_throughput and TPOT against the backend-optimized baseline (NOT the original baseline)
3. If improvement > 1%, mark as **KEEP**
4. **Test ALL kept parameters combined in a single experiment** — synergies can be super-linear

| Result | Action |
|--------|--------|
| throughput > backend_baseline + 1% | KEEP, include in combination test |
| throughput within ±1% | NEUTRAL, skip |
| throughput < backend_baseline - 1% | DISCARD |

### Combination testing (CRITICAL)

**Always test the full combination of all winning backends + all winning parameters.** Do NOT assume gains are additive — they can be super-linear or destructive:

```bash
# Combine all winners from Phase 5.5 (backends) + Phase 6 (params)
EXPERIMENT="combined_all_wins" \
EXTRA_SERVER_ARGS="$WINNING_BACKEND_ARGS $WINNING_PARAM1 $WINNING_PARAM2" \
EXTRA_ENV="$WINNING_ENV_VARS" \
bash rapid_experiment.sh
```

If the combined result is worse than individual winners, test subsets to find conflicting pairs. If better than the sum of individuals, you have synergy — this is common when mixing scheduling changes with kernel backend changes.

After finding best combined config, use it as the new baseline for Phase 7 (GEAK) and Phase 9 (sweep). See [`KNOWLEDGE-BASE.md`](KNOWLEDGE-BASE.md) for model-specific validated results.

## Phase 7: Multi-Round GEAK Optimization Loop

**⚠️ FLOW GUARD: Do NOT skip this phase** unless Phase 5 confirmed no GEAK candidates (all checklist items passed, all kernels >3% are vendor C++). If candidates exist, you MUST complete Phase 7-8 BEFORE proceeding to Phase 9. Running the sweep with unoptimized kernels wastes 1-2 hours of compute. The correct order is strictly: Phase 5 (identify) → Phase 6 (server tune) → Phase 7 (GEAK optimize) → Phase 8 (integrate + benchmark + decide) → Phase 9 (sweep with final optimized version).

This phase follows the **THINK → TRY → MEASURE → DECIDE → REPEAT** pattern. Each round: profile → find hot kernels → GEAK optimize → patch → benchmark → re-profile. Repeat until stopping criteria.

### 7a. Locate kernel source

**Two strategies** depending on whether torch.compile is enabled:

**Strategy A (torch.compile mode):** Extract from STANDALONE kernel files in Inductor cache.

```bash
# Find STANDALONE kernel files (NOT graph modules — those are NOT what Inductor compiles)
# Standalone = has @triton_heuristics, does NOT have async_compile or def call(
find /tmp/torchinductor_root -name "*.py" | while read f; do
    if grep -q "@triton_heuristics" "$f" && \
       ! grep -q "async_compile\|def call(" "$f"; then
        for kernel in triton_red_fused triton_tem_fused_mm triton_poi_fused; do
            if grep -q "$kernel" "$f"; then
                echo "STANDALONE ($kernel): $f"
            fi
        done
    fi
done
```

**Strategy B (no torch.compile):** Find framework source kernels directly.

```bash
# SGLang Triton kernels
find /opt/venv -path "*/sglang/srt/layers/*.py" -exec grep -l "@triton.jit" {} \;
# vLLM kernels
find /opt/venv -path "*/vllm/model_executor/layers/*.py" -exec grep -l "@triton.jit" {} \;
```

### 7b. GEAK prompt template

**CRITICAL: The prompt MUST explicitly require TRUE single-pass with NO second loop.** Without this, GEAK defaults to "hoist rsqrt" which is a no-op when R0_BLOCK = r0_numel (loop executes once).

**Use this template for RMSNorm reduction kernels (the highest-impact target):**

```
Tool: geak_create_task
Args: {
    "input_type": "file",
    "files": [{"filename": "<kernel_name>.py", "content": "<ENTIRE standalone .py file content>"}],
    "prompt": "Optimize this Triton kernel for AMD MI355X (gfx950).

HARDWARE: gfx950, 304 CUs, HBM3e ~8TB/s, MFMA bf16, wavefront 64, 65536 VGPRs per CU (256KB register file).
SHAPES: xnumel={xnumel} (batch rows), r0_numel=2048 (hidden_dim). This is a fused RMSNorm kernel.
CURRENT: {gpu_pct}% of GPU time, called {call_count} times per forward pass in LLM decode.

MANDATORY CONSTRAINTS (violation = rejected):
1. Function name MUST be EXACTLY: `{original_function_name}`. Do NOT rename.
2. Function signature MUST be IDENTICAL to original (same params, same order, same types).
3. The `@triton_heuristics.reduction` decorator MUST be preserved.
4. R0_BLOCK <= 2048. Do NOT increase beyond original value.
5. MUST produce numerically identical output (RMSNorm: x * rsqrt(mean(x^2) + eps) * weight).

CRITICAL OPTIMIZATION — TRUE SINGLE-PASS (this is what produces +9% E2E, not hoisting):

The original kernel has TWO loops that BOTH read from in_ptr0 (and in_ptr1 for 5-ptr variant):
  Loop 1: load input → compute sum of squares → store residual (out_ptr1, 5-ptr only)
  Loop 2: RE-LOAD input → normalize with rsqrt → multiply weight → store (out_ptr2)

This is wasteful: in_ptr0 (and in_ptr1) are read TWICE. Since R0_BLOCK = r0_numel = 2048 and
xnumel is small (1-8), each loop executes exactly ONCE. The data is 16KB which fits in the
256KB register file. Therefore:

**⚠️ CORRECTNESS GUARD**: This single-pass optimization is ONLY valid when `R0_BLOCK = r0_numel`
(the loop executes exactly once). If `r0_numel > R0_BLOCK` (loop runs multiple times), you
CANNOT eliminate all loops — rsqrt depends on the FULL reduction sum which requires all
iterations. See the r0_numel safety check below.

ELIMINATE THE SECOND LOOP ENTIRELY. In a SINGLE loop body:
  1. Load ALL inputs (in_ptr0, in_ptr1 if 5-ptr, in_ptr2/weight)
  2. Compute residual sum (in_ptr0 + in_ptr1 for 5-ptr)
  3. Compute sum of squares: tl.sum(x * x)
  4. Compute rsqrt: libdevice.rsqrt(sum_sq / 2048.0 + 1e-06)
  5. Normalize: x * rsqrt_val * weight
  6. Store ALL outputs (out_ptr1 residual if 5-ptr, out_ptr2 normalized)

The result should have ZERO `for r0_offset in range(...)` loops. All computation in one straight-line block.

WARNING: Do NOT just 'hoist rsqrt out of the second loop'. That is a NO-OP because the loop
runs exactly once (R0_BLOCK = r0_numel). You MUST eliminate the second memory load entirely.

OUTPUT: Write the COMPLETE file (all imports, decorator, kernel function) to output dir as {output_filename}.",
    "step_limit": 50, "gpu_count": 1,
    "workspace_id": "control-plane-prod"
}
```

**For ANY dual-loop kernel (not just RMSNorm), use this general single-pass template:**

The core principle is universal: **if a kernel has two loops that read the same memory, merge them into one.** This applies to any Inductor reduction kernel where `R0_BLOCK = r0_numel` (loop runs once, data fits in registers).

```
"prompt": "Optimize this Triton kernel for AMD MI355X (gfx950).
HARDWARE: gfx950, 304 CUs, HBM3e ~8TB/s, MFMA bf16, wavefront 64, 65536 VGPRs/CU (256KB register file).
SHAPES: {shapes_from_trace}
CURRENT: {gpu_pct}% of GPU time, called {call_count} times per forward pass.

MANDATORY CONSTRAINTS:
1. Function name MUST be EXACTLY: `{original_function_name}`.
2. Function signature MUST be IDENTICAL to original.
3. The decorator MUST be preserved.
4. R0_BLOCK/BLOCK sizes: do NOT increase beyond original values.

CRITICAL OPTIMIZATION — ELIMINATE REDUNDANT MEMORY LOADS:
Analyze the kernel for loops that read the SAME memory address. If the kernel has two `for`
loops and both load from the same pointer(s) with the same index, the second loop is a
redundant memory read. Since R0_BLOCK = r0_numel (loop runs once) and the data fits in
the 256KB register file, you can:
1. Load ALL inputs ONCE (no second loop)
2. Do ALL computation (reduction + normalization/transform) in straight-line code
3. Store ALL outputs at the end

The optimized kernel should have ZERO `for` loops if R0_BLOCK = r0_numel. Every tl.load
should appear exactly ONCE per input pointer.

⚠️ IMPORTANT: This optimization is ONLY valid when R0_BLOCK = r0_numel (loop runs once).
If r0_numel > R0_BLOCK (e.g. r0_numel=4096 but R0_BLOCK=2048), the loop must run multiple
times and you CANNOT eliminate it — rsqrt requires the FULL sum across ALL iterations.

VERIFICATION CHECKLIST:
- Count of tl.load calls = number of unique input pointers (no duplicates)
- Count of tl.store calls = number of output pointers
- Zero `for` loops ONLY when R0_BLOCK = r0_numel (check size_hints in decorator)
- Use libdevice.rsqrt (NOT tl.math.rsqrt)

OUTPUT: Write COMPLETE file to output dir as {output_filename}.",
    "step_limit": 50, "gpu_count": 1,
    "workspace_id": "control-plane-prod"
```

**GEAK workspace retry (max 3 attempts per kernel)**: Try `control-plane-prod` first. If failed, retry on default workspace (no `workspace_id`). If still failed, try `control-plane-prod` again. After 3 consecutive failures on a single kernel, mark it as failed and move to next candidate. Log each attempt with workspace used.

### 7c. Multi-round optimization loop (concrete steps)

**SUBMIT ALL top 5 candidates to GEAK IN PARALLEL.** Each kernel is independent — GEAK tasks run on separate pods. Parallel submission total time = max(single task) ≈ 3-5 min, vs serial = 5 × 3 = 15 min.

**Expected GEAK round count per E2E run:**
- Round 1: Submit ALL top 5 candidates in parallel → wait for all to complete → patch each INDIVIDUALLY + benchmark
- Round 2 (if any kept): Re-profile → submit new candidates
- Typical: 5-8 total submissions, ~5-10 min total GEAK wall clock
- Each submission: step_limit=50

**Step-by-step:**

#### Step 1: SUBMIT ALL candidates to GEAK in parallel

Extract source from STANDALONE kernel files for ALL top 5 candidates. Submit ALL to GEAK simultaneously using `geak_create_task` + `geak_submit_task` for each. Poll ALL tasks in a single loop.

```python
# Submit ALL candidates in parallel
tasks = []
for kernel in top_5_candidates:
    standalone_file = find_standalone_file(kernel)
    task = geak_create_task(standalone_file, prompt_for_kernel_type(kernel))
    geak_submit_task(task.id)
    tasks.append((kernel, task.id))

# Poll ALL tasks until all complete (max 15 min)
for _ in range(15):
    sleep(60)
    all_done = True
    for kernel, task_id in tasks:
        status = geak_get_task(task_id)["status"]
        if status in ("running", "pending"):
            all_done = False
    if all_done:
        break
```

#### Step 2: VERIFY + PATCH each completed task individually

For each completed GEAK task, verify output then patch standalone files + benchmark ONE AT A TIME to isolate E2E impact. Use the retry strategy below if GEAK submission fails.

**GEAK submission retry (max 3 attempts per kernel):**

**Retry strategy**: Each kernel gets up to 3 GEAK submission attempts, alternating workspaces. If all 3 fail, mark kernel as failed and move to next candidate.

```python
WORKSPACE_ROTATION = ["control-plane-prod", None, "control-plane-prod"]  # alternate workspaces

def submit_geak_with_retry(kernel, prompt_template, max_retries=3):
    """Submit to GEAK with retry. Returns (task_id, status) or (None, 'failed')."""
    for attempt in range(max_retries):
        ws = WORKSPACE_ROTATION[attempt]
        ws_label = ws or "default"
        print(f"GEAK attempt {attempt+1}/{max_retries} on workspace={ws_label}")
        
        # Create and submit
        kwargs = {"workspace_id": ws} if ws else {}
        task = geak_create_task(kernel, prompt_template, **kwargs)
        geak_submit_task(task.id)
        
        # Poll (max 15 min per attempt)
        for poll in range(15):
            sleep(60)
            task_status = geak_get_task(task.id)
            status = task_status["status"]
            
            if status == "completed":
                return task.id, "completed"
            elif status == "failed":
                print(f"  Attempt {attempt+1} failed on {ws_label}")
                break  # → next retry
            # "running" or "pending" → keep polling
        else:
            # Timed out after 15 min
            print(f"  Attempt {attempt+1} timed out on {ws_label}")
    
    # All retries exhausted
    return None, "failed"

# Usage per kernel:
task_id, status = submit_geak_with_retry(kernel, prompt_template)
if status != "completed":
    log(kernel, "discard", f"GEAK failed after {max_retries} retries")
    # → go to Step 7 (next kernel)
```

#### Step 3: VERIFY GEAK output

Download the optimized kernel and verify:
- Function name matches original exactly
- Parameter list (count + names) matches original exactly
- If mismatch: re-submit with stricter prompt (max 3 attempts per kernel, see 7d)

#### Step 4: PATCH (Phase 7)

- Use `patch_standalone_kernels()` from Strategy A (Phase 8a) — patches STANDALONE files only.
- The function adapts `xnumel` per file and skips graph modules automatically.
- Kill server, wait 10s, clear `.so`/`.json`/Triton cache.

#### Step 5: BENCHMARK

Restart server with same params as baseline. Run same benchmark (same CONC/ISL/OSL/num_prompts).

#### Step 6: DECIDE

```python
gain = (new_tput - baseline_tput) / baseline_tput * 100
if gain > 0:
    # KEEP: update baseline, log, move to next kernel
    baseline_tput = new_tput
elif attempt < 3:
    # REVERT + re-submit GEAK with "REGRESSION" prompt fix
    revert_all_bak_files()
    # → go back to Step 2 with improved prompt
else:
    # REVERT + mark as discard, move to next kernel
    revert_all_bak_files()
```

#### Step 7: CHECK stopping criteria

If not met → go to Step 1 with next kernel. If met → proceed to Phase 9.

### 7d. GEAK re-submission prompt fixes

When re-submitting after a failed attempt, append to the original prompt:

| Issue | Append to prompt |
|-------|-----------------|
| Signature mismatch | `"PREVIOUS ATTEMPT FAILED: you changed the function name or signature. The function name MUST be EXACTLY: {name}. The EXACT original signature is: {signature}. Do NOT change ANY parameter."` |
| Register OOM | `"PREVIOUS ATTEMPT FAILED: register OOM during Triton compilation. You used BLOCK_N={N} which is too large. Use BLOCK_N <= 128, BLOCK_K <= 256. Reduce block sizes."` |
| Compilation error | `"PREVIOUS ATTEMPT FAILED with error: {error_msg}. Fix this error. Common issues: (1) tl.load with other= requires mask, (2) tl.math.rsqrt not available in older Triton — use libdevice.rsqrt."` |
| Regression (reduction kernel) | `"PREVIOUS ATTEMPT was SLOWER ({gain}%). Try STRUCTURAL changes: (1) hoist loop-invariant computations OUT of loops, (2) merge dual-pass into single-pass, (3) eliminate redundant .to(tl.float32) casts."` |
| Regression (template kernel) | `"PREVIOUS ATTEMPT was SLOWER ({gain}%). Do NOT change block sizes — Inductor autotuner already picked near-optimal values. Focus ONLY on loop body optimizations: precompute strides, remove redundant index calculations."` |
| Correctness fail | `"PREVIOUS ATTEMPT produced wrong output (max diff={diff}). Ensure numerical equivalence. Do NOT change the computation logic, only optimize memory access and scheduling."` |
| GEAK task failed | Retry up to 3 times, alternating workspaces (`control-plane-prod` → default → `control-plane-prod`). If all 3 fail, skip this kernel. |

### 7e. Stopping criteria

| Condition | Action |
|-----------|--------|
| All top 5 candidates processed AND re-profile shows no new >3% non-vendor candidates | Stop |
| Cumulative E2E gain > 15% | Stop — excellent result |
| 5 consecutive discards (across all rounds) | Stop — diminishing returns |
| 2+ crashes during patching | Stop — environment unstable |
| Wall clock > 120 min for optimization phase | Stop — time budget |
| Total GEAK submissions > 15 | Stop — cost budget (step_limit=50 per task) |

**Re-profile after each kept optimization.** After patching a kernel that improves E2E, re-profile to find the NEW top bottleneck (kernel rankings shift after optimization). This may reveal new GEAK candidates that were previously masked by the optimized kernel.

**After stopping, always keep all `keep` patches applied and proceed to Phase 9.**

## Phase 8: Integrate → Benchmark → Decide (per kernel)

Phase 7's loop calls this phase for each GEAK-optimized kernel. **Choose the right patching strategy based on the mode.**

### 8a. Integrate — Dual Strategy

**Decision rule:**
- `--enable-torch-compile` is active AND target is an Inductor kernel (`triton_tem_*`, `triton_red_*`) → **Strategy A**
- Target is a framework source kernel (SGLang/vLLM Triton kernel) OR torch.compile not used → **Strategy B**

---

#### Strategy A: Standalone File Patching (torch.compile mode) — VALIDATED +9% E2E

Inductor generates TWO types of files per kernel:
- **Standalone files**: Contain ONLY the kernel (`@triton_heuristics` + `def kernel(...)`, NO `async_compile` or `def call(`). These are what Inductor **actually compiles and executes**.
- **Graph module files**: Contain `async_compile.triton('name', '''inline_source''')` + `def call(...)`. These reference standalone files but patching their inline source does NOT affect the compiled kernel.

**CRITICAL (validated 2026-03-21): You MUST patch STANDALONE files, NOT graph module inline source.** Graph module patching produces 0% E2E gain. Standalone patching produces **+9% E2E**.

**Use this Python script to patch standalone kernel files:**

```python
import os, re, sys, glob, shutil

CACHE_DIR = "/tmp/torchinductor_root"

def is_standalone_kernel(content):
    """Standalone = has @triton_heuristics but NO async_compile or def call("""
    return ("@triton_heuristics" in content and
            "async_compile" not in content and
            "def call(" not in content)

def patch_standalone_kernels(kernel_name, geak_source_path, target_signature_pattern):
    """
    Find all standalone kernel files matching kernel_name and target_signature_pattern,
    replace the kernel body with GEAK-optimized version.
    
    Args:
        kernel_name: e.g. "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0"
        geak_source_path: path to GEAK output .py file
        target_signature_pattern: regex for the signature to match, e.g. "in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2"
    """
    with open(geak_source_path) as f:
        geak_body = f.read()
    
    # Extract just the function body from GEAK output (everything after the def line)
    body_match = re.search(r'(def ' + kernel_name + r'\([^)]+\):)\n(.*)', geak_body, re.DOTALL)
    if not body_match:
        print(f"ERROR: Could not find {kernel_name} in GEAK output")
        return 0, 0
    geak_function_body = body_match.group(2)
    
    patched, skipped = 0, 0
    for root, dirs, files in os.walk(CACHE_DIR):
        for f in files:
            if not f.endswith(".py") or f.endswith(".bak"):
                continue
            fpath = os.path.join(root, f)
            content = open(fpath).read()
            
            if kernel_name not in content or target_signature_pattern not in content:
                continue
            if not is_standalone_kernel(content):
                skipped += 1; continue
            
            # Extract xnumel and r0_numel from file for dimension-adaptive patching
            m = re.search(r"size_hints=\{'x': (\d+)", content)
            xnumel = int(m.group(1)) if m else 4
            
            # SAFETY CHECK: single-pass only valid when R0_BLOCK = r0_numel
            m_r0_hint = re.search(r"'r0_':\s*(\d+)", content)
            m_r0_body = re.search(r"r0_numel\s*=\s*(\d+)", content)
            r0_hint = int(m_r0_hint.group(1)) if m_r0_hint else 2048
            r0_body = int(m_r0_body.group(1)) if m_r0_body else 2048
            if r0_body > r0_hint:
                print(f"  SKIP (r0_numel={r0_body} > R0_BLOCK hint={r0_hint}, single-pass unsafe): {fpath}")
                continue
            
            # Backup
            shutil.copy2(fpath, fpath + ".bak")
            
            # Replace function body, adapting xnumel (regex handles any original value)
            adapted_body = re.sub(
                r'xnumel\s*=\s*\d+', f'xnumel = {xnumel}', geak_function_body, count=1)
            
            sig_pattern = r'(def ' + kernel_name + r'\([^)]+\):)\n.*'
            new_content = re.sub(sig_pattern, r'\1\n' + adapted_body, content, flags=re.DOTALL)
            open(fpath, 'w').write(new_content)
            patched += 1
            print(f"  PATCHED (x={xnumel}): {fpath}")
    
    # Clear ALL binary caches (BOTH Inductor .so/.json AND Triton cache)
    for so in glob.glob(f"{CACHE_DIR}/**/*.so", recursive=True):
        os.remove(so)
    for j in glob.glob(f"{CACHE_DIR}/**/*.json", recursive=True):
        os.remove(j)
    triton_cache = os.path.expanduser("~/.triton/cache")
    if os.path.exists(triton_cache):
        shutil.rmtree(triton_cache)
    
    print(f"\nPatched: {patched} standalone files, Skipped: {skipped} graph modules")
    print("Binary cache cleared. Restart SGLang server to apply changes.")
    return patched, skipped
```

**After patching, kill server and restart:**
```bash
ps aux | grep "python3 -m sglang" | grep -v grep | grep -v bash | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 10
# Relaunch with same params as Phase 2
```

**For reduction kernels (RMSNorm etc.):** Grid does NOT change — only the kernel body changes. This makes reduction kernels the safest patch targets.

---

#### Strategy B: Direct Source Edit (no torch.compile)

For framework source kernels (SGLang/vLLM/aiter Triton kernels), directly replace the kernel function in the source file.

**CRITICAL (validated 2026-03-21 on DSR1): Use Python AST for function boundary detection.** Do NOT use regex or indentation-based detection — aiter source files have module-level variable definitions (`make_kernel_repr(...)`) between kernel functions that will be accidentally deleted by naive end-of-function detection, causing `NameError` at import.

```python
import ast

def get_function_line_range(source, func_name):
    """Use Python AST to find exact function start/end lines."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start = node.lineno - 1  # 0-indexed
            end = node.end_lineno     # exclusive
            if node.decorator_list:
                dec_start = node.decorator_list[0].lineno - 1
                return dec_start, end
            return start, end
    return None, None

def replace_function_ast(original_source, func_name, geak_source):
    """Replace only the function body (def + body), keeping original decorators."""
    lines = original_source.split('\n')
    orig_start, orig_end = get_function_line_range(original_source, func_name)
    geak_start, geak_end = get_function_line_range(geak_source, func_name)
    if orig_start is None or geak_start is None:
        return original_source, False
    
    geak_lines = geak_source.split('\n')
    
    # Find 'def' line in both (skip decorators)
    orig_def = next(i for i in range(orig_start, orig_end) if lines[i].lstrip().startswith('def '))
    geak_def = next(i for i in range(geak_start, geak_end) if geak_lines[i].lstrip().startswith('def '))
    
    # Keep original decorators + GEAK def+body
    result_lines = lines[:orig_def] + geak_lines[geak_def:geak_end] + [''] + lines[orig_end:]
    return '\n'.join(result_lines), True
```

```bash
# 1. Find + backup
cp "$KERNEL_FILE" "${KERNEL_FILE}.bak"

# 2. Patch using AST-based replacement (see Python script above)
# 3. Verify import: python3 -c "import <module>; print('OK')"
# 4. Clear __pycache__: find /sgl-workspace/aiter -name '__pycache__' -exec rm -rf {} +
# 5. Restart server
ps aux | grep "python3 -m sglang" | grep -v grep | grep -v bash | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 10
```

**To revert:** `cp "${KERNEL_FILE}.bak" "$KERNEL_FILE"` + clear `__pycache__` + restart.

---

### 8b. Re-Benchmark (E2E)

**⚠️ CRITICAL — COMPARISON FAIRNESS (validated 2026-03-23 on Kimi-K2.5):**

The re-benchmark MUST use **EXACTLY the same server config AND benchmark params** as baseline. Any difference invalidates the comparison. On Kimi-K2.5, a "40.4% improvement" turned out to be mostly from accidentally changing concurrency (64→128) and server params (decode-steps 4→8), NOT from kernel optimization.

**Server params that MUST match baseline:**
- `--num-continuous-decode-steps` (e.g., 4 vs 8 = +2-40% difference!)
- `--mem-fraction-static` (more KV cache → larger batch → higher throughput)
- `--cuda-graph-max-bs` (must match)
- `--disable-radix-cache` (if baseline had it, re-benchmark must too)
- Same port, same TP

**Benchmark params that MUST match baseline:**
- `--max-concurrency $CONC` (NEVER omit this — omitting it sends ALL prompts at once!)
- `--num-prompts $((CONC * 3))` (same formula)
- `--random-input-len` and `--random-output-len` (same ISL/OSL)
- `--request-rate inf`

**Validation checklist before accepting results:**
- [ ] Server `--num-continuous-decode-steps` matches baseline
- [ ] Server `--mem-fraction-static` matches baseline
- [ ] Server `--cuda-graph-max-bs` matches baseline
- [ ] Server `--disable-radix-cache` matches baseline (if used)
- [ ] Benchmark `--max-concurrency` matches baseline
- [ ] Benchmark `--num-prompts` matches baseline
- [ ] TPOT direction confirms kernel improvement (should DECREASE, not increase)

**Red flag: if TPOT INCREASES after "optimization"**, the throughput gain is from higher batching (concurrency/scheduling change), NOT from faster kernels. True kernel speedup → lower TPOT at same concurrency.

Kill server → restart with patched kernel → run same benchmark config as baseline:
```bash
ps aux | grep "python3 -m sglang" | grep -v grep | grep -v bash | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 10

# Relaunch server with EXACTLY same params as Phase 2 baseline
# Run benchmark with EXACTLY same CONC/ISL/OSL/num_prompts
bash "$SCRIPTS_DIR/run_baseline.sh"  # or manual launch if only benchmarking
```

### 8c. Decide and log

```python
actual_e2e = (new_tput - baseline_tput) / baseline_tput * 100
```

| Outcome | Action |
|---------|--------|
| `actual_e2e > 0` | **KEEP**. Update baseline_tput. |
| `actual_e2e <= 0` | **REVERT**. Restore backup. |
| Crashed | **REVERT**. Log as crash. Re-submit GEAK with error context. |

**Log to results.tsv:**
```
round | attempt | kernel | gpu_pct | kernel_speedup | actual_e2e | status | description
1     | 1       | fused_mm_0 | 22.2% | +35% | +8.1% | keep | GEAK: BLOCK_M=4 simplified grid
1     | 2       | rmsnorm    | 6.7%  | +60% | +4.2% | keep | GEAK: hoisted rsqrt
1     | 3       | fused_mm_1 | 7.2%  | +20% | -0.1% | discard | GEAK: no E2E gain
2     | 4       | topkGate   | 4.7%  | +15% | +0.5% | keep | GEAK round 2: new top kernel
```

### Cumulative gain and re-profile

After each round of kernels:

```python
cumulative_gain = (current_tput - original_baseline_tput) / original_baseline_tput * 100
```

If any patches were kept, **re-profile** to find the next bottleneck:

```
Tool: run_comparative_analysis
Args: {
    "gpu1_kineto": "/shared_nfs/inference-optimization/traces/<baseline>/filtered-TP-0.trace.json.gz",
    "gpu1_name": "baseline",
    "gpu2_kineto": "/shared_nfs/inference-optimization/traces/<optimized>/filtered-TP-0.trace.json.gz",
    "gpu2_name": "optimized"
}
```

Then return to Phase 7 for the next round.

### Critical lessons (validated 2026-03-20/21)

- **MUST patch STANDALONE files, NOT graph module inline source** (validated 2026-03-21): Graph module patching → **+0.01% E2E** (no effect). Standalone file patching → **+9.01% E2E** (593→647 tok/s). Standalone files are what Inductor actually compiles. Identify them by: `@triton_heuristics` present, `async_compile` absent, `def call(` absent.
- **"Hoist rsqrt" is a NO-OP when R0_BLOCK = r0_numel** (validated 2026-03-21): Inductor's heuristic sets R0_BLOCK = r0_numel = 2048 for RMSNorm, so the reduction loop runs exactly ONCE. Moving rsqrt before the loop saves zero computation. GEAK's default optimization is to hoist — you MUST explicitly instruct GEAK to produce a TRUE single-pass (eliminate the second loop entirely, load data only once).
- **TRUE single-pass is the key optimization** (validated 2026-03-20/21): Original reads in_ptr0 twice (6 memory ops per element). Single-pass reads once (3 memory ops). This 2x memory reduction → **69.9% kernel speedup → +9-14% E2E**.
- **Same-name kernels can have DIFFERENT signatures.** `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` exists in both 3-ptr (no residual) and 5-ptr (with residual) versions, AND with different xnumel values (1, 2, 4, 8). Adapt xnumel when patching: `xnumel = {value_from_file}`.
- **r0_numel > R0_BLOCK: single-pass is UNSAFE** (validated 2026-03-21 on Qwen3-8B). Qwen3-8B has hidden_size=4096, producing standalone files with `r0_numel=4096`. If `R0_BLOCK=2048` (set by Triton heuristics), the loop runs 2 iterations. Eliminating all loops in this case only processes 2048 of 4096 elements — **silent correctness bug**. Before patching, always verify: `size_hints` in the decorator shows `r0_` value = `r0_numel` in the function body. If `r0_numel > size_hints['r0_']`, do NOT apply single-pass. Instead, keep the loop structure but merge the two passes within each iteration (load once per iteration, accumulate sum, then in a second loop after reduction, normalize with the saved data — but this still requires two loops).
- **5-ptr (residual) variant is the high-impact target.** Most transformer layers use the 5-ptr variant (residual add + RMSNorm fused). The 3-ptr variant is used less frequently.
- **Reduction kernels are the safest and most productive GEAK targets.** Grid doesn't change, structural optimizations have real E2E impact. Prioritize over template/GEMM kernels.
- **GEMM block size tuning rarely helps E2E** (validated 2026-03-21): GEAK changed BLOCK_M=16→4, BLOCK_N=64→128 on fused_mm_0, result was **-4.1% E2E**. Inductor autotuner already picks near-optimal config.
- **Micro-optimizations (rsqrt function swap, divide→multiply) alone have NO measurable E2E impact.** Only structural changes (eliminating memory loads) matter.
- **Submit GEAK tasks in parallel, but patch + benchmark ONE kernel at a time** to isolate E2E impact.
- **`--mem-fraction-static`** is model-dependent — check official test configs. With torch.compile, use a lower value (e.g. 0.6) due to extra compilation memory. The scripts auto-detect when `--enable-torch-compile` is present.
- **Wait 10+ seconds** between server kill and relaunch for GPU memory and NCCL cleanup.
- **Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** after profiling.
- **Trace files can be truncated** if read before SGLang finishes writing. Wait for file size to stabilize (check every 10s, 6 retries). Use the filtered trace when available.
- **GEAK workspace**: `control-plane-prod` is preferred. If it fails (no GPU resources), retry without workspace_id.

After the GEAK optimization loop (Phase 7) is complete and all patches are integrated and verified (Phase 8), proceed to Phase 9.

## Phase 9: Parameter Sweep

After the optimization loop, run the full parameter sweep with the **optimized version** to map the Pareto frontier.

### Using the sweep script

```bash
export MODEL="$MODEL_PATH" TP=8 INFERENCEX_PATH="/shared_nfs/xiaofei/InferenceX"
# Optional overrides:
export CONC_VALUES="4 8 16 32 64"
export ISL_OSL_CONFIGS="1024:1024 8192:1024 1024:8192"
export RESULT_DIR="/shared_nfs/inference-optimization/results/sweep_$(date +%Y-%m-%d-%H-%M)"
# Override adaptive num_prompts with fixed multiplier (optional):
# export NUM_PROMPTS_MULTIPLIER=10

bash "$SCRIPTS_DIR/run_sweep.sh"
```

The sweep script:
- **Single server launch** for ALL ISL/OSL configs (ISL/OSL are request-level params, not server params). Eliminates 8-34 min of unnecessary restarts.
- **Default CONC: 3 values** (`4 16 64`) — enough for Pareto curves. Override with `CONC_VALUES="4 8 16 32 64"` for finer granularity.
- **Adaptive num_prompts**: OSL≤1024 → CONC×5, OSL≤4096 → CONC×3, OSL>4096 → CONC×2. Override with `NUM_PROMPTS_MULTIPLIER` env var.
- **Smart ordering**: Configs sorted by estimated cost (num_prompts × OSL), short configs first for early results.
- **Auto-skip extreme combos**: Configs with `num_prompts × OSL > MAX_OUTPUT_TOKENS` (default 2M) are skipped. Override with `MAX_OUTPUT_TOKENS` env var.
- Shows progress `[N/total +elapsed]` and total wall time at completion
- Generates `results.tsv` with all configs (skipped configs logged as `status=skipped`)

### SaFE MCP parallel sweep (faster alternative)

For maximum speed, create one SaFE workload per config and run all in parallel:

```
Tool: workload_create
Args: {
    "display_name": "sweep-dsr1-1k1k-c4",
    "workspace_id": "control-plane-prod",
    "kind": "PyTorchJob",
    "images": ["harbor.oci-slc.example-internal-host.invalid/proxy/lmsysorg/sglang:v0.5.9-rocm700-mi35x"],
    "resources": [{"replica": 1, "cpu": "96", "gpu": "8", "memory": "1024Gi", "sharedMemory": "256Gi"}],
    "entry_points": ["<base64 of: bash $SKILL_ROOT/scripts/run_baseline.sh>"],
    "env": {"MODEL": "...", "TP": "8", "CONC": "4", "ISL": "1024", "OSL": "1024",
            "INFERENCEX_PATH": "/shared_nfs/xiaofei/InferenceX",
            "RESULT_DIR": "/shared_nfs/inference-optimization/results/sweep_<timestamp>"},
    "is_tolerate_all": true,
    "ttl_seconds_after_finished": 600
}
```

15 configs × 15 nodes = all parallel, ~10 min total vs ~75 min serial.

After the sweep completes, proceed to Phase 10 to generate the final report.

## Phase 10: Optimization Report

Write to `$WORK_DIR/optimization_report.md`:

```markdown
# Inference Optimization Report — {Model Name}

## Executive Summary
- **Model**: {model_id} ({param_count})
- **Hardware**: {gpu_count}x {gpu_type}
- **Framework**: {framework} v{version}
- **Optimization attempts**: {N} ({kept} kept, {discarded} discarded, {crashed} crashed)
- **Total improvement**: {total_pct}% (baseline → optimized)
- **Best output throughput**: {best_tput} tok/s at CONC={conc}

## TraceLens Bottleneck Analysis (Baseline)
| Category | GPU Time (ms) | % | GEAK Candidate? |
|----------|--------------|---|-----------------|
| ... | ... | ... | ... |

## GEAK Kernel Optimization (Top 5 Candidates)

### Gain Breakdown (per kernel)
| # | Kernel | GPU % | Kernel Speedup | Theoretical E2E | Actual E2E | Status |
|---|--------|-------|---------------|----------------|-----------|--------|
| 1 | (name) | X% | Y% | Z% | W% | keep/discard |
| ... | ... | ... | ... | ... | ... | ... |
| N | (name) | X% | Y% | Z% | W% | keep/discard |

Gain formulas:
- **Kernel Speedup** = (orig_ms - geak_ms) / orig_ms × 100% (micro-benchmark)
- **Theoretical E2E** = Kernel Speedup × GPU time % (upper bound)
- **Actual E2E** = (new_tput - prev_tput) / prev_tput × 100% (measured)
- **Cumulative E2E** = (final_tput - original_baseline) / original_baseline × 100%

**Cumulative gain: +X.X%** (baseline {baseline_tput} → final {final_tput} tok/s)

### Micro-benchmark Results
| Kernel | Original (ms) | GEAK (ms) | Speedup | Correctness |
|--------|-------------|-----------|---------|-------------|
| ... | ... | ... | ... | PASS/FAIL |

### GEAK Task IDs
| Kernel | Task ID | step_limit | Duration |
|--------|---------|-----------|----------|
| ... | ... | 50 | ...min |

## Optimization Journey (Full Log)
| # | Description | tok/s | Change vs Baseline | Status |
|---|-------------|-------|-------------------|--------|
| 0 | Baseline | ... | — | baseline |
| 1 | GEAK: kernel_1 | ... | +X% | keep |
| 2 | GEAK: kernel_2 | ... | +Y% | keep |
| 3 | GEAK: kernel_3 | ... | -Z% | discard |

## Parameter Sweep (Optimized Version)
### ISL=1024 / OSL=1024
| CONC | Output tok/s | TPOT (ms) | Interactivity | TTFT (ms) |
|------|-------------|-----------|---------------|-----------|
| ... | ... | ... | ... | ... |

(repeat for each ISL/OSL combo)

## Comparison with InferenceX
...

## Recommendations
1. ...
```

## Knowledge Base

For model-specific configurations, validated server-parameter tables, benchmark results, and detailed lessons learned, see [`KNOWLEDGE-BASE.md`](KNOWLEDGE-BASE.md) in this directory.

The following reference sections are kept here for use during execution:

### Process Management (CRITICAL)

- **Never use `pkill -f "sglang.launch_server"` inside scripts** — it kills the script itself if the command line contains that string. Use `ps aux | grep ... | grep -v grep | grep -v bash` instead.
- **GPU memory leaks**: After killing SGLang, lingering `multiprocessing.spawn` workers can hold 200+ GB/GPU. Always verify with `torch.cuda.mem_get_info()`.
- **SGLang "unbalanced memory" error**: If GPUs show different free memory, some have residual allocations. Kill all Python workers.
- **Wait 8+ seconds between server kill and relaunch** for GPU memory and NCCL cleanup.
- **Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** after profiling — inherited env vars cause 30x slowdown.

### TraceLens Tips

- **Traces are written directly to NFS** by the profiling scripts (`SGLANG_TORCH_PROFILER_DIR` points to `/shared_nfs/inference-optimization/traces/<timestamp>/`). No manual copy needed.
- **Always use `filtered-TP-0.trace.json.gz`** for TraceLens analysis (generated by `run_baseline.sh`). Raw traces are 300MB+ (97% python_function events), filtered traces are ~5MB. Merged multi-TP traces can exceed TraceLens memory limits.
- `platform` must be uppercase: `"MI355X"` not `"mi355x"`.
- **TraceLens does NOT support `rocprofv3` trace format** (validated 2026-03-23). Even with `trace_type="rocprof"`, TraceLens internally expects PyTorch Kineto format (`traceEvents` key). Feeding a `rocprofv3` JSON trace causes `KeyError: 'traceEvents'`. Only use PyTorch profiler traces (`.json.gz` with Kineto format) for TraceLens analysis.
- TraceLens output CSVs are saved to `tracelens_output/` subdirectory alongside the traces.

### vLLM Integration (validated 2026-03-23 on Qwen3-8B, MI355X)

All scripts (`run_baseline.sh`, `run_sweep.sh`, `run_profile.sh`) support `FRAMEWORK=vllm`.

**Parameter mapping (SGLang → vLLM):**

| SGLang | vLLM | Notes |
|--------|------|-------|
| `--model-path` | `vllm serve <model>` (positional) | — |
| `--mem-fraction-static 0.8` | `--gpu-memory-utilization 0.85` | Default in script |
| `--disable-radix-cache` | `--no-enable-prefix-caching` | For random benchmarks |
| `--cuda-graph-max-bs N` | Auto (compilation_config) | vLLM captures up to 512 by default |
| `--attention-backend aiter` | Auto-detected on ROCm | vLLM uses Triton attention on ROCm |
| `--enable-torch-compile` | Default ON (level=3) | Disable with `--enforce-eager` |
| `--chunked-prefill-size N` | `--max-num-batched-tokens N` | — |
| `--kv-cache-dtype fp8_e4m3` | `--kv-cache-dtype fp8_e4m3` | Same flag |
| `--num-continuous-decode-steps N` | No direct equivalent | vLLM scheduler differs |
| `SGLANG_TORCH_PROFILER_DIR` | `VLLM_TORCH_PROFILER_DIR` | Set before server launch |
| `SGLANG_USE_AITER=1` | Not needed | vLLM auto-detects |

**vLLM-specific env vars:** `VLLM_EXTRA_ARGS` carries model-specific args (same role as `SGLANG_EXTRA_ARGS`).

**vLLM baseline results (Qwen3-8B, TP=1, ISL=1024, OSL=256, MI355X):**

| CONC | Output tok/s | TPOT (ms) | TTFT (ms) |
|------|-------------|-----------|-----------|
| 4 | 565.3 | 6.73 | 93.3 |
| 16 | 1649.3 | 9.09 | 161.3 |

**Profiling:** vLLM supports `/start_profile` (POST) and `/stop_profile` (POST) endpoints, same as SGLang. Requires `VLLM_TORCH_PROFILER_DIR` set before server launch.

### Trace Size and Filtering (CRITICAL)

SGLang's profiler captures **Python function tracing** by default, producing 97% useless events:
```
Raw trace: 349MB (21.9M events, 97.6% python_function)
Filtered:    5.8MB (524K events, GPU kernels + CPU ops only)
```

**Always filter traces before TraceLens analysis.** Add this step after stop_profile:
```python
import gzip, json
with gzip.open(raw_trace) as f:
    trace = json.load(f)
keep = {'kernel', 'gpu_memcpy', 'gpu_memset', 'cpu_op', 'cuda_runtime', 'ac2g', 'user_annotation', 'gpu_user_annotation'}
trace['traceEvents'] = [e for e in trace['traceEvents'] if e.get('cat', '') in keep]
with gzip.open(filtered_trace, 'wt') as f:
    json.dump(trace, f)
```

Target: filtered trace < 10MB for reliable TraceLens analysis.

### torch.compile + GEAK: The Best Optimization Path

SGLang supports `--enable-torch-compile` which uses PyTorch Inductor to generate optimized Triton kernels. Inductor generates standardized Triton kernels that are ideal GEAK targets.

**Always enable `--enable-torch-compile` before trying GEAK.** Inductor's autotuner already optimizes most kernels. GEAK provides incremental gains on Inductor-generated kernels.

**Validated CONC-dependent GEAK end-to-end numbers, the full Inductor + GEAK step list, timing estimate, and caveats** are documented in [`KNOWLEDGE-BASE.md`](KNOWLEDGE-BASE.md) under **Qwen3-30B-A3B**.

**torch.compile also makes GEAK more effective:** Inductor generates standardized Triton kernels (simple 7-param signatures, hardcoded shapes) that are ideal GEAK targets — unlike framework source kernels (30+ params, multiple quantization paths).

**Launch with torch.compile:**
```bash
--enable-torch-compile --mem-fraction-static 0.6  # torch.compile needs extra memory; value is model-dependent, check official test configs
```

**Inductor kernel extraction for GEAK:**
```bash
# After torch.compile baseline runs, find generated Triton kernels:
find /tmp/torchinductor_root -name "*.py" | xargs grep -l "@triton" | wc -l
# Typical: 200-300 Triton kernel files

# Find standalone kernels (NOT graph modules) for GEAK:
# Standalone = has @triton_heuristics but NO async_compile or def call(
```

### GEAK-Specific Lessons

- **`step_limit=50` recommended** — with 5, GEAK only reads files. With 20, it can read+write but may not iterate enough. With 50, GEAK has room to analyze, write, benchmark, iterate on compilation errors, and produce a verified optimized kernel.
- **Submit ALL top 5 candidates to GEAK in parallel** — they run on separate pods, total time = max(single) ≈ 3-5 min instead of serial 5 × 3 = 15 min.
- **Provide rich context**: hardware (gfx950), dtype (bf16/fp8), input shapes, current TFLOPS.
- **Ask GEAK to write output file explicitly**: "Write optimized kernel to output dir as kernel_optimized.py" — otherwise it may only edit in place.
- **GEAK latency**: 2-30 min per task. Fast if pod cached (~2 min), slow cold start (~30 min).
- **GEAK cannot beat vendor kernels** (hipBLASLt, aiter MLA/MoE). Never submit `Cijk_*` or `aiter::*`.
- **GEAK works best on simple Triton kernels** (< 50 lines, < 10 params). Complex kernels (30+ params like vLLM fused_moe) often get their signatures simplified by GEAK, causing runtime `TypeError: got multiple values for argument`.
- **GEAK output path unreliable (validated 2026-03-23)**: GEAK agent sometimes writes optimized code to the INPUT file path instead of the output directory. Always check `geak_get_outputs` for output files, and if empty, check if the input file was modified.
- **Complex Triton kernels CAN be optimized (validated 2026-03-23 on Kimi-K2.5)**: Despite `fused_moe_kernel_gptq_awq` having 37+ parameters, GEAK successfully optimized it with a GROUP_ALIGNED compile-time conditional path. The key is providing explicit "DO NOT change function signature" constraints in the prompt AND submitting the COMPLETE source file (not just the kernel function).
- **GEAK prompt must include full original signature**: For complex kernels, paste the COMPLETE function signature and explicitly say "DO NOT change the function signature, parameter names, or parameter order. Only optimize the loop body."
- **GEAK model config**: Each user configures their own model backend via `geak_set_model_config`. Recommended: `amd_llm` model class with `claude-opus-4-6` for best optimization results.
- **GEAK workspace**: Always use `workspace_id: "control-plane-prod"` in `geak_create_task`. Default workspace is resource-constrained.
- **GEAK block size changes can cause OOM** (validated 2026-03-21): GEAK may aggressively increase block sizes (e.g., BLOCK_N 64→256), which 4x the per-tile register usage and causes Triton compilation OOM on GPU. The prompt template already constrains this — always use the template from Phase 7b.
- **GEAK function name**: In SGLang v0.5.10+, standalone Inductor files use `def triton_mm(...)` but graph modules use `def triton_tem_fused_mm_0(...)`. The GEAK prompt MUST specify the EXACT original function name. The patching script (Phase 8 Strategy A) handles renaming.
- For full GEAK workflow details, see `GEAK-INFERENCE-KERNEL.md` in this folder.

### CUDA Graph vs GEAK (CRITICAL)

SGLang uses **CUDA Graph** (`--cuda-graph-max-bs N`) which packs all kernels into a single `hipGraphLaunch` call. Without torch.compile, individual kernels are invisible:

```
hipGraphLaunch: 95.1% GPU time (all kernels inside, invisible to profiler)
aiter::rmsnorm:  0.11% GPU time (only graph-external calls)
```

**With torch.compile + piecewise CUDA Graph**: Inductor generates Triton kernels that ARE captured in the graph. Patching the Inductor cache → restarting → new CUDA graphs include the optimized kernels. This is why the torch.compile path works.

### Kernel Replacement in SGLang

- **Inductor cache patching** (best for torch.compile): Patch standalone .py kernel files, clear binary cache, restart server.
- **Direct source edit**: `cp file.py file.py.bak`, then replace kernel function.
- **Monkey-patch**: `importlib.import_module()` + `setattr()` before server launch.
- **Always restart server after kernel changes** — SGLang loads kernels at startup.
- **Server restart takes ~90s without torch.compile, ~180s with torch.compile** (compilation overhead).

### CUDA Graph Coverage (CRITICAL)

SGLang's `--cuda-graph-max-bs N` determines which batch sizes get CUDA graph capture. Batches larger than N fall back to eager mode, re-launching every kernel individually.

**The default `--cuda-graph-max-bs` is often too low.** SGLang auto-detects based on `--cuda-graph-bs` list, but for models with small per-kernel compute, CUDA graph eliminates kernel launch overhead that dominates decode time.

**Validated results (gpt-oss-120b, TP=1, ISL=1024, OSL=256, with decode-steps=8):**

| Config | CONC=4 | CONC=8 | CONC=16 |
|--------|--------|--------|---------|
| `cuda-graph-max-bs=4` | 793 tok/s | 986 tok/s | 1745 tok/s |
| `cuda-graph-max-bs=16` | **1073 tok/s (+35%)** | **1231 tok/s (+25%)** | **1917 tok/s (+10%)** |

**Note**: These baselines (793/986/1745) already include `--num-continuous-decode-steps 8`. The +35% is the **isolated** CUDA graph effect.

**Rule**: Always set `--cuda-graph-max-bs` to at least the maximum expected concurrent batch size. For CONC=N, use `--cuda-graph-max-bs N` or higher.

**How to detect this issue in traces**: If the profiler shows many small kernels (4-5μs each) being launched individually instead of inside `hipGraphLaunch`, the CUDA graph coverage is insufficient. Check `cuda_graph_bs` in server log — if the benchmark's actual batch sizes exceed the max captured bs, that's the problem.

### Benchmark Metrics

| Metric | Unit | Meaning |
|--------|------|---------|
| `output_throughput` | tok/s | Output tokens generated per second |
| `mean_tpot_ms` | ms | Time Per Output Token (decode latency) |
| `mean_ttft_ms` | ms | Time to First Token (prefill latency) |
| `mean_itl_ms` | ms | Inter-Token Latency |
| Interactivity | tok/s/user | `1000 / TPOT_ms` |
