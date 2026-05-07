---
name: geak-mlperf-kernel-reference
description: Deep reference for GEAK kernel optimization in MLPerf training. Covers MCP tool details, kernel extraction methods, integration paths, edge cases, and troubleshooting. Referenced by actions/kernel-opt.md Step 2.
---

# GEAK — Kernel Optimization Backend

GEAK backend for kernel optimization via the GEAK MCP (`geak`).
Runs on a remote GPU pod with the Primus training image — GEAK can compile,
benchmark, and validate kernels on-pod. Results are downloaded and verified
locally before integration via `actions/integrate.md`.

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GEAK_IMAGE` | `harbor.core42.example-internal-host.invalid/sync/tasimage/primus:gpt-oss-20b_training_6.0_2026-04-07-19-47-24_dev` | Primus training image for GEAK pod |
| `GEAK_WORKSPACE` | `core42-sandbox` | GEAK workspace for reliable scheduling |
| `GEAK_STEP_LIMIT` | 100 | Max agent steps per GEAK task |
| `GEAK_MAX_RETRIES` | 3 | Max submission retries per kernel |
| `GEAK_MAX_SUBMISSIONS` | 25 | Total GEAK submissions budget per run |
| `GEAK_TOP_CANDIDATES` | 10 | Number of top kernel candidates to submit |
| `GEAK_POLL_INTERVAL_S` | 30 | Seconds between GEAK task status polls |
| `GEAK_POLL_TIMEOUT_MIN` | 30 | Max minutes to poll a single GEAK task |

**GEAK image must be the Primus training image** (not an inference image like SGLang/vLLM).
MLPerf kernels come from Primus/Megatron/TransformerEngine — the GEAK pod needs those
dependencies to compile and validate optimized kernels.

## Comparison with Other Backends

| | GEAK (this) | OOB-Claude | OOB-Codex |
|---|---|---|---|
| **MCP** | GEAK | OOB Agent | OOB Agent |
| **Latency (full round)** | 10–30 min | 3–15 min | 2–6 min |
| **GPU on pod** | Yes | No | No |
| **Output** | Verified on pod | Locally verified | Locally verified |
| **Tool use** | Bash, profiling, submit | File I/O, shell, multi-step | File I/O, shell |
| **Best for** | Complex HIP, hardware-verified | Multi-step autonomous | Fast Triton rewrites |

## MLPerf Training Kernel Categories

### Kernel identification from TraceLens

| Kernel pattern | Framework | Source available? | GEAK target? |
|----------------|-----------|-------------------|-------------|
| `Cijk_Ailk_Bljk_*` | hipBLASLt | No (compiled) | No — vendor BLAS |
| `aiter::fmha_v3_*` | aiter | No (.so) | No — vendor attention |
| `triton_*` / `_permute_kernel` | Primus/TE | Yes (Python) | **Yes** |
| `cast_transpose` Triton | TransformerEngine | Yes (Python) | **Yes** |
| Custom HIP `__global__` | Primus/Megatron | Yes | **Yes** |
| NCCL kernels | RCCL | No | No — communication |

### Where to find kernel source

**Primus/TransformerEngine Triton kernels:**
```bash
PRIMUS_ROOT=$(python3 -c "import primus_turbo; import os; print(os.path.dirname(primus_turbo.__file__))")
rg "@triton.jit" "$PRIMUS_ROOT" --files-with-matches
rg "def <kernel_name>" "$PRIMUS_ROOT"

TE_ROOT=$(python3 -c "import transformer_engine; import os; print(os.path.dirname(transformer_engine.__file__))")
rg "@triton.jit" "$TE_ROOT" --files-with-matches
```

**torch.compile Inductor kernels:**
```bash
ls /tmp/torchinductor_*/*/triton/*.py | head -20
```

**Custom HIP kernels:**
```bash
rg "void <kernel_name>" "$PRIMUS_ROOT/" --glob "*.{hip,cu,cuh}"
rg "void <kernel_name>" /workspace/Primus/Megatron-LM/ --glob "*.{hip,cu,cuh}"
```

## GEAK MCP Tool Reference

### Authentication

Requires `GEAK_AUTH_KEY` — Bearer token for GEAK endpoint (configured in `.cursor/mcp.json`).

### Tool sequence

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `geak_set_model_config` | Configure LLM backend (once per session) |
| 2 | `geak_create_task` | Create task with source + instructions |
| 3 | `geak_submit_task` | Start optimization |
| 4 | `geak_get_task` | Poll status (every `GEAK_POLL_INTERVAL_S`) |
| 5 | `geak_get_outputs` | List output files |
| 6 | `geak_download_file` | Download optimized code |
| - | `geak_list_tasks` | Debug: list all tasks |
| - | `geak_get_model_config` | Debug: check LLM config |

### geak_create_task — critical details

- `input_type` is **required** — use `"file"`
- The instruction field is `prompt`, NOT `instructions`
- `step_limit` controls agent iterations (**use 100** — GEAK needs room to analyze, write, compile, fix errors, benchmark, and iterate. 20 is often not enough; 5 is completely insufficient)
- `gpu_count` defaults to 1
- **`workspace_id`**: Always specify `GEAK_WORKSPACE` (`"core42-sandbox"`) for reliable scheduling
- Include ALL dependent files in the `files` array (GEAK needs self-contained code)

### Configure LLM backend

```python
CallMcpTool(
    server="geak",
    toolName="geak_set_model_config",
    arguments={"temperature": 0.3}
)
```

### Create and submit task

```python
result = CallMcpTool(
    server="geak",
    toolName="geak_create_task",
    arguments={
        "input_type": "file",
        "prompt": PROMPT,  # see Prompt Template below
        "files": [{"name": kernel_filename, "content": kernel_source}],
        "image": GEAK_IMAGE,
        "workspace_id": GEAK_WORKSPACE,
        "step_limit": GEAK_STEP_LIMIT,
        "gpu_count": 1,
    }
)
task_id = result["task_id"]

CallMcpTool(
    server="geak",
    toolName="geak_submit_task",
    arguments={"task_id": task_id}
)
```

### Polling

```python
import time
start = time.time()
while (time.time() - start) < GEAK_POLL_TIMEOUT_MIN * 60:
    result = CallMcpTool(
        server="geak",
        toolName="geak_get_task",
        arguments={"task_id": task_id}
    )
    if result["status"] in ("completed", "failed"):
        break
    time.sleep(GEAK_POLL_INTERVAL_S)
```

If stuck beyond `GEAK_POLL_TIMEOUT_MIN` with no `updated_at` change, cancel and
retry (up to `GEAK_MAX_RETRIES`).

### Downloading Results

```python
outputs = CallMcpTool(
    server="geak",
    toolName="geak_get_outputs",
    arguments={"task_id": task_id}
)
for f in outputs["files"]:
    CallMcpTool(
        server="geak",
        toolName="geak_download_file",
        arguments={"task_id": task_id, "filename": f["name"]}
    )
```

## Prompt Template

```
Optimize this Triton kernel for AMD MI355X (gfx950, CDNA4).

Hardware: 304 CUs, 256 VGPR/CU, HBM3e ~8 TB/s, MFMA instructions.
Context: MLPerf GPT-OSS-20B training (Primus/Megatron).
Input shapes: [{shapes_from_trace}]
Data types: bf16/fp8 hybrid (E4M3 activations/weights, E5M2 gradients).
Currently: {kernel_time_ms}ms per call, {gpu_pct}% of total GPU time.
Called {count} times per training iteration.

MANDATORY CONSTRAINTS (violation = rejected):
1. The output function name MUST be EXACTLY: {original_function_name}. Do NOT rename it.
2. The function signature (parameter names, order, types) MUST be IDENTICAL to the original.
3. Block size limits: BLOCK_M <= 16, BLOCK_N <= 128, BLOCK_K <= 256. Larger blocks cause register OOM on MI355X.
4. Do NOT increase any block dimension beyond 2x its original value.
5. Do NOT add @triton.autotune or change @triton_heuristics decorators.

OPTIMIZATION TARGETS (prioritized):
1. STRUCTURAL: Hoist loop-invariant computations out of loops.
2. STRUCTURAL: Merge dual-pass into single-pass where possible.
3. TUNING: Adjust BLOCK sizes to match exact dimensions (e.g., BLOCK_M=M when M is small).
4. TUNING: Simplify grid indexing when dimensions are small.
5. MICRO: Use libdevice.rsqrt (NOT tl.math.rsqrt — unavailable in some Triton versions), multiply by reciprocal.

The kernel MUST be optimized to at least 1.5x speedup.
Use homogeneous mode. Set max_rounds to 1.
Do NOT search the filesystem with find / or grep -r /.
Write the COMPLETE file (imports, decorator, function) to the output directory.
```

**GEAK prompt rules — apply to ALL kernel types (MANDATORY for every geak_create_task):**

1. **Kernel path — conditional on image availability:**
   - If the kernel source file **exists in the Docker image** (e.g., `/workspace/Primus/...`, `/opt/...`), **MUST include** the kernel's absolute file path and repo path in the prompt.
   - If the kernel source is **runtime-generated** (e.g., `/tmp/torchinductor_root/...`), **DO NOT include** `kernel_url` or `kernel_repo` in the prompt. These files do not exist in the GEAK pod's image. Rely on `files[].content` only.
   - **How to tell:** paths under `/tmp/`, `/root/.cache/`, or any `torchinductor_*` directory are runtime-generated. Paths under `/workspace/`, `/opt/`, `/usr/` are part of the image.
2. **MUST specify homogeneous mode and max_rounds** — Always include: `"Use homogeneous mode. Set max_rounds to 1."` in the prompt.
3. **MUST specify 1.5x minimum speedup target** — Always include: `"The kernel MUST be optimized to at least 1.5x speedup."` in the prompt.
4. **Always say "Do NOT search the filesystem with find / or grep -r /"** — GEAK agents default to broad filesystem searches which hang 30+ min on NFS.
5. **Always pass framework image** — Use `GEAK_IMAGE` (Primus training image).
6. **Always embed full source in `files[].content`** — GEAK always receives the kernel source via `files[].content`. If the path also exists in the image, include it in the prompt for GEAK's preprocessor.

### GEAK latency breakdown

| Phase | Duration | Notes |
|-------|----------|-------|
| Pod scheduling | 2-15 min | Depends on cluster load, GPU availability |
| Docker image pull | 1-5 min | Primus image is ~15GB |
| Agent execution | 3-10 min | Depends on step_limit |
| **Total** | **10-30 min** | Poll every 30s |

The `updated_at` timestamp stays frozen until the pod starts. Once it changes, the agent has started. If stuck >30 min with no update, the cluster may be overloaded — cancel and retry.

## Kernel Integration Paths

### Path A: Source file patch (primary for distributed training)

Replace the kernel function in the Primus/Megatron/TE source file.

```bash
cp "$ORIGINAL_KERNEL_PATH" "${ORIGINAL_KERNEL_PATH}.bak"
# Replace kernel body with GEAK output (use agent's StrReplace tool)
# ...
# To rollback:
cp "${ORIGINAL_KERNEL_PATH}.bak" "$ORIGINAL_KERNEL_PATH"
```

### Path B: Monkey-patch at import time (least invasive)

```python
import importlib, sys

sys.path.insert(0, "/path/to/geak/outputs")
from geak_optimized import optimized_kernel_func

target_module = importlib.import_module("primus_turbo.kernels.xxx")
original = getattr(target_module, "target_kernel_name")
setattr(target_module, "target_kernel_name", optimized_kernel_func)
```

After patching, dispatch to `actions/integrate.md` for end-to-end benchmark
via `run_mlperf_trial`.

## Correctness Verification

Always verify before benchmarking:

```python
import torch

test_input = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)

orig_out = original_kernel(test_input)
geak_out = geak_kernel(test_input)

max_diff = (orig_out - geak_out).abs().max().item()
if not torch.allclose(orig_out, geak_out, atol=1e-2, rtol=1e-2):
    print(f"INCORRECT: max diff = {max_diff}")
else:
    print(f"Correct: max diff = {max_diff}")
```

## Micro-Benchmark

Before full training benchmark, quickly test kernel speed in isolation:

```python
import torch

x = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)

for _ in range(50): original_kernel(x); geak_kernel(x)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(200): original_kernel(x)
end.record(); torch.cuda.synchronize()
orig_ms = start.elapsed_time(end) / 200

start.record()
for _ in range(200): geak_kernel(x)
end.record(); torch.cuda.synchronize()
geak_ms = start.elapsed_time(end) / 200

print(f"Original: {orig_ms:.3f} ms | GEAK: {geak_ms:.3f} ms | Speedup: {(orig_ms-geak_ms)/orig_ms*100:.1f}%")
```

If micro-benchmark shows regression, skip full integration.

## GEAK Output Validation Checklist

Before patching any GEAK output into the training stack:

- [ ] Function name matches original exactly (not renamed)
- [ ] Function signature (parameters, constexprs) matches original
- [ ] Decorators preserved (`@triton_heuristics`, `@triton.jit`, etc.)
- [ ] No new imports that don't exist in the target environment
- [ ] Block sizes within constraints (not exceeding 2x original)
- [ ] Source code is actual code, not comments or path references
- [ ] `files[].content` contains the full source (not truncated)

## Troubleshooting

### GEAK output doesn't compile
- Fix obvious issues (missing imports, wrong types)
- If unfixable, log as `crash` and move to next candidate

### GEAK output is slower
- Common with Triton on AMD CDNA4 — GEAK may suggest block sizes that cause register pressure
- Log as `discard` and revert
- Try providing more specific hardware constraints in the prompt

### Training hangs after kernel patch
- Revert to backup: `cp "${ORIGINAL_KERNEL_PATH}.bak" "$ORIGINAL_KERNEL_PATH"`
- Clear Python cache: `find "$PRIMUS_ROOT" -name "__pycache__" -exec rm -rf {} +`

### GEAK task stuck in pending
- Pod scheduling can take 15+ min if cluster is loaded
- Check with `geak_get_task` — if `updated_at` hasn't changed in 30 min, cancel and retry
- Always use `workspace_id: GEAK_WORKSPACE` for reliable scheduling
