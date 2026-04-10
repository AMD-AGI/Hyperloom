---
name: geak-inference-kernel-reference
description: Deep reference for GEAK kernel optimization in inference serving. Covers MCP tool details, kernel extraction methods, integration paths, edge cases, and troubleshooting. Referenced by SKILL.md Phase 5 (identify candidates), Phase 7 (GEAK optimization loop), and Phase 8 (integrate + benchmark).
---

# GEAK Inference Kernel Optimization — Deep Reference

This document provides detailed reference material for GEAK kernel optimization in the inference optimization loop defined in `SKILL.md`. The main skill covers the workflow; this file covers the details.

## Inference Kernel Categories

### Kernel identification from TraceLens

| Kernel pattern | Framework | Source available? | GEAK target? |
|----------------|-----------|-------------------|-------------|
| `Cijk_Ailk_Bljk_*` | hipBLASLt | No (compiled) | No — vendor BLAS |
| `aiter::fmha_v3_fwd` | aiter | No (.so) | No — vendor attention |
| `aiter::mha_fwd` | aiter | No (.so) | No — switch backend instead |
| `moe_ck2stages_gemm*` | aiter | No (.so) | No — vendor fused MoE |
| `triton_*` from SGLang | SGLang | Yes (Python) | **Yes** |
| `triton_poi_*`, `triton_red_*` | torch.compile | Yes (Inductor cache) | **Yes** |
| `vectorized_elementwise_kernel` | PyTorch | No (C++) | Maybe — try torch.compile first |
| Custom HIP `__global__` | User code | Yes | **Yes** |

### Where to find kernel source

**SGLang Triton kernels:**
```bash
SGLANG_PATH=$(python3 -c "import sglang; import os; print(os.path.dirname(sglang.__file__))")
rg "@triton.jit" "$SGLANG_PATH" --files-with-matches
# Common locations:
# srt/layers/attention/triton_ops/
# srt/layers/moe/triton_kernels/
# srt/layers/quantization/
```

**torch.compile Inductor kernels:**
```bash
ls /tmp/torchinductor_*/*/triton/*.py | head -20
```

**aiter kernels (NOT GEAK targets — reference only):**
```bash
AITER_PATH=$(python3 -c "import aiter; import os; print(os.path.dirname(aiter.__file__))")
ls "$AITER_PATH/jit/"    # Compiled .so files
ls "$AITER_PATH/ops/"    # Python dispatch wrappers
```

## GEAK Configuration — DO NOT MODIFY (IR-10)

**GEAK is external read-only infrastructure.** The skill MUST NOT modify any GEAK
configuration files, server settings, workspace configs, test data, or results files.
The ONLY allowed interaction is through GEAK MCP tool calls listed below. Modifying
GEAK config (e.g., `server/config.py`, `cursor_mcp_config.json`, `tests/test_data/`,
workspace settings) is an Iron Rule violation and invalidates the entire run.

## GEAK MCP Tool Reference

### Authentication

Requires two keys:
- `GEAK_AUTH_KEY` — Bearer token for GEAK endpoint (set in `.env`)
- `LITELLM_API_KEY` — Used internally by GEAK to call its LLM backend

### Tool sequence

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `geak_set_model_config` | Configure LLM backend (once per session) |
| 2 | `geak_create_task` | Create task with source + instructions |
| 3 | `geak_submit_task` | Start optimization |
| 4 | `geak_get_task` | Poll status (every 30s) |
| 5 | `geak_get_outputs` | List output files |
| 6 | `geak_download_file` | Download optimized code |
| - | `geak_list_tasks` | Debug: list all tasks |
| - | `geak_get_model_config` | Debug: check LLM config |

### geak_create_task — critical details

- `input_type` is **required** — use `"file"`
- The instruction field is `prompt`, NOT `instructions`
- `step_limit` controls agent iterations (**use 100** for kernel optimization — GEAK needs room to analyze, write, compile, fix errors, benchmark, and iterate. 20 is often not enough for a verified result; 5 is completely insufficient)
- `gpu_count` defaults to 1
- **`workspace_id`**: Always specify `GEAK_WORKSPACE` (constant from `SKILL.md`; default `"control-plane-moe"`) for reliable scheduling. Default workspace is often resource-constrained.
- Include ALL dependent files in the `files` array (GEAK needs self-contained code)

### Prompt template for inference kernels

```
Optimize this Triton kernel for AMD MI355X (gfx950, CDNA4).

Hardware: 304 CUs, 256 VGPR/CU, HBM3e ~8 TB/s, MFMA instructions.
Context: LLM inference serving (decode path).
Input shapes: [exact shapes from TraceLens profile]
Data types: bf16 activations, fp8_e4m3 weights/KV cache.
Currently: {kernel_time_ms}ms per call, {gpu_pct}% of total GPU time.
Called {count} times per batch of {batch_size} requests.

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

Write the COMPLETE file (imports, decorator, function) to the output directory.
```

**GEAK prompt rules — apply to ALL kernel types (MANDATORY for every geak_create_task):**

1. **Kernel path — conditional on image availability:**
   - If the kernel source file **exists in the Docker image** (e.g., `/sgl-workspace/aiter/...`, `/opt/venv/...`), **MUST include** the kernel's absolute file path and repo path in the prompt. Example: `"The kernel source file is at /sgl-workspace/aiter/jit/core/compile.py"`, `"The kernel repo is at /sgl-workspace/aiter/"`.
   - If the kernel source is **runtime-generated** and only exists at runtime (e.g., `/tmp/torchinductor_root/...` from `torch.compile` Inductor cache), **DO NOT include** `kernel_url` or `kernel_repo` in the prompt. These files do not exist in the GEAK pod's image. Instead, copy the kernel files to a shared NFS path and reference the NFS path, OR omit these paths entirely and rely solely on `files[].content`.
   - **How to tell:** paths under `/tmp/`, `/root/.cache/`, or any `torchinductor_*` directory are runtime-generated. Paths under `/sgl-workspace/`, `/opt/`, `/usr/` are part of the image.
2. **MUST specify homogeneous mode and max_rounds** — Always include: `"Use homogeneous mode. Set max_rounds to 1."` in the prompt.
3. **MUST specify 1.5x minimum speedup target** — Always include: `"The kernel MUST be optimized to at least 1.5x speedup."` in the prompt.

Additional rules:
4. **Always say "Do NOT search the filesystem with find / or grep -r /"** — GEAK agents default to broad filesystem searches which hang 30+ min on NFS.
5. **Always pass framework image** — In claw mode, use `GEAK_IMAGE_SGLANG_RAY` (for SGLang) or `GEAK_IMAGE_VLLM` (for vLLM). In local mode, use `GEAK_IMAGE_SGLANG` or `GEAK_IMAGE_VLLM`.
6. **Always embed full source in `files[].content`** — GEAK always receives the kernel source via `files[].content`. If the path also exists in the image, include it in the prompt for GEAK's preprocessor. If the path is runtime-generated, the `files[].content` is the sole source of truth.

### GEAK Image Selection

| Condition | Local Mode | Claw Mode |
|-----------|-----------|-----------|
| User specified a custom image | Use user-specified | Use user-specified |
| `FRAMEWORK=sglang` (default) | `GEAK_IMAGE_SGLANG` | `GEAK_IMAGE_SGLANG_RAY` |
| `FRAMEWORK=vllm` | `GEAK_IMAGE_VLLM` | `GEAK_IMAGE_VLLM` |

**Claw mode MUST use `GEAK_IMAGE_SGLANG_RAY`** (not `GEAK_IMAGE_SGLANG`) for SGLang.

**Always pass `workspace_id: GEAK_WORKSPACE` (default `"control-plane-moe"`) in `geak_create_task`.** User can override.

### GEAK latency breakdown

| Phase | Duration | Notes |
|-------|----------|-------|
| Pod scheduling | 2-15 min | Depends on cluster load, GPU availability |
| Docker image pull | 1-5 min | ROCm image is ~15GB |
| Agent execution | 3-10 min | Depends on step_limit |
| **Total** | **10-30 min** | Poll every 30s |

The `updated_at` timestamp stays frozen until the pod starts. Once it changes, the agent has started. If stuck >30 min with no update, the cluster may be overloaded — cancel and retry.

## Kernel Integration Paths

### Path A: Monkey-patch at import time (recommended for quick iteration)

```python
#!/usr/bin/env python3
"""patch_kernel.py — Apply GEAK-optimized kernel before server launch."""
import importlib
import sys

def apply_patches():
    # Load the GEAK-optimized kernel
    sys.path.insert(0, "/path/to/geak/outputs")
    from geak_optimized import optimized_kernel_func

    # Patch it into SGLang
    target_module = importlib.import_module("sglang.srt.layers.xxx.kernels")
    original = getattr(target_module, "target_kernel_name")

    setattr(target_module, "target_kernel_name", optimized_kernel_func)
    print(f"Patched {target_module.__name__}.target_kernel_name")

    return original  # Keep reference for rollback

if __name__ == "__main__":
    apply_patches()
```

Launch with patch:
```bash
python3 -c "import patch_kernel; patch_kernel.apply_patches(); import sglang; ..." # won't work for server

# Instead, modify the launch:
python3 -c "
import patch_kernel
patch_kernel.apply_patches()
from sglang.srt.server import launch_server
# ... launch args ...
"
```

### Path B: Direct source edit (recommended for validation)

```bash
# Backup
cp "$SGLANG_PATH/srt/layers/xxx/kernels.py" "$WORK_DIR/kernels/kernels.py.bak"

# Replace kernel body (use agent's StrReplace tool)
# ...

# To rollback:
cp "$WORK_DIR/kernels/kernels.py.bak" "$SGLANG_PATH/srt/layers/xxx/kernels.py"
```

### Path C: Inductor cache patching (for torch.compile kernels)

If the kernel was extracted from torch.compile's Inductor cache:

```python
# Patch the Inductor-generated .py file
INDUCTOR_FILE = "/tmp/torchinductor_root/xx/xxx.py"
# Backup original
shutil.copy(INDUCTOR_FILE, f"{WORK_DIR}/kernels/{os.path.basename(INDUCTOR_FILE)}.bak")
# Replace kernel function body with GEAK output
# Clear Triton binary cache to force recompilation
import shutil
shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)
# Run benchmark in a FRESH process (Inductor loads at import time)
```

**Caveats for Path C:**
- Must use `torch.compile(mode='default')`, NOT `mode='reduce-overhead'`
- GEAK must preserve the exact function signature (args + constexprs)
- Clear `.json` metadata files to force Inductor to reload source

**Recommended: Use `patch_inductor.py` (IR-8):**

```bash
# Patch kernel source + update .best_config tiling parameters
python3 $SCRIPTS_DIR/patch_inductor.py patch \
    --kernel-name <name> \
    --geak-file <geak_output.py> \
    --target-file <inductor_standalone_file.py> \
    --best-config '{"XBLOCK": 4, "R0_BLOCK": 2048, "num_warps": 4}'

# Revert (restores both .py and .best_config from .bak):
python3 $SCRIPTS_DIR/patch_inductor.py revert --target-file <inductor_standalone_file.py>
```

`patch_inductor.py` preserves `@triton_heuristics`, `inductor_meta`, and launcher config while only replacing the `@triton.jit def` function body.

**CRITICAL:** Always pass `--best-config` when the GEAK-optimized kernel uses different block sizes or warp counts than the original. The `.best_config` file controls Inductor's autotuner launch parameters — a mismatch between kernel code and `.best_config` causes numerical corruption.

## Correctness Verification

Always verify before benchmarking:

```python
import torch

# Prepare test input matching the kernel's expected shapes
test_input = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)

# Run both
orig_out = original_kernel(test_input)
geak_out = geak_kernel(test_input)

# Compare with relaxed tolerance (bf16 precision)
max_diff = (orig_out - geak_out).abs().max().item()
if not torch.allclose(orig_out, geak_out, atol=1e-2, rtol=1e-2):
    print(f"INCORRECT: max diff = {max_diff}")
    # Do NOT proceed to benchmark — log as "crash" and revert
else:
    print(f"Correct: max diff = {max_diff}")
```

## Micro-Benchmark (Optional)

Before full server restart, quickly test kernel speed in isolation:

```python
import torch, time

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

Before patching any GEAK output into the serving environment:

- [ ] Function name matches original exactly (not renamed)
- [ ] Function signature (parameters, constexprs) matches original
- [ ] Decorators preserved (`@triton_heuristics`, `@triton.jit`, etc.)
- [ ] No new imports that don't exist in the target environment
- [ ] Block sizes within IR-8 constraints (not exceeding 2x original)
- [ ] Source code is actual code, not comments or path references
- [ ] `files[].content` contains the full source (not truncated)
- [ ] `.best_config` values identified from GEAK output (XBLOCK, R0_BLOCK, BLOCK_N, BLOCK_K, num_warps, num_stages) and passed via `--best-config`

## Troubleshooting

### GEAK output doesn't compile
- Fix obvious issues (missing imports, wrong types)
- If unfixable, log as `crash` and move to next candidate

### GEAK output is slower
- Common with Triton on AMD CDNA4 — GEAK may suggest block sizes that cause register pressure
- Log as `discard` and revert
- Try providing more specific hardware constraints in the prompt

### Server won't start after kernel patch
- Revert to backup: `cp "$WORK_DIR/kernels/xxx.bak" "$SGLANG_PATH/..."`
- Clear Python cache: `find "$SGLANG_PATH" -name "__pycache__" -exec rm -rf {} +`

### GEAK task stuck in pending
- Pod scheduling can take 15+ min if cluster is loaded
- Check with `geak_get_task` — if `updated_at` hasn't changed in 30 min, cancel and retry
- Always use `workspace_id: GEAK_WORKSPACE` for reliable scheduling (default `"control-plane-moe"` from `SKILL.md`; default workspace is resource-constrained)
