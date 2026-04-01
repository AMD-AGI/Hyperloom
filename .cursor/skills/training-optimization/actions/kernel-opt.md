# Action: Kernel Optimization (GEAK)

## Overview

Submits hot GPU kernels to GEAK MCP for AI-driven kernel-level optimization.
This action is **optional** — only triggered when profiling identifies custom kernels
(Triton, HIP) that consume >2% of total GPU time and are NOT vendor-optimized.

For the full GEAK workflow, see `GEAK-KERNEL-OPTIMIZATION.md`.

## Inputs
- `kernel_candidates` from profile step
- Profile trace for shape/dtype context
- Current kept_overrides

## Eligibility Rules

| Kernel Type | GEAK? | Reason |
|-------------|-------|--------|
| `Cijk_*` (hipBLASLt GEMM) | **No** | Vendor BLAS, hand-tuned MFMA |
| `aiter::fmha_v3_*` | **No** | Vendor attention, optimized for gfx950 |
| `triton_*` / `_permute_kernel` | **Yes** | Triton kernels have Python source |
| Custom HIP `__global__` | **Yes** | Primary GEAK target |
| `vectorized_elementwise_kernel` | **Maybe** | Try torch.compile first |
| NCCL kernels | **No** | Communication, not compute |

**Decision rule:** top-5 by GPU time + has modifiable source + NOT vendor = GEAK candidate.

## Procedure

### Step 1: For each candidate, find source

**Triton kernels:**
```bash
rg "@triton.jit" "$PRIMUS_ROOT/"
rg "def <kernel_name>" "$PRIMUS_ROOT/"
# Check installed packages
python3 -c "import primus_turbo; import os; print(os.path.dirname(primus_turbo.__file__))"
```

**torch.compile generated Triton:**
```bash
ls /tmp/torchinductor_*/*/triton/*.py | head -20
```

**Custom HIP kernels:**
```bash
rg "void <kernel_name>" "$PRIMUS_ROOT/" --glob "*.{hip,cu,cuh}"
```

### Step 2: Extract kernel source with context

Create a standalone file with:
1. The kernel function source
2. Input shapes and dtypes (from profiler trace)
3. Hardware context (MI355X / gfx950 / CDNA4)
4. Current performance (GPU time %, call count)

### Step 3: Submit to GEAK via MCP

See `GEAK-KERNEL-OPTIMIZATION.md` Phase 3 for the full MCP tool sequence:
1. `geak_set_model_config` — configure LLM backend
2. `geak_create_task` — with `input_type: "file"`, kernel source, and optimization prompt
3. `geak_submit_task` — start optimization
4. Poll `geak_get_task` every 30s until complete (10–30 min)
5. `geak_get_outputs` + `geak_download_file` — retrieve optimized kernel

**Submit one kernel per GEAK task.** Multi-kernel tasks produce worse results.

### Step 4: Validate GEAK output

Before integration:
1. **Compile check:** import/compile the optimized kernel
2. **Correctness check:**
   ```python
   orig_out = original_kernel(test_input)
   geak_out = geak_kernel(test_input)
   assert torch.allclose(orig_out, geak_out, atol=1e-2, rtol=1e-2)
   ```
3. **Micro-benchmark:** compare latency of original vs GEAK kernel

If micro-benchmark shows regression, skip integration.

### Step 5: Dispatch to integrate action

If GEAK kernel passes validation, dispatch to `actions/integrate.md` for
full training benchmark.

## Outputs
- Per-kernel GEAK results (task_id, micro-benchmark speedup, status)
- Validated optimized kernel files
- Integration candidates for `actions/integrate.md`

## Failure Handling
- GEAK task stuck >30 min: cancel and retry, or skip
- GEAK output doesn't compile: fix obvious issues, or skip
- GEAK output produces wrong results: skip, log to KB
- Micro-benchmark regression: skip, log to KB
