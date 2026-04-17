# Action: Kernel Optimization (GEAK)

## Overview

Submits hot GPU kernels to GEAK CLI (REST API) for AI-driven kernel-level optimization.
This action is **optional** — only triggered when profiling identifies custom kernels
(Triton, HIP) that consume >2% of total GPU time and are NOT vendor-optimized.

For the full GEAK workflow, see the training-optimization skill's
`GEAK-KERNEL-OPTIMIZATION.md`.

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
| `cast_transpose` Triton | **Yes** | FP8 cast+transpose, can be optimized |
| NCCL kernels | **No** | Communication, not compute |

## Procedure

Same as training-optimization skill's `actions/kernel-opt.md`:
1. Find kernel source
2. Extract with context (shapes, dtypes)
3. Submit to GEAK via CLI
4. Validate GEAK output (compile, correctness, micro-benchmark)
5. If passes, dispatch to `actions/integrate.md`

## Outputs
- Per-kernel GEAK results
- Validated optimized kernel files
- Integration candidates

## Failure Handling
- GEAK task stuck >30 min: cancel and retry, or skip
- GEAK output doesn't compile: fix obvious issues, or skip
- GEAK output produces wrong results: skip, log to KB
