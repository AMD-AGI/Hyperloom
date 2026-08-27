---
title: gemm_epilogue_fused on ck — SOTA card
kind: sota_card
operator: gemm_epilogue_fused
backend: ck
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, mxfp4]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-int8-gemm-sq/README.html
  - https://github.com/ROCm/composable_kernel
---

# gemm_epilogue_fused × ck

## TL;DR
> CK is the **richest epilogue-fusion substrate** on AMD — `DeviceGemmMultipleD_Xdl_CShuffle` +
> `CDEElementwise` chains bias/residual/activation/scale/quant into the CShuffle store. Choose it to
> author any non-trivial fused GEMM (gated act, residual-add, fp8/int8/mxfp4 output) that hipBLASLt's
> fixed epilogues don't cover. Build-time; engage via your op or as an aiter candidate.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `DeviceGemmMultipleD_Xdl_CShuffle` + `CDEElementwise` | `ROCm/composable_kernel` (→ rocm-libraries) | gfx942/950; bf16/fp16/fp8/int8/mxfp4 | qualitative SOTA for custom epilogues (no single TFLOP figure; matmul core ≈ hipBLASLt, epilogue ≈ free) | gated act / residual / quant fusion not in hipBLASLt |
| `03_gemm_bias_relu` (reference pattern) | ROCm CK GEMM tutorial | gfx942 | the canonical bias+relu fusion template | learning / extending the epilogue |
| CK int8 SmoothQuant fused GEMM | ROCm CK-int8 blog | gfx942; int8 W8A8 | qualitative serving win | int8 quantized models with fused epilogue |

## Config space / knobs
- GEMM core: `MPerBlock/NPerBlock/KPerBlock`, `MPerXDL/NPerXDL` (16×16), XdlPerWave, BlockGemmPipeline.
- Epilogue: `CDEElementwise` functor composing the ops; `CShuffleM/NXdlPerWavePerShuffle`,
  `CDEBlockTransferScalarPerVector` sized to output dtype; `OPTIMIZE_EPILOGUE` equivalent via CShuffle.
- Apply activation in fp32 on the accumulator before down-cast (see [numerics.md](numerics.md)).

## Numerics / parity
fp32 epilogue before down-cast → equal-or-better than unfused; quant epilogue task-gated. See
[numerics.md](numerics.md).

## Integration (rebind seam)
Build-time C++ extension; call directly or register as an aiter candidate. No env-overlay into serving.

## Pitfalls & anti-patterns
- Mismatched instance tile → matmul below hipBLASLt; enumerate via ckProfiler.
- Building all DTYPES → huge compile; restrict.
- Writing the epilogue in low precision → accuracy loss.

## How to verify
`ckProfiler` for the fused op vs (hipBLASLt GEMM + separate elementwise) on the same shape; adopt on a
win AND a call seam; gate quant on task eval.

## Alternatives / cross-links
aiter (live, limited epilogues) · hipblaslt · triton ·
hip · [overview.md](overview.md) · language ref languages/composable_kernel/....

## Sources
- CShuffle / MultipleD / CDEElementwise: ROCm "Optimizing with Composable Kernel".
- CK int8 SmoothQuant fused GEMM: ROCm CK-int8-GEMM-SQ blog.
- CK repo: https://github.com/ROCm/composable_kernel.
