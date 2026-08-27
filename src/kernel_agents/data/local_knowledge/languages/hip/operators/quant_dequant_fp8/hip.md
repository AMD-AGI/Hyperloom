---
title: quant_dequant_fp8 on hip — SOTA card
kind: sota_card
operator: quant_dequant_fp8
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/quant_kernels.cu
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# quant_dequant_fp8 × hip

## TL;DR
HIP/C++ is the **editable source layer** beneath aiter and vLLM FP8 quant — the actual `__global__`
kernels live in `csrc/kernels/quant_kernels.cu` (aiter) and `csrc/quantization/fp8/common.cu` (vLLM).
Reach for HIP directly when you need a *custom* fusion the Python/Triton layer can't express, or to read
exactly what the cast/saturation does. It is bandwidth-bound; the craft is wide vectorized loads/stores
and a clean amax reduction. Use the arch-matching FP8 helpers (`__hip_fp8_*` CDNA3 / `__amd_fp8_*` gfx950).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `scaled_quant_impl` / `scaled_quant_vgpr_impl` | `aiter@a6bb49937:csrc/kernels/quant_kernels.cu:248,308` | gfx942/950, e4m3 | `thread_data_size=16/32`, wide `global_load`/`store` | the aiter HIP backend kernels |
| `static_per_tensor_quant` / `dynamic_per_token_scaled_quant` (host) | `:595,710` | gfx942/950 | dispatch + grid | the bound entrypoints |
| vLLM `scaled_fp8_conversion[_vec]` | `vllm:csrc/quantization/fp8/common.cu:43,148` | gfx942/950, e4m3fnuz | float4 vectorized, scale-inverted | vLLM's own path |

## Config space / knobs
- **`thread_data_size`** (16/32) — elements/thread → vector width; cap to avoid VGPR spill.
- **block size** multiple of 64 (256 typical); **grid ≥ 1024 WGs** to fill 304 CUs.
- `is_scale_inverted` → multiply not divide.
- `-munsafe-fp-atomics` for `atomicMaxFloat` in per-tensor dynamic amax.
- `__restrict__` for wider `global_load_dwordx4`.

## Numerics / parity
fp32 amax/scale; FP8 cast saturates to dialect max (FNUZ 240 / OCP 448; vLLM dynamic uses 224). Zero-scale
clamp to 1. Never bit-copy across FNUZ/OCP. → numerics.md, hardware/shared/dtype_numerics,
hip_levers.

## Integration (rebind seam)
Edit the `.cu`, rebuild aiter JIT (`AITER_LOG_MORE=1` shows the build) or vLLM. For a one-off custom
fusion, register via `compile_ops` (aiter) or `torch_bindings` (vLLM). Tier-C edit seam.

## Pitfalls & anti-patterns
- VGPR spill from too-large `thread_data_size` → scratch (HBM) → 3–5× slower.
- Forgetting `warpSize==64` in the row-reduce.
- FNUZ↔OCP helper mismatch.

## How to verify
`-Rpass-analysis=kernel-resource-usage` (VGPR/scratch); rocprofv3 bandwidth vs HBM peak; round-trip error.

## Alternatives / cross-links
aiter · vllm_kernels · triton ·
hip_levers · [overview.md](overview.md).

## Sources
- aiter HIP quant kernels: `ROCm/aiter@a6bb49937:csrc/kernels/quant_kernels.cu`.
- vLLM HIP fp8 conversion: `vllm-project/vllm@HEAD:csrc/quantization/fp8/common.cu`.
- HIP wave64/launch_bounds: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
