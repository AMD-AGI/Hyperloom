---
title: quant_fp4_mxfp — fusion
kind: operator_overview
operator: quant_fp4_mxfp
gens: [gfx950]
dtypes: [mxfp4, mxfp6]
regimes: [both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/fused_mxfp4_quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# quant_fp4_mxfp — fusion

> MXFP quant fuses exactly like FP8: into the producing norm/act, the consuming block-scaled GEMM, and the
> MoE sort. aiter ships a dedicated `fused_mxfp4_quant.py` family. Each fusion removes a full read+write of
> the activation and a launch.

## 1. norm + MXFP4 → operators/fused_norm_quant
- `fused_rms_mxfp4_quant` — RMSNorm + per-32-block MXFP4 cast + E8M0 scale, one pass.
- `fused_reduce_rms_mxfp4_quant` — residual-add + RMSNorm + MXFP4 (residual-stream variant).
- `fused_flatten_mxfp4_quant` — flatten + MXFP4 (for reshaped activations).
The RMS reduction and the per-block amax happen with the row already in registers; the E8M0 scale costs
nothing extra (`aiter/ops/triton/quant/fused_mxfp4_quant.py`).

## 2. act_and_mul + MXFP4 → operators/act_and_mul_silu_gelu
- `fused_reduce_act_mul_and_mxfp4_quant` — SiLU·mul (SwiGLU) + MXFP4 cast for the down-proj input. Saves a
  full pass over `[M, inter]`.

## 3. MoE sort + MXFP4 → operators/fused_moe_grouped_gemm / operators/moe_dispatch_combine
- `fused_dynamic_mxfp4_quant_moe_sort` — fuses the expert-sort/permute with the MXFP4 cast so each routed
  token group is quantized in the same pass. `fused_quant_fp8_sort` is the FP8 analog.
- `mxfp4_moe_sort` / `mxfp4_quant_moe_sort_kernel` (HIP) — the sort+quant kernels.

## 4. block-scaled GEMM (consumer) → operators/scaled_quant_gemm
The block-scaled MFMA `v_mfma_scale_f32_*_f8f6f4` **consumes the E8M0 scales directly** — the "dequant" is
the scale-apply inside the matrix core (after the dot, before accumulate), so there is no separate dequant
op. The quant just needs the scales in the **shuffled** Ax/Bx layout (`shuffle=True`); then quant → GEMM
is seamless. This is the tightest fusion: the scale never round-trips as a separate dequant pass.

## Fusion decision
| producer of the MXFP4 input | fuse into |
|---|---|
| RMSNorm output → linear | `fused_rms_mxfp4_quant` (#1) |
| residual-add + RMSNorm | `fused_reduce_rms_mxfp4_quant` (#1) |
| SiLU·mul → down-proj | `fused_reduce_act_mul_and_mxfp4_quant` (#2) |
| MoE routed tokens | `fused_dynamic_mxfp4_quant_moe_sort` (#3) |
| weights (offline) | Quark `w_mxfp4_*` (baked, no runtime quant) |
| GEMM with block-scaled MFMA | feed E8M0 scales directly (#4) |

## Pitfalls
- Fused norm + MXFP4 must use the **residual-aware** variant if a residual add follows the norm.
- The fused output scales must be **shuffled** if they feed the HW MFMA (set `shuffle=True`).
- MoE: per-expert / per-group block scales must follow the routed permutation.
- All of this is **gfx950-only** for a HW win; on gfx942 the fused quant only saves footprint (GEMM
  simulates).

## Sources
- Fused MXFP4 norm/act/MoE quant: `ROCm/aiter@a6bb49937:aiter/ops/triton/quant/fused_mxfp4_quant.py`.
- MoE sort+quant HIP kernels: `ROCm/aiter@a6bb49937:csrc/kernels/quant_kernels.cu` (`mxfp4_*moe_sort*`), `aiter/ops/quant.py`.
- Block-scaled MFMA consumes E8M0 directly: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
