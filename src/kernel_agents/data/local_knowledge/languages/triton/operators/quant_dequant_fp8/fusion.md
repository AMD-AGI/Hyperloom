---
title: quant_dequant_fp8 — fusion
kind: operator_overview
operator: quant_dequant_fp8
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/quant/fused_fp8_quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/rmsnorm_quant_kernels.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/gated_rmsnorm_quant_kernels.cu
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
---

# quant_dequant_fp8 — fusion

> Standalone FP8 quant is a wasted HBM round-trip. The win is to fold the cast into the op that **produces**
> the activation (norm/act) or the op that **consumes** it (GEMM epilogue / KV store). Every fusion below
> removes one full-tensor read+write and one kernel launch.

## 1. norm + quant (the biggest one) → operators/fused_norm_quant
The activation entering a linear is almost always RMSNorm's output. Fuse:
`y = rmsnorm(x) ; (y_fp8, s) = quant(y)` → **one kernel** that normalizes and emits FP8 + scale.
aiter ships several, per granularity:
- `fused_rms_fp8_per_tensor_static_quant` — RMSNorm + static per-tensor FP8.
- `fused_rms_fp8_group_quant` — RMSNorm + per-group (block) FP8.
- `fused_reduce_rms_fp8_group_quant` — residual-add + RMSNorm + group FP8 (the residual stream variant).
- HIP: `csrc/kernels/rmsnorm_quant_kernels.cu`, `gated_rmsnorm_quant_kernels.cu`
  (`gated_rmsnorm_fp8_group_quant` for the gated-MLP residual path).
All compute the RMS reduction and the amax in the **same pass** (the row is already in registers/LDS), so
the scale costs nothing extra. Engaged in vLLM by `VLLM_ROCM_USE_AITER_RMSNORM=1` (default 1, gated by
the `VLLM_ROCM_USE_AITER=1` master switch).

## 2. act_and_mul + quant
SwiGLU/GeGLU up·gate then SiLU·mul produces the down-proj input — fuse the FP8 cast into the activation:
- `fused_silu_mul_fp8_per_tensor_static_quant`, `fused_reduce_act_mul_fp8_group_quant`
  (`aiter/ops/triton/quant/fused_fp8_quant.py`). Saves a full pass over `[M, inter]` (inter ~14k–35k).
→ operators/act_and_mul_silu_gelu.

## 3. GEMM dequant epilogue (consumer side) → operators/scaled_quant_gemm
The FP8 GEMM accumulates in fp32; the **dequant `* s_a * s_b` is fused into the epilogue** (`gemm_a8w8`
applies the row scale `s_a[M,1]` and column/tensor scale `s_b` before writing bf16 out). The output never
materializes in fp8 — quant (input) and dequant (output) bracket the matrix core with zero extra passes.
This is the canonical FP8 linear: `quant(x) → gemm_a8w8(x_fp8, w_fp8, s_a, s_b) → bf16`.

## 4. KV-store quant → operators/kv_cache_quant
After RoPE, the K/V are cast to FP8 **as they are written into the paged cache** — fused into the
reshape/cache-write so the bf16 KV never hits HBM:
- `fused_qk_norm_rope_cache_quant_shuffle`, `fused_qk_norm_rope_cache_pts_quant_shuffle` (per-tensor),
  `fused_qk_norm_rope_cache_block_quant_shuffle` (block) — QK-norm + RoPE + KV-write + FP8 quant + layout
  shuffle, all in one kernel (`aiter/ops/fused_qk_norm_rope_cache_quant.py`).
- vLLM `reshape_and_cache`/`reshape_and_cache_flash` do `fp8::scaled_convert<cache_t, scalar_t>(...,
  k_scale/v_scale)` inline. → operators/paged_kv_copy.

## Fusion decision
| producer of the FP8 input | fuse into |
|---|---|
| RMSNorm output → linear | `fused_rms_fp8_*_quant` (#1) |
| residual-add + RMSNorm | `fused_reduce_rms_fp8_group_quant` (#1) |
| SiLU·mul (SwiGLU) → down-proj | `fused_silu_mul_fp8_*_quant` (#2) |
| GEMM output → next linear | the GEMM's **own** quant epilogue (#3) |
| K/V after RoPE → cache | `fused_qk_norm_rope_cache_quant` / `reshape_and_cache` (#4) |
| nothing fusible (rare) | standalone `dynamic_per_token_scaled_quant` (tuning.md) |

## Pitfalls
- Fusing the **amax** with the producer is only correct if the producer emits the *final* activation — if
  a residual add happens after the norm, use the `reduce_*` (residual-aware) variant or the scale is wrong.
- Block/group fusion must agree on group size (128 for fp8 block) end-to-end with the consuming GEMM.
- Per-token amax inside a fused norm needs the **full row** in the kernel — fine for RMSNorm (row-wise),
  not for column-parallel splits.

## Sources
- Fused norm/act + FP8 quant ops: `ROCm/aiter@a6bb49937:aiter/ops/triton/quant/fused_fp8_quant.py`,
  `csrc/kernels/rmsnorm_quant_kernels.cu`, `csrc/kernels/gated_rmsnorm_quant_kernels.cu`.
- Fused QK-norm+RoPE+KV-write+quant: `ROCm/aiter@a6bb49937:aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- vLLM AITER RMSNorm/quant gate: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
