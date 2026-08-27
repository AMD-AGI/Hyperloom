---
title: fused_add_rmsnorm — fusion neighbors
kind: technique
operator: fused_add_rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/rmsnorm.py
  - https://github.com/sgl-project/sglang/issues/18466
---

# fused_add_rmsnorm — fusion

This op is already a fusion (add+norm). The stacking ladder:

## 1. + fp8/int8 dynamic quant → residual+norm+quant triple → fused_norm_quant
`rmsnorm2d_fwd_with_add_dynamicquant` / `add_rmsnorm_quant`: `r'=x+r` → `y=rmsnorm(r')` → `y_fp8 =
quant(y)` in one kernel. `residual_out` stays bf16; `y` is fp8 → down-proj/qkv GEMM reads ½ bytes. Part of
SGLang's **1–6% e2e** Qwen3 norm+quant fusion (#18466). Cross-link quant_dequant_fp8, quant_int8.

## 2. + all-reduce (TP) → fused_allreduce_rmsnorm
In tensor-parallel, the sublayer output is all-reduced before the residual-add+norm. Fusing the AR with
the add+norm hides the norm under the collective. aiter custom-AR + norm path. Cross-link
fused_allreduce_rmsnorm, allreduce.

## 3. QK-norm form (attention entry)
For Q/K RMSNorm before RoPE, the residual-add isn't present, but the same fused-norm machinery feeds
`fused_qk_norm_rope_cache_quant` — see rmsnorm fusion.md §3 and rope.

## Fusion table
| form | aiter entrypoint | adds |
|---|---|---|
| add + rmsnorm | `rmsnorm2d_fwd_with_add` / `add_rmsnorm` | baseline (this op) |
| + dynamic fp8 quant | `rmsnorm2d_fwd_with_add_dynamicquant` / `add_rmsnorm_quant` | fp8 y |
| + smoothquant int8 | `rmsnorm2d_fwd_with_add_smoothquant` | int8 y |
| + all-reduce (TP) | custom-AR + norm | hide under collective |

## torch.compile
Custom-op registered (`direct_register_custom_op`) → Inductor fuses around; ROCm fusion pass
(`rocm_aiter_fusion.py`) stitches add+rms+quant chains in the compiled graph. See
backends/vllm_kernels/aiter_integration.

## Sources
- aiter add+norm+quant variants: `/sgl-workspace/aiter/aiter/ops/rmsnorm.py` (`rmsnorm2d_fwd_with_add_dynamicquant`, `add_rmsnorm_quant`, `_with_add_smoothquant`).
- 1–6% e2e Qwen3 norm+quant: https://github.com/sgl-project/sglang/issues/18466.
