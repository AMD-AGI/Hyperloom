---
title: fused_add_rmsnorm — overview
kind: operator_overview
operator: fused_add_rmsnorm
gens: [gfx908, gfx90a, gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/rmsnorm.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/normalization/rmsnorm.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://github.com/vllm-project/vllm/pull/14959
---

# fused_add_rmsnorm  (`r' = x + r; y = rmsnorm(r')` in one kernel)

## TL;DR
This is the **dominant on-serving-path form of RMSNorm**: a transformer block does `h = h + sublayer(...)`
and the next norm reads `h`, so the residual-add is fused into the norm. One read of `x`+`residual`, one
write of `residual_out`+`y` — instead of add(write h)→norm(read h). It is **memory-bound** and runs **2×
per layer**. This op IS the rmsnorm serving path; read rmsnorm first, this is the fused variant.

## Math contract
Inputs `x[M,N]` (sublayer output), `residual_in[M,N]`. Outputs: `residual_out = x + residual_in` (the new
residual stream, written for the next block), `y = rmsnorm(residual_out)·γ`. dtype: bf16/fp16 in, **fp32**
sum-of-squares accumulate, bf16/fp16 out; `residual_out` kept in the IO dtype (it's the running residual).
**Both outputs are written** — that's the key: the residual must persist for the skip connection.

## Shape regimes
Identical to rmsnorm: prefill `M=tokens` (row-per-program), decode `M=batch` (persistent
`min(M,num_sms)`), `N ∈ {4096,5120,8192}`. The extra `residual_in` read + `residual_out` write doubles the
traffic vs plain rmsnorm — making the fusion (vs two kernels) even more worthwhile.

## Where it matters (Amdahl)
It's the **2-per-layer** norm, so ~2–4% GPU time on a dense LLM, and the fusion saves the `h` round-trip on
every one. Combined further with fp8 quant (fused_norm_quant) it's part of SGLang's **1–6% e2e** Qwen3
gain. Engaged by `VLLM_ROCM_USE_AITER_RMSNORM=1` (vLLM PR #14959 wired `rmsnorm2d_fwd_with_add`).

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (HIP/opus `rmsnorm2d_fwd_with_add` / `add_rmsnorm`) | [aiter.md](aiter.md) |
| triton | 🟢 sota (`_fused_add_rmsnorm_kernel`; gfx1250 Gluon `fused_rmsnorm_add`) | triton |
| hip | 🟢 sota (vLLM `fused_add_rms_norm_kernel`) | hip |
| vllm_kernels | 🟢 sota (HIP + AITER wiring) | vllm_kernels |

## Fusion neighbors
This is itself a fusion. Stack further: `+fp8/int8 dynamic quant` → `rmsnorm2d_fwd_with_add_dynamicquant`
/ `add_rmsnorm_quant` (residual+norm+quant triple) → fused_norm_quant; `+all-reduce` (TP) →
fused_allreduce_rmsnorm. See [fusion.md](fusion.md).

## Numerics
fp32 accumulate over `residual_out`; γ fp32-promote; the add happens in IO dtype then the Σx² in fp32.
See [numerics.md](numerics.md).

## How to bench
`python3 op_tests/test_rmsnorm2d.py` with the `_with_add` path; oracle = `rmsnorm(x+r)` fp64; e2e A/B
toggling `VLLM_ROCM_USE_AITER_RMSNORM`.

## Sources
- aiter `rmsnorm2d_fwd_with_add` / `add_rmsnorm`: `aiter/ops/rmsnorm.py`.
- Triton `_fused_add_rmsnorm_kernel`: `aiter/ops/triton/normalization/rmsnorm.py`.
- vLLM HIP `fused_add_rms_norm_kernel`: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu.
- vLLM AITER with_add integration: https://github.com/vllm-project/vllm/pull/14959.
