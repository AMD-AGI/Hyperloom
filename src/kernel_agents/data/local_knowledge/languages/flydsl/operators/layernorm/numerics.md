---
title: layernorm — numerics & parity
kind: technique
operator: layernorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp32]
regimes: [both, training]
updated: 2026-06-08
---

# layernorm — numerics & parity

## TL;DR
Inherits RMSNorm's rules (fp32 accumulate, fp32-promote γ/β, ε inside the rsqrt, fnuz fp8 on gfx942) plus LayerNorm edges: use **two-pass or Welford** variance (never the one-pass `Σx²/N−μ²`, which cancels and can go negative in bf16), use **biased `/N`** variance, and promote **both γ and β** to fp32.

Inherits all of [rmsnorm/numerics.md](../rmsnorm/numerics.md) (fp32 accumulate, fp32-promote γ, ε inside
the rsqrt, fnuz fp8 on gfx942) plus three LayerNorm-specific edges.

## 1. Mean subtraction → catastrophic cancellation risk
`σ² = Σ(x−μ)²/N`. The **naive one-pass** `σ² = Σx²/N − μ²` subtracts two large nearly-equal numbers →
cancellation, can go **negative** for large-magnitude rows in bf16. Avoid it. Use either:
- **two-pass**: compute μ first, then `Σ(x−μ)²` (aiter's Triton impl does exactly this — `mean` then
  `x_block - mean` accumulated), or
- **Welford**: running `(mean, M2)`, `σ² = M2/N` — most stable, single read.
Both keep the squared term centered → no cancellation. The fp32 accumulator is mandatory for both.

## 2. Biased vs unbiased variance
Inference LayerNorm uses the **biased** variance (`/N`, not `/(N−1)`). Match the framework — a `/(N−1)`
mismatch is a subtle accuracy bug. aiter/vLLM/Triton all use `/N`.

## 3. γ AND β must be fp32-promoted
Same regression class as rmsnorm (vLLM #42325): `y = (fp32(x)−μ)·rstd·fp32(γ) + fp32(β)`, convert to out
dtype **after** the bias add. β added in fp32 too.

## 4. saved mean/rstd (training)
The forward saves `mean` and `rstd` (fp32) for the backward; their dtype and the variance convention must
match the backward kernel's expectation, or gradients are wrong. aiter stores both as fp32.

## 5. fp8 quant variant
`layernorm2d_fwd_with_dynamicquant` / `_smoothquant`: scale computed in fp32 over the post-norm row;
fnuz fp8 on gfx942; **task-level gate** (gsm8k/MMLU), not allclose. See [[fused_norm_quant]],
[[quant_dequant_fp8]].

## Parity gate
1. isolated vs fp64: rel-err band (bf16 ~1e-2), check σ²≥0 (no cancellation).
2. e2e greedy parity after backend swap (two-pass/Welford/CK reduction orders differ).
3. fp8 variant: task eval delta within noise + fnuz dialect confirmed.
