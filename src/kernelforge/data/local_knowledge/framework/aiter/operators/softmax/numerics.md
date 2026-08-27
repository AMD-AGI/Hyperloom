---
title: softmax — numerics & parity
kind: technique
operator: softmax
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/_triton_kernels/softmax.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# softmax — numerics & parity

## 1. Max-subtraction is mandatory
`exp(x_i)` overflows fp32 at x>88. Always subtract the row max: `exp(x_i − m)`. The online formulation
keeps a running max and corrects (`sum *= exp(m_old − m_new)`) — algebraically identical, numerically
safe. Skipping max-subtraction = NaNs on large logits.

## 2. fp32 exp + accumulate
exp and the sum accumulate in **fp32** even with bf16/fp16 IO; the final divide and convert-to-out happen
last. bf16 exp/accumulate loses precision in the tail (small probabilities) → wrong sampling.

## 3. Online correction = exact
The online (flash) softmax is **not** an approximation — the correction factor makes it bit-equivalent to
the two-pass up to fp rounding. But the **reduction/block order differs** from a naive softmax → last-bit
differences. After swapping softmax (or attention backend), re-gate greedy/temp=0 parity.

## 4. Sampling sensitivity
For sampling (top-p/top-k) the softmax tail matters: a bf16-accumulated sum shifts low-probability tokens.
Use fp32 accumulate; gate sampling correctness with a fixed-seed distribution check, not allclose. See
sampling_topk_topp.

## 5. Attention softmax
Inside FMHA the softmax is fused with a running `(m, l)` (max, normalizer) per the flash-attention scheme;
fp32 `m,l`; the PV accumulation rescales by `exp(m_old − m_new)`. Parity differences across AITER/Triton/
CK attention come mostly from this reduction order — re-gate after a backend swap. See
attention_prefill_fmha.

## Parity gate
1. isolated vs fp64: rel-err band; no NaN on x∈[−1e4, 1e4].
2. routing/sampling: fixed-seed distribution match.
3. attention: greedy e2e parity after backend swap.

## Sources
- online max-subtract + correction: `aiter/ops/triton/_triton_kernels/softmax.py`.
- wave64 reduce / fp32: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
