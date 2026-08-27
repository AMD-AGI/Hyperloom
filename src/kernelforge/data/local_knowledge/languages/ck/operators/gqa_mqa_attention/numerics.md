---
title: gqa_mqa_attention — numerics
kind: operator_overview
operator: gqa_mqa_attention
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://arxiv.org/abs/2305.13245
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/Dao-AILab/flash-attention
---

# gqa_mqa_attention — numerics

## GQA is numerically identical to MHA-with-shared-KV
GQA/MQA changes only *which* KV head a query head attends — it does not change the softmax math. So the
output is bit-identical between (a) in-register KV broadcast and (b) physically replicated KV
(`repeat_kv`), because both feed the **same K/V values** into the same `q·Kᵀ` / `P·V` GEMMs. The broadcast
is a memory/bandwidth optimization with **zero numerical cost**. fp32 online-softmax accumulate, same as
MHA. This means GQA needs no special numerical gate beyond the underlying FMHA/decode numerics.

## What still applies (inherited from FMHA / decode)
- fp32 online-softmax accumulate; bf16/fp16 storage of P/O.
- Cross-backend bf16 argmax tie-flips on long greedy decode are benign (different KV-tile reduction
  order) — gate with a ≥10-prompt parity probe, accept post-near-tie divergence.
- Causal/SWA/ALiBi masking discipline (mask before row-max).

## fp8 KV-cache with GQA (the one real interaction)
fp8 KV is the common decode quant; with GQA the fp8 K/V head is *shared* by R query heads, so a single
fp8 quant error is amplified across the group. FNUZ on gfx942, OCP on gfx950 (wrong dialect off by 2×).
This makes GQA + fp8 KV slightly more accuracy-sensitive than MHA + fp8 KV — the error correlates across
the query-head group rather than averaging out. **Task-accuracy gate** (gsm8k/eval), never byte parity vs
bf16.

## Verify
- In-register-broadcast vs `repeat_kv` reference: should be **bit-identical** in the same dtype (a sanity
  check that the broadcast is implemented correctly — if it differs, the head pairing is wrong).
- Cross-backend: greedy temp=0 parity ≥10 prompts.
- fp8 KV + GQA: gsm8k/eval accuracy gate; confirm dialect matches gen.

## Sources
- GQA = shared KV head, same softmax: https://arxiv.org/abs/2305.13245
- FNUZ (CDNA3) vs OCP (CDNA4) fp8, 2× wrong-dialect trap: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html ; `hardware/shared/dtype_numerics.md`
- MQA/GQA + fp8 KV as FA features: https://github.com/Dao-AILab/flash-attention
