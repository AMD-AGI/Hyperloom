---
title: attention_prefill_fmha — numerics
kind: operator_overview
operator: attention_prefill_fmha
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, fp8_e5m2]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://arxiv.org/abs/2205.14135
  - https://github.com/Dao-AILab/flash-attention
---

# attention_prefill_fmha — numerics

## The accumulation contract
FlashAttention prefill computes `O = softmax(QKᵀ·scale + mask)·V` **without** materializing the
`S=[sq,sk]` score matrix. The numerically load-bearing rule: the online softmax (running max `m`,
running sum `ℓ`, rescale of the partial `O` by `exp(m_prev − m)`) **accumulates in fp32** regardless of
the bf16/fp16/fp8 storage dtype. Both GEMMs (`QKᵀ` and `PV`) accumulate into fp32 MFMA accumulators
(AGPR). Output is cast back to bf16/fp16 on store. This is identical to the FA-2 reference; every AMD
backend (CK-Tile, Triton/aiter, TileLang, asm) follows it.

## Why cross-backend outputs differ (and why it's usually fine)
Online softmax is **not associative** — the order in which KV tiles are reduced changes the last bits.
CK-Tile, Triton, TileLang and asm tile the KV loop differently and reduce in different orders, so bit
patterns differ even at fp32 accumulate. In bf16 this surfaces as **argmax tie-flips on long greedy
decode**: two tokens whose logits are within bf16 ULP can swap which one is selected. This is a
*numerical-equivalence-class* difference, **not a regression**. Gate with a parity probe over ≥10
prompts (greedy, temp=0, fixed seed) and accept divergence that begins only after a near-tie — do not
gate on exact token-by-token match across backends.

## Masking numerics
- **Causal**: masked positions get `−inf` (or a large negative) before the row-max, so they contribute
  0 after `exp`. The masked variant *skips* upper-triangle KV tiles entirely (perf + avoids `exp` of
  large negatives). Get the masking applied **before** the row-max, not after, or the running max is
  poisoned.
- **Sliding-window / ALiBi**: same `−inf` discipline; ALiBi adds a per-(head,distance) bias to S before
  softmax (Triton/aiter FA path supports it; core CK FA does not — see
  fa_rocm).
- **Softcap** (`logit_cap`): `S = softcap·tanh(S/softcap)` before mask/softmax — used by Gemma-style
  models; present in the unified-attention Triton kernel.

## fp8 attention (CDNA3/CDNA4 relevant)
- **Dialect**: CDNA3 (gfx942) matrix core consumes **FNUZ** fp8 (`e4m3fnuz`/`e5m2fnuz`); CDNA4 (gfx950)
  adds **OCP** fp8 (`e4m3`/`e5m2`). The FNUZ exponent bias differs by one from OCP — reading fp8 bytes
  in the wrong dialect is off by **exactly 2×** and produces silent garbage, not an error. Always tag
  the dialect with the gen.
- **Scaling**: fp8 FMHA scales Q/K/V per-tensor (or per-tile); the descales (`q_descale`, `k_descale`)
  are applied inside the kernel before the fp32 softmax. A mismatched scale → silent wrong output.
- **What to gate**: fp8 attention is a **task-accuracy** gate (gsm8k / perplexity / eval), never byte
  parity vs bf16. fp8 KV-cache (storing K/V as fp8, dequant on read) is the most common production fp8
  use — it trades ~half the KV bandwidth for quant error; validate the model, not the kernel.

## Tolerances (rule of thumb)
- bf16/fp16 FMHA vs an fp32 reference: relative error ~1e-2 (bf16) / ~1e-3 (fp16) at the output — driven
  by the bf16/fp16 storage of P and O, not the fp32 accumulate. CK examples ship a built-in reference
  comparison (`-v 1`).
- fp8 FMHA vs bf16: expect visibly larger per-element error; only the downstream task metric is
  meaningful.

## Verify
- CK: `./bin/tile_example_fmha_fwd -b=1 -h=8 -s=4096 -d=128 -v=1` (built-in reference).
- Cross-backend: greedy temp=0 fixed-seed parity over ≥10 prompts; accept post-near-tie divergence.
- fp8: task-accuracy gate (gsm8k/eval), and confirm the dialect matches the gen (fnuz on gfx942).

## Sources
- FA online-softmax / fp32 accumulate, not associative: FlashAttention paper https://arxiv.org/abs/2205.14135 ; CK-Tile FA blog https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- FNUZ (CDNA3) vs OCP (CDNA4) fp8, 2× wrong-dialect trap: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html ; `hardware/shared/dtype_numerics.md`
- fp8 / ALiBi / softcap / arbitrary head dim feature split (Triton vs CK FA): https://github.com/Dao-AILab/flash-attention
