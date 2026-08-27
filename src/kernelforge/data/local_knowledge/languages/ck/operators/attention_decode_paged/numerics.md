---
title: attention_decode_paged — numerics
kind: operator_overview
operator: attention_decode_paged
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, fp8_e5m2]
regimes: [decode]
updated: 2026-06-08
sources:
  - https://arxiv.org/abs/2205.14135
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# attention_decode_paged — numerics

## Accumulation contract
Decode attention is the same online softmax as prefill but over a streamed paged KV history with `sq=1`.
fp32 accumulate for the running max `m`, sum `ℓ`, and `O` partial; bf16/fp16 storage of K/V (or fp8
KV-cache). With **splitKV**, each split computes a partial `(O_i, m_i, ℓ_i)` and a **reduce kernel**
combines them with the same log-sum-exp correction — the combine must use the per-split `m_i` to rescale,
or you get wrong softmax normalization. This is the numerically delicate part of flash-decoding.

## Why split-count changes the bits (and that's OK)
The reduce order over KV splits is not associative in fp32, so different `num_kv_splits` (or
auto-vs-manual) produce slightly different last bits. Different backends (aiter asm / vLLM-HIP / Triton)
split and reduce differently → bf16 argmax tie-flips on long greedy decode. **Benign** — gate with a
parity probe over ≥10 prompts (greedy, temp=0), accept post-near-tie divergence. Do not gate exact match
across backends or across split counts.

## fp8 KV-cache (the main decode quant)
Storing K/V as fp8 halves KV-cache bandwidth — the single biggest long-context decode speedup — at the
cost of quant error on every K/V read.
- **Dialect**: FNUZ on CDNA3 (gfx942), OCP on CDNA4 (gfx950). Reading fp8 KV bytes in the **wrong
  dialect is off by exactly 2×** (silent garbage, not an error). `--kv-cache-dtype fp8_e4m3` is fnuz on
  MI300X.
- **Scaling**: K/V are quantized with a (per-tensor or per-head) scale on write and dequantized on read
  inside the kernel before the fp32 softmax. Mismatched scale → silent wrong output.
- **Gate**: task accuracy (gsm8k / perplexity / eval), never byte parity vs bf16. fp8 KV has caused real
  eval regressions on some models — always accuracy-gate.
- **Coverage**: aiter **MLA** decode does not support fp8 KV-cache in vLLM upstream (per vLLM ROCm blog);
  check per backend.

## Masking
Decode has no causal mask within a step (the query is the newest token), only the **valid-length mask**
(`seq_lens[b]`): KV positions beyond the sequence length are masked to `−inf` before the row-max. Sliding
window applies an additional lower bound on attended positions. Softcap (`logit_cap`) applies
`softcap·tanh(S/softcap)` before mask/softmax if the model uses it.

## Tolerances
- bf16 KV vs fp32 reference: output rel-error ~1e-2; fp16 KV ~1e-3 — driven by KV storage dtype.
- fp8 KV vs bf16: visibly larger per-element error; only the downstream task metric is meaningful.

## Verify
- Cross-backend: greedy temp=0 fixed-seed parity over ≥10 prompts; accept post-near-tie divergence.
- fp8 KV: task-accuracy gate; confirm dialect matches gen (fnuz on gfx942).
- Confirm the splitKV reduce used per-split `m_i` (a wrong combine shows as wrong logits, not tie-flips).

## Sources
- flash-decoding / online softmax (split + reduce, log-sum-exp combine): https://arxiv.org/abs/2205.14135
- FNUZ (CDNA3) vs OCP (CDNA4) fp8, 2× wrong-dialect trap: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html ; `hardware/shared/dtype_numerics.md`
- aiter MLA decode no fp8 KV in vLLM upstream; fp8 KV accuracy: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
