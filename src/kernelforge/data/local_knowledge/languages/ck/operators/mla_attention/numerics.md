---
title: mla_attention — numerics
kind: operator_overview
operator: mla_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://arxiv.org/abs/2405.04434
---

# mla_attention — numerics

## Matrix absorption is algebraically exact (in fp32)
Weight-absorbed MLA folds the KV up-projection (`Wuk`, `Wuv`) into Q and into the output. Mathematically
`(q·Wuk)·c_KVᵀ = q·(Wuk·c_KVᵀ)` and `(P·c_KV)·Wuv = P·(c_KV·Wuv)` — so the absorbed MQA form is **exactly
equal** to materializing K/V and running MHA, in exact arithmetic. In bf16 it is parity-safe: the folded
GEMM accumulates in fp32, and the absorption does not introduce a new lossy step. So absorbed-vs-
unabsorbed should agree to within bf16 reduction-order noise — not a regression.

## The two-part score and fp32 accumulate
The score has two contributions summed in fp32: `q_nope·c_KVᵀ` (over the 512-wide latent) and
`q_rope·k_ropeᵀ` (over the 64-wide decoupled RoPE). Both GEMMs accumulate fp32 into the MFMA accumulator;
the online softmax (running max/sum, `O` rescale) is fp32. splitKV decode reduces partials with per-split
`m_i` — same log-sum-exp discipline as flash-decoding.

## Cross-backend / split-count differences (benign)
asm `mla_decode_fwd`, the Triton MLA reference (`mla_decode.py`), and the unabsorbed prefill path reduce
in different orders → bf16 argmax tie-flips on long greedy decode. Benign equivalence-class differences —
gate with a ≥10-prompt greedy temp=0 parity probe, accept post-near-tie divergence. The Triton MLA
reference exists precisely for correctness cross-checks (it is much slower).

## fp8 latent / fp8 KV (the accuracy risk)
- `mla_decode_fwd` takes `q_scale` / `kv_scale` for fp8 Q and fp8 latent/KV. FNUZ on gfx942, OCP on
  gfx950 — wrong dialect off by **exactly 2×** (silent garbage).
- **MLA is accuracy-sensitive to fp8**: AITER MLA has shown real eval regressions — gsm8k loss with
  Kimi-K2 DP2TP4 (aiter #1455). The latent is a compressed representation, so fp8 quant error on it
  propagates more than fp8 on a full KV head. **Always task-accuracy gate** fp8 MLA, never byte parity.
- aiter MLA decode does **not** support fp8 KV-cache in vLLM upstream — check coverage before assuming it.

## logit_cap / softcap
`mla_decode_fwd` currently asserts `logit_cap <= 0` (softcap not yet supported in the asm decode path) —
models needing softcap must use a path that supports it.

## Tolerances
- bf16 absorbed vs bf16 unabsorbed: agree to reduction-order noise (~1e-2 rel at output).
- bf16 vs fp32 reference: ~1e-2 rel (bf16 storage of P/O).
- fp8 vs bf16: only the downstream task metric is meaningful; expect visible per-element error.

## Verify
- Absorbed-vs-unabsorbed bf16 agreement (sanity for the absorption).
- Triton MLA reference vs asm `mla_decode_fwd`: greedy temp=0 parity ≥10 prompts, accept post-near-tie.
- fp8 MLA: gsm8k/eval accuracy gate; confirm dialect matches gen.

## Sources
- absorption algebra + 17× (vendor): https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html ; MLA def https://arxiv.org/abs/2405.04434
- `mla_decode_fwd` fp8 `q_scale`/`kv_scale`, `logit_cap<=0` assert: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py`.
- AITER MLA eval regression (#1455), fp8 KV unsupported in vLLM upstream: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
