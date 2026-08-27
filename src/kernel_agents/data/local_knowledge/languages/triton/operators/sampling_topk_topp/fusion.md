---
title: sampling_topk_topp — fusion
kind: technique
operator: sampling_topk_topp
gens: [gfx942, gfx950]
dtypes: [fp32, bf16]
regimes: [decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/cpp_itfs/sampling/sampling.cuh
  - https://flashinfer.ai/2025/03/10/sampling.html
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
  - https://vllm.ai/blog/mrv2
---

# sampling_topk_topp — fusion

## TL;DR
The whole point of the modern sampler is **one fused kernel**: temperature → (softmax) → top-k → top-p →
min-p → draw, with **no intermediate `[M,V]` writes and no host sync**. The biggest fusion is
**softmax-into-sample** (Gumbel-max avoids materializing softmax) and **filter-into-sample** (the rejection
loop applies the top-k/top-p threshold *while* sampling, instead of mask-then-renormalize-then-sample).
Penalties and logit-bias stay **unfused** (applied as separate logits processors before the sampler).

## Fusion neighbors
| neighbor | type | done? | note |
|---|---|---|---|
| temperature scale | pre-scale | ✅ folded in | `logits /= T` before softmax; greedy path skips it |
| softmax | activation | ✅ avoided/fused | Gumbel-max samples without explicit softmax (MRV2 Triton); rejection sampler works on probs/logits directly |
| top-k + top-p + min-p | filter | ✅ in the rejection loop | dual-pivot accepts/rejects against the threshold *during* sampling — no sort, no separate mask pass |
| inverse-transform draw | CDF + locate | ✅ same kernel | parallel prefix-sum (CDF) + locate token, in-block |
| **penalties** (rep/freq/presence), **logit-bias**, **min-tokens** | logits processors | ❌ separate, **before** | non-argmax-invariant processors run pre-sample; min-p is argmax-invariant (run only for random rows) |
| greedy argmax (temp=0) | reduction | ✅ separate fast path | whole-batch-greedy → argmax, skip the sampler entirely → argmax_topk |
| lm_head_logits | upstream GEMM | ❌ separate | sampler is the consumer; the head's greedy-argmax fusion handles temp=0 |

## The single-kernel design (why it's fused)
The naive pipeline is: softmax (`[M,V]` write) → sort (`[M,V]`) → cumsum → mask (`[M,V]` write) →
renormalize → multinomial (host sync). Each `[M,V]` pass at V=256k is a full bandwidth hit, and the
multinomial syncs. The fused rejection sampler collapses all of it into **one threadblock per row** that
streams the `V`-row once or a few times (one dual-pivot round each), keeps partials in LDS/registers
(hipCUB BlockScan/BlockReduce), and emits the token id — **zero intermediate `[M,V]` materialization, zero
host sync**. This is the entire performance argument.

## Gumbel-max (softmax-free sampling)
The no-sync sample is `argmax(logprobs + gumbel_noise)` (equivalently exponential-trick:
`random_sample_outer_exponential`). It avoids `torch.multinomial`'s sync **and** avoids materializing a
softmax — MRV2's Gumbel-Max Triton kernel does exactly this with stateless in-kernel RNG. Combined with the
rejection filter, the result is a single fused, sync-free sampler.

## What stays unfused (and why)
- **Penalties / logit-bias / min-tokens**: applied as logits processors *before* sampling because they
  affect the greedy result too (non-argmax-invariant) and are cheap elementwise ops over `[M,V]`. min-p is
  argmax-invariant → applied only on the random branch.
- **logprobs**: gathered from **raw** logits (a separate fused log_softmax+gather, ~1.4× faster than sparse
  gather per vLLM) — not produced by the rejection kernel.

## Cross-links
[overview.md](overview.md) · [tuning.md](tuning.md) · [numerics.md](numerics.md) · softmax ·
argmax_topk · lm_head_logits · cumsum_scan · aiter.

## Sources
- One fused kernel (no multi-pass, no extra launches), sorting-free: https://flashinfer.ai/2025/03/10/sampling.html
- aiter HIP fused sampler (BlockScan/BlockReduce in-block, philox): `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/cpp_itfs/sampling/sampling.cuh`.
- Gumbel `random_sample` (no multinomial sync), processors before sample: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py
- MRV2 Gumbel-Max Triton kernel (no explicit softmax, stateless RNG): https://vllm.ai/blog/mrv2
