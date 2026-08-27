---
title: softmax — overview
kind: operator_overview
operator: softmax
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/softmax.py
  - /sgl-workspace/aiter/aiter/ops/triton/_triton_kernels/softmax.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# softmax  (`y_i = exp(x_i − max) / Σ exp(x_j − max)`)

## TL;DR
Standalone softmax is a **memory-bound, max-subtracted row reduction**. The single most important fact:
in real LLMs softmax almost never appears alone — it is **fused into attention** (online/flash softmax,
the QKᵀ→softmax→PV inner loop). Standalone softmax matters only for **MoE routing logits**, **sampling
logits**, and **classifier heads**. Use the **online (single-pass, running max+sum)** formulation; on
MI300X round the reduced width to a power-of-2 so the wave64 reduce is full.

## Math contract
Row `x[N]`: `m = max(x)`, `y_i = exp(x_i − m) / Σexp(x_j − m)` (max-subtraction for stability).
fp32 compute (exp + accumulate); bf16/fp16 IO. Reduce over the contiguous last dim. **Online softmax**
fuses the max and the sum into one pass with a correction factor `sum *= exp(m_old − m_new)`.

## Shape regimes
- **fused-in-attention (the dominant case)**: N = seqlen / KV length; computed inside FMHA, *not* a
  separate kernel — see attention_prefill_fmha / attention_decode_paged.
- **standalone routing/sampling**: `M = tokens`, `N = n_experts` (8..256) or `N = vocab` (32k..256k).
  Vocab softmax (lm_head) is wide → lm_head_logits / sampling_topk_topp.

## Where it matters (Amdahl)
Standalone softmax is **<1%** of GPU time on a dense LLM (it lives inside attention). The optimization
budget belongs to the **fused attention softmax** — see the attention operators. Standalone wins only on
wide-vocab logits or MoE routing where it's a genuine separate pass.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota (aiter online-softmax impl) | [triton.md](triton.md) |
| aiter | 🟢 sota (Triton + fused-into-attn/topk) | aiter |
| hip | 🟢 sota (hand-written wave64 reduce) | hip |
| flydsl | 🧪 experimental (max/sum reduce **primitives** `reduce_vec_max`/`reduce_vec_sum`/`make_block_reduce`; no standalone op) | flydsl |

## Fusion neighbors
**Into attention** (online/flash softmax — the whole point); `+topk` (MoE routing: aiter
`topk_softmax` / `rocm_aiter_topk_softmax`); `+sampling` (logits → top-k/top-p). See [fusion.md](fusion.md).

## Numerics
Max-subtraction is mandatory; fp32 exp+accumulate; online correction factor. See [numerics.md](numerics.md).

## How to bench
Isolated row softmax at `(M, N)` for the target (routing N=experts, vocab N); fp64 oracle; median ≥3 reps.
For the real workload, bench attention (softmax is inside it), not standalone.

## Sources
- aiter Triton online softmax (`_softmax_kernel_online`, running max/sum + correction): `/sgl-workspace/aiter/aiter/ops/triton/softmax.py`, `_triton_kernels/softmax.py`.
- wave64 reduce / power-of-2 reduced dim: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- Softmax fused into FMHA: perf_knowledge attention_prefill_fmha.
