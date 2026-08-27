---
title: softmax — fusion neighbors
kind: technique
operator: softmax
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/moe_op.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/softmax.py
  - https://github.com/vllm-project/vllm/pull/16752
---

# softmax — fusion

Softmax is **defined by its fusions** — standalone it's negligible. The three that matter:

## 1. Into attention (online / flash softmax) — the whole point
The QKᵀ→softmax→PV inner loop never materializes the full softmax; it runs an **online softmax** with a
running `(m, l)` so the score matrix stays in registers/LDS. This is the dominant softmax on any LLM and
lives entirely inside FMHA. Optimize the attention kernel, not softmax. See attention_prefill_fmha,
attention_decode_paged, mla_attention.

## 2. softmax + topk → MoE routing
MoE routers compute `topk(softmax(logits))` (or softmax over the top-k). aiter fuses this:
`topk_softmax` (and vLLM's `rocm_aiter_topk_softmax`, PR #16752; biased/grouped variants for DeepSeek).
One kernel: softmax over n_experts + top-k selection + renormalize. Cross-link moe_routing_topk,
argmax_topk.

## 3. softmax + sampling → logits → token
After the lm_head, `softmax(logits)` feeds top-p/top-k sampling. Often kept separate (vocab is wide) but
the softmax+sort can fuse for greedy/argmax. Cross-link sampling_topk_topp, lm_head_logits.

## Fusion table
| form | where | op |
|---|---|---|
| online softmax | inside FMHA | attention_prefill_fmha |
| softmax + topk | MoE router | aiter `topk_softmax` → moe_routing_topk |
| softmax + sample | post-lm_head | → sampling_topk_topp |

## Sources
- aiter topk_softmax fusion: `aiter/ops/moe_op.py`; vLLM registered op: https://github.com/vllm-project/vllm/pull/16752.
- online softmax in attention: perf_knowledge attention_prefill_fmha.
