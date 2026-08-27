---
title: softmax on aiter — SOTA card
kind: sota_card
operator: softmax
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/softmax.py
  - ROCm/aiter@b467ce342:aiter/ops/moe_op.py
  - ROCm/aiter@b467ce342:aiter/ops/mha.py
---

# softmax × aiter

## TL;DR
aiter's softmax appears in three forms, none of them a tunable standalone GEMM-style op: (1) the Triton
online softmax (`ops/triton/softmax.py`), (2) **fused into FMHA/MLA** attention (the dominant case), and
(3) **fused with topk** for MoE routing (`topk_softmax`). On the serving path, softmax is whatever the
attention or router kernel does — you don't call it directly.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton online softmax | `aiter/ops/triton/softmax.py` | gfx942/950, bf16/fp16/fp32 | bandwidth-bound | standalone row softmax |
| fused-in-attention online softmax | `aiter/ops/mha.py` (flash_attn), MLA decode | gfx942/950 | the real softmax; see attn cards | attention (default) |
| `topk_softmax` (MoE routing) | `aiter/ops/moe_op.py`; vLLM `rocm_aiter_topk_softmax` | gfx942/950 | softmax+topk+renorm one kernel | MoE router → moe_routing_topk |

## Config space / knobs
- Standalone: same as triton (`num_warps`, `BLOCK_SIZE=next_pow2(N)`, fp32).
- Attention softmax: governed by the attention backend (`--attention-backend`, `schedule_hint`,
  `num_stages=1`) — see attention_prefill_fmha.
- Routing: `topk_softmax` takes n_experts, top_k; biased/grouped variants for DeepSeek.

## Numerics / parity
Max-subtract; fp32 exp/accumulate; online = exact; reduction order varies across attn backends → greedy
re-gate. Routing softmax tail affects expert selection → task gate. See [numerics.md](numerics.md).

## Integration (rebind seam)
- Attention: `--attention-backend {ROCM_AITER_FA, ...}` (vLLM) / SGLang attention registry.
- Routing: aiter MoE path picks `topk_softmax` automatically when `VLLM_ROCM_USE_AITER_MOE=1`.
- Standalone: `from aiter.ops.triton.softmax import softmax`.

## Pitfalls & anti-patterns
- ⚠ Treating softmax as a separate tuning target — it isn't on the serving path; tune attention / routing.
- fnuz fp8 if the attention path uses fp8 KV → re-gate.

## How to verify
For attention: rocprofv3 confirms the FMHA kernel (softmax inside); greedy parity. For routing: confirm
`topk_softmax` engaged (`AITER_LOG_MORE=1`).

## Alternatives / cross-links
triton · hip · attention_prefill_fmha · moe_routing_topk ·
backends/aiter/attn_mla.

## Sources
- aiter Triton softmax + topk_softmax + mha (`ROCm/aiter@b467ce342`): `aiter/ops/triton/softmax.py`,
  `aiter/ops/moe_op.py` (`topk_softmax`), `aiter/ops/mha.py`.
