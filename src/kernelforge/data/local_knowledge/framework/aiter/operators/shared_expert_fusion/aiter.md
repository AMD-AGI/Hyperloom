---
title: shared_expert_fusion on aiter — SOTA card
kind: sota_card
operator: shared_expert_fusion
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/fused_moe_dp_shared_expert.py
  - https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
---

# shared_expert_fusion × aiter

## TL;DR
> aiter owns shared-expert fusion on AMD: `fused_moe_dp_share_expert` computes the shared dense MLP in the
> fused-MoE pipeline and **atomic-adds** it into the routed result; under Wide-EP the shared expert is
> injected as a synthetic routed expert for a single fused dispatch. Use it for DeepSeek-V2/V3/R1 — but
> **don't** combine the vLLM shared-fusion flag with MoRI (incompatible).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `fused_moe_dp_share_expert` (fused + atomic-add) | `aiter/fused_moe_dp_shared_expert.py` | gfx942/950; bf16, fp8 | part of Wide-EP 32.3k in / 12.4k out tok/s/node (32× MI300X, AMD-reported) | DeepSeek shared+routed in one pipeline |
| shared-as-synthetic-routed (EP) | Wide-EP path (`grouped_topk` injection) | gfx942/950 | EP16 1.3× over EP8 (AMD-reported) | distributed DeepSeek EP |

## Config space / knobs
- DP token-range split: `get_dp_shared_expert_token_range(token_num, dp_size, rank)`.
- Shared sorting layout: `get_dp_shared_expert_stage1/stage2_moe_sorting_result`; `get_padded_M` bucketing
  (align the shared module's copy with the tier logic in `fused_moe.py` — they must agree).
- `block_size_M` for the (dense, all-token) shared GEMM — big tile to fill 304 CUs.
- `quant_type` shared = routed (shared dequant rides the routed epilogue).
- Overlap: separate HW queue (`GPU_MAX_HW_QUEUES=2`) + atomic-add.

## Numerics / parity
Math-preserving; atomic-add order → benign bf16 deltas (no byte parity). Shared weight = 1, not renormed
with routed. fp8 shared is a quant gate. See [numerics.md](numerics.md).

## Integration (rebind seam)
- vLLM: `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` (⚠ **incompatible with MoRI** — under MoRI the
  fusion is MoRI-side).
- SGLang: aiter shared-expert path under `SGLANG_USE_AITER=1`.
- The result buffer is passed in and atomic-added — wire the no-shared MoE output as the accumulator.

## Pitfalls & anti-patterns
- Setting the vLLM shared-fusion flag **and** MoRI → conflict.
- Applying a routed weight / renorm to the shared expert → wrong output.
- Atomic-add contention if shared and routed write the same token ranges concurrently.

## How to verify
`AITER_LOG_MORE=1` for the shared kernels; isolated shared-only vs torch dense MLP; e2e DeepSeek tok/s
fusion on/off; greedy parity.

## Alternatives / cross-links
hip · triton · `backends/aiter/fmoe.md` ·
`backends/mori_rccl/mori_ep.md` · [overview.md](overview.md).

## Sources
- on-box: `ROCm/aiter@b467ce342:aiter/fused_moe_dp_shared_expert.py` (`fused_moe_dp_share_expert`,
  `get_dp_shared_expert_*`).
- Wide-EP shared fusion + numbers + MoRI incompatibility: https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html
