---
title: moe_routing_topk on Triton — SOTA card
kind: sota_card
operator: moe_routing_topk
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/topk.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# moe_routing_topk × Triton

## TL;DR
> Triton is the **editable fallback** router: vLLM/SGLang's fused-MoE Triton path includes
> softmax/sigmoid + top-k + grouped select, and aiter ships a Triton `topk` (`aiter/ops/triton/topk.py`).
> Use it when AITER's HIP path is wrong/missing for your scoring/group config, or to prototype a new
> routing variant. On the steady-state hot path the HIP/DPP kernel usually wins (Triton has no DPP
> cross-lane and round-trips the reduction through LDS), but Triton is the correctness-first universal path.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM `fused_grouped_topk` / `grouped_topk` (Triton/torch) | `vllm/.../fused_moe/fused_moe.py` | gfx942/950 | — (reference correctness path) | softmax models; debug fallback (`VLLM_ROCM_USE_AITER=0`) |
| aiter Triton `topk` | `aiter/ops/triton/topk.py` | gfx942/950 | — | portable routing inside aiter Triton MoE |
| SGLang Triton fused-MoE routing | sglang fused_moe | gfx942/950 | — | when Triton fused-MoE is selected |

Recommend: HIP/DPP for production; Triton when the HIP path mis-scores (softmax) or for a new variant.

## Config space / knobs
- `BLOCK` over the expert dim `E`; `num_warps` (wave64 — start 2–4, **not** 8, to avoid VGPR spill).
- fp32 accumulation for the softmax/sigmoid reduction (`tl.float32`).
- `matrix_instr_nonkdim` irrelevant (no `tl.dot`); this is a reduction kernel — tune `num_warps`,
  `num_stages=1`, and use `knobs.amd.use_buffer_ops` for the masked logit load.
- For grouped top-k, the group reduction is a two-level argmax/sum — keep it in registers, reduce across
  the wave with `tl.max`/`tl.sum` over the group axis.

## Numerics / parity
fp32 reduce; tie-break flips benign. Triton path **honors the requested `scoring_func`** (softmax or
sigmoid) — this is why it's the correct fallback when the HIP biased path's sigmoid hardcode is wrong.
See [numerics.md](numerics.md).

## Integration (rebind seam)
- vLLM: `VLLM_ROCM_USE_AITER=0` (or `_MOE=0`) falls back to the Triton/torch `grouped_topk`.
- SGLang: Triton fused-MoE selection.
- aiter: `aiter/ops/triton/topk.py` inside the Triton MoE pipeline.

## Pitfalls & anti-patterns
- Carrying `num_warps=8` from NVIDIA → spill (3–5× slower); this is a small reduction kernel, use 2–4.
- Triton routing is slower than HIP/DPP at steady state — don't ship it as the hot path if the HIP path is
  correct for your model.

## How to verify
Compare ids+weights vs torch reference; e2e parity with `VLLM_ROCM_USE_AITER=0` vs `=1`. rocprof to
confirm the Triton routing kernel ran (carries the Python name).

## Alternatives / cross-links
aiter · hip · `triton_levers/` ·
[overview.md](overview.md).

## Sources
- vLLM reference `fused_grouped_topk`/`grouped_topk`: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
- aiter Triton topk: `ROCm/aiter@a6bb49937:aiter/ops/triton/topk.py`
- Triton AMD knobs (num_warps/wave64): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
