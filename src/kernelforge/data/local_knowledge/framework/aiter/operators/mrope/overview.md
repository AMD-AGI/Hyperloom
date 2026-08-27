---
title: mrope — overview
kind: operator_overview
operator: mrope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode, both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_mrope_cache_quant.py
  - https://github.com/vllm-project/vllm/pull/22593
  - https://github.com/vllm-project/vllm/pull/22593
---

# mrope  (multimodal / 3D rotary position embedding, e.g. Qwen2.5-VL)

## TL;DR
MRoPE is RoPE with **multiple position axes** — text/temporal, height, width — so image/video tokens get a
3D position. The head dimension is partitioned into **sections** (`mrope_section`), each rotated by its own
axis's position. It is rope plus a section-split, same memory-bound profile, and the same fusion story
(fused into the QK-norm+RoPE+KV-write attention entry). Read rope first.

## Math contract
Like RoPE but the cos/sin angle for dim `i` uses the position of the **section** dim `i` belongs to.
`mrope_section = [s_t, s_h, s_w]` partitions `rotary_dim` into temporal/height/width spans; positions is a
`[3, ...]` tensor (one row per axis). Each section rotates with its axis's `θ`. `is_neox` style; partial
rotation (rotary_dim < head_size); dtype bf16/fp16, cos/sin fp32.

## Shape regimes
- VLM prefill: `[batch, seq, h, d]` with a `[3, batch, seq]` position tensor (text tokens use the same
  position on all 3 axes; image tokens differ). Many tokens → fills the chip.
- decode: one token; latency-bound; fuse.

## Where it matters (Amdahl)
Only on **VLMs** (Qwen2-VL/2.5-VL, GLM-4.1V). Same magnitude as RoPE (1–3% kernelized; catastrophic if
naive — vLLM #22593/#16457 moved MRoPE to a Triton kernel and fixed partial-rotation). Negligible on
text-only models.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (fused QK-norm+mrope+KV+quant) | [aiter.md](aiter.md) |
| triton | 🟢 sota (vLLM MRoPE Triton kernel) | triton |
| hip | 🟡 competitive (extend the RoPE HIP kernel) | hip |

## Fusion neighbors
`QK-norm + mRoPE + KV-cache write + quant` → aiter `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`
(fused_norm_quant, kv_cache_quant); same fusion family as rope. See [fusion.md](fusion.md).

## Numerics
section partition must match the model's `mrope_section`; per-axis positions; partial-rotation bound
(#22593 illegal access); `is_neox`; deterministic parity. See [numerics.md](numerics.md).

## How to bench
mrope op test at `[b, s, h, d]` + `[3, b, s]` positions; fp64 reference per-section rotation; e2e on a VLM
with image inputs.

## Sources
- aiter fused QK-norm+mrope+KV+quant: `aiter/ops/fused_qk_norm_mrope_cache_quant.py`.
- MRoPE Triton kernel + partial-rotary fix (Qwen2-VL/GLM-4.1V): https://github.com/vllm-project/vllm/pull/22593.
- naive RoPE/MRoPE latency → Triton: https://github.com/vllm-project/vllm/pull/16457.
