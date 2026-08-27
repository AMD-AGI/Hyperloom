---
title: mrope on aiter — SOTA card
kind: sota_card
operator: mrope
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/fused_qk_norm_mrope_cache_quant.py
  - https://github.com/sgl-project/sglang/issues/18466
---

# mrope × aiter

## TL;DR
aiter ships the **fused** mRoPE attention entry for VLMs:
`fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` (QK-norm + 3D mRoPE + KV-write + quant + shuffle). This is
the SOTA serving path for Qwen2.5-VL-class models on MI300X; there's no reason to run a standalone mRoPE.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle` | `aiter/ops/fused_qk_norm_mrope_cache_quant.py` | gfx942/950, bf16/fp16/fp8 | VLM attention-entry mega-fusion (analog of Qwen3 #18466) | **VLM serving** |
| (standalone mRoPE) | via aiter rope + section split | gfx942/950 | bandwidth-bound | rare; debugging |

## Config space / knobs
- `mrope_section_: List[int]` ([s_t, s_h, s_w]) — must match the model.
- `is_neox`, `rotary_dim` (partial), positions `[3, ...]`, fp8 quant toggle, KV shuffle layout.
- JIT `module_fused_qk_norm_mrope_cache_quant_shuffle` on first call.

## Numerics / parity
cos/sin fp32; `mrope_section` and per-axis positions correct; partial-rotary bound; fnuz fp8 on gfx942 for
the quant; deterministic → token-identical greedy parity. See [numerics.md](numerics.md).

## Integration (rebind seam)
- SGLang: Qwen3-VL/Qwen2.5-VL attention routes here when `SGLANG_USE_AITER=1`.
- vLLM: `VLLM_ROCM_USE_AITER=1` (+ rope gate) for the VLM attention entry.
- Verify: `AITER_LOG_MORE=1` dispatch; rocprofv3 fused mrope kernel.

## Pitfalls & anti-patterns
- ⚠ Wrong `mrope_section` → scrambled image/video positions (subtle accuracy loss).
- ⚠ Partial-rotary OOB; per-axis position mis-broadcast.
- ⚠ fp8 fnuz on gfx942.

## How to verify
mrope op test; isolated vs fp64 per-section; VLM greedy parity with image inputs; rocprofv3.

## Alternatives / cross-links
triton · hip · rope/backends/aiter · fused_norm_quant ·
kv_cache_quant.

## Sources
- aiter fused mrope kernel (`ROCm/aiter@b467ce342`): `aiter/ops/fused_qk_norm_mrope_cache_quant.py`.
- analogous RoPE fusion win: https://github.com/sgl-project/sglang/issues/18466.
