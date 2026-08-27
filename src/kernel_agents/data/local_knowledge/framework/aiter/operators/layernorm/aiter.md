---
title: layernorm on aiter — SOTA card
kind: sota_card
operator: layernorm
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/norm.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/normalization/norm.py
---

# layernorm × aiter

## TL;DR
On the AMD serving stack, aiter is the live LayerNorm path with a CK/asm tier (`layernorm2d_fwd` via
`module_norm`) plus the Triton fallback. As with rmsnorm, the *fused* entrypoints (`_with_add`,
`_with_smoothquant`) are what serving actually calls; standalone LayerNorm is mostly vision-tower / encoder.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `layer_norm` / `layernorm2d_fwd` (CK/asm) | `aiter/ops/norm.py` (`module_norm`) | gfx942/950, bf16/fp16 | bandwidth-bound floor | standalone / encoder norm |
| `layernorm2d_fwd_with_add` | same | gfx942/950 | one read+write incl. residual | ViT/encoder block norm |
| `layernorm2d_fwd_with_smoothquant` / `_with_add_smoothquant` | same | gfx942/950, int8 out | SmoothQuant per-channel scale | int8 quant path → fused_norm_quant |
| Triton fallback (`layer_norm`, `_with_dynamicquant`) | `aiter/ops/triton/normalization/norm.py` | gfx942/950, fp8 out | two-pass, blocked | no CK/asm path / portability |

(Note: in `aiter/ops/norm.py` the `layernorm2d_fwd_with_dynamicquant` C++ entrypoints are commented out —
the dynamic-quant LayerNorm path is currently the **Triton** impl; smoothquant has a CK/asm path.)

## Config space / knobs
- Tier: CK default (`ENABLE_CK=1`); fusion select = entrypoint (`_with_add`, `_with_smoothquant`).
- Triton path: `BLOCK_SIZE=min(65536//elt, next_pow2(N))`, blocked when N>BLOCK, `num_warps` heuristic,
  persistent grid for decode.
- JIT-compiled `module_norm` on first call.

## Numerics / parity
fp32 μ/σ² (two-pass, no cancellation); γ,β fp32-promote; biased variance; smoothquant int8 / dynamic fp8
fnuz on gfx942 → task gate. CK vs Triton reduction order differs → greedy re-gate. See
[numerics.md](numerics.md).

## Integration (rebind seam)
- vLLM: `VLLM_ROCM_USE_AITER=1` (+ `_RMSNORM` gate also governs the norm family); LayerNorm models route
  `forward_hip` to aiter when available.
- SGLang: on by default with `SGLANG_USE_AITER=1`.
- Verify: `AITER_LOG_MORE=1` dispatch + rocprofv3 CK/asm norm kernel name.

## Pitfalls & anti-patterns
- ⚠ Dynamic-quant LayerNorm has **no C++/CK entrypoint** (commented out) → it runs Triton; don't assume an
  asm path exists for that fusion.
- ⚠ fnuz/OCP fp8 dialect on gfx942.
- gfx942 shape with no CK tune → Triton fallback (confirm in trace).

## How to verify
`python3 op_tests/test_layernorm2d.py`; rocprofv3 kernel name; greedy parity; σ²≥0.

## Alternatives / cross-links
triton · hip · vllm_kernels · [miopen.md](miopen.md) ·
[overview.md](overview.md) · fused_norm_quant.

## Sources
- aiter C++ layernorm dispatch + fused variants (note commented dynamicquant), `ROCm/aiter@b467ce342`:
  `aiter/ops/norm.py`.
- Triton layernorm bodies: `aiter/ops/triton/normalization/norm.py`.
