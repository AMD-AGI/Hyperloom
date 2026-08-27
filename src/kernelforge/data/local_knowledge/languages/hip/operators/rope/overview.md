---
title: rope — overview
kind: operator_overview
operator: rope
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/rope.py
  - /sgl-workspace/aiter/aiter/ops/triton/rope/rope.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu
---

# rope  (rotary position embedding: rotate Q/K by position-dependent angles)

## TL;DR
RoPE applies a position-dependent **2D rotation** to pairs of head dimensions of Q and K, just before
attention. It is **memory-bound** (read Q/K, read cos/sin cache, write Q/K) and runs **once per layer**.
The dominant optimization is **fusion into the attention entry** (QK-norm + RoPE + KV-cache write + quant
in one kernel) — done standalone it was 40–60% of inference latency in a naive PyTorch impl (vLLM #16457
moved it to Triton for that reason). Cross-link mrope (3D/multimodal) and fused_norm_quant.

## Math contract
For position `p` and dim pair `(2i, 2i+1)` with angle `θ_i = p · base^(−2i/d)`:
`x'_{2i} = x_{2i}cosθ_i − x_{2i+1}sinθ_i`, `x'_{2i+1} = x_{2i}sinθ_i + x_{2i+1}cosθ_i`. Two styles:
**NeoX** (rotate-halves: pair `i` with `i+d/2`) and **GPT-J** (rotate adjacent pairs) — `is_neox` flag.
Inputs: `q[*, h, d]`, `k[*, h, d]`, `cos_sin_cache[max_pos, d]` (precomputed). dtype bf16/fp16, cos/sin
fp32. Partial rotation: only the first `rotary_dim ≤ head_size` dims are rotated (GLM/partial models).

## Shape regimes
- **prefill**: `[batch, seq, h, d]`, seq 1k–64k → many elements, fills the chip.
- **decode**: `[batch, 1, h, d]` (one token) → few elements, latency-bound; fuse to hide it.
- `d` = head_size (64/128); `rotary_dim` may be < d (partial). `h` = n_heads (Q) / n_kv_heads (K).

## Where it matters (Amdahl)
Standalone RoPE is 1–3% GPU time when kernelized, but a **naive PyTorch RoPE was 40–60% of latency** for
Qwen2-VL (vLLM #16457) — the lesson is that RoPE *must* be a fused kernel. Done right + fused into the
attention entry it's near-free.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (CK/asm rope_fwd family + fused QK-norm+RoPE) | aiter |
| triton | 🟢 sota (aiter Triton rope + vLLM Triton rope) | triton |
| hip | 🟢 sota (vLLM `rotary_embedding` HIP) | [hip.md](hip.md) |
| vllm_kernels | 🟢 sota (HIP pos_encoding + AITER rope) | vllm_kernels |

## Fusion neighbors
`QK-norm + RoPE + KV-cache write + quant` → aiter `fused_qk_norm_rope_cache_quant` (fused_norm_quant,
kv_cache_quant); `+KV cache write` (`rope_cached_*`, fused_kv_cache); inside attention prefill.
3D/multimodal variant → mrope. See [fusion.md](fusion.md).

## Numerics
cos/sin in fp32; NeoX vs GPT-J style must match the checkpoint; partial-rotation bounds; scaling variants
(YaRN/NTK/linear) change the angle table. See [numerics.md](numerics.md).

## How to bench
`op_tests/test_rope.py` (aiter) at `[b, s, h, d]`; oracle = fp64 reference rotation; e2e A/B with the fused
QK-norm+RoPE path.

## Sources
- aiter rope ops (cached/uncached, 1c/2c, thd, 2d, positions/offsets): `/sgl-workspace/aiter/aiter/ops/rope.py`.
- aiter Triton rope (`_rope_fwd`, BLOCK_D, BLOCK_S=32, num_warps=4): `/sgl-workspace/aiter/aiter/ops/triton/rope/rope.py`.
- vLLM HIP `rotary_embedding` (is_neox, cos_sin_cache): https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu.
- naive RoPE 40–60% latency → Triton: https://github.com/vllm-project/vllm/pull/16457.
