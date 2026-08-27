---
title: rope — tuning
kind: technique
operator: rope
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-09
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/rope/rope.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# rope — tuning

Memory-bound elementwise-with-table op: read Q/K + cos/sin, rotate, write Q/K. No reduction. Tuning =
saturate bandwidth + reuse the cos/sin table + fuse.

## 1. The aiter Triton rope structure (verified)
`_rope_fwd` (`ops/triton/rope/rope.py`):
```python
BLOCK_D = d//2 (NeoX rotate-halves) or d ; BLOCK_D_HALF = d//4 or d//2
BLOCK_S = 32 ; num_warps = 4
grid = (b, h, cdiv(s, BLOCK_S))     # one program per (batch, head, seq-tile)
```
- 3D grid over `(batch, head, seq-tile)` → for prefill (large s) this is many programs (fills the chip);
  for decode (s=1) it's `b·h` programs — small, latency-bound.
- `BLOCK_D` is the rotated half-width; the rotation reads `(x_2i, x_2i+1)` pairs (GPT-J) or
  `(x_i, x_{i+d/2})` (NeoX).

## 2. cos/sin table reuse
The `cos_sin_cache[max_pos, d]` is shared across all heads at a position. Load the cos/sin for the
seq-tile **once per program** and reuse across heads/Q-K — don't re-read per head. For decode (one
position), the cos/sin is a single row → broadcast.

## 3. Vectorized I/O + in-place
- Read Q/K as `float4`/`__half2` (head_dim 64/128 → 16/32 fp16 = 4/8 dwordx4 loads).
- RoPE is usually **in-place** (`rope_fwd_inplace`) — write back to Q/K, no extra buffer. Halves traffic.
- cos/sin in fp32 (precision); rotate in fp32, write bf16.

## 4. Knob table
| knob | rope setting | note |
|---|---|---|
| `num_warps` | 4 (aiter) | memory-bound |
| `BLOCK_S` | 32 | seq-tile per program |
| `BLOCK_D` | d/2 (NeoX) / d (GPT-J) | rotated width |
| grid | `(b, h, cdiv(s,32))` | fills chip for prefill |
| in-place | yes (`_inplace`) | halve write traffic |
| cos/sin reuse | per seq-tile | avoid per-head reload |

## 5. The real lever: fuse into the attention entry
Standalone RoPE forces a Q/K read+write round-trip. Fusing QK-norm + RoPE + KV-cache write (+ quant) into
one kernel (aiter `fused_qk_norm_rope_cache_quant`, `rope_cached_*`) removes that. For decode especially,
the fusion is what makes RoPE near-free. See [fusion.md](fusion.md). vLLM's #16457 (PyTorch→Triton RoPE)
shows the floor: a naive RoPE was 40–60% of latency. vLLM also ships a **ROCm-only RoPE + KV-cache fusion**
pass (O1+, auto) that folds the cache write into RoPE. ⚠ Inductor torch-op quant now auto-fuses some quant
patterns (so adjacent norm/quant passes can be obsolete except custom-op cases), but RoPE itself remains a
custom op — the win is the fused QK-norm+RoPE+KV(+quant) kernel, not an Inductor pattern.

## Sources
- aiter Triton rope (grid, BLOCK_S=32, num_warps=4, BLOCK_D): `/sgl-workspace/aiter/aiter/ops/triton/rope/rope.py`.
- in-place / cached rope variants: `/sgl-workspace/aiter/aiter/ops/rope.py`.
- 128-bit loads / ≥1024 grid: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- vLLM ROCm RoPE+KV-cache fusion pass (O1+, auto); torch-op quant auto-fuse / custom-op caveat: https://docs.vllm.ai/en/latest/design/fusions/.
