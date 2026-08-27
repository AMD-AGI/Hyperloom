---
title: rope on triton — SOTA card
kind: sota_card
operator: rope
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/rope/rope.py
  - https://github.com/vllm-project/vllm/pull/16457
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# rope × triton

## TL;DR
Triton is the authorable SOTA and the path vLLM took to kill the naive-PyTorch RoPE bottleneck (#16457:
PyTorch RoPE was 40–60% of Qwen2-VL latency → Triton fixed it). aiter ships its own Triton rope
(`ops/triton/rope/rope.py`) plus fused QKV-split+RoPE. Memory-bound → Triton matches CK.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter Triton `_rope_fwd` | `aiter/ops/triton/rope/rope.py` | gfx942/950, bf16/fp16 | `grid=(b,h,cdiv(s,32))`, BLOCK_S=32, num_warps=4 | the Triton tier |
| `fused_qkv_split_qk_rope` | `aiter/ops/triton/rope/fused_qkv_split_qk_rope.py` | gfx942/950 | QKV-split + RoPE one kernel | fused QKV proj output |
| vLLM Triton rope (from flash_attn) | vLLM #16457 | gfx942/950 | replaced 40–60%-latency PyTorch RoPE | vLLM RoPE path |

## Config space / knobs
- `BLOCK_D = d/2` (NeoX) or `d` (GPT-J); `BLOCK_D_HALF`; `BLOCK_S=32`; `num_warps=4`.
- grid `(batch, head, cdiv(seq, BLOCK_S))` — many programs for prefill, `b·h` for decode.
- cos/sin fp32; in-place write; vectorized Q/K loads.
- `is_neox` and `rotary_dim` (partial) as constexpr/args.

## Numerics / parity
cos/sin fp32; `is_neox` matches config; **bound the rotation by `rotary_dim`** (partial — #22593 illegal
access if assumed full). Deterministic → token-identical greedy parity. See [numerics.md](numerics.md).

## Integration (rebind seam)
- Direct import from `aiter.ops.triton.rope`.
- vLLM: `VLLM_ROCM_USE_AITER_TRITON_ROPE=1` selects the Triton rope.
- torch.compile: Inductor can emit RoPE; the fused QKV-split+RoPE is best from the aiter kernel.

## Pitfalls & anti-patterns
- ⚠ Assuming `rotary_dim == head_size` → out-of-bounds for partial-rotary models (#22593).
- ⚠ Wrong `is_neox`.
- `num_warps=8` over-warps (memory-bound) → use 4.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; isolated vs fp64 rotation; partial-rotary shape test; greedy parity.

## Alternatives / cross-links
aiter · hip · vllm_kernels · mrope/backends/triton ·
triton_levers/patterns.

## Sources
- aiter Triton rope: `/sgl-workspace/aiter/aiter/ops/triton/rope/rope.py`, `fused_qkv_split_qk_rope.py`.
- vLLM PyTorch→Triton RoPE (40–60% latency): https://github.com/vllm-project/vllm/pull/16457.
- partial-rotary fix: https://github.com/vllm-project/vllm/pull/22593.
