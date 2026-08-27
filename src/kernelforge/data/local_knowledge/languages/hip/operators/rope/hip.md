---
title: rope on hip — SOTA card
kind: sota_card
operator: rope
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu
  - /sgl-workspace/aiter/aiter/ops/pos_encoding.py
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# rope × hip

## TL;DR
Hand-written HIP is the reference. vLLM's `csrc/pos_encoding.cu` `rotary_embedding` is the canonical
editable kernel: per (token, head) read the head-dim vector + cos/sin from the cache, apply the rotation
(NeoX/GPT-J via `is_neox`), write in place. aiter exposes a HIP `rotary_embedding` /
`batched_rotary_embedding` (`module_pos_encoding`). Reach for HIP to fuse RoPE into a custom attention
entry.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM HIP `rotary_embedding` | `vllm/csrc/pos_encoding.cu` | gfx942/950, bf16/fp16 | bandwidth-bound, in-place | editable HIP / non-AITER |
| aiter HIP `rotary_embedding` / `batched_rotary_embedding` | `aiter/ops/pos_encoding.py` (`module_pos_encoding`) | gfx942/950 | floor | aiter pos-encoding path |
| AMD elementwise template | hip_levers/patterns §2 | all | reference | from-scratch |

## Config space / knobs
- Grid: one program per (token, head) or a tile; block ×64. Decode (1 token) → `b·h` programs.
- Read head-dim vector (`float4`/`__half2`), cos/sin from `cos_sin_cache[pos]` (fp32), rotate in fp32,
  write in place (`__restrict__`, 16-B alignment).
- `is_neox` selects rotate-halves vs adjacent-pairs; loop bounded by `rotary_dim` (partial).
- `__launch_bounds__(block, 4)`; `hipcc --offload-arch=gfx942 -O3`.

## Numerics / parity
cos/sin fp32; `is_neox` matches config; `rotary_dim` bound; deterministic → token-identical parity. See
[numerics.md](numerics.md).

## Integration (rebind seam)
- vLLM: edit `csrc/pos_encoding.cu`, rebuild; op `_C::rotary_embedding` in `torch_bindings.cpp`.
- aiter: JIT `module_pos_encoding`; called via `rotary_embedding(...)`.
- Standalone: `torch.utils.cpp_extension`.

## Pitfalls & anti-patterns
- ⚠ `rotary_dim < head_size` not handled → OOB (partial-rotary).
- ⚠ Wrong `is_neox`.
- ⚠ `dim3 grid(0)` for empty batch → crash (guard).

## How to verify
`--save-temps` ISA; isolated vs fp64 rotation; partial-rotary test; greedy parity (token-identical).

## Alternatives / cross-links
aiter · triton · vllm_kernels ·
hip_levers/patterns §2 · mrope/backends/hip.

## Sources
- vLLM HIP rotary_embedding: https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu.
- aiter HIP pos_encoding: `/sgl-workspace/aiter/aiter/ops/pos_encoding.py`.
- wave64 / vectorized I/O: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html.
