---
title: grouped_gemm_moe on triton — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
---

# grouped_gemm_moe × triton

## TL;DR
> Triton fused-MoE GEMM (the `fused_moe` triton kernel used by vllm/sglang) is the **authorable,
> portable** grouped path: a single kernel walks tiles over a sorted token layout using an `expert_ids`
> offset table. Competitive and the right place to prototype tile/shape changes; for peak on AMD prefer
> aiter asm.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| triton `fused_moe` (sorted tokens + expert_ids offsets) | vllm/sglang triton MoE kernel; aiter also ships triton MoE | gfx942/950; bf16, fp8, mxfp4 | no first-party number reproduced; use as authorable baseline vs aiter asm | prototyping, shapes aiter lacks |

## Config space / knobs
- `BLOCK_M / BLOCK_N / BLOCK_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `waves_per_eu`,
  `matrix_instr_nonkdim` (16 → mfma_16x16 for small groups, else 32).
- MoE align block size must equal `BLOCK_M` so each expert's padded M maps cleanly to tiles
  (align&sort design).
- For fp8/mxfp4 use the block-scaled triton matmul → ../scaled_quant_gemm/backends/triton.md.

## Numerics / parity
- fp32 accumulate; mask padded rows; fp32 routing combine → [numerics.md](numerics.md).

## Integration (rebind seam)
- Overlay the triton kernel module the framework imports (vllm/sglang `fused_moe`); verify it engages by
  the kernel name in a rocprof trace and the autotune cache key.

## Pitfalls & anti-patterns
- Mismatched align-block vs BLOCK_M → wrong tile→expert mapping or wasted padding.
- Autotune over a uniform synthetic M distribution mis-tunes for skewed real routing.

## How to verify
- A/B vs aiter asm with the per-expert dense oracle ([numerics.md](numerics.md)).

## Alternatives / cross-links
aiter · ck · hip · tilelang · [overview.md](overview.md)

## Sources
- AITER (ships triton MoE kernels): https://github.com/ROCm/aiter
- MoE align & sort + tile mapping: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
