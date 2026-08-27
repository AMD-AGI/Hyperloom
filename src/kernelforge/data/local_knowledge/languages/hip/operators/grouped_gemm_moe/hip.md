---
title: grouped_gemm_moe on hip — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/aiter
---

# grouped_gemm_moe × hip

## TL;DR
> A hand-written HIP/C++ grouped-GEMM kernel (MFMA intrinsics + an `expert_offsets` table) is the route
> when you need a fusion or layout the library/Triton paths don't expose, or as the reference for an asm
> rewrite. Most production serving should use aiter; HIP is the authoring substrate.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| HIP MFMA grouped GEMM (custom epilogue/layout) | author against `__builtin_amdgcn_mfma_*` per Matrix Core blog | gfx942/950; bf16, fp8 | no published number — use for custom fusions / as asm reference | bespoke fusion not in aiter/ck |

## Config space / knobs
- MFMA instr choice (`mfma_16x16` vs `32x32`), LDS tiling/double-buffer, register-tile per wave,
  per-expert offset indexing, CU tile balancing (304 MI300X / 256 MI350X).
- Fuse act-and-mul + quant epilogue inline → [fusion.md](fusion.md).

## Numerics / parity
- fp32 accumulate; honor per-expert scale order; mask padded rows → [numerics.md](numerics.md).

## Integration (rebind seam)
- Expose via a custom op / `hipModule`; wire into the MoE layer call site. Verify by kernel name in trace.

## Pitfalls & anti-patterns
- Reimplementing align&sort/offset logic incorrectly corrupts token routing; reuse the framework's
  `expert_ids`/`num_tokens_post_pad`.
- Hand-HIP rarely beats tuned aiter asm on covered shapes — justify with a measured win.

## How to verify
- Per-expert dense oracle ([numerics.md](numerics.md)) + A/B vs aiter.

## Alternatives / cross-links
aiter · triton · ck · tilelang · [overview.md](overview.md)

## Sources
- Matrix Core programming (MFMA intrinsics): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AITER (production grouped path): https://github.com/ROCm/aiter
