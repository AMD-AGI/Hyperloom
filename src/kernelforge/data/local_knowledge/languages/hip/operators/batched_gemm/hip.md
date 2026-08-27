---
title: batched_gemm on hip — SOTA card
kind: sota_card
operator: batched_gemm
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: legacy
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/hipBLASLt
---

# batched_gemm × hip

## TL;DR
> A raw HIP/MFMA batched GEMM (hand-written kernel or a loop over `hipblasGemmStridedBatched`) is a
> **baseline / learning** path — for production it does not beat tuned hipBLASLt/CK/asm and lacks an
> autotuner. Use it only to understand the strided-batched mapping or to wire a quick custom kernel; for
> the live path use aiter/hipblaslt.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Raw HIP MFMA batched kernel (`__builtin_amdgcn_mfma_*`, batch on blockIdx.z) | hand-authored | gfx942/950; bf16/fp16 | no figure competitive with tuned hipBLASLt → **use hipblaslt** | learning / custom one-off |
| `hipblasGemmStridedBatched` (library call) | `ROCm/hipBLASLt`/hipBLAS | gfx942/950 | competitive only as a thin wrapper over hipBLASLt solutions | non-aiter stacks |

## Config space / knobs
- Map batch to `blockIdx.z`; per-block tile multiples of 8; 16×16 MFMA; LDS double-buffer; ≥1024 total WGs.
- No autotuner — you hand-pick tile/pipeline like an asm kernel but with less control.
- `hipblasGemmStridedBatched` exposes `algo`/`computeType` only; real tuning lives in hipBLASLt solutions.

## Numerics / parity
fp32 accumulate per batch; bf16 parity-safe. See [numerics.md](numerics.md).

## Integration (rebind seam)
Build-time; call directly. No env-overlay into serving GEMM.

## Pitfalls & anti-patterns
- A host-side loop of B GEMM launches → launch overhead + CU starvation (the classic anti-pattern).
- Naive global-memory inner loop (no LDS staging / no MFMA) → orders of magnitude off peak.
- Hand-tuning when hipBLASLt already has a faster solution.

## How to verify
Microbench vs `hipblaslt-bench --batch_count`; expect to lose — keep only for learning/custom needs.

## Alternatives / cross-links
hipblaslt · asm (peak) · ck · aiter ·
[overview.md](overview.md) · language ref hip_levers/....

## Sources
- Raw MFMA batched mapping: ROCm matrix-cores-cdna blog.
- Strided-batched API: https://github.com/ROCm/hipBLASLt.
