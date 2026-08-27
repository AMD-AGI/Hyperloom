---
title: batched_gemm on triton — SOTA card
kind: sota_card
operator: batched_gemm
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# batched_gemm × triton

## TL;DR
> Triton is the easiest way to author a **fused** batched GEMM (custom per-batch epilogue, odd shapes,
> research kernels) and is competitive on MI300X — but on plain uniform batched GEMM it generally loses
> to tuned hipBLASLt dispatched by aiter. Reach for it when you need fusion/flexibility, not for the
> last few % on vanilla batched matmul.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton batched matmul (batch on program axis) | Triton matmul tutorial + AMD autotune | gfx942/950; bf16/fp16/fp8 | competitive but typically < tuned hipBLASLt on plain batched (no public win figure) → use hipblaslt for plain | fused / custom-epilogue batched GEMM |

## Config space / knobs
- `BLOCK_M/N/K` (8-multiples; small M/N for small batched), `matrix_instr_nonkdim=16` (force 16×16 MFMA),
  `num_stages` (deeper for K-deep), `num_warps`, `waves_per_eu`, `GROUP_SIZE_M`, `SPLIT_K` for decode.
- Map batch onto a program-id axis so total programs ≥1024.
- Autotune over these per (B,M,N,K) — Triton's perf is configuration-sensitive.

## Numerics / parity
fp32 accumulate per batch; bf16 parity-safe. fp8 needs task gating. See [numerics.md](numerics.md).

## Integration (rebind seam)
JIT kernel callable from Python — wire directly into your op, or register as an aiter candidate. No
env-overlay into sglang's library GEMM path.

## Pitfalls & anti-patterns
- Default (untuned) configs → well below hipBLASLt; always autotune.
- `matrix_instr_nonkdim` left at 32 on small head_dim shapes wastes lanes.
- Too-large tiles on small batched → < 1024 programs, CU starvation.

## How to verify
Bench the kernel vs `hipblaslt-bench --batch_count` on the same shape; adopt only on a win or for fusion.

## Alternatives / cross-links
hipblaslt · aiter · ck · asm ·
[overview.md](overview.md) · language ref triton_levers/....

## Sources
- Triton matmul tutorial: triton-lang.org.
- MFMA/tile levers (matrix_instr_nonkdim, ≥1024 WG): ROCm workload guide.
