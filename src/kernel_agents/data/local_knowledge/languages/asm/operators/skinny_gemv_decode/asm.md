---
title: skinny_gemv_decode on asm — SOTA card
kind: sota_card
operator: skinny_gemv_decode
backend: asm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: competitive
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
---

# skinny_gemv_decode × asm

## TL;DR
> Hand-tuned asm is what makes aiter's **wvSplitK / skinny** decode kernels near-bandwidth-optimal: a
> split-K GEMV that streams W coalesced, uses mfma_16x16 or VALU dots for the tiny M, and reduces in fp32.
> This is the peak decode-GEMM path; author/tune through aiter rather than raw asm.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter asm skinny / wvSplitK GEMV | `ROCm/aiter@HEAD` (asm skinny path) | gfx942/950; bf16, fp16, fp8 | no first-party number reproduced — selected by aiter for small `padded_M` | decode M=1..8, bandwidth-bound |

## Config space / knobs
- Split-K factor (fill CUs: 304 MI300X / 256 MI350X), mfma_16x16 vs VALU dot, coalesced W load pattern,
  fp32 reduction (atomic/LDS). Selection via aiter 9-tuple key (small `padded_M`).

## Numerics / parity
- fp32 accumulate + split-K reduction; fp8 weight scale after dot → [numerics.md](numerics.md).

## Integration (rebind seam)
- Engages through aiter dispatch on small `padded_M`; verify aiter tuned-config log + asm kernel name in
  trace.

## Pitfalls & anti-patterns
- asm is gfx-specific (gfx942 vs gfx950 MFMA layouts differ).
- Uncoalesced W reads kill the bandwidth-bound win — the whole point is W streaming.

## How to verify
- HBM GB/s vs peak + A/B vs triton skinny + dense fp32 oracle ([numerics.md](numerics.md)).

## Alternatives / cross-links
aiter · triton · hip · [overview.md](overview.md) ·
split-K: ../splitk_streamk_gemm/backends/asm.md

## Sources
- AITER asm skinny/wvSplitK: https://github.com/ROCm/aiter
- vLLM ROCm decode GEMV custom kernels: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
