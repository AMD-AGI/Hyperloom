---
title: splitk_streamk_gemm on asm — SOTA card
kind: sota_card
operator: splitk_streamk_gemm
backend: asm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill]
status: competitive
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter
  - https://arxiv.org/abs/2301.03598
---

# splitk_streamk_gemm × asm

## TL;DR
> Hand/asm MFMA kernels with built-in split-K are aiter's weapon for the **small-M decode GEMMs** where
> library tiles leave CUs idle — the skinny/wvSplitK-style asm path splits K across workgroups and reduces
> in fp32. Use for the hottest captured decode shapes; author/tune via aiter, not by hand from scratch.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter asm split-K GEMM (skinny/wvSplitK-style) | `ROCm/aiter@HEAD` (asm GEMM + skinny path) | gfx942/950; bf16, fp8 | no first-party number reproduced — selected by aiter for small `padded_M` | decode small-M, large-K |

## Config space / knobs
- Split-K factor, MFMA instr (mfma_16x16 preferred for skinny), LDS/atomic reduction, register tiling.
  Selection is driven by aiter's 9-tuple key (small `padded_M` → skinny/asm) →
  ../skinny_gemv_decode/backends/asm.md.

## Numerics / parity
- fp32 accumulate + fp32 reduction; atomic order non-deterministic → [numerics.md](numerics.md).

## Integration (rebind seam)
- Engages through aiter dispatch when the shape's `padded_M` is small; verify via aiter tuned-config log
  marker + asm kernel name in trace.

## Pitfalls & anti-patterns
- asm kernels are gfx-specific (gfx942 vs gfx950 MFMA layouts differ) — don't assume portability.
- Hand-asm rarely beats aiter's tuned asm on covered shapes; tune through aiter.

## How to verify
- A/B vs triton split-K + dense fp32 oracle ([numerics.md](numerics.md)).

## Alternatives / cross-links
triton · ck · hipblaslt · hip · [overview.md](overview.md)

## Sources
- AITER (asm GEMM / skinny split-K): https://github.com/ROCm/aiter
- Stream-K: https://arxiv.org/abs/2301.03598
