---
title: splitk_streamk_gemm — overview
kind: operator_overview
operator: splitk_streamk_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode]
updated: 2026-06-05
---

# splitk_streamk_gemm

## TL;DR
> A GEMM **decomposition strategy** (not a new math op): partition the K reduction (split-K) or the whole
> tile space (stream-K) across more workgroups to fill all CUs when the natural tile count is too small to
> saturate the GPU — the key fact is **it only helps when you're CU-underutilized** (small M·N or huge K).

## Math contract
- Computes the same `C = A @ B (+ bias)` as dense GEMM. Decomposition changes *who computes which partial
  sums*, not the result.
- **Split-K**: split the K dimension into `SPLIT_K` chunks; each workgroup computes a partial `C`; combine
  by atomic add or a workspace + reduction kernel.
- **Stream-K**: flatten all (tile × k-iteration) work into a 1-D iteration space and distribute it evenly
  across a fixed number of persistent workgroups (≈ CU count), so partial tiles balance across CUs; fix-up
  partial tiles with a reduction. (Osama et al., "Stream-K", arXiv:2301.03598.)
- Accumulate fp32; output dtype as dense GEMM.

## Shape regimes
- **Small-M / skinny** (decode-ish, M small, N·K moderate): few output tiles → many idle CUs → split-K /
  stream-K recover utilization. See [../skinny_gemv_decode/overview.md](../skinny_gemv_decode/overview.md).
- **Large-K, small M·N**: classic split-K territory (the reduction dominates).
- **Large M·N**: plenty of tiles already → split-K/stream-K usually unnecessary or harmful (extra
  reduction cost). Use plain dense GEMM ([../dense_gemm/overview.md](../dense_gemm/overview.md)).

## Where it matters (Amdahl)
- Matters for the GEMM shapes that otherwise leave the 304 CUs (MI300X) / 256 CUs (MI350X) idle — small
  batch decode projections and tall-skinny problems. Stream-K's value is wave-quantization removal: it
  turns a "1.2 waves" launch into balanced full occupancy.

## Backend landscape
Competitive positioning for this operator; this folder's card is [flydsl.md](flydsl.md). Other backends
live in their own folders (`framework/aiter/`, `languages/{triton,ck,asm,hip}/`).

| backend | status |
|---|---|
| triton | 🟢 sota |
| ck | 🟡 competitive |
| hipblaslt | 🟡 competitive |
| asm | 🟡 competitive |
| hip | 🟡 competitive |
| flydsl | 🟡 competitive |

## Fusion neighbors
- The reduction/fix-up epilogue can fold bias/activation; for fp8 it can fold dequant scale →
  [../scaled_quant_gemm/overview.md](../scaled_quant_gemm/overview.md),
  [../gemm_epilogue_fused/overview.md](../gemm_epilogue_fused/overview.md).

## Numerics
- Atomic-add reduction order is non-deterministic; workspace reduction is deterministic →
  [numerics.md](numerics.md).

## How to bench
- Sweep `SPLIT_K`/stream-K on/off across the target small-M / large-K shapes; median ≥3 warm reps; oracle =
  dense fp32 reference. Always compare against plain dense GEMM to confirm the decomposition actually wins
  for that shape.
