---
title: gemm_epilogue_fused — overview
kind: operator_overview
operator: gemm_epilogue_fused
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
---

# gemm_epilogue_fused  (`D = act(α·A·Bᵀ + bias + β·residual)`, optionally → quant)

## TL;DR
Epilogue fusion folds bias / activation / residual / output-quant into the GEMM's C write-out, removing
a full [M,N] memory round-trip per fused op. The single most important fact: the GEMM is compute-bound
so the appended epilogue is **nearly free**, while the separate elementwise kernel it replaces is a
memory-bound [M,N] pass — fuse the MLP up/gate `act_and_mul` and the output-quant first. Use
`OPTIMIZE_EPILOGUE=1` to keep the fused C write off the 512B Tagram hotspot.

## Math contract
`D[M,N] = act( α · (A[M,K]·B[N,K]ᵀ) + bias[N] + β · residual[M,N] )`, then optional fp8/fp4 quant of D
with a computed/given scale. act ∈ {identity, relu, gelu, silu, silu·mul (gated)}. dtype: bf16/fp16 in,
fp32 accumulate; out bf16 or quantized fp8/fp4. The fused flags (bias/act/residual/quant) are part of
the tuning lookup key.

## Shape regimes
- **MLP up/gate (prefill)**: K=hidden, N=inter (large, up to ~34816), fuse `silu·mul` to skip a giant
  [M,inter] pass — the highest-value fusion.
- **attn out / down proj**: fuse `+residual` (skip-add) into the GEMM.
- **decode (skinny M)**: epilogue cost is tiny; fusion still removes the extra kernel launch + pass.

## Where it matters (Amdahl)
The GEMM mass itself is the dense_gemm 78–81%; epilogue fusion doesn't change the matmul FLOPs but
deletes the extra elementwise/quant passes (each a [M,N] read+write). On wide MLP N this is a real
memory-traffic win and removes kernel-launch overhead; typically a few % when the unfused passes were
non-trivial.

## Backend landscape
Competitive positioning for this operator; this folder's card is [flydsl.md](flydsl.md). Other backends
live in their own folders (`framework/aiter/`, `languages/{triton,ck,asm,hip}/`).

| backend | status |
|---|---|
| aiter | 🟢 sota (live path, bias/scale epilogue) |
| hipblaslt | 🟢 sota (executed fused kernels) |
| ck | 🟢 sota (CShuffle CDEElementwise — richest fusion) |
| triton | 🟡 competitive (easy custom epilogue) |
| hip | 🟤 legacy (raw fused kernel) |
| flydsl | 🟡 competitive (MFMA row + LDS CShuffle epilogue) |

## Fusion neighbors
This *is* the fusion operator. Producer-side: norm+quant before the GEMM
([[operators/fused_norm_quant/overview.md]]); activation op
([[operators/act_and_mul_silu_gelu/overview.md]]); output quant
([[operators/scaled_quant_gemm/overview.md]], [[operators/quant_dequant_fp8/overview.md]]). See
[fusion.md](fusion.md).

## Numerics
fp32 accumulate; act applied in fp32 before down-cast; quant epilogue needs task-accuracy gating →
[numerics.md](numerics.md).

## How to bench
Isolated: fused GEMM vs (GEMM + separate elementwise) at the exact (M,N,K,flags). e2e: same-session A/B,
gate delta>0.5% + non-overlap + engagement.
