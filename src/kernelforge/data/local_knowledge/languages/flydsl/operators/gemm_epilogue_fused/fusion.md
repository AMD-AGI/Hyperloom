---
title: gemm_epilogue_fused — fusion
kind: technique
operator: gemm_epilogue_fused
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
---

# gemm_epilogue_fused — fusion

## TL;DR
This operator *is* the fusion. The recipe: keep the GEMM accumulator in registers/LDS and apply every
downstream elementwise op (bias → residual → activation → output-quant) before the single C store —
turning N separate [M,N] memory-bound passes into zero extra HBM traffic. Highest-value targets: MLP
up/gate `silu·mul`, attn-out `+residual`, and fp8/fp4 output quant feeding the next GEMM.

## Fusable epilogue ops (compose in order)
1. **scale `α`** (per-tensor / per-row).
2. **+bias[N]** (free in CShuffle).
3. **+β·residual[M,N]** (skip-add; folds an [M,N] read+write).
4. **activation**: relu / gelu / silu / **silu·mul (gated)** — the gated `act_and_mul` on up/gate is the
   biggest win (removes a [M, inter] pass). See [[operators/act_and_mul_silu_gelu/overview.md]].
5. **output quant**: write D as fp8/fp4 with computed scale → next GEMM consumes it directly. See
   [[operators/scaled_quant_gemm/overview.md]], [[operators/quant_fp4_mxfp/overview.md]].

## Producer-side neighbors (fused before the GEMM)
- **RMSNorm + quant → GEMM**: norm and input-quant fused on the producer side →
  [[operators/fused_norm_quant/overview.md]], [[operators/rmsnorm/overview.md]].

## Why it pays (Amdahl)
The matmul is compute-bound; each appended epilogue op is ~free vs the FLOPs but each *separate* kernel
it replaces is a full [M,N] memory pass + a launch. On wide MLP N (up to ~34816) the saved silu·mul pass
and the saved quant pass are the main wins.

## How it's expressed per backend
- **CK**: `DeviceGemmMultipleD_Xdl_CShuffle` + a `CDEElementwise` op chaining bias/residual/act/scale —
  the richest fusion surface.
- **triton**: write the epilogue inline after MFMA, before store; `OPTIMIZE_EPILOGUE=1`.
- **aiter/hipBLASLt**: bias + scale (+ limited act) as epilogue args on the live tuned path.

## Pitfalls
- Fused flags change the tuning key → tune the fused variant or it won't engage.
- Down-cast before activation → accuracy loss (apply act in fp32, see [numerics.md](numerics.md)).
- Epilogue write hitting the 512B Tagram hotspot → use `OPTIMIZE_EPILOGUE`.
- Over-fusing rare paths → kernel-variant + tuning-time blowup.
