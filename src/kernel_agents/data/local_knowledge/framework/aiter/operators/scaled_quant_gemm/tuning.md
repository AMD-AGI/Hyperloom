---
title: scaled_quant_gemm — tuning
kind: technique
operator: scaled_quant_gemm
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp4_e2m1, fp6_e2m3, fp6_e3m2]
regimes: [prefill, decode]
updated: 2026-07-14
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
  - https://arxiv.org/abs/2511.08083
---

# scaled_quant_gemm — tuning

## TL;DR
> Beyond normal GEMM tiling, the decisive new knob is the **scale pipeline**: scales arrive from global
> memory in a layout the scaled-MFMA cannot consume, so you must stage them (GlobalRead → LDS write to
> re-layout → LDS read) in parallel with the tile pipeline. Getting that pipeline right is what separates
> near-peak from mediocre on CDNA4.

## Current SOTA bars (gfx950, AMD/HK-measured)
On CDNA4 the scaled-GEMM ceiling is now near-peak; aim at these:
- **HIP/C++ 8-wave ping-pong FP8: 3204 TFLOPS** @ M=N=K=8192 (MI355X, ROCm 7.1) — *beats* hipBLASLt
  (3130) **with no assembly**. The **4-wave interleave** variant (one wave/SIMD, full 512-VGPR budget,
  128×128 tile, no `#pragma unroll`) is its robustness/perf successor (HK 4-wave FP8 3327 TFLOPS).
- **Gluon BF8: 3257 TFLOPS @ 99.72% MFMA eff** (4096×4096×16384); **Gluon MXFP4: 5255 TFLOPS @ 92.41%**
  (4096×4096×32768, native scaled-MFMA `v_mfma_scale_f32_16x16x128_f8f6f4`).

Both the 8-wave ping-pong and 4-wave interleave **scheduling patterns originate from HipKittens**
(arXiv 2511.08083) and were adopted into AMD's own CDNA4 GEMM blogs. NVIDIA-style wave specialization
underperforms on CDNA (static register allocation starves producers → ~80% peak) — use ping-pong/
interleave; see optimization/mfma_scheduling.md and the new Gluon/HipKittens cards.

## The levers
- **MFMA scaled instr**: `32x32x64` (4 packed MFMA ops, K=64 each) for throughput on large tiles vs
  `16x16x128` for smaller/skinny tiles. Gluon exposes `gl.amd.cdna4.mfma_scaled`.
- **Scale pipeline depth**: stage scales through LDS to match the MFMA's expected layout; pipeline with the
  A/B tile loads (there is no instruction to feed scales directly from registers).
- **Block size = 32 (fixed by MXFP)**: one E8M0 per 32 K-elements; your K-tiling must align to 32-element
  blocks.
- **FP6 at FP4 rate**: on CDNA4, FP6 issues at FP4 throughput — prefer FP6 over FP8 when accuracy needs the
  extra mantissa but you want FP4-class speed.
- **Standard GEMM knobs**: BLOCK_M/N/K, num_stages, num_warps, waves_per_eu, split-K for large-K
  (splitk_streamk_gemm/overview.md).

## CDNA3 vs CDNA4
- gfx942: fp8 e4m3/e5m2 FNUZ MFMA, tensor/coarse scaling — no native 32-elem block-scaled MFMA; emulate
  block scaling in software if needed.
- gfx950: native MXFP block-scaled MFMA — use it; this is where FP6=FP4-rate and 10 PFLOPS MXFP4 apply.

## Pitfalls
- Skipping LDS re-layout of scales (feeding the global-memory layout straight to MFMA) → wrong results.
- Misaligning K-tiles to 32-element blocks → scale/data misindex.

## Verify
- Achieved TFLOPS vs dtype peak; accuracy gate vs bf16 ([numerics.md](numerics.md)). Gluon tutorial reports
  BF8 99.72% efficiency and MXFP4 92.41% as reference ceilings (gfx950).

## Sources
- Gluon GEMM tutorial (scale pipeline, BF8 3257@99.72%, MXFP4 5255@92.41%): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- HIP/C++ 8-wave ping-pong FP8 3204@8192 (>hipBLASLt 3130, no asm): AMD CDNA4 GEMM blog (https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html).
- 4-wave interleave successor (1 wave/SIMD, full 512 VGPR, 128×128, no `#pragma unroll`): AMD 4-wave FP8 GEMM blog (https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html).
- Scheduling patterns originate from HipKittens; wave-spec underperforms on CDNA (~80% peak): arXiv 2511.08083.
- Triton block-scaled matmul (32x32x64 packing): https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
