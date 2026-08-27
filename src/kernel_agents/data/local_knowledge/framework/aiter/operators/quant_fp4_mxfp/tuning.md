---
title: quant_fp4_mxfp — tuning
kind: operator_overview
operator: quant_fp4_mxfp
gens: [gfx950]
dtypes: [mxfp4, mxfp6]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/triton/quant/quant.py
  - ROCm/aiter@b467ce342:aiter/ops/quant.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://arxiv.org/abs/2511.08083
---

# quant_fp4_mxfp — tuning

> The MXFP **quant** is bandwidth-bound (read bf16, write 4-bit + E8M0 scale). The group size is **fixed at
> 32** by the OCP spec — *not* a tunable. What you tune is the kernel tiling and the scale-shuffle path;
> the real perf lever is the downstream block-scaled GEMM (operators/scaled_quant_gemm) and not paying
> for the quant at all by fusing it (fusion.md).

## Downstream block-scaled GEMM is where the MXFP4 win cashes out (gfx950 SOTA)
The cast is bandwidth-bound and small; the payoff is the block-scaled GEMM it feeds. Current bars
(AMD/HK-measured, MI355X):
- **Gluon MXFP4: 5255 TFLOPS @ 92.41% MFMA eff** (4096×4096×32768, native scaled-MFMA
  `v_mfma_scale_f32_16x16x128_f8f6f4`) — the practical MXFP4 ceiling.
- For FP8 block-scale: HIP/C++ **8-wave ping-pong 3204 TFLOPS** @ 8192 (*beats* hipBLASLt 3130, no asm);
  **4-wave interleave** is the successor. Both scheduling patterns originate from **HipKittens** (arXiv
  2511.08083) and were adopted into AMD's CDNA4 GEMM blogs.

So tune the quant cast to ~HBM peak, then chase these GEMM bars via operators/scaled_quant_gemm/tuning
and the optimization/mfma_scheduling ping-pong/interleave prior.

## Performance model
`x[M,N]` bf16 → MXFP4: traffic ≈ `2·M·N` read + `0.5·M·N` write (4-bit) + `M·N/32` E8M0 scales. Output is
~4× smaller than the bf16 input → the kernel is read-dominated; hit ~HBM peak. MXFP6 writes `0.75·M·N`.

## Triton `dynamic_mxfp4_quant` knobs (sourced)
The kernel auto-picks tiling by shape (`aiter/ops/triton/quant/quant.py`):
- **`MXFP4_QUANT_BLOCK_SIZE = 32`** — fixed by spec; comment: "Do not tune this."
- `M ≤ 32`: `BLOCK_SIZE_M=next_pow2(M)`, `BLOCK_SIZE_N=32`, `NUM_WARPS=1`, `NUM_ITER=1`, `NUM_STAGES=1`.
- `M > 32`: `BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, NUM_WARPS=4, NUM_ITER=4, NUM_STAGES=2`; if `N ≤ 16384` →
  `32×128`.
- `N ≤ 1024`: `BLOCK_SIZE_N=min(256,next_pow2(N))` (≥32 mult), `BLOCK_SIZE_M=min(8,next_pow2(M))`,
  `NUM_ITER=1, NUM_STAGES=1, NUM_WARPS=4`.
- Output scale stored as `[(N+31)//32, M].T` (uint8 / `fp8_e8m0`).

## HIP `dynamic_per_group_scaled_quant_fp4` knobs
- group size 32; `shuffle_scale` (bool) → `e8m0_shuffle` for HW MFMA layout; scale padded to
  `((m+255)//256*256, ((n+31)//32+7)//8*8)` when shuffled (`per_1x32_mx_quant_hip`).
- `num_rows` / `num_rows_factor` for ragged/MoE batches.
- `thread_data_size`, block size as usual.

## The scale-shuffle decision
- **shuffle=True** when the MXFP4 tensor feeds the HW block-scaled MFMA directly (production GEMM) — the
  scales must be in Ax/Bx layout.
- **shuffle=False** for `tl.dot_scaled` (Triton handles its own scale layout) or for simulation/debug.
Picking wrong → either wasted re-shuffle or silent corruption (numerics.md).

## Decode vs prefill
- prefill (large M): the `M>32` config; bandwidth-bound.
- decode (M≤32): the small-M config (`NUM_WARPS=1`); launch-bound → fuse into norm.

## CDNA3 caveat
On gfx942 there is no FP4 HW; the "quant" is only useful for **footprint / simulation**, and the GEMM must
dequant-on-the-fly (fused Quark kernel) with no throughput win. Don't tune MXFP4 quant for speed on gfx942.

## Sources
- Triton tiling per shape, group 32 fixed: `ROCm/aiter@b467ce342:aiter/ops/triton/quant/quant.py`.
- HIP group quant + shuffle padding: `ROCm/aiter@b467ce342:aiter/ops/quant.py` (`per_1x32_mx_quant_hip`; `per_1x32_f4_quant_hip` is a thin wrapper).
- Scaled MFMA / scale layout: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- Downstream MXFP4 GEMM ceiling 5255 TFLOPS @ 92.41%: AMD Gluon GEMM tutorial; FP8 8-wave ping-pong 3204@8192 (>hipBLASLt): AMD cdna4-gemm-kernels blog; scheduling origin: arXiv 2511.08083.
