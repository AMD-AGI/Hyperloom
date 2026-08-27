---
title: batched_gemm — tuning
kind: technique
operator: batched_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# batched_gemm — tuning

## TL;DR
The dominant batched-GEMM regime is **many small matmuls**, so the tuning goal is **machine fill**: map
the whole batch into one launch with **≥1024 total workgroups** and place same-batch workgroups on the
same XCD. Per-matmul tile choices matter less than getting all 304 CUs busy. For uniform shapes the live
lever is the same aiter DB as dense GEMM (aiter).

## What dominates time
- **Small/decode batched (head_dim 64/128, M≈1)**: occupancy- and launch-bound. One strided-batched
  kernel >> a loop of B GEMMs. Split-K across the K=head_dim and batch-parallelism fill the machine.
- **Large-M prefill batched**: compute-bound, behaves like B dense GEMMs → use the dense knobs.

## Knob space
- **Grid mapping**: workgroups = ceil(M/BLOCK_M)·ceil(N/BLOCK_N)·B; ensure total ≥1024. Small per-matmul
  tiles (BLOCK_M/N 16–64) so B alone gives enough WGs.
- **MFMA**: prefer `16×16×16` (head_dim 64/128 maps cleanly; 32×32 wastes lanes on small N).
- **SPLIT_K**: for decode M≈1 + small K, split the K reduction to add parallelism →
  operators/splitk_streamk_gemm/overview.md.
- **XCD/L2 placement**: keep a batch's tiles on one XCD; ≥1024 WGs for stable timing.
- **b_preshuffle**: pre-permute per-batch B if the layout is reused.
- triton: `BLOCK_M/N/K`, `matrix_instr_nonkdim=16`, `num_stages`, `waves_per_eu`, `GROUP_SIZE_M`.

## How to tune (live lever)
Uniform batched GEMM goes through aiter's tuned_gemm just like dense; capture with `AITER_TUNE_GEMM=1`,
race with gradlib (`err_ratio<0.05`), deploy by `AITER_CONFIG_GEMM_BF16`. Verify
`grep -c 'is tuned on cu_num'`. (See aiter.)

## Pitfalls
- Looping B separate GEMM launches (host overhead + CU starvation) instead of one batched kernel.
- Large tiles on small (M,N) → most lanes idle; total WGs < 1024.
- For *variable* per-batch shapes, batched GEMM pads to max → wasted FLOPs; use grouped GEMM instead.

## Sources
- Occupancy / ≥1024 WG / XCD placement: ROCm workload guide.
- MFMA 16×16 mapping: ROCm matrix-cores-cdna blog.
- Live tuning path: `ROCm/aiter@HEAD` (see backends/aiter.md).
