---
title: quant_dequant_fp8 — tuning
kind: operator_overview
operator: quant_dequant_fp8
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3]
regimes: [both]
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:csrc/kernels/quant_kernels.cu
  - ROCm/aiter@b467ce342:aiter/ops/triton/quant/quant.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# quant_dequant_fp8 — tuning

> Standalone FP8 quant is **bandwidth-bound** — the whole game is read the activation once, write the FP8
> tensor + scale once, at peak HBM. Beyond that, the dominant lever is *not tuning the quant kernel* but
> **fusing it away** (fusion.md). This page covers the knobs when you do run it standalone.

## The performance model
For `x[M,N]` bf16 → fp8: traffic ≈ `M·N·2` (read bf16) + `M·N·1` (write fp8) + `M·(scale)`. A correct
kernel hits ~HBM bandwidth (~5.3 TB/s MI300X). If you are far below that, you are launch-bound (decode,
tiny M) or under-vectorized.

## Knobs by backend
### HIP (`csrc/kernels/quant_kernels.cu`)
- **`thread_data_size`** — elements per thread (template param, 16 or 32 in the aiter kernels). Higher →
  wider `global_load`/`global_store` (`dwordx4`), fewer launches; cap so VGPR doesn't spill.
- **block size** — multiple of 64; 256 threads/block typical. Grid ≥ 1024 WGs to fill 304 CUs.
- **vectorized cast** — pack the per-element FP8 cast over a `float4`/8-wide vector (vLLM
  `scaled_fp8_conversion_vec`); the cast itself is cheap, memory is the wall.
- **reduction pattern** — per-token amax is a row-reduce (one block per row, wave shuffle); per-tensor
  amax needs `atomicMaxFloat` into a global scale (`segmented_max_reduction`).

### Triton (`aiter/ops/triton/quant/quant.py`)
- `dynamic_per_token_quant_fp8_i8`: grid = `(rows,)` (one program per token row), `NUM_COL_POW2 =
  next_power_of_2(cols)` so the row fits a single block reduce. `DTYPE_MAX` from `torch.finfo`.
- `BLOCK_SIZE` ≈ padded `hidden`; `num_warps` 4–8 for wide hidden, 1–2 for narrow.
- For very wide hidden, tile the column reduce; for decode (few rows) the kernel is launch-bound — fuse.

### aiter dispatch
- `dynamic_per_token_scaled_quant`, `static_per_tensor_quant`, `dynamic_per_tensor_quant` dispatch to the
  HIP kernels; the Triton variants exist for the fused/portable path. Pick per-token dynamic for
  activations (accuracy), per-tensor static for weights (speed).

## Decode vs prefill
- **prefill** (large M): bandwidth-bound; maximize vector width + occupancy, grid ≥ 1024.
- **decode** (M≤256): the standalone quant is dominated by **launch latency** → the only real win is to
  fuse it into the preceding RMSNorm/act or the GEMM (no separate launch). See fusion.md.

## When NOT to tune this kernel
If the activation is produced by RMSNorm/act_and_mul immediately before the linear, **fuse the quant into
that op** (`fused_rms_fp8_*_quant`) — you save a full read+write of the activation and a kernel launch.
Tuning a standalone quant kernel that should have been fused is a local optimum.

## Sources
- HIP kernel knobs (`thread_data_size`, block size, scaled-quant impl): `ROCm/aiter@b467ce342:csrc/kernels/quant_kernels.cu`.
- Triton per-token grid/`NUM_COL_POW2`/`DTYPE_MAX`: `ROCm/aiter@b467ce342:aiter/ops/triton/quant/quant.py`.
- ≥1024 WG grid, vectorization, 304 CU: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
