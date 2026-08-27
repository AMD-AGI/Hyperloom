---
title: gemm_epilogue_fused — tuning
kind: technique
operator: gemm_epilogue_fused
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - ROCm/aiter@HEAD:gradlib/gradlib/gemm_tuner.py
---

# gemm_epilogue_fused — tuning

## TL;DR
Tune the **fused variant as its own shape** — the epilogue flags (bias/act/residual/quant) are part of
the lookup key, so a DB tuned on the bare GEMM does not engage the fused call. The GEMM-core knobs are
identical to dense (`mfma_16x16`, ≥1024 WGs, 8-multiple tiles); the epilogue-specific lever is
**`OPTIMIZE_EPILOGUE=1`** to route the C write off the 512B Tagram hotspot.

## What dominates time
The matmul dominates FLOPs; the epilogue is a thin pass on the accumulator before store. So tuning is
~90% the dense-GEMM problem plus: (1) don't let the epilogue serialize the store (Tagram/CShuffle), (2)
keep the fused output dtype (bf16 vs fp8) matching the next consumer to avoid an extra cast.

## Knob space
- **GEMM core**: `BLOCK_M/N/K` (8-multiples), `matrix_instr_nonkdim=16`, `num_stages`, `waves_per_eu`,
  `SPLIT_K` (decode) — same as operators/dense_gemm/tuning.md.
- **`OPTIMIZE_EPILOGUE=1`**: avoids the 512B Tagram hotspot on the fused C write (CShuffle path).
- **CShuffle params (CK)**: `CShuffleMXdlPerWavePerShuffle`, `CShuffleNXdlPerWavePerShuffle`,
  `CDEBlockTransferScalarPerVector` — size the epilogue write to match output dtype/vector width.
- **Output dtype**: fuse the quant so D is fp8/fp4 if the next GEMM consumes it (saves a cast pass).
- **act in fp32**: apply activation on the fp32 accumulator before down-cast (accuracy + no extra pass).

## How to tune (live lever)
Capture the **fused** shapes live (`AITER_TUNE_GEMM=1`) so bias/scale flags match, race with gradlib
(`err_ratio<0.05`), deploy `AITER_CONFIG_GEMM_BF16`, verify `grep -c 'is tuned on cu_num'`. See
aiter.

## Pitfalls
- Tuning the bare GEMM then expecting the fused call to hit it → key miss, 0 engagement (same failure
  class as bias mismatch).
- Epilogue write hitting the Tagram hotspot (forgot `OPTIMIZE_EPILOGUE`) → store-bound stalls.
- Down-casting before activation → accuracy loss.
- Over-enumerating rare fused variants → tuning-time blowup for little Amdahl.

## Sources
- mfma/tile/Tagram/OPTIMIZE_EPILOGUE levers: ROCm workload guide.
- CShuffle epilogue params: ROCm "Optimizing with Composable Kernel".
- Live tuning + err_ratio gate: `ROCm/aiter@HEAD` (see backends/aiter.md).
