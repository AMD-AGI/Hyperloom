---
title: kv_cache_quant — numerics
kind: operator_overview
operator: kv_cache_quant
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, fp8_e5m2, int8]
regimes: [both]
updated: 2026-07-14
sources:
  - vllm-project/vllm@HEAD:csrc/cache_kernels.cu
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/amd/quant_utils.cuh
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/aiter
---

# kv_cache_quant — numerics

> KV quant accuracy is unusual: the quantized KV is read *many times* (every decode step attends to all
> past tokens), so an error in a stored K/V propagates through the whole softmax. The store is one-shot but
> the read is repeated → KV quant accuracy matters more per-bit than activation quant. Cross-refs:
> operators/quant_dequant_fp8/numerics, hardware/shared/dtype_numerics.

## Which FP8 for KV
- **E4M3** is the default KV dtype (more mantissa; KV values are well-scaled after RoPE). E5M2 only if the
  KV range exceeds E4M3 after scaling (rare).
- **FNUZ on gfx942 / OCP on gfx950** — the same off-by-2× trap: a KV cache written with one dialect and
  read by an attention kernel expecting the other is silently ~2× wrong. vLLM uses `Float8_e4m3fnuz` on
  ROCm; the FA/paged backends must agree.

## k_scale / v_scale
- **per-tensor static** (default): scalar `k_scale`, `v_scale` from calibration. `reshape_and_cache` takes
  them as `double` and does `scaled_convert<cache_t, scalar_t>(kv, scale)` on store; the attention kernel
  multiplies back on read. Cheapest, paged-friendly.
- **block / per-token**: aiter `fused_qk_norm_rope_cache_block_quant_shuffle` stores per-block scales;
  better for models with high per-token KV variance. `pts` = per-tensor-scale variant.
- **Calibration**: a too-small scale saturates large KV (clamps to ±448/±240); too-large underflows small
  KV. vLLM has a `--calculate-kv-scales` path; static scales come from the quant config.

## Online-softmax interaction
The paged-attention kernel accumulates the softmax in **fp32** regardless of KV dtype — the KV is
dequantized to fp32 before the QK·V math. So the only loss is the KV *storage* rounding, not the softmax.
This is why FP8 KV is relatively safe: the high-dynamic-range part (softmax) stays fp32.

## INT8 KV
INT8 KV (`int8` cache dtype) uses a symmetric scale like activation INT8; uniform spacing makes it more
outlier-sensitive than FP8, so FP8 is the usual KV choice on CDNA3/4. INT8 KV is mainly for INT8-native
deployments. MXFP/block-scaled KV is emerging on CDNA4 (per-block E8M0) but per-tensor FP8 dominates today.

## Stochastic rounding
Not standard for KV store (RNE is used). The KV is written once; the repeated-read concern is addressed by
finer scale granularity, not SR.

## Accuracy gates (real, observed)
- **Gate on task accuracy** (gsm8k/mmlu), never byte parity. FP8 KV typically costs < ~0.5 pt on large
  models with per-tensor scales.
- **Observed failure:** AITER MLA + KV caused a gsm8k loss with Kimi-K2 DP2TP4 (aiter #1455) — a real
  reminder that KV+MLA+quant combinations need an explicit accuracy gate, not assumed parity.
- Compare greedy/temp=0 outputs before/after enabling `--kv-cache-dtype fp8`.

## Pitfalls
- **FNUZ↔OCP** mismatch between the cache write and the attention read (~2× error).
- **Stale/wrong k_scale,v_scale** — saturation or underflow of stored KV.
- **KV layout shuffle mismatch** (`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`) between write and FA read.
- **Assuming softmax is lossy** — it stays fp32; only the KV storage is quantized.
- **INT8 KV outlier sensitivity** — prefer FP8 unless INT8-native.

## Verify
- Round-trip a K/V tensor through the cache dtype; check max-rel error.
- e2e: gsm8k with and without `--kv-cache-dtype fp8`, same seed/temp=0; confirm the attention backend reads
  the same dialect it was written with.

## Sources
- vLLM scaled_convert on store, k_scale/v_scale, kv_cache_dtype dispatch: `vllm-project/vllm@HEAD:csrc/cache_kernels.cu`, `csrc/quantization/fp8/amd/quant_utils.cuh`.
- FNUZ vs OCP, FP8 ranges: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AITER MLA+KV gsm8k regression (#1455): https://github.com/ROCm/aiter (issues).
