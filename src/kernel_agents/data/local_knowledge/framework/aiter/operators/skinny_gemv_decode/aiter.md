---
title: skinny_gemv_decode on aiter — SOTA card
kind: sota_card
operator: skinny_gemv_decode
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:csrc/kernels/custom_kernels.cu
  - https://github.com/ROCm/aiter
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
---

# skinny_gemv_decode × aiter

## TL;DR
> aiter has a **dedicated skinny / wvSplitK GEMM path** that the per-shape dispatch selects whenever the
> 10-tuple key's `padded_M` is small (decode M=1..8). It's the live decode-GEMM path on sglang/vllm and the
> SOTA on AMD — bandwidth-bound, so it splits K across all CUs (64 lanes/wave) rather than chasing MFMA
> throughput. To improve decode projections you tune aiter's DB with the **real small-M shapes** (capture
> M=1..8 explicitly; tuning only large M misses the decode regime entirely).

## SOTA implementation
There are two engagement modes: a tuned CSV row (`libtype=skinny`) and a built-in default that fires for
canonical decode shapes even with no tuned row. From `aiter/tuned_gemm.py`
(`ROCm/aiter@b467ce342`):

```python
def is_skinny_default_shape(M, N, K, dtype, cu_num=None):
    return (dtype in [dtypes.fp16, dtypes.bf16] and K % 8 == 0 and (
        (((M == 1 and N <= 2*cu_num) or (M > 1 and M <= 4 and N <= cu_num)) and K <= 9216)
        or ((M > 4 and M <= 8 and N <= cu_num) and K <= 5120)
        or ((M > 8 and M <= 16 and N <= cu_num) and K <= 256)))
# executor:
def skinny_gemm(inp, weights, solidx, ...):
    if solidx == 0: ops.wvSpltK(weights, inp, out, inp.shape[0], get_cu_num())
    elif solidx == 1: ops.LLMM1(weights, inp, out, 4)
    if solidx == 2: ops.wv_splitk_small_fp16_bf16(weights, inp, out, inp.shape[0], get_cu_num())
```

The default config picks `libtype=skinny, solidx=2` (`wv_splitk_small_fp16_bf16`). The kernel
(`csrc/kernels/custom_kernels.cu`) does **64-lane K-splitting per wave** and fetches B in interleaved K-split
to coalesce HBM reads — bandwidth-optimal for tall-skinny weights.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `wvSpltK` (solidx 0) | `ops.wvSpltK` | gfx942/950; bf16/fp16 | bandwidth-bound; passes `CuCount` to fill all CUs | M=1..8, moderate N |
| `LLMM1` (solidx 1) | `ops.LLMM1` | gfx942/950; bf16/fp16 | LL-style GEMV | very thin M=1 |
| `wv_splitk_small_fp16_bf16` (solidx 2, **default**) | `ops.wv_splitk_small_fp16_bf16` | gfx942/950; bf16/fp16 | the default skinny kernel; no first-party isolated number reproduced | the `is_skinny_default_shape` envelope |

Beyond the skinny path, a tuned CSV row may instead select **`libtype=opus`** (split-K GEMM) for small-M
bf16 — the dispatcher races it against skinny/asm/hipBLASLt at tune time. See
operators/dense_gemm/backends/opus.md.

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| `padded_M` (from 10-tuple) | small (≤8 after bucketing) | routes the shape to the skinny path | — |
| `solidx` | 0 / 1 / 2 | wvSpltK / LLMM1 / wv_splitk_small (default) | 2 |
| `CuCount` | `get_cu_num()` | split K across this many CUs | all CUs (304 MI300X) |
| `K % 8 == 0` | required | skinny default needs K aligned to 8 | — |
| N bound | `N ≤ cu_num` (or `2·cu_num` for M=1) | skinny default envelope on N | — |
| K bound | ≤9216 (M≤4) / ≤5120 (M≤8) / ≤256 (M≤16) | skinny default envelope on K | — |
| `AITER_CONFIG_GEMM_BF16` | path | deploy tuned small-M rows | default CSV |

**Constraint:** `skinny_gemm` asserts `not bpreshuffle` ("bpreshuffle is not supported in skinny_gemm!") —
preshuffled weights can't use the skinny path.

## Numerics / parity
- fp32 accumulate + split-K reduction; fp8 weight scales applied after dot → [numerics.md](numerics.md).
  bf16/fp16 skinny is parity-safe vs the dense kernel up to reduction order.

## Integration (rebind seam)
Live call site: `aiter.tuned_gemm` (`gemm_a16w16` / scaled variants) — the skinny impl is one of the
candidates `solMap["skinny"]` dispatches to, plus the `is_skinny_default_shape` auto-default. Deploy by
`AITER_CONFIG_GEMM_BF16` env. Verify via `grep 'is tuned on cu_num'` and the skinny kernel name in a rocprof
trace.

## Pitfalls & anti-patterns
- **bias/scaleAB mismatch** vs live calls → lookup miss → falls back to a non-skinny kernel (slower). Capture
  live small-M shapes.
- **Tuning only large M misses decode entirely** — capture M=1..8 explicitly.
- **bpreshuffle + skinny is unsupported** (asserts) — if weights are preshuffled, the shape can't use skinny.
- Outside the `is_skinny_default_shape` envelope (N>cu_num, K too large, K%8≠0), the default won't fire — only
  a tuned row will route there.

## How to verify (worked example)
```bash
# A/B the skinny kernel vs the dense kernel at the same small-M shape, report HBM GB/s vs peak
rocprofv3 --stats -- python bench_decode_gemv.py    # expect wv_splitk_small / wvSpltK kernel
# achieved bandwidth: bytes_moved / t  -> compare to ~5.3 TB/s HBM3 peak (MI300X)
# parity oracle in ../numerics.md
```

## Alternatives / cross-links
operators/skinny_gemv_decode/backends/triton · operators/skinny_gemv_decode/backends/asm ·
operators/dense_gemm/backends/aiter (parent dispatcher / tune flow) ·
operators/grouped_gemm_moe/backends/aiter (small per-group M) ·
operators/splitk_streamk_gemm/backends/asm (split-K technique) ·
operators/skinny_gemv_decode/overview

## Sources
- On-box: `aiter/tuned_gemm.py` (`is_skinny_default_shape`, `skinny_gemm`),
  `csrc/kernels/custom_kernels.cu` (`wv_splitk_small_fp16_bf16_kernel`, 64-lane K-split) — `ROCm/aiter@b467ce342`.
- AITER (skinny / wvSplitK path, padded_M selection): https://github.com/ROCm/aiter
- vLLM ROCm decode GEMV kernels: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
