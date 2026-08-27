---
title: autotuning methodology (triton / hipBLASLt / aiter)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, both]
updated: 2026-06-05
---

# autotuning methodology

## TL;DR
Three tuner tiers exist, but **only one engages the live sglang/vllm GEMM path: aiter's per-shape DB**.
The recipe is: capture real shapes with **`AITER_TUNE_GEMM`**, race candidates with **gradlib
`gemm_tuner.py`** gated at **`err_ratio < 0.05`**, deploy via **`AITER_CONFIG_GEMM_BF16=<tuned.csv>`**,
and verify engagement in the log. The lookup key is a **9-tuple** — get any field (esp. `bias`) wrong
and you hit 0% engagement. Validated **+2.23% e2e** on Qwen3.5-27B / sglang 0.5.11 / aiter, 2026-06-08.
triton `@autotune` and `hipblaslt-bench` are useful for *authoring* kernels but don't reach the live
dispatch unless rebound through aiter. See `[[operators/dense_gemm/tuning.md]]` and
`[[operators/dense_gemm/backends/aiter.md]]`.

## The three tiers
| tier | tool | reaches live serving path? |
|---|---|---|
| author-time kernel | triton `@triton.autotune` over configs | only if the kernel is the live dispatch |
| library offline | `hipblaslt-bench` / TensileLite; PyTorch `TunableOp` (`HIPBLASLT_TUNING_FILE`) | **no** for sglang/aiter — aiter bypasses these hooks |
| **live dispatch** | **aiter gradlib per-shape DB** | **yes** — this is the lever |

## aiter live-tuning recipe (the lever)
1. **Capture** real shapes on a *warm* server: `AITER_TUNE_GEMM=1` (or replace `F.linear`→`tgemm.mm`);
   shapes append to `aiter/configs/bf16_untuned_gemm.csv`. **Bias/scale/dtype must match live calls.**
2. **Race**: `python3 gradlib/gradlib/gemm_tuner.py --input_file …/bf16_untuned_gemm.csv
   --tuned_file …/bf16_tuned_gemm.csv --indtype bf16 --mp <ngpus>`; gate accepted kernels at
   **`err_ratio < 0.05`** (max tolerable numeric error vs reference). If OOM, set
   `CACHE_INVALIDATE_BUFFERS` to a small prime (e.g. 11/7/3/1).
3. **Deploy**: `AITER_CONFIG_GEMM_BF16=<tuned.csv>` and restart; **verify** with
   `grep -c 'is tuned on cu_num'` in the server log (non-zero ⇒ engaged).

### The 9-tuple lookup key
aiter selects by `(cu_num, M, N, K, bias, dtype_in, dtype_out, scaleAB, bpreshuffle)`. The tuned CSV
columns mirror this: `cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle, libtype, solidx,
splitK, soltimes, kernelName, tflops, bw`. `libtype ∈ {hipblaslt, rocblas, asm, triton}`; `splitK` is
valid when `libtype==asm`. **A single mismatched field (commonly `bias=true` tuned vs `bias=false`
live) ⇒ 100% lookup miss, 0 engagement** — the most common silent failure.

## Search-space pruning
- **Bucket M**: live M varies per batch; round/bucket to a small set instead of tuning every M
  (racing ~1365 hipBLASLt solutions/shape is slow and can fork-storm the host).
- **Constrain knobs** to MI300X-good defaults before searching: `mfma_16x16` (`matrix_instr_nonkdim=16`),
  8-multiple tiles, ≥1024 WGs, `OPTIMIZE_EPILOGUE=1` (`[[optimization/mfma_scheduling.md]]`,
  `[[optimization/xcd_l2_locality.md]]`). For decode, prioritize small `BLOCK_M` + SPLIT_K
  (`[[operators/splitk_streamk_gemm/overview.md]]`).
- **Parallelize** with `--mp <ngpus>` and prune obviously-bad configs (spilling, sub-1024 WG) early via
  the ISA dump (`[[optimization/occupancy_and_registers.md]]`).

## Caching / reuse
- Commit the tuned CSV per (model, dtype, hardware) and load via the env var — do **not** edit
  site-packages. Re-tune when shapes, dtype, ROCm/aiter version, or GPU SKU (304 vs 256 CUs) change.
- triton: persist the `@autotune` cache keyed on shape; the first call per shape pays the search.

## Pitfalls
- Tuning synthetic `bias=true` shapes when live is `bias=false` ⇒ 0 engagement (verify with `grep`).
- Relying on `TunableOp` / `HIPBLASLT_TUNING_FILE` for an aiter-dispatched server — aiter bypasses it.
- Reporting theoretical peak as the bar; the real target is the best *tuned library* kernel
  (`[[optimization/roofline_and_bottlenecks.md]]`).
- Forgetting to re-tune after a ROCm/aiter bump or SKU change.

## Verify
- Engagement: `grep -c 'is tuned on cu_num' server.log` > 0.
- Gate: accepted kernels satisfy `err_ratio < 0.05`; e2e delta beyond noise band with non-overlapping
  repeats (`[[operators/dense_gemm/tuning.md]]`).
- Counters: tuned config should show higher MFMA busy / closer-to-roofline TFLOP/s (`[[profiling/]]`).
