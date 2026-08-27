---
title: FlyDSL — kernel families in aiter
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1, mxfp4]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
---

# FlyDSL — kernel families

The shipped FlyDSL kernels in `ROCm/aiter@/sgl-workspace/aiter:aiter/ops/flydsl/`. Each family has a
high-level API (the `*_kernels.py` wrapper) and a DSL body under `kernels/`.

## 1. HGEMM (dense bf16/fp16) — `splitk_hgemm.py`
- API: `flydsl_hgemm` (gemm_kernels.py); body `kernels/splitk_hgemm.py` (`compile_hgemm_kernel`).
- 2-stage LDS pipeline, MFMA-16, XOR-swizzled LDS, optional split-K (global-semaphore reduce),
  `b_preshuffle` or `b_to_lds`. fp32 accumulate.
- Maps to `act_and_mul`-free dense GEMM (operators: `dense_gemm`, `splitk_streamk_gemm`,
  `skinny_gemv_decode` via split-K). Default tile 128×128×64, warps 1×4.

## 2. Small-M HGEMM (decode) — `small_m_hgemm.py`
- `kernel_family="small_m"`; **bf16 only**, `tile_m=16`, `block_m_warps=1`, `b_preshuffle=False`.
- Extra knobs: `n_tile_repeat`, `persistent_n_tiles`, `waves_per_eu`, `b_to_lds_unroll`.
- Target: decode GEMV-like shapes (M=1..16) where the dense HGEMM tiling wastes the matrix core.
- `iter_small_m_registry_configs(dtype, out, m, n, k)` supplies tuned small-M configs, merged into
  the kernel registry when `(m,n,k)` are known. Operator: `skinny_gemv_decode`.

## 3. Preshuffle GEMM A8 (scaled fp8/int8) — `preshuffle_gemm.py`, `mfma_preshuffle_pipeline.py`
- API: `flydsl_preshuffle_gemm_a8(XQ, WQ, x_scale, w_scale, Out, tile_*, lds_stage,
  use_cshuffle_epilog, use_async_copy, waves_per_eu)`.
- W8A8 / int8 GEMM with per-row/col scales; output bf16/fp16. `use_cshuffle_epilog` = keep MFMA
  layout through epilogue. Operators: `scaled_quant_gemm`, `gemm_epilogue_fused`.

## 4. 2-stage fused MoE — `moe_gemm_2stage.py`, `mixed_moe_gemm_2stage.py`
- API: `flydsl_moe_stage1` / `flydsl_moe_stage2` (moe_kernels.py).
- Grouped GEMM for fused MoE: stage1 (up/gate proj + act) and stage2 (down proj), with token sorting
  by expert (`sort_block_m`). `mixed_moe_gemm_2stage.py` = mixed-precision (W4A16, fp4/fp8 output).
- Kernel-name suffixes: `_fp4` / `_fp8` (output dtype + `a_scale_one`) / `_sbm{N}` (sort block M).
- Env: `FLYDSL_W4A16_HYBRID=w2_bf16` → stage1 W4A16, stage2 BF16 (accuracy/perf trade).
- **This is the Kimi-K2.5 win**: the fused MoE was 87.8% (concurrency 2) / 89.7% (concurrency 40) of
  GPU time; rewriting it in FlyDSL drove the e2e numbers below. Operators: `fused_moe_grouped_gemm`,
  `moe_dispatch_combine`, `grouped_gemm_moe`.

  Vendor-reported (AMD blog, MI300X gfx942, ROCm 7.2.0, PyTorch 2.9.1, aiter
  0.1.5.post5.dev409+g6b157bbb2, 2026-03-24), Kimi-K2.5 e2e:
  | metric | baseline | FlyDSL | Δ |
  |---|---|---|---|
  | throughput @ concurrency 40 | 135.39 tok/s | 355.35 tok/s | **+162.4%** |
  | TPOT @ concurrency 40 | 230.37 ms | 70.86 ms | **−69.2%** |
  | TTFT @ concurrency 2 | 2918 ms | 1014 ms | **−65.3%** |
  | throughput @ concurrency 2 | 45.04 tok/s | 66.24 tok/s | +47.1% |

  Fused-MoE kernel time (vendor): bf16 0.13/0.60/2.25/8.68 ms; W4A16 0.11/0.69/2.42/9.77 ms for
  512/2048/4096/16384 tokens. (CK reported gpu_fault/unsupported on large W4A16.)

## 5. Linear attention / GDR decode — `linear_attention_kernels.py`, `kernels/gdr_decode.py`
- API: `flydsl_gdr_decode` (gated-delta-rule decode). Tuned configs in `gdr_decode_tuned.jsonl`
  (`NUM_BLOCKS_PER_V_DIM`, `NUM_WARPS`, `WARP_THREADS_K`).
- Target: gated linear/delta attention decode (operator: `linear_attention_gated_delta`).

## 6. Fused activation+quant — `kernels/silu_and_mul_fq.py`, `kernels/reduce.py`
- `silu_and_mul_fq`: fused SiLU·mul + fp-quant (operator: `act_and_mul_silu_gelu`,
  `fused_norm_quant`). `reduce.py`: reduction primitives used by split-K / MoE.

## Family → operator map (for the SOTA matrix)
| FlyDSL family | operators it serves |
|---|---|
| HGEMM / split-K | `dense_gemm`, `splitk_streamk_gemm` |
| small-M HGEMM | `skinny_gemv_decode` |
| preshuffle A8 | `scaled_quant_gemm`, `gemm_epilogue_fused` |
| 2-stage MoE | `fused_moe_grouped_gemm`, `grouped_gemm_moe`, `moe_dispatch_combine` |
| GDR decode | `linear_attention_gated_delta` |
| silu_and_mul_fq | `act_and_mul_silu_gelu`, `fused_norm_quant` |
