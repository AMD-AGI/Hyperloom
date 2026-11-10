---
title: FlyDSL — what already ships inside aiter, so you don't rewrite it
kind: language
lever: flydsl_kernel_library
gens: [gfx950]
updated: 2026-08-28
---

# The FlyDSL kernels aiter already ships

**This is the using-FlyDSL side, not the writing-FlyDSL side.** Everything below is already built and
sitting in `aiter/ops/flydsl/`. Each family follows the same two-file shape: a Python wrapper in
`*_kernels.py`, and the DSL body under `kernels/`.

## Route here when
- You are about to author a FlyDSL kernel and should first check whether one exists.
- You know which family you want and need to know what it takes and what it costs.

**Go to `flydsl_authoring_method.md`** if you have already established that nothing here fits and you
are writing a fresh `@flyc.kernel`.

## The families

### Dense bf16/fp16 — HGEMM
Entry `flydsl_hgemm` in `gemm_kernels.py`; body `kernels/splitk_hgemm.py` (`compile_hgemm_kernel`).

What it is built from: a 2-stage LDS pipeline, the 16-wide MFMA atom, XOR-swizzled LDS, fp32
accumulation, and B supplied either pre-shuffled or staged through LDS. Split-K is available and its
reduction goes through a **global semaphore rather than atomics**, so results are reproducible run to
run. Defaults land at a 128×128×64 tile with a 1×4 warp grid.

### Decode-shaped — small-M HGEMM
Selected with `kernel_family="small_m"`. Narrower than the dense path in every sense: **bf16 only**,
`tile_m` pinned to 16, `block_m_warps=1`, and `b_preshuffle` off.

It exists because dense HGEMM tiling is wasteful when M is 1–16 — most of each MFMA tile is padding.
Four extra arguments come with it: `n_tile_repeat`, `persistent_n_tiles`, `waves_per_eu`,
`b_to_lds_unroll`. When `(m, n, k)` are known ahead of time,
`iter_small_m_registry_configs(dtype, out, m, n, k)` supplies tuned configurations that get merged into
the registry.

### Scaled fp8/int8 — preshuffle GEMM A8
```python
flydsl_preshuffle_gemm_a8(XQ, WQ, x_scale, w_scale, Out,
                          tile_*, lds_stage, use_cshuffle_epilog,
                          use_async_copy, waves_per_eu)
```

W8A8 and int8 GEMM with per-row and per-column scales, producing bf16 or fp16. `use_cshuffle_epilog`
holds the result in MFMA layout all the way through the epilogue — the same idea as Triton's
`OPTIMIZE_EPILOGUE`. This is also where scaled GEMM has to go: `flydsl_hgemm` asserts its scale
arguments are `None`.

### Fused MoE, two stages — the family that justifies the whole path
`flydsl_moe_stage1` and `flydsl_moe_stage2` in `moe_kernels.py`; bodies in `moe_gemm_2stage.py` and
`mixed_moe_gemm_2stage.py` (the latter covers mixed-precision W4A16 with fp4 or fp8 output).

Structurally it is a grouped GEMM: stage 1 does the up and gate projections plus the activation, stage
2 does the down projection, and tokens are sorted by expert first (`sort_block_m`). The kernel names
encode their variant — `_fp4`, `_fp8` (output dtype plus `a_scale_one`), `_sbm{N}` for the sort block
size. Setting `FLYDSL_W4A16_HYBRID=w2_bf16` runs stage 1 as W4A16 and stage 2 as bf16, trading a little
speed for accuracy.

**Why this one matters.** On Kimi-K2.5 the fused MoE was not one hot spot among several — it was
**87.8% of GPU time at concurrency 2 and 89.7% at concurrency 40**. Rewriting it in FlyDSL moved the
whole model:

> Vendor-reported: AMD blog, **MI300X / gfx942**, ROCm 7.2.0, PyTorch 2.9.1,
> aiter 0.1.5.post5.dev409+g6b157bbb2, 2026-03-24.

| Metric | Before | After | Change |
|---|---|---|---|
| throughput @ concurrency 40 | 135.39 tok/s | 355.35 tok/s | **+162.4%** |
| TPOT @ concurrency 40 | 230.37 ms | 70.86 ms | **−69.2%** |
| TTFT @ concurrency 2 | 2918 ms | 1014 ms | **−65.3%** |
| throughput @ concurrency 2 | 45.04 tok/s | 66.24 tok/s | +47.1% |

Kernel-level times, same source, for 512 / 2048 / 4096 / 16384 tokens: bf16 at 0.13 / 0.60 / 2.25 /
8.68 ms, W4A16 at 0.11 / 0.69 / 2.42 / 9.77 ms. CK either faulted or reported the shape unsupported on
the large W4A16 cases.

**Take the method, not the number.** A 162% gain is not a property of FlyDSL — it is what happens when
you rewrite the op that owns nine tenths of the runtime. Applied to a 5% op the same effort yields at
most 5%. Profile before you pick a target; that is the reusable part of this story.

### Linear attention — GDR decode
`flydsl_gdr_decode` (`linear_attention_kernels.py`, body `kernels/gdr_decode.py`) implements
gated-delta-rule decode. Tuned configurations live in `gdr_decode_tuned.jsonl`, keyed on
`NUM_BLOCKS_PER_V_DIM`, `NUM_WARPS`, and `WARP_THREADS_K`.

### Activation and reduction primitives
`kernels/silu_and_mul_fq.py` fuses SiLU·mul with the following quantization.
`kernels/reduce.py` holds the reduction primitives that split-K and the MoE stages build on.

## Which family serves which operator
| Family | Operators |
|---|---|
| HGEMM / split-K | dense GEMM; split-K and stream-K GEMM |
| small-M HGEMM | skinny GEMV decode |
| preshuffle A8 | scaled-quant GEMM; fused GEMM epilogue |
| 2-stage MoE | fused MoE grouped GEMM; grouped GEMM MoE; MoE dispatch/combine |
| GDR decode | gated-delta linear attention |
| `silu_and_mul_fq` | activation+mul; fused norm+quant |

## Verify
| Check | Why it matters |
|---|---|
| The family actually dispatched | A FlyDSL kernel that exists but is not selected changes nothing at all. The selection gates are in `../../../../../framework/aiter/skills/optimize/aiter_levers/aiter_flydsl_libtype.md` — and one of them is whether the FlyDSL package is even installed. |
| You re-measured on your hardware | Every vendor number above is **MI300X / gfx942**. Expect the ordering to survive the move to gfx950 and the magnitudes not to. |

## Related
`flydsl_knob_space.md` (the arguments these families accept) ·
`flydsl_authoring_method.md` (if none of them fit) ·
`../../../../../framework/aiter/skills/optimize/aiter_levers/aiter_flydsl_libtype.md` (whether aiter will pick FlyDSL at all)
