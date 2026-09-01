---
title: tl.dot_scaled on gfx950 — block-scaled MXFP8 dot API & codegen behavior
kind: api_reference
gens: [gfx950]
dtypes: [fp8_e4m3, mxfp8, mxfp4]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html
---

# `tl.dot_scaled` on gfx950 (MI350 / MI355X) — Know-Hows

Source: MX-FP8 grouped GEMM work for Primus-Turbo `feat/mxfp8-grouped-gemm-e8m0`.

## What it does (at the instruction level)

`tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, fast_math=True)` on
gfx950 emits **native `v_mfma_scale_f32_32x32x64_f8f6f4`** — verify with
`AMDGCN_ENABLE_DUMP=1` on any scaled kernel build. No bf16 emulation.

Scale format the instruction expects:
- `uint8` e8m0 (biased exponent, bias 127; 0x00 = zero, 0xFF = NaN)
- Groups of 32 K-elements share one scale byte
- Layout for the B operand: `[N, K // 32]` (N-first, NOT K-first). Do NOT
  transpose B's scale from this layout — the MFMA instruction bakes in the
  access pattern.

## The codegen tax (24% below fp8 tensorwise ceiling)

`tl.dot_scaled` wall-clock is ~24% slower than a matching plain `tl.dot`
on e4m3 operands at the SAME MFMA count. Confirmed via AMDGCN diff:

- `+24 ds_read_u8` per MFMA cluster (scale re-distribution to lanes)
- `+14 s_waitcnt lgkmcnt(N)` fences per cluster (scale-LDS serialisation)
- `+5 scratch_store_dword` (spill pressure)
- `+12 SGPR`, same VGPR count

Identity-scale probe (all scales = 0x7F = 2^0) proves the overhead is
**NOT data-dependent** — wall-clock identical to random-scale inputs.

Downstream fixes empirically refuted:
- prepareOperands `kBase` packing (MFMA.cpp:441-454) — VALU 1.24→1.25×
  (LLVM already folds insert_element).
- LinearEncodingAttr scale vecSize routing (LowerLoops.cpp:228-240) —
  source inspection: already calls `composeSharedLayoutForOperand` with
  proper vecSize.

Fix path: **upstream Triton scale-LDS waitcnt density fix** (MFMA.cpp:750-815)
or a custom HIP/CK kernel bypassing `tl.dot_scaled` entirely.

## Knob sweep verdicts (13 configs tested)

No config beats baseline (BLK=256x256x128, GM=4, num_stages=2, num_warps=8,
nonkdim=32) on the gpt_oss_20B shape. Key results:

- `waves_per_eu=2` ties baseline (noise).
- `num_stages=1` matches `num_stages=2` → pipeline isn't the bottleneck.
- `BLK_K=64` with `num_stages=3/4` regresses — loop overhead > pipeline gain.
- `BLK_M=128` regresses ~26% — tile-level throughput loss.
- `waves_per_eu=3` catastrophic (0.19× bf16) — occupancy overprovisioned.
- `matrix_instr_nonkdim=16` regresses 4-14% vs `=32`.

Wgrad variable-K sweep: baseline also optimal. Memory-bound kernels (stall
ratio ~0.91) are insensitive to tile shape changes.

## Triton 3.6 vs 3.7

3.7's `opSel` / `prepareOperands` kBase-packing fix drops VALU ratio from
1.42× → 1.24× but wall-clock is ~1% (scale-LDS waitcnt unchanged). Requires
source build from `release/3.7.x` (not on pypi).

## MX-FP8 precision

FP8 E4M3 quant-noise floor is **~28 dB SNR** (3 mantissa bits = ~4% RMS
relative error). Same floor as tensorwise and rowwise fp8.

Do NOT expect SNR >> 30 dB on random inputs — it's physically impossible
with fp8 e4m3. Industry standard correctness gate is **25 dB for e4m3**,
**20 dB for e5m2** (e.g. Transformer Engine, NVIDIA MX kernels, AMD
ROCm Primus-Turbo tests).

## Shape-dependent surprises on gpt_oss_20B (M=65536, K=2880, N=5760, G=32)

- `dgrad` is FASTER than `fwd` per-kernel (1.62× vs 1.43× bf16) because
  its output K=2880 < fwd's N=5760 — fewer output tiles at same BLK config.
- Quant dominates training step: pure kernels = 4.62 ms but full autograd
  step = 8.92 ms. Quant/save-for-backward = ~half the step.
- Weight prequant (hoist B quant once per optimiser step vs per forward)
  is the largest single lever in the full-step picture: 1.22× → 1.53× bf16
  at k=8 gradient accumulation.

## Files / references

- Forward kernel: `primus_turbo/triton/grouped_gemm/grouped_gemm_mxfp8_kernel.py`
- Variable-K wgrad: `primus_turbo/triton/grouped_gemm/grouped_gemm_mxfp8_variable_k_kernel.py`
- Quant kernels: `primus_turbo/triton/quantization/mxfp8_quant_kernels.py`
- Autograd Function: `primus_turbo/pytorch/ops/grouped_gemm_fp8.py` (class `FP8GroupedGemmMXFunc`)
- Bench: `benchmark/ops/mxfp8/bench_grouped_gemm_mxfp8.py`
- Upstream Triton issue draft: `memory/reference_triton_mxfp8_upstream_issue_v2.md`
