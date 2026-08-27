---
title: aiter FlyDSL path — split-K HGEMM and A4W4 MoE (with CK fallback)
kind: backend
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp4_e2m1]
regimes: [prefill, decode]
status: experimental
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/flydsl/gemm_kernels.py
  - ROCm/aiter@b467ce342:aiter/ops/flydsl/moe_kernels.py
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:csrc/gemm_a16w16/gemm_a16w16_tune.py
---

# aiter FlyDSL path

## TL;DR
**FlyDSL is one of aiter's GEMM/MoE libtypes** — a Python-authored, name-encoded kernel family for
**split-K HGEMM (bf16/fp16)** and **A4W4 (FP4 weight) MoE**. It is selected only when (a) the tuned DB row
says `libtype="flydsl"` *and* (b) the `flydsl` package is importable *and* (c) a catalog kernel matches the
shape. If any condition fails, aiter silently drops the FlyDSL config and falls through — for A4W4 MoE the
fallback is **CK**. Treat FlyDSL as an opt-in, gfx950-leaning accelerant, not a default.

## Concepts

### Availability gate
`aiter/ops/flydsl/utils.py:is_flydsl_available()` is just
`importlib.util.find_spec("flydsl") is not None`. FlyDSL is a **separate package**, not vendored — if it's
not installed, every FlyDSL path is inert. The GEMM dispatch double-checks this: in
`tuned_gemm.get_GEMM_A16W16_config`, a `libtype=="flydsl"` row is kept only if `is_flydsl_available()` and
`get_flydsl_splitk_hgemm_kernel_params(kernelName)` resolves; otherwise `config=None` and the lookup
continues to the next `padded_M` / default.

### Name-encoded kernels
FlyDSL kernels are identified by a structured name parsed with a regex
(`gemm_kernels.py`), e.g. tile/split-K/warps/preshuffle are encoded in the string:
`flydsl_gemm{stage}_a{dtype}_w{dtype}_{out}_t{TM}x{TN}x{TK}_split_k{SK}_block_m_warp{..}_block_n_warp{..}_async_copy{..}_b_to_lds{..}_b_preshuffle{..}[_wpe{N}]`.
The DB stores the `kernelName`; at call time `get_flydsl_splitk_hgemm_kernel_params(name)` decodes it back
into the launch params.

### HGEMM launch (`flydsl_gemm` in tuned_gemm.py → `flydsl_hgemm`)
Params passed through from the decoded config:
`tile_m, tile_n, tile_k, split_k, block_m_warps, block_n_warps, n_tile_repeat, persistent_n_tiles,
waves_per_eu, b_to_lds_unroll, stages (default 2), async_copy, b_to_lds, b_preshuffle, c_to_lds`.
Constraint: `b_to_lds=False` is required when `b_preshuffle=True`. **No scaling** — `flydsl_hgemm`
asserts `scale_a/scale_b/scale_c is None`; bias is fused only when dtype/otype align, else added after.

### A4W4 MoE + CK fallback
`fused_moe` routes 4-bit-weight (A4W4 / FP4) MoE to FlyDSL when available; the FlyDSL MoE kernels
(`moe_kernels.py`) are two-stage (`flydsl_moe1_*` stage-1 gate/up, `flydsl_moe2_*` stage-2 down) with
their own name encoding and a `sort_block_m` knob. **When FlyDSL is absent, A4W4 fused MoE falls back to
CK** grouped-GEMM instances — the same coverage caveats apply (an uncovered shape raises the CK
"does not support this GEMM problem" error). See [fmoe.md](fmoe.md).

## The levers
- **Tile/split-K**: `tile_m/n/k`, `split_k` — `gemm_kernels.py` enumerates split-K options subject to
  `k % split_k == 0` and `(k//split_k) % tile_k == 0`; `tile_m` options are capped relative to M.
- **Occupancy/pipeline**: `waves_per_eu`, `stages`, `async_copy`, `b_to_lds`.
- **Layout**: `b_preshuffle` (preshuffled weights; mutually exclusive with `b_to_lds`).
- These are tuned by the multi-backend tuner `csrc/gemm_a16w16/gemm_a16w16_tune.py --libtype flydsl` (or
  `all`); the FlyDSL catalog for a shape comes from `get_flydsl_splitk_hgemm_kernels(in_dt, out_dt, m, n, k)`.

## Numerics / parity
bf16/fp16 HGEMM: same math, tuner-gated `err_ratio < 0.05`. A4W4 is block-scaled FP4 — accuracy depends on
the quant recipe; validate against the model's reference, not just GEMM tolerance.

## Pitfalls
- FlyDSL not installed → all FlyDSL DB rows silently no-op; a DB shipped from a FlyDSL-enabled box will
  fall back to CK/hipBLASLt on a box without it. Don't assume portability.
- A4W4 → CK fallback inherits CK instance-coverage gaps; confirm a CK A4W4 instance covers your
  (M,N,K, layout) or pad to a covered shape.
- No scaling support in `flydsl_hgemm` — scaled GEMM never uses FlyDSL.

## How to verify
With `AITER_LOG_TUNED_CONFIG=1`, a FlyDSL hit logs `libtype is flydsl, kernel name is <encoded name>`.
If you expect FlyDSL but see `hipblaslt`/`torch`, check `python -c "import flydsl"` and that a catalog
kernel exists for the shape.

## Alternatives / cross-links
[tuning_db.md](../../../overall/tuning_db.md) (dispatch + tuner) · [fmoe.md](fmoe.md) (A4W4 MoE) ·
`languages/flydsl/` (authoring FlyDSL) ·
`operators/dense_gemm/backends/flydsl.md`.

## Sources
- On-box: `ROCm/aiter@b467ce342`:
  `aiter/ops/flydsl/gemm_kernels.py` (name regex, tile/split-K options, `flydsl_hgemm`,
  `get_flydsl_splitk_hgemm_kernel_params`), `aiter/ops/flydsl/moe_kernels.py` (two-stage A4W4 MoE),
  `aiter/ops/flydsl/utils.py` (`is_flydsl_available`), `aiter/tuned_gemm.py` (flydsl gate + dispatch),
  `csrc/gemm_a16w16/gemm_a16w16_tune.py` (`--libtype flydsl`).
- FlyDSL A4W4 → CK fallback for fused MoE: https://github.com/ROCm/aiter (README, fused MoE backend selection).
