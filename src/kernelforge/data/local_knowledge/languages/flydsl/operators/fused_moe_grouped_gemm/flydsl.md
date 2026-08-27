---
title: fused_moe_grouped_gemm on FlyDSL (fp4 a4w4) — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: flydsl
gens: [gfx950]
dtypes: [fp4_e2m1, fp8_e4m3_fnuz, bf16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-09
---

# fused_moe_grouped_gemm × FlyDSL (fp4 a4w4)

## TL;DR
The fp4 **a4w4 block-scaled** MoE path: `flydsl_moe_stage1` (fused gate+up, A·W1) followed by
`flydsl_moe_stage2` (down-proj A·W2), with `a_dtype="fp4"`, `b_dtype="fp4"`, per-1×32 MXFP4 weights/acts and
e8m0 scales. Stage1 can **fuse activation + requantization** in-kernel (silu_and_mul + fp4/fp8 quant + tiled
scale write), so the fp8-input intermediate is produced ready for stage2 without a separate pass. When
`b_dtype=="fp4"` the compile dispatches to `mixed_moe_gemm_2stage`. This is the path aiter FusedMoE uses for
mixed precision (A4W4) and the kind of MoE GEMM AMD reported on Kimi-K2.5. On MI355X the A4W4/MXFP4 path
cuts latency **1.6×** at concurrency 512; MXFP4 GEMMs are ≈**62%** of Llama2-70B e2e, so this is the
dominant low-bit MoE lever. Targets CDNA4 (gfx950) for fp4 MFMA. Gate on `is_flydsl_available()`.

## SOTA implementation
`flydsl_moe_stage1` infers fp4/fp8 fused-quant from `out_dtype`, runs the (optionally split-K) fp4 GEMM,
then for the split-K / gate-up-interleave fused cases calls the cached fused silu+quant module. From
`/sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py`:

```python
_need_fp4 = out_dtype == "fp4"
_need_fp8 = out_dtype == "fp8"
_fuse_any_quant = _need_fp4 or _need_fp8
...
if _gui_sk_fused:                                  # gate-up-interleave split-K + fused quant
    _quant_mode = "fp4" if _need_fp4 else "fp8"
    _silu_fused_k = _get_compiled_silu_fused(inter_dim, topk, _quant_mode, gui_layout=True)
    _run_compiled(_silu_fused_k, (tmp_out.view(-1, inter_dim * 2),
        out.view(-1).view(torch.uint8), out_scale_sorted_flat,
        sorted_token_ids, num_valid_ids, token_num, num_sorted_rows,
        torch.cuda.current_stream()))
```

The mixed kernel builder exposes a `GateMode` enum
(`kernels/mixed_moe_gemm_2stage.py`): `SEPARATED` (default, two B-tile streams), `INTERLEAVE`
(weight rows interleave gate/up; pack_N=2 routes even/odd N subtiles), `MOCK_GATE_ONLY` (single B stream
over full `[0,2*inter_dim)`, requires split-K), `GATE_ONLY` (reserved). Weights/scales follow the CK
preshuffle for fp4 — per `test_flydsl_moe_a4w4.py`: `shuffle_weight(w1_qt, (16,16))`,
`shuffle_weight_a16w4(w2_qt, 16, False)`, with `moe_mxfp4_sort` for the scales.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `compile_mixed_moe_gemm1` (gate+up fp4 a4w4) | `kernels/mixed_moe_gemm_2stage.py::compile_mixed_moe_gemm1` | gfx950; a4w4 (fp4×2), e8m0 scale | no isolated number on-box (test asserts ≥95% close, atol 1.0) | fp4 MoE stage1 |
| `compile_mixed_moe_gemm2` (down-proj fp4) | `kernels/mixed_moe_gemm_2stage.py::compile_mixed_moe_gemm2` | gfx950; a4w4 | no isolated number on-box | fp4 MoE stage2 |
| fused silu_and_mul + fp4/fp8 quant | `kernels/silu_and_mul_fq.py::build_silu_and_mul_fq_module` | gfx950 | no isolated number on-box | split-K / interleave stage1 post-process |
| full fp4 fused-MoE GEMM (vendor) | aiter FusedMoE on FlyDSL | gfx942/950 | A4W4/MXFP4 path **1.6× latency cut @ concurrency 512 (MI355X)**; Kimi-K2.5 MI300X: up to **+162% tput, −69% TPOT, −65% TTFT** (vendor) | mixed-precision MoE block |

## Config space / knobs
From `flydsl_moe_stage1` / `flydsl_moe_stage2` and `get_flydsl_stage{1,2}_kernels` (fp4 branch).

| param | range / typical | effect | default |
|---|---|---|---|
| `a_dtype` / `b_dtype` | fp4 / fp4 (a4w4); a_dtype fp8 also valid for w4 | activation / weight precision | s1 fp8/fp4, w fp4 |
| `out_dtype` | bf16 / fp4 / fp8 | bf16 = plain; fp4/fp8 trigger fused requant + e8m0 sorted scale | bf16 |
| `tile_m`/`tile_n`/`tile_k` | s1 n∈{32,64,128} (fp4 b); s2 m∈{16,32,64,128}, n∈{128,256}, k=256 | tiling | s1 32×256×256, s2 32×128×256 |
| `k_batch` (split-K) | 1,2,4,7,14 (fp4-a stage1) | split-K → atomic partials, fused silu reduces | 1 |
| `gate_mode` | separated / interleave / mock_gate_only | gate/up strategy (`GateMode`) | separated |
| `gui_layout` (silu_fq) | False=separated, True=block-interleaved (block 16) | input layout of fused activation | per gate_mode |
| `quant_mode` (silu_fq) | fp4 / fp8 / none | output of fused activation (MXFP4 / MXFP8 e4m3fn / bf16) | inferred from out_dtype |
| `mode` (s2) | atomic / reduce | accumulation strategy | atomic |
| `xcd_swizzle` | 0 / 4 | XCD remap | 0 |

## Numerics / parity
fp4 (E2M1) MXFP4 with **per-1×32 e8m0 block scales**. The fused `silu_and_mul_fq` kernel computes SiLU via
`llvm.amdgcn.exp2.f32`/`rcp.f32`, takes the per-32-lane abs-max (warp shuffle-xor reduction), derives the
e8m0 scale with a fixed headroom (`_fp_headroom = 2` for fp4, `8` for fp8), then quantizes: fp4 via a
hand-coded `_f32_to_e2m1`, fp8 via `rocdl.cvt_pk_fp8_f32`. Scales are written in a tiled sorted layout
(the 6-index `d0..d5` swizzle). GEMM accumulates fp32. The a4w4 test (`test_flydsl_moe_a4w4.py`) checks
≥95% of elements close at `atol=1.0, rtol=0.05` vs a dequant reference — fp4 is lossy by construction, so
verify at the task-accuracy level, not bit-exact. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Entry points `aiter.ops.flydsl.flydsl_moe_stage1` / `flydsl_moe_stage2` with `a_dtype="fp4"/"fp8"`,
`b_dtype="fp4"`. Caller supplies moe-sorted ids, CK-preshuffled fp4 weights, e8m0 weight scales, and
(for fused requant) receives `(out, out_scale_sorted)` from stage1 to feed stage2. No env-CSV overlay; the
mixed builders are `lru_cache`d on the full shape/dtype tuple. Symbols only import under
`is_flydsl_available()` (flydsl ≥ 0.1.3).

## Pitfalls & anti-patterns
- **CDNA4 (gfx950) for fp4 MFMA** — the fp4 a4w4 path targets gfx950; do not assume it on gfx942.
- Weights and scales must use the exact CK preshuffle (`shuffle_weight`, `shuffle_weight_a16w4`,
  `shuffle_scale_a16w4`) and `moe_mxfp4_sort` layout the kernel decodes — wrong layout = silent garbage.
- `inter_dim` must be divisible by 32 (MXFP4 block) for the fused activation kernel (asserted).
- fp4 is lossy; never gate correctness on bit-exact parity — use the ≥95%-close / task-accuracy bar.
- `mock_gate_only` requires `k_batch>1`; misusing it raises `ValueError`.
- Optional dependency + per-shape JIT cost as in the standard path.

## How to verify
```bash
pytest -sv /sgl-workspace/aiter/aiter/ops/flydsl/test_flydsl_moe_a4w4.py
# test_flydsl_stage1_a4w4 / stage2 build moe-sorted fp4 inputs and check ≥95% close.
```

## Alternatives / cross-links
[[operators/grouped_gemm_moe/backends/flydsl]] (standard fp8/bf16 2-stage) ·
[[operators/fused_moe_grouped_gemm/backends/aiter]] (DB-driven A4W4 dispatch) ·
[[operators/fused_moe_grouped_gemm/backends/ck]] (block-scale stage2) ·
[[operators/act_and_mul_silu_gelu/backends/flydsl]] (the fused silu+quant kernel) ·
[[operators/quant_fp4_mxfp]] (MXFP4 quantization) · [[operators/dense_gemm/backends/flydsl]].
