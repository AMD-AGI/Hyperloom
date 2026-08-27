---
title: gemm_epilogue_fused on FlyDSL — SOTA card
kind: sota_card
operator: gemm_epilogue_fused
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-09
---

# gemm_epilogue_fused × FlyDSL

## TL;DR
FlyDSL ships a reusable, dialect-agnostic **MFMA-16x16 epilogue toolkit** (`mfma_epilog`) that the preshuffle
GEMM kernels use to fuse the accumulator→output mapping: a lightweight **row epilogue** (load scales once,
loop N, store) or an **LDS CShuffle epilogue** (stage tile to LDS, remap threads to (MLane,NLane)=(8,32),
emit half2 global store **or atomic**). The CShuffle epilogue is gated by `use_cshuffle_epilog` in the
preshuffle compiler. Reached only through the FlyDSL preshuffle/scaled GEMM family, gated by
`is_flydsl_available()`.

## SOTA implementation
`mfma_epilog` is one entrypoint dispatching on `use_cshuffle`; callers supply the dialect modules and
per-row callbacks, so the same epilogue drives fp8/int8/fp4 preshuffle kernels. From
`/sgl-workspace/aiter/aiter/ops/flydsl/kernels/mfma_epilogues.py` (`ROCm/aiter@a6bb4993`):

```python
def mfma_epilog(*, use_cshuffle, arith, range_constexpr, m_repeat, lane_div_16, bx_m,
                body_row=None,                       # default row-epilog callback
                vector=None, gpu=None, scf=None,     # cshuffle path
                tile_m=None, tile_n=None, e_vec=2, cshuffle_nlane=32, block_size=256,
                write_row_to_lds=None, store_pair=None, ...):
    if not use_cshuffle:
        if body_row is None:
            raise ValueError("mfma_epilog(use_cshuffle=False) requires `body_row`.")
        return default_epilog(arith=arith, range_constexpr=range_constexpr, m_repeat=m_repeat,
                              lane_div_16=lane_div_16, bx_m=bx_m, body_row=body_row)
    return c_shuffle_epilog(arith=arith, vector=vector, gpu=gpu, scf=scf, ...,
                            tile_m=tile_m, tile_n=tile_n, e_vec=e_vec,
                            cshuffle_nlane=cshuffle_nlane, block_size=block_size, ...)
```

`default_epilog` walks the canonical MFMA row map `row = bx_m + mi*16 + lane_div_16*4 + ii`
(`mi∈[0,m_repeat)`, `ii∈[0,4)`). `c_shuffle_epilog` writes each MFMA row to LDS, barriers, then re-reads as
half2 under a (MLane,NLane) remap; `store_pair` can emit a global store **or an atomic** (for split-K
combine), and a split-LDS mode halves the tile across two wave groups when LDS overflows. The B/scale
preshuffle staging used by the same kernels is in `mfma_preshuffle_pipeline.py`.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FlyDSL MFMA epilogue (row + CShuffle) | `mfma_epilogues.py::mfma_epilog` / `default_epilog` / `c_shuffle_epilog` | gfx942/950; fp8/int8/fp4 + bf16/fp16 out | no isolated flydsl number; folded into aiter GEMM tune (`use_cshuffle_epilog` is a tune knob) | preshuffle/scaled GEMM needing fused dequant-scale + store; split-K atomics |

## Config space / knobs
The epilogue is selected/parameterized through the preshuffle compiler
(`preshuffle_gemm.py::compile_preshuffle_gemm_a8`) and `mfma_epilog` args:

| param | range / source | effect | default |
|---|---|---|---|
| `use_cshuffle_epilog` | `0 \| 1` (compiler arg; appends `_csh` to kernel name) | LDS CShuffle epilogue vs direct row store | 0 |
| `cshuffle_nlane` | `block_size % cshuffle_nlane == 0`; `cshuffle_mlane = block_size/nlane` | thread N-lane remap | 32 |
| `e_vec` | `tile_n % (cshuffle_nlane*e_vec) == 0` | half2/vector store width | 2 |
| `block_size` | divisible by `cshuffle_nlane` | threads per block | 256 |
| `m_repeat` | `tile_m // 16` | MFMA row repeats | derived |
| `frag_elem_type` | f16 / bf16 | LDS load element type | f16 |
| `store_pair` semantics | global store or atomic | direct write vs split-K combine | per kernel |

Validation: `tile_m % cshuffle_mlane == 0`, `tile_n % (cshuffle_nlane*e_vec) == 0`; split-LDS mode
(`lds_out_split`) requires the `scf` module.

## Numerics / parity
Accumulate fp32 in the MFMA fragment; the epilogue applies dequant scales/bias per row before the
bf16/fp16 store. For the dense hgemm path, bias is fused only when `bias.dtype == inp.dtype` and the output
dtype matches (`flydsl_gemm` in `tuned_gemm.py`), else added in a follow-up cast — the hgemm path itself
takes no scales. The scaled epilogue lives in the preshuffle family.

## Integration (rebind seam)
Not a standalone op: the epilogue is compiled into the preshuffle/scaled GEMM kernel and chosen by
`use_cshuffle_epilog`, which is part of the a8w8 bpreshuffle tune
(`gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`, the `cshuffle` field of `kernelInstance`). Reached via
the FlyDSL preshuffle driver `flydsl_preshuffle_gemm_a8`, all gated by `is_flydsl_available()`. The dense
bias-fusion seam is `aiter.tuned_gemm.flydsl_gemm` (CSV `libtype=flydsl`).

## Pitfalls & anti-patterns
- CShuffle has hard divisibility constraints (`tile_m % cshuffle_mlane`, `tile_n % (cshuffle_nlane*e_vec)`)
  — an arbitrary tile won't compile with `use_cshuffle_epilog=1`.
- The dense hgemm path silently does **non-fused** bias (cast-then-add) when dtypes mismatch — don't assume
  the epilogue fused it.
- `use_cshuffle=False` requires a `body_row` callback; `use_cshuffle=True` requires `write_row_to_lds` +
  `store_pair` — partial wiring raises.
- Split-LDS epilogue needs `scf`; omitting it raises.

## How to verify
```bash
python -c "from aiter.ops.flydsl.utils import is_flydsl_available; print(is_flydsl_available())"
pytest -q aiter/ops/flydsl/test_flydsl_moe_a4w4.py   # exercises preshuffle GEMM + epilogue end-to-end
```

## Alternatives / cross-links
[[operators/gemm_epilogue_fused/backends/ck]] (CShuffle CDEElementwise, richest fusion) ·
[[operators/gemm_epilogue_fused/backends/aiter]] (live bias/scale epilogue) ·
[[operators/scaled_quant_gemm/backends/flydsl]] (the kernels that use this epilogue) ·
[[operators/layout_shuffle/backends/flydsl]] (B/scale preshuffle staging).
