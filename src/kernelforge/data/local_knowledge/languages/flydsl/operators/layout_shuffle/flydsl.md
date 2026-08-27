---
title: layout_shuffle on FlyDSL — SOTA card
kind: sota_card
operator: layout_shuffle
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-09
---

# layout_shuffle × FlyDSL

## TL;DR
FlyDSL's GEMM kernels consume **pre-shuffled (bpreshuffle) weight layouts** so MFMA loads are coalesced and
swizzle-free. Two pieces: (1) on the host, `flydsl_hgemm`/preshuffle drivers expect B already shuffled to
the FlyDSL layout `(16*pack_n, 16)` via aiter `shuffle_weight`; (2) inside the kernel, `layout_utils`
parses fly layout strings and emits **shift/mask instead of div/rem** for power-of-2 strides, and
`mfma_preshuffle_pipeline` builds the matching B and MXFP-scale layouts. It is a FLIR/ROCDL MLIR-Python DSL
(CuTe-inspired) — layouts are first-class `(shape):(stride)` objects.

## SOTA implementation
The host-side B-shuffle contract lives in `gemm_kernels.py`; the FlyDSL layout is `(16*pack_n, 16)`. From
`/sgl-workspace/aiter/aiter/ops/flydsl/gemm_kernels.py` (`ROCm/aiter@a6bb4993`):

```python
def _get_flydsl_shuffle_layout(pack_n: int) -> tuple[int, int]:
    return (16 * pack_n, 16)
...
if b_preshuffle and not getattr(b, "is_shuffled", False):
    if auto_shuffle_b:
        b = shuffle_weight(b, layout=_get_flydsl_shuffle_layout(pack_n))
    else:
        raise ValueError("`b_preshuffle=True` expects `b` to be pre-shuffled. "
                         f"Use `shuffle_weight(b, layout={_get_flydsl_shuffle_layout(pack_n)})` first ...")
```

The device-side index math is optimized in `layout_utils.py`: power-of-2 strides/shapes lower to `shrui`/
`andi` (1 VALU cycle) instead of `divui`/`remui` (10–15 cycles on CDNA):

```python
def _div_pow2(val, divisor):
    shift = _math.log2(divisor)
    assert shift == int(shift), f"{divisor} is not a power of 2"
    return arith.shrui(val, arith.index(int(shift)))   # vs arith.divui
```

`mfma_preshuffle_pipeline.make_preshuffle_b_layout` builds the B layout matching aiter/CK preshuffle
(`kpack_bytes` 8 or 16; N-major `(0,1,3,4,2,5)` vs K-major `(0,3,1,4,2,5)` permutation), and
`make_preshuffle_scale_layout` builds the MXFP block-scale layout `(c_mn1, c_k1, 4, 16)` with
`scale_block_size=32` and a `swizzle_xor16` K-swizzle.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FlyDSL bpreshuffle B layout + pow2 index lowering | `gemm_kernels.py::_get_flydsl_shuffle_layout`, `layout_utils.py`, `mfma_preshuffle_pipeline.py::make_preshuffle_b_layout` | gfx942/950; bf16/fp16/fp8/int8/fp4 | no isolated flydsl number; offline weight prep, folded into GEMM tune | preparing W once for repeated preshuffle/scaled GEMM |

## Config space / knobs
| param | range / source | effect | default |
|---|---|---|---|
| host shuffle `layout` | `(16*pack_n, 16)` (`pack_n` from kernel) | aiter `shuffle_weight` target layout | `(16,16)` (pack_n=1) |
| `b_preshuffle` | bool | kernel consumes preshuffled B (mutually exclusive with `b_to_lds`) | True |
| `auto_shuffle_b` | bool | shuffle B on the fly if not pre-shuffled | False |
| `kpack_bytes` | `8 \| 16` | K-pack granularity of B layout | 16 |
| `k_major` | bool | block-level K-major vs N-major permutation | False (N-major) |
| `scale_block_size` | 32 (MXFP) | scale layout `(c_mn1,c_k1,4,16)` block | 32 |
| `swizzle_xor16` | XOR row-swizzle at 16B K granularity | LDS-bank-conflict-free B reads | — |

## Numerics / parity
Layout shuffle is value-preserving (a permutation of weight storage) — no numeric change to the GEMM result;
it only changes memory order so MFMA loads are contiguous. The MXFP scale layout pairs the per-32 block
scales with their B tiles. `is_shuffled` is a tensor attribute the kernel checks to avoid double-shuffling.

## Integration (rebind seam)
The shuffled-B contract is implicit in the FlyDSL GEMM rebind seam: a tuned `libtype=flydsl` row with
`b_preshuffle=True` requires B pre-shuffled to `(16*pack_n,16)` (or `auto_shuffle_b=True`). The exported
entrypoint `flydsl_preshuffle_gemm_a8` (from `aiter.ops.flydsl.__init__`, available only when
`is_flydsl_available()`) consumes already-quantized + preshuffled W. aiter's `bpreshuffle` GEMM key (see
[[operators/layout_shuffle/backends/aiter]]) ties the same CSV-row dimension.

## Pitfalls & anti-patterns
- `b_preshuffle=True` with un-shuffled B raises unless `auto_shuffle_b=True` — pre-shuffle weights once at
  load time, not per call.
- `b_to_lds=True` and `b_preshuffle=True` together are rejected (`flydsl_kernel_name`).
- `kpack_bytes` must be 8 or 16; `elem_bytes` must be 1 or 2 in `make_preshuffle_b_layout` — other widths
  raise.
- MXFP scale layout requires `elem_bytes == mn_pack*k_pack` — mismatched scale packing raises.
- Use the FlyDSL layout `(16*pack_n,16)`, not an arbitrary aiter shuffle layout, or the kernel reads garbage.

## How to verify
```bash
python -c "from aiter.ops.shuffle import shuffle_weight; import torch; \
b=torch.randn(256,512,dtype=torch.bfloat16,device='cuda'); \
bs=shuffle_weight(b, layout=(16,16)); print(getattr(bs,'is_shuffled',None), bs.shape)"
pytest -q aiter/ops/flydsl/test_flydsl_splitk_hgemm.py -k b_to_lds   # exercises shuffle vs non-shuffle B
```

## Alternatives / cross-links
[[operators/layout_shuffle/backends/aiter]] (`shuffle_weight`, bpreshuffle key) ·
[[operators/scaled_quant_gemm/backends/flydsl]] (consumes preshuffled W + MXFP scales) ·
[[operators/gemm_epilogue_fused/backends/flydsl]] (same preshuffle pipeline) ·
[[operators/dense_gemm/backends/flydsl]] (b_preshuffle hgemm).
