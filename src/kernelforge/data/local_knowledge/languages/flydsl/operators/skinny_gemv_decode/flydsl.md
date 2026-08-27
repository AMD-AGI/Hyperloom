---
title: skinny_gemv_decode on FlyDSL — SOTA card
kind: sota_card
operator: skinny_gemv_decode
backend: flydsl
gens: [gfx950]
dtypes: [bf16]
regimes: [decode]
status: competitive
updated: 2026-06-09
---

# skinny_gemv_decode × FlyDSL

## TL;DR
FlyDSL has a dedicated **small-M kernel family** (`KERNEL_FAMILY_SMALL_M`) for decode-shaped GEMM where
`1 ≤ M < 17` (`SMALL_M_KERNEL_MAX = 17`). It is a FLIR/ROCDL MLIR-Python DSL kernel with a fixed `tile_m=16`,
a wide N-tile catalog, and persistent/repeat-N variants to keep all CUs busy at tiny M. **It is gated to
gfx950 only** — `iter_small_m_registry_configs` returns nothing on gfx942. Reached through the same
`aiter.tuned_gemm` flydsl seam as the dense/split-K hgemm.

## SOTA implementation
The registry only emits small-M configs for bf16→bf16 on non-gfx942, within the small-M range. From
`/sgl-workspace/aiter/aiter/ops/flydsl/kernels/small_m_hgemm.py` (`ROCm/aiter@a6bb4993`):

```python
def iter_small_m_registry_configs(dtype, out_dtype, *, m, n, k):
    if dtype != "bf16" or out_dtype != "bf16":
        return
    gpu_arch = get_rocm_arch()
    if gpu_arch == "gfx942" or not (1 <= m < SMALL_M_KERNEL_MAX):   # SMALL_M_KERNEL_MAX = 17
        return
    ...
    for tile_n in SMALL_M_TILE_N_OPTIONS:
        for tile_k in _small_m_tile_k_options(k):
            for split_k in _small_m_split_k_options(k, tile_k):
                for variant in _small_m_registry_variants():
                    config = {"kernel_family": "small_m", "tile_m": TILE_M, ...}  # TILE_M = 16
```

The kernel name carries the small-M shape and its variant tags via `small_m_kernel_name`:
`smallm_hgemm_{dtype}_16x{tile_n}x{tile_k}_S2TN_AS_BNW{block_n_warps}` plus optional `_NR{n_tile_repeat}`,
`_PN{persistent_n_tiles}`, `_SPK{split_k}`, `_BS` (b_to_lds) `_WPE{waves_per_eu}` `_UR{unroll}`, `_BIAS`.
`_small_m_split_k_options` keeps only `split_k` that evenly divide K with `(K/split_k) % tile_k == 0`.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FlyDSL small-M hgemm | `kernels/small_m_hgemm.py` (`iter_small_m_registry_configs`, `compile_small_m_hgemm_kernel`) | gfx950; bf16→bf16 | no isolated flydsl number; folded into aiter GEMM tune | decode / token-gen GEMM, `1 ≤ M < 17`, wide N |

## Config space / knobs
From the `SMALL_M_*` option tables in `small_m_hgemm.py` (fixed `TILE_M = 16`, `STAGES = 2`,
`BLOCK_M_WARPS = 1`, `WARP_SIZE = 64`):

| param | range / source | effect | default |
|---|---|---|---|
| `tile_n` | `SMALL_M_TILE_N_OPTIONS = (32,64,96,128,160,192,224,256,384,512,768,1024)` | N tile (very wide allowed) | — |
| `tile_k` | `SMALL_M_TILE_K_OPTIONS = (32,64,96,128,160,192,256)` | K tile (≥32, %32==0) | — |
| `split_k` | search `1..SMALL_M_MAX_SPLIT_K (32)`, divisibility-filtered | K split across CUs | — |
| `block_n_warps` | base `(1,2,3,4)`; repeat `(1,2)`; persistent `(2,3,4)` | warps over N | — |
| `n_tile_repeat` | `SMALL_M_N_TILE_REPEAT_OPTIONS = (1,2,4)` | N tiles per WG iter (no `b_to_lds`) | 1 |
| `persistent_n_tiles` | `SMALL_M_PERSISTENT_N_TILE_OPTIONS = (2,4,8)` (needs b_to_lds, tile_n≥128, block_n_warps≥2) | persistent-kernel N tiling | 1 |
| `b_to_lds` + `b_to_lds_unroll` | `SMALL_M_B_TO_LDS_UNROLL_OPTIONS = (8,16)` | stage B via LDS + unroll | False / 0 |
| `waves_per_eu` | `(0,2,4)` | occupancy hint (0 = compiler) | 0 |

Validation (`_validate_small_m_registry_config`) enforces `tile_n % (block_n_warps*16) == 0`, LDS ≤
`MAX_LDS_BYTES = 163840`, and the repeat/persistent compatibility rules above.

## Numerics / parity
fp32 MFMA accumulate (16×16 atoms, `STAGES=2`); `b_preshuffle=False` and `async_copy=True` in the emitted
small-M configs. bf16-only (registry refuses any non-bf16 in/out). Split-K combine same deterministic
signal-state mechanism as the split-K hgemm card.

## Integration (rebind seam)
Same path as the other FlyDSL GEMM cards: an `aiter.tuned_gemm` CSV row with `libtype=flydsl` whose
`kernelName` decodes (here via the small-M name grammar parsed back through
`get_flydsl_splitk_hgemm_kernel_params` → `_parse_hgemm_kernel_params` with the `_small_m` suffix), gated by
`is_flydsl_available()`. `flydsl_gemm` forwards `kernel_family`, `n_tile_repeat`, `persistent_n_tiles`,
`waves_per_eu`, `b_to_lds_unroll` from the decoded config into `flydsl_hgemm`.

## Pitfalls & anti-patterns
- **gfx942 gets nothing** from the small-M registry — small-M FlyDSL is effectively gfx950-only. On MI300X
  the decode shape falls to skinny/asm/CK.
- Small-M kernels are bf16-only here; fp16 / fp8 decode does not use this family.
- `n_tile_repeat>1` and `persistent_n_tiles>1` have strict gating (see validation) — most (tile_n,
  block_n_warps) combos are rejected; rely on the registry, don't hand-build names.
- `M ≥ 17` must use the regular hgemm tile_m path, not this family.

## How to verify
```bash
python -c "from aiter.jit.utils.chip_info import get_gfx; print(get_gfx())"   # confirm gfx950
python -c "from aiter.ops.flydsl.kernels.small_m_hgemm import iter_small_m_registry_configs as it; \
print(len(list(it('bf16','bf16', m=8, n=7168, k=512))))"
```

## Alternatives / cross-links
[[operators/skinny_gemv_decode/backends/aiter]] (sota decode path) ·
[[operators/skinny_gemv_decode/backends/asm]] · [[operators/dense_gemm/backends/flydsl]] (M≥17 hgemm) ·
[[operators/splitk_streamk_gemm/backends/flydsl]] (split-K combine shared).
