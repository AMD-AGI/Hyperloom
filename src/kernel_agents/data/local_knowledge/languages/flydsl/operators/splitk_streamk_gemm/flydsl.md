---
title: splitk_streamk_gemm on FlyDSL — SOTA card
kind: sota_card
operator: splitk_streamk_gemm
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-09
---

# splitk_streamk_gemm × FlyDSL

## TL;DR
FlyDSL's split-K hgemm is a **FLIR/ROCDL MLIR-Python DSL** (CuTe-inspired) kernel that partitions the K
reduction across workgroups, combining partials through a per-stream global signal/semaphore state rather
than naive atomics. It's the same authoring backend AMD used on Kimi-K2.5; for the dense hgemm path it's
reached only through `aiter.tuned_gemm` when a tuned CSV row carries `libtype=flydsl` **and**
`is_flydsl_available()` is true (else aiter silently falls back to CK/asm). Only **split-K** is in this
source — no separate stream-K kernel.

## SOTA implementation
The split-K decomposition is driven by `_hgemm_split_k_options`: a candidate `split_k` is kept only when it
**evenly divides K** and the per-split K is a whole multiple of `tile_k` (so partials are aligned), plus an
extra-loops window. From `/sgl-workspace/aiter/aiter/ops/flydsl/gemm_kernels.py` (`ROCm/aiter@a6bb4993`):

```python
def _hgemm_split_k_options(k: Optional[int], tile_k: int) -> tuple[int, ...]:
    if k is None:
        return HGEMM_BASE_SPLIT_K_OPTIONS         # (1, 2, 4, 8, 16)
    options = set()
    for split_k in range(1, HGEMM_MAX_SPLIT_K + 1):   # up to 32
        if k % split_k != 0 or (k // split_k) % tile_k != 0:
            continue
        if split_k in HGEMM_BASE_SPLIT_K_OPTIONS:
            options.add(split_k); continue
        block_k_loops = k // (split_k * tile_k)
        if HGEMM_EXTRA_BLOCK_K_LOOPS_MIN <= block_k_loops <= HGEMM_EXTRA_BLOCK_K_LOOPS_MAX:  # 2..8
            options.add(split_k)
    return tuple(sorted(options))
```

`_hgemm_tile_m_options` clamps tile_m to `max(96, align_up(2*m, 16))` so small-M shapes don't waste tiles.
The runtime executor `flydsl_hgemm` only advances the split-K signal state when `split_k > 1`
(`_advance_split_k_signal_state`), i.e. the reduction-combine path is engaged exactly for true split-K. The
MFMA microkernel is `WmmaHalf_m16n16k16` / `_m16n16k32` emitting `rocdl.mfma_f32_16x16x16bf16_1k` /
`mfma_f32_16x16x16f16` in `kernels/splitk_hgemm.py`.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FlyDSL split-K hgemm | `gemm_kernels.py::flydsl_hgemm` + `kernels/splitk_hgemm.py` | gfx942/950; bf16, fp16 | no isolated flydsl number; folded into aiter GEMM tune (chosen per-shape vs asm/CK/hipblaslt) | skinny / large-K bf16 hgemm where K split refills idle CUs |

## Config space / knobs
From `flydsl_hgemm` signature + the option tables in `gemm_kernels.py`:

| param | range / source | effect | default |
|---|---|---|---|
| `tile_m` | `HGEMM_TILE_M_OPTIONS = (16,32,48,64,80,96,112,128,160,256)` | output M tile | 128 |
| `tile_n` | `HGEMM_TILE_N_OPTIONS = (64,128,160,192,256)` | output N tile | 128 |
| `tile_k` | `HGEMM_TILE_K_OPTIONS = (64,96,128,160,256)`; must be ≥32 and %32==0 | K tile | 64 |
| `split_k` | base `(1,2,4,8,16)`, search up to `HGEMM_MAX_SPLIT_K = 32` | K-dim split across CUs | 1 |
| `block_m_warps` / `block_n_warps` | warp grid (16×16 atoms) | warps inside block | 1 / 4 |
| `b_to_lds` / `b_preshuffle` | bool / bool (mutually exclusive: `b_to_lds=False` required when `b_preshuffle=True`) | stage B via LDS vs consume preshuffled B | False / True |
| `stages` | must equal `FIXED_STAGE = 2` | software-pipeline depth | 2 |
| `async_copy` / `c_to_lds` | `KERNEL_ASYNC_COPY = get_rocm_arch()!='gfx942'` / fixed | async g→LDS / C staging | arch-derived / fixed |

`split_k` extra-loops window: `HGEMM_EXTRA_BLOCK_K_LOOPS_MIN=2`, `..._MAX=8`.

## Numerics / parity
Split-K accumulates partials in **fp32**; the combine is mediated by a 3-state per-stream signal
(`SPLIT_K_SIGNAL_STATE_COUNT = 3`) advanced only when `split_k > 1`, giving deterministic reduction (not
free atomics). Regression cases in `test_flydsl_splitk_hgemm.py` use bf16, `atol=rtol=1e-2`, pass ≥99.9%
close (relaxed to 99.0% + bounded `max_delta` for large split_k=16/8 + b_to_lds cases). Reference is fp32
`torch.mm`.

## Integration (rebind seam)
Reached through `aiter.tuned_gemm`: a CSV row with `libtype=flydsl` + a `kernelName` that
`get_flydsl_splitk_hgemm_kernel_params(name)` resolves (registry lookup, else `_parse_hgemm_kernel_params`),
**and** `is_flydsl_available()` true. The dispatcher (`tuned_gemm.py` ~L133) sets `config=None` and falls
through to the next granularity / default if either check fails. The hgemm path **asserts no scaling**
(`flydsl_gemm`: `scale_a/scale_b/scale_c is None`). Deploy = same env path as the dense aiter card
(`AITER_CONFIG_GEMM_BF16=<csv>`); no standalone FlyDSL env overlay.

## Pitfalls & anti-patterns
- A `split_k` that doesn't evenly divide K, or whose per-split K isn't a multiple of `tile_k`, is silently
  excluded by `_hgemm_split_k_options` — never hand-pick split_k without checking divisibility.
- `b_preshuffle=True` requires pre-shuffled B (`shuffle_weight(b, layout=(16*pack_n,16))`) or
  `auto_shuffle_b=True`; otherwise `flydsl_hgemm` raises. And `b_to_lds=True` with `b_preshuffle=True` is
  rejected by `flydsl_kernel_name`.
- Only `stages == FIXED_STAGE (2)` compiles — legacy stage/async_copy/c_to_lds metadata raises.
- A typo in the CSV `kernelName` → `get_flydsl_splitk_hgemm_kernel_params` returns `None` → row silently
  disabled.

## How to verify
```bash
python -c "from aiter.ops.flydsl.utils import is_flydsl_available; print(is_flydsl_available())"
pytest -q aiter/ops/flydsl/test_flydsl_splitk_hgemm.py   # split-K precision regressions
```

## Alternatives / cross-links
[[operators/splitk_streamk_gemm/backends/triton]] (stream-K + split-K, the sota authoring path) ·
[[operators/splitk_streamk_gemm/backends/ck]] (the fallback) ·
[[operators/dense_gemm/backends/flydsl]] (dense hgemm, same kernel family) ·
[[operators/skinny_gemv_decode/backends/flydsl]] (small-M sibling kernel).
