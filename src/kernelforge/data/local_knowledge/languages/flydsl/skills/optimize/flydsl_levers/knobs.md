---
title: FlyDSL — the full knob set (verified against aiter source)
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: competitive
updated: 2026-06-08
---

# FlyDSL — knobs

The `flydsl_hgemm` signature is the authoritative knob set. Every name below is from
`gemm_kernels.py::flydsl_hgemm` / `_compile_flydsl_hgemm`; constraints from `_validate_hgemm_tiling`.

## 0. `flydsl_hgemm` knobs
| Knob | Type | Default | Meaning / constraint |
|---|---|---|---|
| `tile_m` | int | 128 | output tile rows; `tile_m % (block_m_warps·16) == 0` (MFMA warp atom = 16) |
| `tile_n` | int | 128 | output tile cols; `tile_n % (block_n_warps·16) == 0`; **`N % tile_n == 0`, `N ≥ tile_n`** |
| `tile_k` | int | 64 | K-block; **`% 32 == 0` and `≥ 32`**; `(K/split_k) % tile_k == 0` |
| `split_k` | int | 1 | K-reduction parallelism; `K % split_k == 0`; uses global-semaphore reduce; tiles ≤ 128 |
| `block_m_warps` | int | 1 | warps along M (×64 lanes) |
| `block_n_warps` | int | 4 | warps along N; `block_threads = bmw·bnw·64` |
| `b_preshuffle` | bool | True | expect B pre-shuffled to `(16·pack_n, 16)`; **requires `b_to_lds=False`** |
| `b_to_lds` | bool | False | stage B through LDS (vs preshuffle); mutually exclusive with preshuffle |
| `async_copy` | bool | False | direct-to-LDS async copy; **forced from arch** (gfx950 True, gfx942 False) — validated, can't override |
| `stages` | int | 2 | LDS pipeline depth; **FIXED at 2** in current kernel (`FIXED_STAGE`) |
| `c_to_lds` | bool | False | route C through LDS in epilogue; **FIXED False** (raises if True) |
| `pack_n` | int | 1 | weight pack factor; **only 1 supported** |
| `n_tile_repeat` | int | 1 | (small-M) repeat N tile per workgroup |
| `persistent_n_tiles` | int | 1 | (small-M) persistent-kernel N tiles per WG |
| `waves_per_eu` | int | 0 | (small-M) occupancy hint, 0 = compiler |
| `b_to_lds_unroll` | int | 0 | (small-M) unroll factor for B→LDS staging |
| `auto_shuffle_b` | bool | False | shuffle B inside the call (one-shot) when `b_preshuffle=True` |
| `bias` | Tensor? | None | 1-D `[N]`, fused only when out dtype == input dtype |

## 1. Tiling — the primary lever
`tile_m × tile_n × tile_k` with `block_m_warps × block_n_warps` warps. Because the MFMA atom is
**16×16**, tile_m/tile_n must be multiples of `warps·16`. aiter's default search space:
- `tile_m ∈ {16,32,48,64,80,96,112,128,160,256}` (capped near 2·M),
- `tile_n ∈ {64,128,160,192,256}` (must divide N),
- `tile_k ∈ {64,96,128,160,256}`,
- warp variants: `(bmw,bnw,b_to_lds)` ∈ `{(1,2,F),(1,4,F),(2,2,F),(1,4,T),(2,2,T)}`.
Note non-power-of-2 tiles (160, 192, 48, 80, 112) — FLIR layouts make these legal, unlike Triton's
pow2-biased space. This is a real expressivity advantage for odd N (e.g. N=160 GEMMs).

## 2. `split_k` — skinny/decode parallelism
Same role as Triton SPLIT_K but reduced via a **global semaphore + signal-state ring**
(deterministic, not raw atomic_add). Only split_k that divide K and leave 2–8 block-K loops are
offered. Capacity guard: `ceil(M/tile_m)·(N/tile_n) ≤ 128` for split_k>1.

## 3. `b_preshuffle` vs `b_to_lds`
- **`b_preshuffle=True`** (default, fastest for serving): weight pre-laid-out to MFMA fragment order
  `(16·pack_n, 16)` — removes in-kernel relayout. Shuffle once at model load.
- **`b_to_lds=True`**: stage B through LDS in-kernel; pay relayout per call but no offline shuffle.
  Adds `stages·tile_n·tile_k·2B` to the LDS budget (`_estimate_hgemm_lds_bytes`).
Mutually exclusive.

## 4. `async_copy` / `stages` (arch-pinned in current kernel)
`async_copy` is **not free to set** — `_normalize_supported_kernel_metadata` forces it to
`get_rocm_arch() != "gfx942"` (so gfx950 uses direct-to-LDS async, gfx942 doesn't) and raises if you
pass a mismatch. `stages` is pinned at **2**. Treat these as architecture facts, not tunables, in the
current build — the codegen variants were collapsed into the kernel.

## 5. small-M extra knobs
`n_tile_repeat`, `persistent_n_tiles`, `waves_per_eu`, `b_to_lds_unroll` only apply to the small-M
family (`kernel_family="small_m"`, bf16, `tile_m=16`, `block_m_warps=1`, `b_preshuffle=False`). These
are the decode-GEMV occupancy/persistence levers.

## 6. preshuffle-a8 (scaled fp8/int8) knobs
`flydsl_preshuffle_gemm_a8(..., lds_stage=2, use_cshuffle_epilog=0, use_async_copy=0, waves_per_eu=0)`:
- `lds_stage` — LDS pipeline depth (settable here, unlike hgemm).
- `use_cshuffle_epilog` — keep result in MFMA layout through epilogue (≈ Triton OPTIMIZE_EPILOGUE).
- `use_async_copy` — direct-to-LDS.
- `waves_per_eu` — occupancy hint (0 = compiler).

## 7. FlyDSL autotuner (`flydsl.autotune`)
The DSL also ships a Triton-style autotuner:
```python
from flydsl import Config, autotune
Config(num_warps=4, waves_per_eu=3, maxnreg=128, **kernel_kwargs)
```
`Config.compiler_opts()` separates **compiler-level** options (`waves_per_eu`, `maxnreg`) from user
kwargs injected into the `@jit` call. `@autotune` benchmarks configs and picks the fastest (caches to
disk). In aiter, GEMM tuning instead uses an offline sweep that writes the per-shape CSV consumed by
`tuned_gemm` — same "bake the winner, don't autotune in the hot path" discipline as Triton.

## 8. Pitfalls
- `b_preshuffle=True` without a shuffled B → raises (use `shuffle_weight` or `auto_shuffle_b=True`).
- `N % tile_n != 0` → unsupported (FlyDSL HGEMM requires N a multiple of tile_n).
- Passing `async_copy`/`stages`/`c_to_lds` off their arch-fixed values → `ValueError`.
- Scaling (`scale_a/b/c`) on `flydsl_hgemm` → asserts; use `flydsl_preshuffle_gemm_a8`.
- Tuned config CSV is build-specific — re-tune per ROCm/aiter version.
