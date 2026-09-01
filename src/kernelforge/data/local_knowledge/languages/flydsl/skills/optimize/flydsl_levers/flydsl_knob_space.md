---
title: FlyDSL — what is actually tunable on a shipped kernel, and what only looks tunable
kind: language
lever: flydsl_knob_space
gens: [gfx950]
updated: 2026-08-28
---

# FlyDSL's tunable surface

## Route here when
You are tuning a kernel FlyDSL already ships — Path A. You want to know which arguments move
performance, which ones will raise, and what the legal ranges are.

**Not here if you are authoring.** When you write your own kernel, knobs are the last thing you touch,
and `flydsl_authoring_method.md` explains why reaching for them early wastes the run.

## Where these facts come from
The signature of `flydsl_hgemm` is the specification — there is no separate knob document to drift out
of sync with it. Names below are read off `gemm_kernels.py::flydsl_hgemm` and
`_compile_flydsl_hgemm`; the legality rules are enforced in `_validate_hgemm_tiling`.

## Three arguments exist but are not yours to set
This trips people first, so it goes first. Passing any of these off its pinned value raises a
`ValueError` — it does not silently take a slow path.

| Argument | Actual behaviour |
|---|---|
| `async_copy` | Derived from the architecture. `_normalize_supported_kernel_metadata` computes it as `get_rocm_arch() != "gfx942"`, meaning **gfx950 gets direct-to-LDS async and gfx942 does not**, and then raises if what you passed disagrees. |
| `stages` | Pinned to 2 (`FIXED_STAGE`) in the HGEMM kernel as it currently ships. |
| `c_to_lds` | Pinned to `False`. Passing `True` raises. |

These read like tuning parameters because the codegen used to emit variants for them. Those variants
were folded into the kernel; the parameters survive as validated constants. Treat them as facts about
the architecture and the build.

One exception worth knowing: `lds_stage` **is** settable on `flydsl_preshuffle_gemm_a8`. The pinning is
specific to HGEMM, not to FlyDSL.

## The `flydsl_hgemm` arguments
| Argument | Type | Default | What it does, and what constrains it |
|---|---|---|---|
| `tile_m` | int | 128 | output tile rows; needs `tile_m % (block_m_warps · 16) == 0` — the 16 is the MFMA warp atom |
| `tile_n` | int | 128 | output tile columns; needs `tile_n % (block_n_warps · 16) == 0`, **plus `N % tile_n == 0` and `N ≥ tile_n`** |
| `tile_k` | int | 64 | K block; must be `≥ 32` and a multiple of 32, and `(K / split_k) % tile_k == 0` |
| `split_k` | int | 1 | K-dimension parallelism; needs `K % split_k == 0`; see the capacity guard below |
| `block_m_warps` | int | 1 | warps along M, 64 lanes each |
| `block_n_warps` | int | 4 | warps along N; the block is `block_m_warps · block_n_warps · 64` threads |
| `b_preshuffle` | bool | True | B is expected already laid out as `(16 · pack_n, 16)`; **requires `b_to_lds=False`** |
| `b_to_lds` | bool | False | stage B through LDS instead; mutually exclusive with preshuffle |
| `auto_shuffle_b` | bool | False | perform the shuffle inside the call, once, when `b_preshuffle=True` |
| `pack_n` | int | 1 | weight pack factor — **1 is the only supported value** |
| `bias` | Tensor? | None | 1-D `[N]`; fused into the epilogue only when the output dtype matches the input dtype |
| `n_tile_repeat` | int | 1 | small-M path: N tiles repeated per workgroup |
| `persistent_n_tiles` | int | 1 | small-M path: N tiles per workgroup in persistent mode |
| `waves_per_eu` | int | 0 | small-M path: occupancy hint; 0 leaves it to the compiler |
| `b_to_lds_unroll` | int | 0 | small-M path: unroll depth for B→LDS staging |

## Tiling is the lever that matters
Everything else is secondary to `tile_m × tile_n × tile_k` and the `block_m_warps × block_n_warps` warp
grid. Because the MFMA atom is 16×16, both output tile dimensions have to be multiples of
`warps × 16` — that is where most rejected configurations fail.

The space aiter searches:

- `tile_m` — 16, 32, 48, 64, 80, 96, 112, 128, 160, 256, capped somewhere near 2·M
- `tile_n` — 64, 128, 160, 192, 256, and **it has to divide N**
- `tile_k` — 64, 96, 128, 160, 256
- `(block_m_warps, block_n_warps, b_to_lds)` — (1,2,F), (1,4,F), (2,2,F), (1,4,T), (2,2,T)

> **Look at the non-power-of-two entries: 48, 80, 112, 160, 192.** FLIR's layout system makes these
> legal, where Triton's space is effectively pow2-biased. For an awkward N — say 160 — Triton has to pad
> and eat the waste, and FlyDSL does not. This is one of the few places where FlyDSL is more expressive
> rather than just different, and it is worth remembering when picking a backend for odd shapes.

## `split_k`, and why its reduction is worth knowing about
Functionally this plays the same role as Triton's `SPLIT_K`: cut K into pieces so a skinny or
decode-shaped GEMM has enough parallelism to fill the device.

The implementation differs in a way that matters. The partial results are combined through a **global
semaphore and a signal-state ring**, not through raw `atomic_add`. That makes the reduction
**deterministic** — run to run, the same inputs give bit-identical output. If you are gating on
reproducibility, or debugging a numeric difference against a non-deterministic baseline, this is the
detail that explains why FlyDSL behaves differently.

Two limits: only `split_k` values that divide K *and* leave between 2 and 8 block-K loops are offered,
and when `split_k > 1` the tile count is capped —
`ceil(M / tile_m) · (N / tile_n) ≤ 128`.

## Choosing between `b_preshuffle` and `b_to_lds`
They are mutually exclusive, and the choice is really about when you can afford to pay the relayout.

| | `b_preshuffle=True` (default) | `b_to_lds=True` |
|---|---|---|
| What happens to B | pre-arranged into MFMA fragment order `(16 · pack_n, 16)` | staged through LDS inside the kernel |
| When you pay | once, at model load | **on every call** |
| Extra LDS | none | `stages · tile_n · tile_k · 2` bytes (`_estimate_hgemm_lds_bytes`) |
| Pick it when | serving — this is the fast answer | you have no opportunity to shuffle offline |

## The scaled fp8/int8 path has its own knobs
```python
flydsl_preshuffle_gemm_a8(..., lds_stage=2, use_cshuffle_epilog=0,
                          use_async_copy=0, waves_per_eu=0)
```

| Argument | What it does |
|---|---|
| `lds_stage` | LDS pipeline depth — genuinely settable here, unlike on HGEMM |
| `use_cshuffle_epilog` | keeps the result in MFMA layout through the epilogue; the analogue of Triton's `OPTIMIZE_EPILOGUE` |
| `use_async_copy` | direct global→LDS |
| `waves_per_eu` | occupancy hint; 0 defers to the compiler |

Note that `flydsl_hgemm` asserts its scale arguments are `None`. Scaled GEMM belongs here, not there.

## About the built-in autotuner
FlyDSL ships a Triton-shaped autotuner:

```python
from flydsl import Config, autotune
Config(num_warps=4, waves_per_eu=3, maxnreg=128, **kernel_kwargs)
```

`Config.compiler_opts()` splits the compiler-level options (`waves_per_eu`, `maxnreg`) from the kwargs
that get injected into the `@jit` call; `@autotune` times the candidates and caches the winner to disk.

**aiter does not use it for GEMM.** aiter runs an offline sweep and writes the per-shape CSV that
`tuned_gemm` reads at dispatch. The principle is the same one Triton work follows: decide offline, bake
the answer, never search in the serving path.

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| Raises on the call | `b_preshuffle=True` but B was never shuffled | call `shuffle_weight` beforehand, or pass `auto_shuffle_b=True` |
| "Unsupported" for your shape | `N % tile_n != 0` | HGEMM needs N to be a whole multiple of `tile_n` — pick a dividing tile |
| `ValueError` naming a knob | `async_copy`, `stages` or `c_to_lds` passed off its pinned value | these are not tunable in this build |
| Assertion about scaling | scale arguments handed to `flydsl_hgemm` | use `flydsl_preshuffle_gemm_a8` |
| A tuned config regressed after an upgrade | the CSV is tied to the build that produced it | re-tune per ROCm/aiter version |
| Non-deterministic split-K expected, got determinism | the reduction uses a semaphore ring, not atomics | this is by design; do not "fix" it |

## Related
`flydsl_kernel_library.md` (which family to call in the first place) ·
`flydsl_authoring_method.md` (what to do when no knob setting is enough) ·
`../../../../../hardware/mi350_lds.md` (the LDS budget these tiles are spending)
