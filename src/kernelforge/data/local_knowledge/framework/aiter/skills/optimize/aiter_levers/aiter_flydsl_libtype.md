---
title: aiter's flydsl libtype — when it engages, and why it usually doesn't
kind: lever
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp4_e2m1]
regimes: [prefill, decode]
status: experimental
updated: 2026-08-28
sources:
  - ROCm/aiter@b467ce342:aiter/ops/flydsl/gemm_kernels.py
  - ROCm/aiter@b467ce342:aiter/ops/flydsl/moe_kernels.py
  - ROCm/aiter@b467ce342:aiter/ops/flydsl/utils.py
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:csrc/gemm_a16w16/gemm_a16w16_tune.py
---

# aiter's `flydsl` libtype

## Route here when
- A tuned CSV row says `libtype="flydsl"` and you need to know whether it will actually run.
- You tuned with `--libtype flydsl` (or `all`) and the win vanished on a different box.
- You are deciding whether to *include* FlyDSL in a tuning sweep at all.
- A4W4 (FP4-weight) MoE is slower than expected, or raises a CK "does not support this GEMM
  problem" error.

**Skip this if** you are authoring FlyDSL kernels — that is `languages/flydsl/`. This card is about
aiter's *dispatch* to FlyDSL, not the language.

## The one thing to internalize
`flydsl` is the only aiter libtype that can be **present in the DB and still not run**. Every other
libtype (`hipblaslt`, `asm`, `skinny`, `triton`, `opus`, `torch`) resolves from the row alone. A
`flydsl` row additionally requires a package that is *not vendored* and a kernel name that must still
exist in the current catalog. Either check failing makes aiter silently drop the row and continue to
the next lookup granularity — no warning, no error, just a different kernel.

That is why a tuned CSV can measure +X% on the tuning box and 0% in production.

## The three gates, in dispatch order

| # | Gate | Where | Fails when |
|---|---|---|---|
| 1 | DB row says `libtype == "flydsl"` | `tuned_gemm.get_GEMM_A16W16_config` | shape was never tuned, or another libtype won |
| 2 | `is_flydsl_available()` | `aiter/ops/flydsl/utils.py` | FlyDSL package not installed |
| 3 | `get_flydsl_splitk_hgemm_kernel_params(kernelName)` resolves | `aiter/ops/flydsl/gemm_kernels.py` | the encoded kernel name is not in this build's catalog |

Gate 2 is literally `importlib.util.find_spec("flydsl") is not None` — an import check, nothing more.
On failure at gate 2 or 3 the dispatcher sets `config = None` and falls through to the next
`padded_M` granularity, then to the default (`hipblaslt`/`asm` when `bpreshuffle`, `skinny` for
small-M default shapes, else `torch`).

## How kernels are named
FlyDSL kernels carry their entire launch configuration **in the name**, parsed back out by a regex.
The DB stores only that string:

```
flydsl_gemm{stage}_a{dtype}_w{dtype}_{out}
  _t{TM}x{TN}x{TK}_split_k{SK}
  _block_m_warp{..}_block_n_warp{..}
  _async_copy{..}_b_to_lds{..}_b_preshuffle{..}[_wpe{N}]
```

`get_flydsl_splitk_hgemm_kernel_params(name)` decodes it into launch params at call time. This is why
gate 3 exists: the name is a *contract with a specific FlyDSL build*. Upgrade FlyDSL, and a name that
no longer parses (or no longer maps to a built kernel) drops the row.

**Consequence for tuning:** a `flydsl` row is more version-brittle than a hipBLASLt `solidx`, which is
already the most brittle thing in the DB. Re-tune on every FlyDSL bump, not just every ROCm bump.

## What you can tune
| Knob | Values / constraint | Notes |
|---|---|---|
| `tile_m` / `tile_n` / `tile_k` | enumerated by `gemm_kernels.py` | `tile_m` options are capped relative to M |
| `split_k` | requires `k % split_k == 0` **and** `(k // split_k) % tile_k == 0` | both conditions, not either |
| `stages` | default 2 | pipeline depth |
| `async_copy`, `b_to_lds` | bool | staging strategy |
| `b_preshuffle` | bool | **mutually exclusive with `b_to_lds`** — `b_to_lds=False` is required when true |
| `waves_per_eu` | int | occupancy hint |
| `n_tile_repeat`, `persistent_n_tiles`, `b_to_lds_unroll`, `c_to_lds` | int / bool | passed straight through from the decoded name |

You do not set these by hand. The multi-backend tuner enumerates them:

```bash
python csrc/gemm_a16w16/gemm_a16w16_tune.py --libtype flydsl \
    -i aiter/configs/bf16_untuned_gemm.csv -o /tmp/tuned.csv --errRatio 0.05
```

The candidate set for a shape comes from
`get_flydsl_splitk_hgemm_kernels(in_dt, out_dt, m, n, k)`.

## Hard limit: no scaling
`flydsl_hgemm` **asserts `scale_a`, `scale_b`, and `scale_c` are all `None`**. Scaled GEMM
(fp8/fp4 with per-tensor or per-token scales) can never reach FlyDSL through this path. If you are
tuning a scaled-GEMM DB, `--libtype flydsl` is wasted sweep time.

Bias is fused only when the input and output dtypes align; otherwise it is added afterwards.

## A4W4 MoE and the CK fallback
`fused_moe` routes 4-bit-weight (A4W4 / FP4) MoE to FlyDSL when it is available. The FlyDSL MoE path
is two-stage with its own name encoding and a `sort_block_m` knob:

- `flydsl_moe1_*` — stage 1, gate + up
- `flydsl_moe2_*` — stage 2, down

**Without FlyDSL, A4W4 fused MoE falls back to CK grouped-GEMM instances.** This is the failure mode
worth remembering: it is not a slowdown, it is a *coverage* change. CK's instance set does not cover
every `(M, N, K, layout)`, so an odd expert/inter dimension that worked on the FlyDSL box raises
`device_gemm does not support this GEMM problem` on a box without FlyDSL. See
[aiter_moe_pipeline.md](aiter_moe_pipeline.md).

## Verify
| Check | Command / signal | Pass condition |
|---|---|---|
| Package present | `python -c "import flydsl"` | no ImportError |
| Row selected | `AITER_LOG_TUNED_CONFIG=1` | log line `libtype is flydsl, kernel name is <encoded name>` |
| Row *not* silently dropped | same log | you see `flydsl`, not `hipblaslt` / `torch` for that shape |
| Catalog still has the kernel | the encoded name appears in the log | a fall-through means gate 3 failed |

If you expect FlyDSL and see `hipblaslt` or `torch`, work gates 2 → 3 in that order.

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| Tuned CSV gives 0% in production, +X% on the tuning box | FlyDSL installed on one, not the other | Install FlyDSL, or re-tune on the target box with `--libtype` excluding flydsl |
| Log shows `torch` for a shape the CSV covers | gate 2 or 3 failed | check `import flydsl`; if it imports, the kernel name is stale — re-tune |
| A4W4 MoE raises `does not support this GEMM problem` | FlyDSL absent → CK fallback → CK instance gap | install FlyDSL, or pad to a CK-covered shape |
| `--libtype flydsl` sweep finds nothing for a scaled GEMM | `flydsl_hgemm` asserts scales are `None` | scaled GEMM cannot use this path; drop flydsl from the sweep |
| Row worked before a FlyDSL upgrade, now doesn't | encoded name no longer in the catalog | re-tune; names are build-specific |

## Deeper
[tuning_db.md](../../../overall/tuning_db.md) (capture → tune → deploy, and the full dispatch key) ·
[config_files_and_merge.md](../../../overall/config_files_and_merge.md) (how the CSV is resolved and
merged) · [aiter_moe_pipeline.md](aiter_moe_pipeline.md) (A4W4 MoE) ·
`languages/flydsl/` (authoring FlyDSL kernels rather than dispatching to them).

## Sources
- On-box `ROCm/aiter@b467ce342`: `aiter/ops/flydsl/gemm_kernels.py` (name regex, tile/split-K
  enumeration, `flydsl_hgemm` and its `scale_* is None` assert,
  `get_flydsl_splitk_hgemm_kernel_params`, `get_flydsl_splitk_hgemm_kernels`),
  `aiter/ops/flydsl/moe_kernels.py` (two-stage A4W4 MoE, `sort_block_m`),
  `aiter/ops/flydsl/utils.py` (`is_flydsl_available` = `find_spec("flydsl")`),
  `aiter/tuned_gemm.py` (the flydsl gate inside `get_GEMM_A16W16_config`, fall-through to default),
  `csrc/gemm_a16w16/gemm_a16w16_tune.py` (`--libtype flydsl`).
- A4W4 → CK fallback for fused MoE: https://github.com/ROCm/aiter (README, fused-MoE backend selection).
