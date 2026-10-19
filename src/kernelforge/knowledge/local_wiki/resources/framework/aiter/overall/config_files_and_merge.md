---
title: aiter config files — where a tuned CSV comes from, and the merge rules that decide what wins
kind: reference
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-08-28
sources:
  - ROCm/aiter@b467ce342:aiter/jit/core.py
  - ROCm/aiter@b467ce342:aiter/configs/
  - ROCm/aiter@b467ce342:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce342:aiter/fused_moe.py
---

# aiter config files and merge semantics

## Route here when
- You have a tuned CSV and need to know **how to make aiter actually read it**.
- Two config sources disagree and you need to know which one wins.
- A DB hit works on one box and misses on another.
- You need the exact column list for a CSV you are about to generate or diff.

**Skip this if** the question is *how to produce* a tuned CSV — that is
[tuning_db.md](tuning_db.md). This card is about resolution, schema, and merge, i.e. everything
between "I have a CSV" and "the kernel changed".

## The model in three sentences
Every aiter op family has a `(tuned, untuned)` CSV pair under `aiter/configs/`. Each tuned file is
reachable by an `AITER_CONFIG_*` environment variable that **replaces** the shipped path and accepts a
`:`-joined list. Deploying a tuning win is therefore an env var, never a `site-packages` edit.

The part that surprises people is what happens when you *don't* set the env var: aiter auto-discovers
per-model overlay CSVs and merges them on top of the shipped default. Setting the env var turns that
off.

## The files
| Op family | tuned CSV | untuned CSV | env override |
|---|---|---|---|
| dense bf16/fp16 GEMM | `bf16_tuned_gemm.csv` | `bf16_untuned_gemm.csv` | `AITER_CONFIG_GEMM_BF16` |
| bf16 batched GEMM | `bf16_tuned_batched_gemm.csv` | `bf16_untuned_batched_gemm.csv` | `AITER_CONFIG_BF16_BATCHED_GEMM` |
| a8w8 (fp8/int8) GEMM | `a8w8_tuned_gemm.csv` | `a8w8_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8` |
| a8w8 bpreshuffle | `a8w8_bpreshuffle_tuned_gemm.csv` | `a8w8_bpreshuffle_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| a8w8 block-scale | `a8w8_blockscale_tuned_gemm.csv` | `a8w8_blockscale_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` |
| a8w8 blockscale + bpreshuffle | `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | `…_untuned_…` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |
| a8w8 batched GEMM | `a8w8_tuned_batched_gemm.csv` | `a8w8_untuned_batched_gemm.csv` | `AITER_CONFIG_A8W8_BATCHED_GEMM` |
| a4w4 block-scale GEMM | `a4w4_blockscale_tuned_gemm.csv` | `a4w4_blockscale_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A4W4` |
| fused MoE (2-stage) | `tuned_fmoe.csv` | `untuned_fmoe.csv` | `AITER_CONFIG_FMOE` |
| grouped fused MoE (FlyDSL, gfx1250) | `tuned_grouped_fmoe.csv` | `untuned_grouped_fmoe.csv` | `AITER_CONFIG_GROUPED_FMOE` |
| asm a8w8 catalog | `asm_a8w8_gemm.csv` | — | none (it is a kernel catalog, not a tuning result) |

Each `AITER_CONFIGS.*_FILE` property resolves env → default path → shipped CSV. The singleton is
`AITER_CONFIGS = AITER_CONFIG()`.

`aiter/configs/model_configs/` holds roughly 104 **per-model overlay** CSVs — `dsv4_bf16_tuned_gemm.csv`,
`qwen3_235b_bf16_tuned_gemm.csv`, `dsv4_fp8fp4_tuned_fmoe.csv`, and so on.

## Resolution: the branch that catches people
`get_config_file(env_name, default_file, tuned_file_name)` has exactly two behaviours:

| Env var | What aiter uses |
|---|---|
| **SET** | Precisely the `:`-joined paths you listed. The shipped default is **not** prepended. Model overlays are **not** discovered. |
| **UNSET** | Auto-discovers `configs/model_configs/*{tuned_file_name}*.csv` (excluding `untuned`). If any match, the shipped `default_file` is **prepended** and all are merged. If none match, only the shipped default is used. |

```bash
export AITER_CONFIG_GEMM_BF16=/abs/my_tuned.csv            # exactly this file, nothing else
export AITER_CONFIG_GEMM_BF16=/abs/a.csv:/abs/b.csv        # exactly a + b, merged
# unset                                                     # shipped default + any matching model overlay
```

**The consequence worth writing down:** setting the env var to your own file *disables the model
overlays you were implicitly getting*. If a model overlay was carrying good rows for your workload and
you deploy a narrow tuned file, you can lose more than you gain. Either list the shipped default in
the `:` chain yourself, or verify hit counts before and after.

## Merge rules (`update_config_files`)
When several files merge, in order:

1. **Column union** with fill defaults (`xbf16`, `run_1stage`, `ksplit` default to `0`).
2. **`gfx` backfill** — a missing `gfx` column is derived from `cu_num` via `gfx_from_cu_num`
   (256 → gfx950; 80 / 304 → gfx942).
3. **De-duplicate** by the untuned file's key columns plus `cu_num` (plus `gfx`).
4. **Duplicate shapes resolve to the lowest `us`** — fastest measurement wins, regardless of which
   file it came from. Later in the `:` chain does *not* mean higher priority.
5. Write the merged result to `/tmp/aiter_configs/{merge_name}.csv` under a file lock.

Rule 4 is the one to remember: merge order does not decide the winner, measured time does. If you want
a specific row to win, it has to be *faster*, not later.

Rule 2 explains a portability trap — a legacy CSV with no `gfx` column only gets backfilled **during a
merge**. Regenerate on the target box rather than relying on it.

## Schemas (real headers)

**`bf16_tuned_gemm.csv`**
```
gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle,   # key
libtype, solidx, splitK, us, kernelName, err_ratio, tflops, bw       # result
```
Serving-time lookup uses the **10-tuple** `(gfx, cu_num, M(padded), N, K, bias, dtype, outdtype,
scaleAB, bpreshuffle)`. `gfx` is the **first index field**, not provenance metadata — a row tuned on
another arch simply misses. `M` is the bucketed `padded_M`, so one tuned row covers a range of live M.

**`bf16_untuned_gemm.csv`** (capture output; no result columns)
```
M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle
```
Written by `AITER_TUNE_GEMM=1` (`save_shapes`), deduplicated. `bias` is `bias is not None` — a boolean
derived from the live call, which is why hand-authoring this file reliably produces a DB that never hits.

**`a4w4_blockscale_tuned_gemm.csv`**
```
gfx, cu_num, M, N, K, kernelId, splitK, us, kernelName, tflops, bw, errRatio
```
Real rows are gfx950 FP4 BpreShuffle kernels, e.g. `_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E`.

**`tuned_fmoe.csv`**
```
cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype,
q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1, block_m, ksplit,   # key
us1, kernelName1, err1, us2, kernelName2, err2, us, run_1stage, tflops, bw, _tag   # result
```
The shipped header has **no `gfx` column**, but the runtime lookup in `fused_moe.py` keys on
`(gfx, cu_num, token, …)` and backfills `gfx` from `cu_num` for legacy files. `token` is the M-bucket,
not the raw token count. The grouped path (`tuned_grouped_fmoe.csv`, gfx1250 FlyDSL) uses a much wider
tile-config schema: `…, gate_mode, max_m, tile_m/n/k[2], m_warp, n_warp, num_buffers, split_k1/2, …,
kernelName1, kernelName2`. See
[aiter_moe_pipeline.md](../skills/optimize/aiter_levers/aiter_moe_pipeline.md).

## Verify
| Check | Signal | Pass condition |
|---|---|---|
| The file was found and parsed | `AITER_LOG_TUNED_CONFIG=1` | `… is tuned on cu_num = N in <file>, libtype is …` names *your* file |
| A shape hit | same | count of `is tuned on cu_num` lines > 0 |
| A shape missed | same | `… not found tuned config in <file>, will use default config!` |
| The merge did what you expected | read `/tmp/aiter_configs/{merge_name}.csv` | the row you care about is present, with the `us` you measured |

That last one is the fastest way to settle an argument about merge behaviour: the merged file is on
disk, so read it instead of reasoning about it.

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| CSV deployed, zero hits | key mismatch — `gfx`, `cu_num`, or `bias` differs from the live call | capture live with `AITER_TUNE_GEMM=1`; never hand-author the untuned file |
| Hits dropped after deploying your own CSV | env var set → model overlays no longer discovered | include the shipped default in the `:` chain |
| A row you added is ignored | another file had a lower `us` for the same key | merge order is irrelevant; make it faster or remove the competitor |
| Worked on the tuning box, misses in production | different arch or CU count in the key | re-tune on the target box |
| Row references a kernel that no longer exists | `solidx` / `kernelName` are tied to the aiter+ROCm build | re-tune after any upgrade; never ship a tuned table as portable |
| A `flydsl` row does nothing | FlyDSL package absent or kernel name stale | [aiter_flydsl_libtype.md](../skills/optimize/aiter_levers/aiter_flydsl_libtype.md) |

## Deeper
[tuning_db.md](tuning_db.md) (how to produce these files) ·
[dispatch_and_rebind.md](dispatch_and_rebind.md) (how the resolved row becomes a kernel call) ·
[jit_and_build.md](jit_and_build.md) (why a build change invalidates a tuned table) ·
[aiter_moe_pipeline.md](../skills/optimize/aiter_levers/aiter_moe_pipeline.md) (the MoE DB) ·
[aiter_flydsl_libtype.md](../skills/optimize/aiter_levers/aiter_flydsl_libtype.md) (the one libtype that can be dropped after selection).

## Sources
- On-box `ROCm/aiter@b467ce342`: `aiter/jit/core.py` (`AITER_CONFIGS.*_FILE` resolution,
  `get_config_file` set-vs-unset branch, `update_config_files` column-union / dedup / lowest-`us`
  resolution / `/tmp/aiter_configs` output, `gfx_from_cu_num` backfill),
  `aiter/configs/*.csv` and `aiter/configs/model_configs/*.csv` (real headers and rows),
  `aiter/tuned_gemm.py` (`save_shapes`, the 10-tuple index columns),
  `aiter/fused_moe.py` (the gfx-first runtime fmoe key).
