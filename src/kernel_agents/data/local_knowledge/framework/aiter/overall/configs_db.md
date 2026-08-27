---
title: aiter configs DB — CSV schema, env overrides, merge semantics
kind: reference
backend: aiter
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:aiter/configs/
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:aiter/jit/core.py
---

# aiter configs DB

## TL;DR
aiter's per-shape tuned configs live as **CSV files under `aiter/configs/`**, one (tuned, untuned) pair
per op family. Each is reachable by an `AITER_CONFIG_*` env var that **overrides** the shipped path and is
**`:`-mergeable** (your tuned rows overlay the shipped table without editing site-packages). This is the
deploy seam for every aiter tuning win.

## The files (`aiter/configs/`)

| Op family | tuned CSV | untuned CSV | env override |
|---|---|---|---|
| dense bf16/fp16 GEMM | `bf16_tuned_gemm.csv` | `bf16_untuned_gemm.csv` | `AITER_CONFIG_GEMM_BF16` |
| bf16 batched GEMM | `bf16_tuned_batched_gemm.csv` | `bf16_untuned_batched_gemm.csv` | `AITER_CONFIG_BF16_BATCHED_GEMM` |
| a8w8 (fp8/int8) GEMM | `a8w8_tuned_gemm.csv` | `a8w8_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8` |
| a8w8 bpreshuffle | `a8w8_bpreshuffle_tuned_gemm.csv` | `a8w8_bpreshuffle_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| a8w8 block-scale | `a8w8_blockscale_tuned_gemm.csv` | `a8w8_blockscale_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` |
| a8w8 blockscale+bpreshuffle | `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | `..._untuned_...` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |
| a8w8 batched GEMM | `a8w8_tuned_batched_gemm.csv` | `a8w8_untuned_batched_gemm.csv` | `AITER_CONFIG_A8W8_BATCHED_GEMM` |
| a4w4 block-scale GEMM | `a4w4_blockscale_tuned_gemm.csv` | `a4w4_blockscale_untuned_gemm.csv` | `AITER_CONFIG_GEMM_A4W4` |
| fused MoE (2-stage) | `tuned_fmoe.csv` | `untuned_fmoe.csv` | `AITER_CONFIG_FMOE` |
| grouped fused MoE (FlyDSL, gfx1250) | `tuned_grouped_fmoe.csv` | `untuned_grouped_fmoe.csv` | `AITER_CONFIG_GROUPED_FMOE` |
| asm a8w8 list | `asm_a8w8_gemm.csv` | — | (asm kernel catalog; no env) |

(Each `AITER_CONFIGS.*_FILE` property in `aiter/jit/core.py:141-214` resolves env → default path → shipped CSV.
The singleton accessor is `AITER_CONFIGS = AITER_CONFIG()` at `:377`.)
**`configs/model_configs/`** holds ~104 per-model overlay CSVs (e.g. `dsv4_bf16_tuned_gemm.csv`,
`qwen3_235b_bf16_tuned_gemm.csv`, `dsv4_fp8fp4_tuned_fmoe.csv`); when the matching `AITER_CONFIG_*` env is
**unset**, files whose name matches `*{tuned_file_name}*` are auto-discovered and merged on top of the
shipped default (see merge semantics below).

## Schemas (real headers)

**`bf16_tuned_gemm.csv`** (key + result):
```
gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle,   # key
libtype, solidx, splitK, us, kernelName, err_ratio, tflops, bw       # result
```
Lookup at serving time uses the **10-tuple** `(gfx, cu_num, M(padded), N, K, bias, dtype, outdtype, scaleAB,
bpreshuffle)` — `gfx` is the **first index field**, not just provenance, so a row tuned on another arch
misses. `M` is the bucketed `padded_M`. See [tuning_db.md](tuning_db.md).

**`bf16_untuned_gemm.csv`** (capture output, no result columns):
```
M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle
```
Written by `AITER_TUNE_GEMM=1` (`save_shapes`), deduped. `bias` = `bias is not None`.

**`a4w4_blockscale_tuned_gemm.csv`**:
```
gfx, cu_num, M, N, K, kernelId, splitK, us, kernelName, tflops, bw, errRatio
```
(real rows are `gfx950` FP4 BpreShuffle kernels, e.g.
`_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E`).

**`tuned_fmoe.csv`**:
```
cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type,
use_g1u1, doweight_stage1, block_m, ksplit,   # key
us1, kernelName1, err1, us2, kernelName2, err2, us, run_1stage, tflops, bw, _tag   # result
```
(stage-1 + stage-2 kernel names; see [fmoe.md](../skills/optimize/aiter_levers/fmoe.md)). The shipped header has **no `gfx` column**, but
the runtime lookup in `fused_moe.py` keys on `(gfx, cu_num, token, …)` and backfills `gfx` from `cu_num` for
legacy CSVs. The grouped path (`tuned_grouped_fmoe.csv`, gfx1250 FlyDSL) has a much wider tile-config schema
(`…, gate_mode, max_m, tile_m/n/k[2], m_warp, n_warp, num_buffers, split_k1/2, …, kernelName1, kernelName2`).

## Env override + merge semantics (`aiter/jit/core.py:216-374`)
`get_config_file(env_name, default_file, tuned_file_name)`:
- **env SET** → the `:`-joined (`os.pathsep`) value is passed straight to `update_config_files` (the shipped
  default is **not** auto-prepended — you merge exactly the paths you list).
- **env UNSET** → auto-discover `configs/model_configs/*{tuned_file_name}*.csv` (excluding `untuned`); if any
  are found, the shipped `default_file` is **prepended** and all are merged; if none, only the shipped
  default is used.
```bash
export AITER_CONFIG_GEMM_BF16=/abs/my_bf16_tuned.csv          # use exactly this file
export AITER_CONFIG_GEMM_BF16=/abs/a.csv:/abs/b.csv           # merge a + b (list what you want merged)
```
`update_config_files` merges the CSVs (column-union with fill defaults `xbf16/run_1stage/ksplit=0`),
**backfills a missing `gfx` column from `cu_num`** (`gfx_from_cu_num`: 256→gfx950, 80/304→gfx942),
de-duplicates by the untuned-file key columns + `cu_num`(+`gfx`), auto-resolves duplicate shapes by lowest
`us`, and writes the merged file to `/tmp/aiter_configs/{merge_name}.csv` under a file lock.
- `AITER_LOG_TUNED_CONFIG=1` logs every DB hit (`… is tuned on cu_num = N in <file>, libtype is …`); a miss
  logs `… not found tuned config in <file>, will use default config!`.

## Pitfalls
- **Version/build-specific**: hipBLASLt `solidx` and asm `kernelName` are tied to the aiter/ROCm build.
  Never ship a hand-copied tuned table as portable — re-tune on upgrade (sourcing rule #2).
- **`gfx`, `cu_num`, and `bias` in the key** — a DB tuned on a different arch (gfx942/gfx950/gfx1250), CU
  count, or bias flag won't hit (see [tuning_db.md](tuning_db.md) pitfalls). Legacy CSVs without a `gfx`
  column only get the `cu_num`→`gfx` backfill during a merge, so regenerate on the target box.
- A `flydsl` row needs the FlyDSL package present to be honored (see [flydsl_path.md](../skills/optimize/aiter_levers/flydsl_path.md)).

## Cross-links
[tuning_db.md](tuning_db.md) · [fmoe.md](../skills/optimize/aiter_levers/fmoe.md) · [flydsl_path.md](../skills/optimize/aiter_levers/flydsl_path.md) ·
[integration.md](dispatch_and_rebind.md).

## Sources
- On-box: `ROCm/aiter@b467ce342`: `aiter/configs/*.csv` + `aiter/configs/model_configs/*.csv` (real
  headers/rows), `aiter/jit/core.py` (`AITER_CONFIGS.*_FILE`, `get_config_file`, `update_config_files`,
  `gfx_from_cu_num` backfill), `aiter/tuned_gemm.py` (`save_shapes`, 10-tuple index columns),
  `aiter/fused_moe.py` (runtime `gfx`-first fmoe key).
