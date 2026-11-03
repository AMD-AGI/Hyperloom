---
title: aiter per-shape DB tuning — the primary aiter optimization lever
kind: language
gens: [gfx942, gfx950, gfx1250]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:aiter/tuned_gemm.py
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:csrc/gemm_a16w16/gemm_a16w16_tune.py
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:gradlib/gradlib/gemm_tuner.py
  - ROCm/aiter@b467ce3425cceeafe4f5587212d36df46feeb265:aiter/jit/core.py
  - https://github.com/ROCm/aiter
---

# aiter per-shape DB tuning

## TL;DR
The main way to make aiter faster on a live workload is **not** editing kernel source — it is **tuning
aiter's per-shape dispatch DB**: capture the real shapes from the running server, race the candidate
library kernels per shape (gradlib), keep the winners in a CSV, and deploy the CSV by env. On sglang/vLLM
this is the **only** GEMM lever that engages the live path (`aiter.tuned_gemm.gemm_a16w16` / `tgemm.mm`);
PyTorch TunableOp and `HIPBLASLT_TUNING_FILE` hook a dispatch layer aiter bypasses and do nothing.
Measured: **+2.23% e2e** on Qwen3.5-27B / MI300X from DB tuning alone (246 engagement hits).

## The DB lookup key (every field must match the live call)
`aiter/tuned_gemm.py` (`get_GEMM_A16W16_config`) resolves a **10-tuple** against the CSV — note the
leading **`gfx`** field (the CSV header now starts `gfx,cu_num,M,N,K,...`):
```
(gfx, cu_num, padded_M, N, K, bias, dtype, otype, scaleAB, bpreshuffle)
```
- `gfx` = `get_gfx()` (e.g. `gfx942`/`gfx950`/`gfx1250`); `cu_num` = `get_cu_num()`; `bias` = `bias is not
  None`; `scaleAB` = `scale_a/scale_b is not None`; `bpreshuffle` = `B.is_shuffled`.
- `padded_M`: lookup tries exact M, then `get_padded_m` `gl=0` (fine: round up to 16/32/64/128 by range),
  then `gl=1` (coarse: `nextPow2(M)`), so one tuned bucket covers a *range* of live M. `get_padded_m` is
  exposed in Python from `aiter/ops/gemm_op_common.py` (compiled op `module_gemm_common`/`getPaddedM`; C++
  impl still in `csrc/py_itfs_cu/gemm_common.cu`).
- A wrong field (classically **`bias`**, now also **`gfx`**/`cu_num`) → every lookup misses → the tuned CSV
  silently does nothing. A legacy CSV without a `gfx` column is backfilled from `cu_num` at merge time
  (`gfx_from_cu_num`: 256→gfx950, 80/304→gfx942) — but only when merged via `AITER_CONFIGS`, so regenerate
  rather than rely on it.

## The capture → tune → deploy → gate recipe
1. **Capture live** (`AITER_TUNE_GEMM=1`): warm the server with real traffic; every `gemm_a16w16` call
   appends its true shape (incl. real `bias`) to `aiter/configs/bf16_untuned_gemm.csv`. **Never guess the
   schema from `meta.json`** — the bias/M there can be wrong for the live path.
2. **Bucket-reduce & order**: `get_padded_m` collapses M to unique buckets; sort **FLOPs-DESC** so gradlib
   (processes input order, writes incrementally) tunes the GPU-dominant large-M prefill shapes FIRST
   (it otherwise tunes M-ascending = decode-first = worst ROI). Partial DBs never regress (uncovered → default).
3. **Tune** — the primary multi-backend tuner is now `csrc/gemm_a16w16/gemm_a16w16_tune.py` (races
   `asm`/`opus`/`flydsl`/`triton`/`skinny`/`torch`; `--libtype` picks the subset, default `all`). hipBLASLt
   is **opt-in** here via `--with-hipblaslt` (which calls into gradlib) and is also available standalone as
   the dedicated hipBLASLt path `gradlib/gradlib/gemm_tuner.py`. Each solution is gated on
   `err_ratio < --errRatio` (default 0.05) and the winner `libtype`+`solidx`(+`kernelName`/`splitK`) is
   written. CLI (shared base-tuner flags): `-i/--untune_file`, `-o/--tune_file`, `--mp`, `--errRatio`,
   `--indtype {f32,f16,bf16,fp8}`, `--all_bias`, plus `--libtype` and `--with-hipblaslt`.
4. **Deploy by env** (reversible, no code edit): `AITER_CONFIG_GEMM_BF16=<csv>` (`:`-joined merges
   multiple), `AITER_LOG_TUNED_CONFIG=1`.
5. **Prove engagement, then A/B gate**: `grep -c 'is tuned on cu_num' server.log` must be **> 0** before
   believing any delta; then same-session 2-launch A/B (accept iff `delta > 0.5% AND cand_min > ref_max AND
   parity holds`).

## Worked example (bf16 dense GEMM)
```bash
# 1) capture live shapes (real traffic)
EXTRA_ENV="AITER_TUNE_GEMM=1 SGLANG_USE_AITER=1" <launch sglang server>   # -> bf16_untuned_gemm.csv
# 2+3) tune across all GPUs, accuracy-gated (multi-backend: asm/opus/flydsl/triton/skinny/torch)
python csrc/gemm_a16w16/gemm_a16w16_tune.py --indtype bf16 --mp 8 \
    -i aiter/configs/bf16_untuned_gemm.csv -o /tmp/tuned.csv \
    --libtype all --errRatio 0.05 --with-hipblaslt   # --with-hipblaslt also races hipBLASLt (via gradlib)
# 4) deploy + 5) prove engagement
EXTRA_ENV="AITER_CONFIG_GEMM_BF16=/tmp/tuned.csv AITER_LOG_TUNED_CONFIG=1" <launch server>
grep -c 'is tuned on cu_num' server.log     # must be > 0 (win run: 246)
```

## Non-GEMM tuning
The same "capture-shape → codegen/tune configs → install CSV" pattern applies to ASM/CK ops via
`hsa/codegen.py -m {pa,fmha,mla}` and each `csrc/ck_*/gen_instances.py` + `--gen-tune` sweep — see
[jit_and_build.md](jit_and_build.md). GEMM is the highest-leverage and best-tooled path (gradlib).

## Pitfalls & anti-patterns
- **bias mismatch = 0 engagement** (the trap that produced a false "GEMM tuning has no benefit"). Capture
  bias live; never synthesize `bias=True`.
- **TunableOp / `HIPBLASLT_TUNING_FILE` are dead ends on sglang** (hook PyTorch `addmm`; aiter calls
  `hipb_mm` directly). Measured −0.11%/−0.30% — wrong lever.
- **Fork-storm**: the hipBLASLt path (`--with-hipblaslt` / gradlib) races ~1365 solutions/shape and across
  big prefill spawns hundreds of `rocm_agent_enumerator` procs → host OOM / corrupted e2e timing.
  Bucket-reduce big M, cap `--mp`, restrict `--libtype` while iterating; serialize heavy nested tunes.
- **The CSV is build-locked** (`solidx`/`kernelName` are ROCm/hipBLASLt/aiter-specific) — regenerate on
  any upgrade; never ship a hand-copied CSV as portable.
- **`flydsl` rows silently drop** if FlyDSL isn't installed (`is_flydsl_available()` false) → falls to next
  granularity/default. Verify FlyDSL before trusting flydsl rows.
- **`SGLANG_USE_AITER=1` is required** to route to `tgemm.mm` at all (else `UnquantizedLinearMethod` runs
  `F.linear`/hipBLASLt default and the DB is never consulted).

## Sources
- Dispatch + 10-tuple key + `get_padded_m`: `ROCm/aiter@b467ce342:aiter/tuned_gemm.py`,
  `aiter/ops/gemm_op_common.py`, `csrc/py_itfs_cu/gemm_common.cu`.
- Multi-backend tuner (race, err gate, `--libtype`, `--with-hipblaslt`): `csrc/gemm_a16w16/gemm_a16w16_tune.py`
  (`ALL_LIBTYPES`, docstring). hipBLASLt-only tuner: `gradlib/gradlib/gemm_tuner.py`, `GemmTuner.py`.
- Config resolve/merge + `gfx` backfill: `aiter/jit/core.py` (`AITER_CONFIGS`, `get_config_file`,
  `update_config_files`).
