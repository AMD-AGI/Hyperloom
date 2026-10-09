---
title: autotuning — which tuner actually reaches the live dispatch
kind: lever
lever: autotune
gens: [gfx950]
bottleneck: any — this is the last lever, after the structure is right
updated: 2026-08-28
---

# Autotuning

## Route here when
- The kernel's structure is already right (correct class, no spills, clean LDS, filled grid) and you
  are searching the **residual** parameter space.
- You need a real speedup on a **deployed sglang/vLLM server**, not on a microbenchmark.

**Apply this lever last.** Autotuning a structurally wrong kernel just finds the best of a bad family.

## The one fact that decides everything

**Only aiter's per-shape config DB engages the live sglang/vLLM GEMM path.**

| Tier | Tool | Reaches live serving? |
|---|---|---|
| Author-time kernel | Triton `@triton.autotune` over configs | only if that kernel *is* the live dispatch |
| Library offline | `hipblaslt-bench` / TensileLite, PyTorch `TunableOp` (`HIPBLASLT_TUNING_FILE`) | **No** — aiter bypasses these hooks |
| **Live dispatch** | **aiter per-shape DB** | **Yes — this is the lever** |

Tuning the wrong tier is the most expensive failure mode here: hours of search, a real measured win in
the microbenchmark, and zero change on the server.

## The recipe

### 1. Capture real shapes from a warm server
```bash
AITER_TUNE_GEMM=1 <launch server and drive real traffic>
# shapes append to aiter/configs/*_untuned_gemm.csv
```
**Bias, scale and dtype must match the live calls exactly.** A synthetic capture with `bias=true`
against a live path that uses `bias=false` is the classic 0%-engagement bug.

### 2. Race candidates
The primary tuner is the multi-backend one; `gradlib` is now the hipBLASLt-only path.
```bash
python csrc/gemm_a16w16/gemm_a16w16_tune.py --indtype bf16 --mp 8 \
       [--libtype all] [--with-hipblaslt]
```
- Races `{torch, hipblaslt, skinny, asm, triton, flydsl, opus}` per shape and writes the winner
  (`libtype` + `solidx`, plus `kernelName`/`splitK` where relevant).
- Each solution is gated on **`err_ratio < --errRatio`** (default **0.05**) against a reference.
- `--mp <ngpus>` parallelizes across visible GPUs.
- On OOM, set `CACHE_INVALIDATE_BUFFERS` to a small prime (11 / 7 / 3 / 1).

### 3. Deploy by env — never edit site-packages
```bash
EXTRA_ENV="AITER_CONFIG_GEMM_BF16=/tmp/tuned.csv AITER_LOG_TUNED_CONFIG=1" <launch server>
```
Multiple CSVs merge with `:`.

### 4. Prove engagement **before** believing any number
```bash
grep -c 'is tuned on cu_num' server.log     # must be > 0
```
Zero hits means the lookup missed and you measured noise.

### 5. Then A/B gate
Non-overlapping same-session repeats, outside the noise band (`measure_protocol.md`).

## The lookup key — where engagement silently dies

The dispatcher resolves a **10-tuple**, `gfx` first:

```
(gfx, cu_num, padded_M, N, K, bias, dtype, otype, scaleAB, bpreshuffle)
```

**One mismatched field ⇒ 100% lookup miss ⇒ 0 engagement**, with no error. The usual culprits:
`bias` tuned true / live false, a dtype string mismatch, or a CSV tuned on a different `gfx` or
`cu_num` (SKU change, or a partitioned GPU reporting fewer CUs).

`padded_M` is a bucketed M: the lookup tries the exact M, then padded granularities. That bucketing is
what makes tuning tractable — you do not need a row per batch size.

## Prune the search space before you start

- **Bucket M.** Live M varies per batch; tune a small bucketed set. Racing every M over ~1000+
  hipBLASLt solutions per shape is slow and can fork-storm the host.
- **Constrain to gfx950-good defaults first** so the search starts inside the good region:
  `mfma_16x16` (`matrix_instr_nonkdim=16`), 8-multiple tiles, ≥1024 workgroups, `OPTIMIZE_EPILOGUE=1`
  (`lever_mfma_sched.md`, `lever_xcd_locality.md`). For decode: small `BLOCK_M` + split-K
  (`lever_grid_sizing.md`).
- **Kill obviously-bad configs early** from the ISA dump — anything that spills is not a candidate
  (`lever_occupancy.md`).

## Caching and re-tuning

Commit the tuned CSV per **(model, dtype, GPU SKU)** and load it by env var. Re-tune when any of these
change: shapes, dtype, ROCm/aiter version, **or the GPU SKU / CU count** — the `cu_num` field is part
of the key, so a CSV tuned on a different partition mode will not match.

For Triton `@autotune`, persist the cache keyed on shape; the first call per shape pays the search.

## Verify

| Check | How | Pass |
|---|---|---|
| **Engagement** | `grep -c 'is tuned on cu_num' server.log` | **> 0** — check this first, always |
| Correctness | tuner's own gate | every accepted solution `err_ratio < 0.05` |
| Real delta | e2e A/B, non-overlapping repeats | outside the noise band (`measure_protocol.md`) |
| Sanity | `rocprof-compute` | tuned config shows higher MFMA busy / closer to the roof |

## Expected magnitude
A well-captured, well-engaged GEMM DB tune is typically a **low single-digit percentage e2e** — real,
reproducible, and cheap, but not transformative. Reference: **+2.23% e2e** on Qwen3.5-27B / sglang
0.5.11 / aiter (1548.9 → 1583.5 tok/s, 5 non-overlapping reps, 246 engagement hits). If you are seeing
a 2× "win" from a DB tune, suspect the measurement.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Tuned CSV deployed, nothing changed | 0 engagement | `grep 'is tuned on cu_num'`; check all 10 key fields, especially `bias` |
| Microbench faster, server unchanged | tuned the wrong tier | only the aiter DB reaches the live path |
| `TunableOp` file ignored | aiter bypasses the hipBLASLt hook entirely | tune through aiter |
| Was engaged, now isn't | ROCm/aiter bump, or SKU / partition-mode change (`cu_num`) | re-tune |
| Accepted a faster-but-wrong kernel | no oracle gate | enforce `err_ratio` (`lever_numerics.md`) |
| Search never finishes | tuning every M against every solution | bucket M, constrain knobs, `--mp` |

## Deeper
`framework/aiter/overall/tuning_db.md` (the full capture→tune→deploy workflow, on-box commands) ·
`framework/aiter/overall/config_files_and_merge.md` (CSV schema, `AITER_CONFIG_*` merge semantics) ·
`framework/aiter/overall/dispatch_and_rebind.md` (how a call resolves to a backend; `get_padded_m`
bucketing is covered in `tuning_db.md`) ·
`measure_protocol.md` (the A/B discipline this depends on)
