---
title: prefetch and software pipelining — direct-to-LDS, stages, overlap
kind: lever
lever: prefetch
gens: [gfx950]
bottleneck: latency-bound; also feeds compute-bound GEMM
updated: 2026-08-28
---

# Prefetch and software pipelining

## Route here when
- Latency-bound: both roofs far, occupancy acceptable, high stall cycles.
- Compute-bound GEMM where MFMA busy is low and the stalls are on `s_waitcnt vmcnt` — the core is
  waiting for operands.
- The kernel does `global_load` → VGPR → `ds_write` (the slow staging path).

**Read `lever_loop_form.md` first if** the loop is a `while`, a gather, or anything with a
data-dependent trip count. Both mechanisms on this page are **silently disabled** by such a loop, and
a `num_stages` sweep will read flat and look like "pipelining doesn't help here."

## gfx950 constants

| Fact | Value |
|---|---|
| Direct global→LDS | `global_load_lds` / `buffer_load ... lds`, DWORD counts **1 / 2 / 4 / 12 / 16** → up to **128 b/lane** |
| LDS budget for stages | **160 KiB/CU** (320-DWORD alloc granule) |
| LDS read BW | 256 B/clk |
| Read-with-transpose `ds` | available — free B-operand transpose |
| Wait counters | `vmcnt` (VMEM), `lgkmcnt` (LDS/SMEM); **count-based, not fences** |

`s_waitcnt vmcnt(N)` means "wait until **≤ N outstanding**", not "wait N instructions". That is what
makes deep overlap expressible: wait only for the specific loads you need now.

## Mechanism 1: direct-to-LDS (skip the register file)

A load whose destination is **LDS, not a VGPR**. CDNA's equivalent of `cp.async`.

Two wins at once:
- **Frees staging registers** → higher occupancy (`lever_occupancy.md`). This is usually the bigger
  effect on tiled GEMM.
- **Overlaps with compute** — the load is in flight while MFMAs run.

**gfx950 widened this 4×** (32 → 128 b/lane). If you are emitting the 1/2/4-DWORD forms, you are
leaving the width on the table. Target the **16-DWORD** form.

## Mechanism 2: software pipelining

An `S`-stage K-loop keeps `S` tiles in flight:

1. **Prologue** — issue `global_load_lds` for tiles `0 .. S-1`.
2. **Steady state** at step `k` — `v_mfma` on tile `k` (from LDS) **while** the load for tile `k+S` is
   in flight **while** `ds_read` for `k+1` overlaps.
3. **Epilogue** — drain the remaining MFMAs.

**Precondition:** a `for`/`tl.range` loop with a loop-invariant bound and load addresses affine in the
induction variable. A *runtime* bound is fine — `tl.cdiv(K, BLOCK_K)` with `K` a kernel argument is the
normal GEMM K-loop. A bound read out of a tensor is not.

## What to change, in order

### 1. Switch the staging path to direct-to-LDS
Verify in the ISA that you get `global_load_lds` / `buffer_load ... lds` at 12 or 16 DWORD, not
`global_load_dwordx4` → `ds_write_b128`.

### 2. Set the stage count
`num_stages` (Triton) or explicit multi-buffering (CK / HIP / asm).

| Stages | When |
|---|---|
| 1 | fused attention; LDS-tight kernels |
| 2 | classic double-buffer — safe default |
| **3–4** | **K-deep prefill GEMM on gfx950** — the 160 KiB budget affords it |

Each stage costs another LDS tile. On a 64 KiB part 2 was often the ceiling; here 3–4 is normal.
**Re-tune this when porting** — an inherited `num_stages=2` is likely leaving overlap unused.

### 3. Tune the prefetch distance
Far enough ahead to cover HBM latency, not so far that LDS overflows and occupancy collapses. The
`num_stages` sweep is the practical handle: latency dips, then rises when LDS runs out.

### 4. Overlap `ds_read` against `ds_write`
Schedule the consumer read of the current tile against the producer write of the next, so the LDS port
stays busy — without creating conflicts on **64 banks** (`lever_lds_banks.md`).

### 5. Use read-with-transpose
Feeds the MFMA B operand without an explicit transpose pass. Free on gfx950.

## Verify

| Check | How | Pass |
|---|---|---|
| Direct path emitted | ISA: `global_load_lds` / `buffer_load ... lds` | present, at 12/16 DWORD |
| Pipeline exists | ISA: unrolled K-loop, `s_waitcnt` placed between stages not before every load | overlap visible |
| Overlap achieved | `rocprof-compute` | high MFMA busy **and** HBM active, low MFMA stall |
| Registers freed | ISA `.vgpr_count` before/after | drops (that is the occupancy win) |
| A/B | sweep `num_stages ∈ {1,2,3,4}` | curve **dips then rises**. A **flat** curve is not a tuning result — it says the loop is never pipelined → `lever_loop_form.md` |

The flat-curve diagnostic is the most useful line on this page. Treat it as a signal, not a shrug.

## Expected magnitude
Plain staging → direct-to-LDS on a tiled GEMM: frees roughly **~100 VGPR/wave** in the reference case
and is often worth **>20%** through the occupancy it unlocks. Adding stages 2 → 4 on a K-deep prefill
GEMM: **10–25%**. On an unpipelinable loop: **0%** — fix the loop first.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `num_stages` sweep completely flat | loop cannot be pipelined | `lever_loop_form.md` — fix the form, then re-sweep |
| Raising stages made it slower | LDS overflow dropped occupancy below the latency-hiding threshold | back off; recompute `occ_lds = floor(163840/L)` |
| Loading through VGPRs | plain `global_load` → `ds_write` | switch to `global_load_lds` |
| Emitting 4-DWORD direct loads | inherited from a 32-b/lane part | request the 16-DWORD form |
| Race / wrong results between stages | missing barrier discipline on the shared LDS buffer | audit `s_waitcnt lgkmcnt` and barriers per stage |
| Prefetch introduced `ds_write` conflicts | swizzle not re-derived for 64 banks | `lever_lds_banks.md` |

## Deeper
`hardware/mi350_lds.md` (direct-to-LDS widths, LDS geometry) ·
`lever_loop_form.md` (**the precondition**) · `lever_lds_banks.md` · `lever_mfma_sched.md` ·
`lever_occupancy.md`
