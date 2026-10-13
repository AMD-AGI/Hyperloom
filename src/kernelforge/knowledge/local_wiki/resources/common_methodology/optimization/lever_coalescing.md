---
title: coalescing and vectorization — 128-bit loads, alignment, lane mapping
kind: lever
lever: coalescing
gens: [gfx950]
bottleneck: bandwidth-bound
updated: 2026-08-28
---

# Coalescing and vectorization

## Route here when
- `lever_bottleneck_class.md` said **bandwidth-bound** (HBM near peak, MFMA idle).
- Measured HBM bandwidth is well below the achievable ceiling on a kernel that should be streaming.
- The kernel is a norm, elementwise, cast, copy, decode GEMV, or KV read.

**This is the cheapest large win on a memory-bound kernel** — usually a few lines of change. Do it
before anything more invasive.

## gfx950 constants

| Fact | Value |
|---|---|
| Wavefront | **64 lanes** — one memory instruction issues 64 addresses |
| Widest load/store | `global_load_dwordx4` = **128-bit / 16 B per lane** |
| Alignment for 128-bit | address must be **16-byte aligned** |
| Cache line | **128 B** |
| HBM3E | 288 GB @ **8.0 TB/s** peak (achievable is below this — measure it) |
| FP16 roofline ridge | ≈ **312 FLOP/byte** |

16 B/lane × 64 lanes = **1024 B per instruction**, exactly 8 cache lines. That is the target shape for
every streaming access.

## The mechanism

The hardware merges lanes that fall in the same 128 B cache line into one transaction. Two independent
things can go wrong, and they need different fixes:

- **Narrow access** — the compiler emitted `dword` (4 B) instead of `dwordx4` (16 B). 4× the
  instructions for the same bytes.
- **Scattered access** — the 64 lanes touch 64 different cache lines. Up to 64 transactions where
  one wave should have taken 8.

A kernel can suffer either or both. Check the ISA for the first, the transaction counters for the second.

## What to change, in order

### 1. Make the access 128-bit wide
- **HIP**: load/store through `float4`, `int4`, or a packed 8×bf16 / 16×fp8 vector type so the
  compiler emits `*_dwordx4`.
- **Triton**: contiguous blocks with the right `BLOCK` divisibility auto-vectorize — but the compiler
  must be able to *prove* alignment. Declare divisibility hints; without them it falls back to narrow.
- **LDS too**: `ds_read_b128` / `ds_write_b128` on the staging path (`lever_lds_banks.md`).

### 2. Align base pointers and strides
Pad leading dimensions to a 16-byte multiple. **An odd row stride breaks vectorization on every
single row** — this is a common silent regression when a tensor is sliced or a head-dim is not a
power of two.

### 3. Fix the lane mapping
Index so **lane `i` reads element `base + i`** — the innermost dimension runs along the wave. If you
need the transposed order, do the transpose **in LDS**, not with strided global reads: one coalesced
read into LDS plus a swizzled read out beats 64 scattered global transactions by a wide margin.

### 4. Grid-stride loops for elementwise / reductions
```cpp
for (size_t i = gid; i < N; i += gridDim.x * blockDim.x) { ... }
```
Each step stays contiguous per wave, and the kernel scales to any `N` with a fixed, occupancy-tuned
grid instead of a size-dependent launch (`lever_grid_sizing.md`).

### 5. Use `buffer_*` for bounds checking
`buffer_load` / `buffer_store` with a descriptor gives hardware out-of-bounds handling — cheaper than
branchy guards in a tiled loop, and it does not break vectorization the way a predicated `if` can.

## Coalescing is not bank conflicts

Two different axes, routinely confused:

| | Coalescing | Bank conflicts |
|---|---|---|
| Memory | **global** (HBM/L2) | **LDS** |
| Granularity | 128 B cache line, across 64 lanes | 4 B bank, within a half-wave |
| Fix | wide + contiguous + aligned | pad or XOR swizzle over **64 banks** |
| Card | this one | `lever_lds_banks.md` |

A kernel can be perfectly coalesced in global and badly conflicted in LDS, or the reverse.

## Verify

| Check | How | Pass |
|---|---|---|
| Width | ISA dump: count `global_load_dwordx4` vs `global_load_dword` in the hot loop | wide forms dominate |
| Transactions | `rocprof-compute` memory chart: fetch size / transaction efficiency | near 128 B per transaction |
| Bandwidth | achieved HBM BW vs the empirical roof from `measure_roofline.md` | close to the measured ceiling, not the 8 TB/s datasheet number |
| A/B | aligned vs deliberately misaligned base pointer | transaction count and BW should visibly jump |

## Expected magnitude
Narrow → 128-bit on a streaming kernel: **up to 4×** fewer memory instructions, commonly **1.5–3×**
end-to-end. Fixing a fully scattered access pattern: can exceed **5×**. If you see less than ~20%,
the kernel probably was not actually bandwidth-bound — re-run `lever_bottleneck_class.md`.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Wide loads in source, narrow in ISA | compiler cannot prove 16 B alignment | add divisibility hints; align the base pointer and stride |
| Column-major read over a row-major tensor | one transaction per lane | stage through LDS, transpose there |
| Vectorized but still slow | scattered *across* lanes, not narrow | check transaction count, fix the lane mapping |
| Tiny tensors got slower | over-wide loads waste tail lanes on predication | match the width to the data |
| Ported CUDA coalescing math | CDNA is **wave64** — the window is 64 lanes, not 32 | re-derive |

## Deeper
`hardware/mi350_memory.md` (the memory ladder) ·
`hardware/mi350_memory.md` (bandwidth ladder, why bytes win) ·
`lever_lds_banks.md` · `lever_xcd_locality.md` (the next lever once access is clean) ·
`lever_fusion.md` (removing the traffic entirely)
