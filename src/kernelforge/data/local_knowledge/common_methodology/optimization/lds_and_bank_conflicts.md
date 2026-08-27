---
title: LDS sizing and bank-conflict avoidance
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
---

# LDS sizing and bank conflicts

## TL;DR
The Local Data Share (LDS) is the on-CU scratchpad that stages GEMM/attention operands for the matrix
cores. Capacity: **64 KB/CU on CDNA3 (MI300X)**, **160 KB/CU on CDNA4 (MI350X/MI355X)** with ~2× LDS
bandwidth. LDS is split into **32 banks of 4 bytes** (a 128-byte row); two lanes in a wave hitting the
same bank on different addresses serialize (**bank conflict**). The two fixes are **padding** (add a
stride so consecutive rows land in different banks) and **XOR swizzle** (permute the column index so
`ds_read`/`ds_write` are conflict-free). LDS capacity directly bounds tile/head-dim size, so the
CDNA3→CDNA4 jump materially relaxes flash-attention head-dim limits. See
`[[hardware/shared/memory_model_lds_bank.md]]` and `[[hardware/cdna3_mi300/memory_hierarchy.md]]`.

## Concepts (the hardware)
- **Capacity**: CDNA3 = 64 KB/CU; CDNA4 = 160 KB/CU (≈2.5×) plus a direct **L1→LDS load path** that
  removes the intermediate-register hop (see `[[optimization/memory_pipelining.md]]`,
  `[[hardware/cdna4_mi350/memory.md]]`).
- **Banks**: 32 banks × 4 B = a 128-B-wide structure. Model LDS as a 2-D array `[rows][32]`; address
  `a` maps to bank `(a/4) % 32`.
- **Conflict**: when multiple lanes of a wave access *different* 4-B words in the *same* bank in one
  instruction, accesses serialize (N-way conflict ⇒ N× the `ds_*` cost). Broadcast (same address) is free.
- **Why it bites GEMM**: storing a tile column-major / reading row-major (or vice-versa for the MFMA
  operand layout) makes lanes stride by the row length, which is frequently a multiple of 32 words ⇒
  worst-case 32-way conflict.

## The levers
### 1. Padding
Add an extra element to each row so the row stride is **not** a multiple of 32 words:
`__shared__ float tile[BM][BK + PAD];` with `PAD` chosen so `(BK+PAD) % 32 != 0` (commonly `+1` for
f32, `+4`/`+8` for 16-/8-bit to also keep 128-bit vector alignment, see
`[[optimization/vectorization_and_coalescing.md]]`). Cost: a little wasted LDS — tight under 64 KB.

### 2. XOR swizzle (preferred for GEMM)
Permute the column index with the row: `col' = col ^ (row & mask)`. This is the CK-Tile approach: the
swizzle is chosen so every lane in a `ds_read`/`ds_write` lands in a distinct bank for the MFMA operand
access pattern, giving **0 conflicts with 0 wasted LDS**. CK-Tile and the AMD blog show XOR swizzle
eliminating LDS bank conflicts in the GEMM operand staging. See
`[[languages/composable_kernel/ck_tile.md]]` and `[[languages/composable_kernel/knobs.md]]`.

### 3. Double-buffering (ping-pong)
Allocate two LDS tiles and alternate: while the matrix core consumes buffer A, the loader writes
buffer B. This overlaps `ds_read` (to MFMA) with the next `global_load`/`ds_write` and is the core of
software pipelining (`[[optimization/memory_pipelining.md]]`, triton `num_stages≥2`). Cost: 2× the LDS
footprint — feasible for many tiles on CDNA3's 64 KB, much easier on CDNA4's 160 KB.

### 4. Use the right operand layout / `b_preshuffle`
Pre-permuting B into the MFMA-native layout (aiter `bpreshuffle`, `[[operators/dense_gemm/tuning.md]]`)
moves the shuffle off the hot path so the LDS read pattern is already conflict-free.

## Sizing budget (capacity → tile)
LDS bytes per stage ≈ `(BM·BK + BK·BN) · sizeof(dtype) · num_stages · (2 if double-buffer)`.
On CDNA3 (64 KB) a bf16 256×64 + 64×128 double-buffered tile is already tight; CDNA4 (160 KB) lets you
either grow the tile, add stages, or fit larger attention head dims. LDS use also competes with
register-driven occupancy (`[[optimization/occupancy_and_registers.md]]`).

## Pitfalls
- Row stride a multiple of 32 4-B words ⇒ silent 32-way conflict; the kernel "works" but is 10–30× slow
  on the staging step.
- Padding that breaks 128-bit alignment ⇒ you trade bank conflicts for non-vectorized `ds_read`.
- Porting an H100 kernel 1:1: H100 has up to ~228 KB programmable shared mem; an MI300X port at 64 KB
  must shrink tiles/head-dim. Re-tune for CDNA4 (160 KB) separately.
- Forgetting double-buffer doubles LDS — can drop occupancy below the latency-hiding threshold.

## Verify
- Omniperf / `rocprof`: LDS bank-conflict counters (`SQ_LDS_BANK_CONFLICT` family) and `ds_*` stall
  cycles — target near-zero conflicts (`[[profiling/]]`).
- ISA dump: confirm `ds_read_b128`/`ds_write_b128` (vectorized) and the swizzle math.
- A/B: same kernel with vs without padding/swizzle; conflict counter should collapse.
