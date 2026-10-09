---
title: LDS — 64-bank conflicts, swizzle, and the 160 KiB budget
kind: lever
lever: lds_banks
gens: [gfx950]
bottleneck: LDS-bound
updated: 2026-08-28
---

# LDS sizing and bank conflicts

## Route here when
- `ds_*` stall cycles are high, or the bank-conflict counter is non-zero.
- MFMA busy is low but there are no spills and the K-loop has multiple accumulators — the core is
  starving on its LDS reads.
- You are **porting a kernel from a 32-bank part** (any pre-CDNA4 AMD GPU). Assume the swizzle is
  wrong until measured; see the warning below.

## gfx950 constants — read this before reusing any padding formula

| Property | gfx950 | Previous gens |
|---|---|---|
| Capacity | **160 KiB/CU** | 64 KiB |
| Banks | **64 × 4 B** (640 entries each) | 32 × 4 B |
| Bank index | **`(byte_addr / 4) mod 64`** | `mod 32` |
| Read bandwidth | **256 B/clk** | 128 B/clk |
| Allocation granule | **320 DWORD** | 128 DWORD |
| Direct global→LDS | **1/2/4/12/16 DWORD** (up to 128 b/lane) | 1/2/4 (32 b/lane) |
| Read-with-transpose `ds` | **yes** | no |

**The 32→64 bank change is the single most likely reason an inherited kernel is slow here.** A padding
or XOR swizzle tuned for 32 banks does not guarantee conflict-freedom on 64. Re-derive it; do not
port it.

## The mechanism

A wave issues LDS in **half-waves of 32 lanes**. Within a half-wave:
- Lanes hitting the **same address** in a bank → **broadcast, free**.
- Lanes hitting **different addresses** in the **same bank** → **N-way conflict**, serialized into
  N cycles.

Why it bites GEMM: staging a tile one way and reading it the other (row-major store, column-major read
for the MFMA operand layout) makes lanes stride by the row length. When that stride is a multiple of
the bank count, every lane in a column lands in one bank.

```cpp
__shared__ float tile[64][64];   // BAD: stride 64 words == 64 banks -> full-width conflict
__shared__ float tile[64][65];   // GOOD: +1 spreads the column across all banks
float v = tile[k][threadIdx.x];
```

What matters is **`(byte_stride / 4) mod 64`**, not the element count — the same trap fires at stride 32
for a `[32][32]` tile of 8-byte elements.

## What to change, in order

### 1. Pad the leading dimension
Choose `PAD` so `((BK+PAD) · sizeof(dtype) / 4) mod 64 != 0`. Commonly `+1` for f32, `+4`/`+8` for
16-/8-bit — but the second constraint is **keep 16-byte alignment** so `ds_read_b128` still fires. A pad
that fixes conflicts and breaks vectorization is a net loss. Cost: a little wasted LDS, which the
160 KiB budget absorbs easily now.

### 2. XOR swizzle (preferred for GEMM)
`col' = col ^ (row & mask)` — permute the column index by the row so every lane in a `ds_read`/`ds_write`
lands in a distinct bank for the MFMA operand pattern. **Zero conflicts, zero wasted LDS.** This is the
CK-Tile approach; CK and Triton generate it automatically. For hand-written kernels, mirror the
register map from `amd_matrix_instruction_calculator --get-register --A-matrix ...` rather than guessing.

### 3. Use the wide and transposing ops
- `ds_read_b128` / `ds_write_b128` — 16 B/lane per instruction; fewer issue slots, fewer conflict
  opportunities.
- **Read-with-transpose `ds` loads (gfx950)** — transpose the B operand on the LDS read and delete the
  explicit transpose pass entirely.

### 4. Double-buffer, and spend the surplus
While MFMA consumes buffer 0, stage buffer 1. On 160 KiB this is cheap: at typical GEMM tile sizes you
can afford **3–4 stages**, not the 2 that fit on a 64 KiB part. Pair with 128-bit `global_load_lds`
(`lever_prefetch.md`).

### 5. Pre-permute the operand off the hot path
`b_preshuffle` (aiter) stores B already in the MFMA-native layout, so the staging read is conflict-free
by construction. Moves the cost to a one-time weight transform.

## Sizing budget

```
LDS bytes/workgroup ≈ (BM·BK + BK·BN) · sizeof(dtype) · num_stages
occ_lds (workgroups/CU) = floor(163840 / LDS_bytes)     # remember the 320-DWORD granule rounds up
```

At 160 KiB the LDS term **rarely binds** — VGPR pressure is usually the occupancy limiter on gfx950
(`lever_occupancy.md`). So treat the budget as room to grow tiles and pipeline depth, not as a
constraint to fight.

## Verify

| Check | How | Pass |
|---|---|---|
| Conflicts | `rocprof-compute` LDS panel — bank-conflict rate over 64 banks | near zero |
| Vectorization survived | ISA: `ds_read_b128` / `ds_write_b128` in the hot loop | wide forms, not `b32` |
| Footprint | ISA `.lds_size` | matches your budget after the 320-DWORD rounding |
| Bandwidth headroom | LDS BW utilization vs the **256 B/clk** ceiling | not saturated |
| A/B | same kernel with and without the pad/swizzle | conflict counter collapses |

## Expected magnitude
Removing a full-width conflict on the staging path: the `ds_*` step goes **10–30× faster**, which
typically shows up as **1.3–2×** on the whole kernel if it was LDS-bound. Switching `b32`→`b128`:
**up to 4×** fewer issue slots on that path.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Ported kernel slow, "worked before" | 32-bank swizzle on 64 banks | re-derive the swizzle |
| Fixed conflicts, still slow | pad broke 16-byte alignment → scalar `ds_read` | re-pad to preserve alignment |
| Occupancy dropped after double-buffering | LDS × stages exceeded the budget | recompute `occ_lds`; on 160 KiB this is rare — check the 320-DWORD granule |
| Kernel "works" but 10–30× slow on staging | silent full-width conflict | check the counter, not the output |
| H100 port overflows LDS | H100 has ~228 KiB shared; gfx950 has 160 KiB | shrink tile or head-dim |

## Deeper
`hardware/mi350_lds.md` (the model and the 64-bank rules) ·
`hardware/mi350_lds.md` (LDS geometry, banks, direct-to-LDS widths) ·
`languages/ck/skills/optimize/ck_levers/ck_frontend_tile.md` (how CK generates the swizzle) ·
`lever_prefetch.md` · `lever_occupancy.md`
