---
title: MI350X — LDS, 64-bank conflicts, direct-to-LDS staging
kind: hardware
topic: lds
gens: [gfx950]
updated: 2026-08-28
---

# LDS — capacity, banks, staging

The on-CU scratchpad that stages GEMM/attention operands for the matrix cores. Three gfx950 numbers
drive kernel design here: **160 KiB**, **64 banks**, **128 b/lane direct-to-LDS**.

## Geometry

| Property | Value |
|---|---|
| Capacity | **160 KiB/CU** (64 banks × 640 entries × 4 B = 163840 B) |
| Banks | **64 × 4 B** |
| **Bank index** | **`(byte_address / 4) mod 64`** |
| Read bandwidth | **256 B/clk** |
| Allocation granule | **320 DWORD** |
| Direct global→LDS | **1 / 2 / 4 / 12 / 16 DWORD** → up to **128 b/lane** |
| Read-with-transpose `ds` | **yes** |

Versus earlier CDNA parts: 64 KiB → 160 KiB, 32 → 64 banks, 128 → 256 B/clk, 32 → 128 b/lane
direct-to-LDS, and read-with-transpose is new.

> **The 32 → 64 bank change is the most likely reason an inherited kernel is slow here.** A padding or
> XOR swizzle tuned for 32 banks does **not** guarantee conflict-freedom on 64. Re-derive it; do not
> port it.

## Bank conflicts

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

What matters is **`(byte_stride / 4) mod 64`**, not the element count — the same trap fires at stride
32 for a `[32][32]` tile of 8-byte elements.

Synchronization uses **`s_waitcnt lgkmcnt`** — count-based, not a fence. Wait only for the specific
outstanding LDS/scalar ops you need, which is what makes deep prefetch overlap expressible.

## The two fixes

### Padding
Choose `PAD` so `((BK+PAD) · sizeof(dtype) / 4) mod 64 != 0`. Commonly `+1` for f32, `+4`/`+8` for
16-/8-bit — but the second constraint is **keep 16-byte alignment** so `ds_read_b128` still fires. A
pad that removes conflicts and breaks vectorization is a net loss. Cost is a little wasted LDS, which
the 160 KiB budget absorbs easily.

### XOR swizzle (preferred for GEMM)
`col' = col ^ (row & mask)` — permute the column index by the row so every lane in a `ds_read`/
`ds_write` lands in a distinct bank for the MFMA operand pattern. **Zero conflicts, zero wasted LDS.**
This is the CK-Tile approach; CK and Triton generate it automatically. Hand-written kernels should
mirror the register map from `amd_matrix_instruction_calculator --get-register`, not guess.

## Direct global→LDS (128 b/lane)

A load whose destination is **LDS, not a VGPR** — CDNA's equivalent of `cp.async`:

```asm
global_load_lds_dwordx4 ...        ; 16 B/lane straight into LDS
buffer_load_dwordx4 ... lds        ; descriptor form
```

Two wins at once: it **frees staging registers** (the bigger effect on tiled GEMM — see
`mi350_execution.md`) and **overlaps with compute**. gfx950 accepts 1/2/4/**12/16** DWORD; the 96- and
128-bit forms are new. If you are emitting the 4-DWORD form you are leaving 4× on the table.

## Occupancy budget

```
LDS bytes/workgroup ≈ (BM·BK + BK·BN) · sizeof(dtype) · num_stages
occ_lds (workgroups/CU) = floor(163840 / LDS_bytes)     # 320-DWORD granule rounds up
```

At 160 KiB the LDS term **rarely binds** — VGPR pressure is usually the limiter
(`mi350_execution.md`). Treat the budget as **room to grow tiles and pipeline depth**, not a
constraint to fight: 3–4 double-buffer stages are affordable at typical GEMM tile sizes, where a
64 KiB part topped out at 2.

## What it means for kernels

1. **Re-derive padding/swizzle for 64 banks** when porting anything.
2. **Use `ds_read_b128` / `ds_write_b128`** — 16 B/lane per instruction, fewer issue slots and fewer
   conflict opportunities.
3. **Use read-with-transpose `ds`** to feed the MFMA B operand and delete the explicit transpose pass.
4. **Emit 128-bit `global_load_lds`** for tile staging; pair with double-buffering.
5. **Spend the surplus capacity** on bigger tiles or deeper pipelines.
6. **Pre-permute the operand off the hot path** (`b_preshuffle`) so the staging read is conflict-free
   by construction.

## Pitfalls
- **Reusing a 32-bank swizzle unchanged** → conflicts on 64 banks.
- **Padding that breaks 16-byte alignment** → you trade bank conflicts for scalar `ds_read`.
- **Scalar/uncoalesced LDS** — a strided `ds_read_b32` per element wastes 4× the issue slots vs `b128`.
- **Sticking to 32-bit direct-to-LDS** — leaves the 128-bit width unused.
- **Forgetting the 320-DWORD allocation granule** — a small `L` still rounds up.
- **Porting an H100 kernel 1:1** — H100 has ~228 KiB programmable shared memory; shrink the tile or
  head-dim here.

## Verify
- `rocprof-compute` LDS panel: **bank-conflict rate over 64 banks**, LDS BW utilization against the
  **256 B/clk** ceiling, % stalls on LDS.
- ISA dump: confirm `ds_read_b128`/`ds_write_b128` (not `b32`), the swizzle math, and that
  `global_load_lds` emits the 12/16-DWORD form.
- `.lds_size` / `-Rpass-analysis=kernel-resource-usage` for the per-kernel footprint after rounding.
- A/B the same kernel with and without the pad/swizzle — the conflict counter should collapse.

## Related
`mi350_execution.md` (the occupancy formula) · `mi350_matrix_core.md` (the lane map you swizzle for) ·
`mi350_memory.md` (the level above) ·
`common_methodology/optimization/lever_lds_banks.md` · `.../lever_prefetch.md`
