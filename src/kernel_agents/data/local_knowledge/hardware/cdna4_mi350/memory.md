---
title: CDNA4 / MI350 (gfx950) — memory hierarchy & LDS
kind: hardware
gens: [gfx950]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350 (gfx950) — memory hierarchy & LDS

> Cross-gen LDS bank rules in [../shared/memory_model_lds_bank.md](../shared/memory_model_lds_bank.md);
> HBM/Fabric in [../shared/hbm_infinity_fabric.md](../shared/hbm_infinity_fabric.md). This file is the
> CDNA4-specific memory deltas vs MI300X.

## TL;DR
> Three memory changes drive CDNA4 kernel design: **LDS 64 KiB → 160 KiB** with **64 banks** and
> **256 B/clk** (vs 32 banks / 128 B/clk); **GLOBAL_LOAD_LDS widened to 128 b/lane**; and HBM3E at
> **288 GB / 8 TB/s**. Re-tune all 32-bank LDS swizzles for 64 banks and exploit the 4×-wider
> direct-to-LDS path.

## Concepts

### The ladder (MI350X/355X)
| Level | Capacity | Scope | Bandwidth | Delta vs MI300X |
|---|---|---|---|---|
| VGPR | 512 ×4 B/SIMD | wave | — | same |
| AGPR | ≤256 ×4 B/SIMD | wave | — | same |
| LDS | **160 KiB/CU**, **64 banks** | workgroup | **256 B/clk** | **2.5× cap, 2× banks, 2× BW** |
| L1 vector | 32 KiB/CU, 128 B line | CU | — | same |
| L2 | per-XCD | per XCD | — | same model (per-XCD) |
| Infinity Cache | 256 MiB | device | — | same size |
| HBM3E | **288 GB** | device | **8.0 TB/s** | +96 GB, +2.7 TB/s |

FP16 roofline ridge ≈ 8 TB/s vs 2.5 PF ≈ **312 FLOP/byte** — more kernels are bandwidth-bound relative
to the bigger matrix core, so byte-cutting matters even more than on MI300X.

### LDS: 160 KiB, 64 banks
- Configured as **64 banks × 640 entries × 4 B = 163840 B/CU**. Bank index = **`(addr/4) mod 64`**.
- **Re-tune padding**: a swizzle tuned for the 32-bank MI300X LDS does **not** guarantee
  conflict-freedom on 64 banks — re-run the matrix calculator's register map and re-pad.
- **256 B/clk** read BW (2× CDNA3's 128 B/clk) — feeds the 2×-throughput matrix core.
- **Read-with-transpose `ds` loads**: CDNA4 can transpose on the LDS read, removing an explicit B
  transpose in GEMM.
- LDS allocation granularity for gfx950 = **320 DWORD blocks** (vs 128 at 64 KiB).

### GLOBAL_LOAD_LDS / buffer_load ... lds: 128 b/lane
CDNA4 widens direct global→LDS from CDNA3's 32 b/lane to **up to 128 b/lane** — accepted DWORD counts
are **1, 2, 4, 12, 16** (vs 1/2/4 on CDNA3), i.e. 96-bit and 128-bit variants added. This moves a full
16 B/lane straight into LDS in one instruction, eliminating `ds_write`, staging VGPRs, and copy index
math entirely — a bigger occupancy win than on MI300X. Pair with double-buffering and `s_waitcnt`.

### Occupancy impact of bigger LDS
```
occ_lds (workgroups/CU) = floor(163840 / L)   # vs floor(65536 / L) on MI300X
```
The 2.5× LDS budget allows **larger tiles or deeper buffering** at the same occupancy — attention/softmax
kernels that were LDS-limited on MI300X gain headroom. (VGPR limit `floor(512/N)` unchanged.) See
[isa_notes.md](isa_notes.md) and [../cdna3_mi300/occupancy.md](../cdna3_mi300/occupancy.md) for the
full model.

## The levers
1. **Re-tune LDS padding for 64 banks** when porting from MI300X.
2. **Use 128-bit GLOBAL_LOAD_LDS** (`global_load_lds` / `buffer_load ... lds` at 16 DWORD) — top GEMM
   occupancy win.
3. **Read-with-transpose `ds`** to feed the B operand without a transpose pass.
4. **Spend the bigger LDS** on larger tiles / more prefetch stages.
5. **Cut bytes** harder — the ridge moved right (312 FLOP/B); decode/norm/attention are even more
   bandwidth-bound.
6. **Coalesce to 128 B `dwordx4`**, keep reuse XCD-local, 8-multiple tiles (8 XCDs) — unchanged rules.

## Pitfalls
- **Reusing MI300X 32-bank swizzles** → conflicts on 64 banks.
- **Sticking to 32-bit direct-to-LDS** → leaves the 128-bit width on the table.
- **Assuming a global L2** — still per-XCD.

## Verify
- ISA dump for `.lds_size` and that `global_load_lds` emits the 12/16-DWORD form.
- `rocprof-compute` LDS panel (bank-conflict rate over 64 banks, 256 B/clk utilization) and HBM BW.
