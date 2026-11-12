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
> CDNA4-specific memory deltas vs MI300X. Numbers in brackets are the sources at the end.

## TL;DR
> Three memory changes drive CDNA4 kernel design: **LDS 64 KiB → 160 KiB** with **64 banks** [1][4] and
> **256 B/clk** (vs 32 banks / 128 B/clk) [2]; **direct global→LDS loads widened to 128 b/lane** [1][4];
> and HBM3E at **288 GB / 8 TB/s** [2]. Re-tune all 32-bank LDS swizzles for 64 banks and use the
> 4×-wider direct-to-LDS path (both follow from [1]).

## Concepts

### The ladder (MI350X/355X)
| Level | Capacity | Scope | Bandwidth | Delta vs MI300X |
|---|---|---|---|---|
| VGPR | 512 per lane per SIMD (512 KiB/CU) [3][5]; ≤256 per wave [1] | wave | — | same [5] |
| AGPR (AccVGPR) | ≤256 per wave, from the same file [1][5] | wave | — | same |
| LDS | **160 KiB/CU**, **64 banks** [1] | workgroup | **256 B/clk** [2] | **2.5× cap, 2× banks, 2× BW** [2][4] |
| L1 vector | 32 KiB/CU, 128 B line, 64-way [2] | CU | — | mostly unchanged [2] |
| L2 | 4 MB per XCD, 16-way, 16 channels [2] | per XCD | 128 B read + 64 B write per channel per clock [2] | same model (per-XCD) |
| Infinity Cache | 256 MB, memory-side [2] | device | — | same size [2] |
| HBM3E | **288 GB** [2] | device | **8.0 TB/s** [2] | +96 GB, +2.7 TB/s (vs 192 GB, 5.3 TB/s [6]) |

FP16 roofline ridge ≈ 2.5 PF / 8 TB/s ≈ **312 FLOP/byte** (peaks from [2]) — more kernels are
bandwidth-bound relative to the bigger matrix core, so byte-cutting matters even more than on MI300X
(derived).

### LDS: 160 KiB, 64 banks
- Configured as **64 banks × 640 entries × 4 B = 163840 B/CU** [1]. Dwords are placed in the banks
  serially [1], so bank index = **`(addr/4) mod 64`** (derived).
- **Re-tune padding**: a swizzle tuned for the 32-bank MI300X LDS [4] does **not** guarantee
  conflict-freedom on 64 banks — re-derive the bank map from the operand layout and re-pad (follows from
  the bank counts).
- **256 B/clk** read BW (2× CDNA3's 128 B/clk) [2] — AMD's reason: feeding the vector and matrix units in
  matrix multiplies [2].
- **Read-with-transpose `ds` loads** (`DS_READ_B64_TR_B4/B8/B16`, `DS_READ_B96_TR_B6`): transpose 16-,
  8-, 6- or 4-bit data on the way from LDS to VGPRs, for a column-major A or row-major B MFMA operand [1].
- LDS allocation granularity for gfx950 = **1280 B (320 DWORD) blocks** [1] (vs 512 B, 128 DWORD, on
  CDNA3 [4]).

### GLOBAL_LOAD_LDS / buffer_load ... lds: 128 b/lane
CDNA4 widens direct global→LDS from CDNA3's 32 b/lane [4] to **up to 128 b/lane**: the forms are
`_UBYTE`/`_SBYTE`, `_USHORT`/`_SSHORT`, `_DWORD`, `_DWORDX3` and `_DWORDX4`, i.e. 1, 2, 4, 12 or 16 bytes
per lane, for `GLOBAL_LOAD_LDS_*` and for `BUFFER_LOAD_*` with `LDS=1` [1]. The 96- and 128-bit forms are
new [1][4]. The data goes into LDS without passing through VGPRs [1], which AMD says reduces vector
register usage and latency [2]. Pair with double-buffering and `s_waitcnt`.

### Occupancy impact of bigger LDS
```
occ_lds (workgroups/CU) = floor(163840 / L)   # vs floor(65536 / L) on MI300X
```
(AMD's formula [3].) The 2.5× LDS budget allows **larger tiles or deeper buffering** at the same
occupancy [3][5]. (VGPR limit `floor(512/N)` waves per SIMD, unchanged [5].) See
[isa_notes.md](isa_notes.md) and [../cdna3_mi300/occupancy.md](../cdna3_mi300/occupancy.md) for the full
model.

## The levers
1. **Re-tune LDS padding for 64 banks** when porting from MI300X.
2. **Use 128-bit direct-to-LDS loads** (`global_load_lds_dwordx4` / `buffer_load_dwordx4 ... lds`) for
   tile staging [1].
3. **Read-with-transpose `ds`** to feed an MFMA operand without a transpose pass [1].
4. **Spend the bigger LDS** on larger tiles / more prefetch stages [3].
5. **Cut bytes** harder — the ridge moved right (312 FLOP/B); bandwidth-bound kernels gain from fewer
   bytes, not from the bigger matrix core (derived).
6. **Coalesce to the 128 B L1 line** [2], keep reuse XCD-local, map workgroups in multiples of 8 (8 XCDs)
   [3] — unchanged rules.

## Pitfalls
- **Reusing MI300X 32-bank swizzles** → conflicts on 64 banks.
- **Sticking to 32-bit direct-to-LDS** → leaves the 128-bit width on the table.
- **Assuming a global L2** — still per-XCD [2].

## Verify
- ISA dump for `.lds_size` and that `global_load_lds` emits the `dwordx3`/`dwordx4` form.
- `rocprof-compute` LDS panel (bank conflicts, LDS bandwidth) and HBM BW.

## Sources
1. AMD, *CDNA4 Instruction Set Architecture Reference Guide*, 5 Aug 2025: §2.2.1 and §11.1 (LDS: 160 kB,
   64 banks of 640 dwords, dwords placed serially), §3.6.4 (VGPRs), §3.6.5 (LDS allocated in 1280-byte
   blocks), §9.1.9 (buffer loads to LDS), §10 (`GLOBAL_LOAD_LDS_*`), §11.4 (MFMA transpose loads).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
2. AMD, *Introducing AMD CDNA 4 Architecture* (white paper): LDS (160 KB, 256 bytes per clock, direct
   loads from L1), L1 and L2, Infinity Cache, HBM3E.
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
3. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (occupancy
   formula, larger LDS tiles, XCD multiples).
   <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
4. AMD, *AMD Instinct MI300 / CDNA3 Instruction Set Architecture Reference Guide*: §2.2.1 (LDS 64 kB,
   32 banks × 512 entries), §3.6.5 (512-byte blocks), §9.1.9 and §12.15 (loads to LDS up to a dword).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
5. ROCm Blogs, *Occupancy Math on the AMD MI355X GPU (CDNA4)* (512-entry VGPR file per SIMD, unchanged
   from CDNA3; AccVGPRs from the same file).
   <https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html>
6. ROCm documentation, *AMD Instinct MI300 Series microarchitecture* (MI300X 5.3 TB/s); MI300X capacity
   from [3].
   <https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html>
