---
title: MI350X / MI355X hardware — knowledge map
kind: index
scope: hardware
gens: [gfx950]
updated: 2026-08-28
---

# MI350X / MI355X hardware — knowledge map

Entry index for `hardware/`. Backend-neutral facts about the metal: what the chip is, what the numbers
are, and what each subsystem does to a kernel.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole.

## Scope: gfx950 only

Target parts are **MI350X** (air, 1000 W) and **MI355X** (liquid, 1400 W), CDNA4, ISA **gfx950**.

**Earlier generations are not covered** — no CDNA1 (gfx908), CDNA2 (gfx90a), or **CDNA3 (gfx942 /
MI300X / MI325X)** cards, and no cross-generation comparison tables. If you are targeting MI300X, the
numbers here are wrong for you; use AMD's CDNA3 documentation instead. CDNA3 appears only as
*porting warnings* ("that value is MI300X's — here it is X").

## Layout — one flat folder, one file per subsystem

There are no subfolders. Each card carries **both** the mental model and the concrete gfx950 numbers
for its subsystem, so a single Read answers a question end to end.

| File | Subsystem |
|---|---|
| **`mi350_overview.md`** | **START HERE** — one-screen cheat sheet, peak tables, roofline ridges, topology, the four porting deltas |
| `mi350_execution.md` | wave64, SIMD/CU hierarchy, VGPR/AGPR file, occupancy formula + worked examples |
| `mi350_matrix_core.md` | MFMA model, shape/cycle table, per-lane registers, block-scaled MFMA, capability list |
| `mi350_dtypes.md` | format table, the **OCP** FP8 trap, FP6/FP4, MXFP E8M0 block scaling, accumulation rules |
| `mi350_lds.md` | 160 KiB / **64 banks**, conflicts, padding vs XOR swizzle, 128-bit direct-to-LDS, read-with-transpose |
| `mi350_memory.md` | bandwidth ladder, HBM3E, Infinity Cache, per-XCD L2, coalescing, roofline ridge |
| `mi350_chiplet.md` | 8 XCDs × 32 CU, L2 locality and CTA swizzle, 512 B stride cliff, clock variance, SPX/DPX/CPX × NPS |
| `mi350_isa.md` | gfx950 target/toolchain, changed instruction families, **the disassembly checklist** |
| `mi350_clocks.md` | MI350X vs MI355X, sustained clock, and what that does to measurements |

## Constants you will look up most

| | |
|---|---|
| 256 CU (8 XCD × 32) · 4 SIMD/CU · 1024 matrix cores | wave64 · 8 slots/SIMD → 32 waves/CU |
| 512 regs/SIMD, 16-granule · ≤256 AGPR, unified pool | LDS **160 KiB/CU, 64 banks**, 256 B/clk, 320-DWORD granule |
| HBM3E **288 GB @ 8 TB/s** · 256 MiB Infinity Cache · **L2 per-XCD** | FP16 **2.5 PF** · FP8 **5 PF** · FP6/FP4 **10 PF** |
| FP16 ridge ≈ **312 FLOP/byte** | tuned GEMM sustains **~45–55% of peak** |
| FP8 is **OCP**, not FNUZ | **TF32 removed** |
| `global_load_lds` up to **128 b/lane** | `mfma_16x16` over `32x32`; ≥1024 WGs; 8-multiple tiles |

## Portable golden rules

- **wave64 everywhere** — all divergence/shuffle/ballot/reduction math is mod 64, never 32.
- **`mfma_16x16` beats `mfma_32x32`** at equal peak — 4 C-registers/lane vs 16.
- **Most inference kernels are HBM-bandwidth-bound** — optimize bytes moved, not FLOPs.
- **L2 is per-XCD, not global** → 8-multiple tiles, ≥1024 workgroups across 256 CUs.
- **FP8 is OCP** — re-cast any FNUZ checkpoint, never bit-copy it.
- **TF32 is gone** — fall back to BF16 or FP32.
- **LDS is 160 KiB over 64 banks** — re-derive any 32-bank swizzle; VGPR pressure, not LDS, is usually
  the occupancy limiter now.
- **Accumulate in FP32/INT32**; never down-convert inside the K-loop.
- **Quote achieved, never peak** — sustained is ~45–55% of peak.

## Problem → file

| Task / symptom | Read |
|---|---|
| Orient me on the chip / one-screen cheat sheet | `mi350_overview.md` |
| Peak numbers, roofline ridge, FLOP·TOPS math | `mi350_overview.md` |
| Write / tune a GEMM (MFMA) | `mi350_matrix_core.md` → `mi350_lds.md` → `mi350_execution.md` |
| Low occupancy / register pressure / few waves/CU | `mi350_execution.md` |
| LDS bank conflicts / `ds_*` stalls / tile won't fit | `mi350_lds.md` |
| Memory-bound / low HBM BW / coalescing | `mi350_memory.md` → `mi350_lds.md` |
| Chiplet locality / L2 reuse / tile swizzle / Tagram cliff | `mi350_chiplet.md` → `mi350_memory.md` |
| Partitioning: SPX / DPX / CPX × NPS | `mi350_chiplet.md` |
| Which dtype? FP8 OCP / FP6 vs FP4 / numerics | `mi350_dtypes.md` |
| Low-bit MXFP4 / FP6 / block scaling | `mi350_dtypes.md` → `mi350_matrix_core.md` |
| Which opcodes / compile target / **read the ISA dump** | `mi350_isa.md` |
| Benchmark variance / clock throttling / peak ≠ sustained | `mi350_clocks.md` → `mi350_chiplet.md` |
| **Porting a kernel written for MI300X** | `mi350_overview.md` (the four deltas) → `mi350_dtypes.md` (FNUZ→OCP) → `mi350_lds.md` (32→64 banks) → `mi350_execution.md` (304→256 CU, 64→160 KiB) |

## Reading depth

- **A single number** (a peak, a cache size, a CU count) — `mi350_overview.md` alone.
- **Designing or tuning a subsystem** — the one card for that subsystem; each is self-contained.
- **Porting from MI300X** — `mi350_overview.md`, then the three delta cards it names.
- **ISA-level authoring** — `mi350_isa.md` + `mi350_matrix_core.md`, and treat
  `amd_matrix_instruction_calculator --architecture cdna4` as authoritative over any table here.

## Cross-links out of this folder

Hardware facts are the substrate. **How to decide what to change** lives in
`common_methodology/optimization/` (the `lever_*` cards) and `common_methodology/profiling/` (the
`measure_*` cards). **How to write it in a given language** lives in
`languages/{hip,triton,gluon,flydsl,ck,asm}/`. The library control plane is `framework/aiter/`.
