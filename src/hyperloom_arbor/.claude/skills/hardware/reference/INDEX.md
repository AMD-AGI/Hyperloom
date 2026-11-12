---
title: AMD GPU hardware knowledge map — index, file roles & problem-routing
kind: index
scope: hardware
updated: 2026-07-14
---

# AMD GPU hardware — knowledge map

This file is the entry index for everything under `hardware/`. It gives (1) what this
knowledge base covers and how it is organized, (2) for a given task/symptom, **which files to read and in
what order**, and (3) the role of every file and folder.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file** — load it whole. Folders without an `INDEX.md` fall back to a generated
> "filename — one-line description" listing.

## What this knowledge base is
Backend-neutral **AMD CDNA hardware facts** for authoring and tuning GPU kernels (GEMM, attention, MoE,
norm, quant). It answers "how does the metal behave, and what are the concrete numbers on this chip?"
independent of any framework or kernel language. Language/framework specifics live elsewhere
(`languages/{hip,triton,flydsl,ck,asm}/`, `framework/`); this folder is the substrate they all sit on.

It is organized on **two axes**:
- **`shared/`** — generation-neutral **mental models** and **cross-generation matrices** (the "how it
  works" + "what each gen supports"). Covers all CDNA gens (gfx908/90a/942/950) in its tables.
- **`cdna3_mi300/` and `cdna4_mi350/`** — per-generation **concrete numbers** and **gen-specific
  instruction tables** for the two chips with dedicated coverage.

**The load-bearing navigation rule:** for any subsystem, read the **`shared/` model doc first** (mental
model + cross-gen capability matrix), then the **target generation's doc** (concrete numbers +
gen-specific instruction/opcode table). One tells you *how it works and what differs across gens*; the
other tells you *the exact values and opcodes on your chip*.

> **Coverage caveat:** only **CDNA3 (gfx942, MI300X/MI325X)** and **CDNA4 (gfx950, MI350X/MI355X)** have
> dedicated per-generation folders. **CDNA1 (gfx908, MI100)** and **CDNA2 (gfx90a, MI250/MI210)** appear
> **only inside the `shared/` cross-gen tables** — there is no per-gen folder for them.

## Portable golden rules (true across the covered gens)
These recur in nearly every card; internalize them before optimizing (numbers in brackets are the
sources at the end):
- **wave64 everywhere** [1][2] — all divergence/shuffle/ballot/reduction math is **mod 64**, never 32.
- **`mfma_16x16` vs `mfma_32x32`** — AMD reports 16×16 ahead on MI300X; on MI350X, benchmark both [3].
- **Kernels left of the roofline ridge are HBM-bandwidth-bound** — the FP16 ridge is ≈247 FLOP/B on
  MI300X and ≈312 on MI355X (peak FLOPs over peak bandwidth [3][4]); optimize **bytes moved** there, not FLOPs.
- **L2 is per-XCD, not global** (chiplet gens) [3][4] → **multiples of 8** in workgroup mapping +
  **≥1024 workgroups** [3].
- **FP8 is FNUZ on CDNA3 but OCP on CDNA4** [6] — re-cast checkpoints, never bit-copy across gens.
- **Accumulate in FP32/INT32** (the MFMA outputs [1][2]); never down-convert inside the K-loop.
- **Quote achieved, never peak** — on MI300X, sustained matrix throughput averaged ~45% of peak
  (arXiv 2510.27583 [5]).

## Start here — problem → files → order
Substitute `<gen>` with your target: `cdna3_mi300` (MI300X/MI325X) or `cdna4_mi350` (MI350X/MI355X).

| Task / symptom | Read in this order |
|---|---|
| "Orient me on chip X / one-screen cheat sheet" | `<gen>/arch.md` |
| "Which chip am I even on? / naming (gfx942 vs gfx950)" | `<gen>/arch.md` (TL;DR + cheat sheet) → `<gen>/isa_notes.md` (target/toolchain) |
| "Write / tune a GEMM (MFMA)" | `shared/matrix_core_mfma_smfmac.md` → `<gen>/matrix_core*.md` → `shared/memory_model_lds_bank.md` → `<gen>/memory*.md` → `<gen>/occupancy.md`¹ |
| "Low occupancy / register pressure / few waves/CU" | `shared/wavefront_simd_vgpr_agpr.md` → `cdna3_mi300/occupancy.md`¹ → (CDNA4 LDS delta) `cdna4_mi350/memory.md` |
| "LDS bank conflicts / shared-memory stalls" | `shared/memory_model_lds_bank.md` → `<gen>/memory*.md` |
| "Memory-bound / low HBM BW / coalescing / roofline" | `shared/hbm_infinity_fabric.md` → `<gen>/memory*.md` → `shared/l2_xcd_swizzle.md` |
| "Chiplet locality / L2 reuse / tile swizzle / Tagram cliff" | `shared/l2_xcd_swizzle.md` → `cdna3_mi300/xcd_chiplet.md`² → `<gen>/memory*.md` |
| "Partitioning: SPX / DPX / QPX / CPX × NPS1/2/4" | `cdna3_mi300/xcd_chiplet.md`² → `<gen>/arch.md` (partition section) |
| "Which dtype? FP8 FNUZ vs OCP / numerics / accuracy" | `shared/dtype_numerics.md` → `<gen>/matrix_core*.md` (numerics section) |
| "Low-bit MXFP4 / FP6 / block scaling (CDNA4 only)" | `cdna4_mi350/fp4_fp6_microscaling.md` → `cdna4_mi350/matrix_core_blockscale.md` → `shared/dtype_numerics.md` |
| "Peak numbers / roofline ridge / FLOP·TOPS math" | `<gen>/peak_tables.md` |
| "Which opcodes / intrinsics / compile target / read ISA dump" | `<gen>/isa_notes.md` → `<gen>/matrix_core*.md` |
| "Benchmark variance / clock throttling / peak≠sustained" | `<gen>/clocks_power.md` → `cdna3_mi300/xcd_chiplet.md`² (per-XCD clock spread) |
| "Port an MI300X kernel to MI350X" | `cdna4_mi350/arch.md` (deltas) → `cdna4_mi350/matrix_core_blockscale.md` + `cdna4_mi350/memory.md` + `cdna4_mi350/isa_notes.md` |

¹ CDNA4 has **no** dedicated `occupancy.md`; the model + worked examples in `cdna3_mi300/occupancy.md`
apply (VGPR limit unchanged [7]), with the **160 KiB / 64-bank LDS delta** [1] covered in `cdna4_mi350/memory.md`.
² CDNA4 has **no** dedicated `xcd_chiplet.md`; the chiplet & partitioning mechanics in
`cdna3_mi300/xcd_chiplet.md` apply (still 8 XCDs), with CDNA4-specific counts in `cdna4_mi350/arch.md`.

## Folder structure & file roles
```
hardware/
├── INDEX.md                              ← this map (load first)
├── shared/                               ← generation-neutral models + cross-gen matrices
│   ├── wavefront_simd_vgpr_agpr.md       # execution model: wave64, SIMD, VGPR/AGPR/SGPR, occupancy math
│   ├── memory_model_lds_bank.md          # LDS scratchpad model & bank-conflict rules (32 vs 64 banks)
│   ├── matrix_core_mfma_smfmac.md        # MFMA/SMFMAC/scaled-MFMA model + cross-gen capability matrix
│   ├── l2_xcd_swizzle.md                 # L2/XCD locality, tile swizzle, 512B Tagram cliff, coalescing
│   ├── hbm_infinity_fabric.md            # HBM, Infinity Fabric/Cache, bandwidth ladder, roofline ridge
│   └── dtype_numerics.md                 # FP8 FNUZ/OCP, FP6/FP4, MXFP (E8M0), TF32, rounding/subnormals
├── cdna3_mi300/                          ← gfx942 · MI300X / MI325X (concrete numbers)
│   ├── arch.md                           # orientation map + one-screen cheat sheet (START HERE for MI300X)
│   ├── peak_tables.md                    # theoretical peak FLOPS/TOPS, memory peaks, sustained reality
│   ├── matrix_core.md                    # CDNA3 MFMA instruction table + HIP intrinsics + FNUZ numerics
│   ├── memory_hierarchy.md               # MI300X memory ladder, coalescing, direct global→LDS, double-buffer
│   ├── occupancy.md                      # occupancy math + worked examples (the reference model for both gens)
│   ├── xcd_chiplet.md                    # 8-XCD scheduling, cross-XCD cost, clock variance, SPX/DPX/CPX×NPS
│   ├── isa_notes.md                      # gfx942 target/toolchain, instruction families, reading the ISA dump
│   └── clocks_power.md                   # 2.1 GHz peak, 750/1000 W, peak≠sustained, benchmark hygiene
└── cdna4_mi350/                          ← gfx950 · MI350X / MI355X (concrete numbers + CDNA4-only features)
    ├── arch.md                           # orientation map + cheat sheet + CDNA3→CDNA4 deltas (START HERE for MI350X)
    ├── peak_tables.md                    # MI355X: FP16 2.5 PF / FP8 5 PF / MXFP6·MXFP4 10 PF, 288 GB @ 8 TB/s
    ├── matrix_core_blockscale.md         # CDNA4 MFMA table + block-scaled v_mfma_scale_* intrinsic + layout
    ├── fp4_fp6_microscaling.md           # FP6/FP4 formats + MXFP 32-elem E8M0 block scaling (CDNA4-only)
    ├── memory.md                         # CDNA4 memory deltas: 160 KiB/64-bank LDS, 128-bit GLOBAL_LOAD_LDS
    ├── isa_notes.md                      # gfx950 ISA deltas vs gfx942 (scaled MFMA, TF32 removed, ROCm 7.0+)
    └── clocks_power.md                   # MI350X 1000 W air vs MI355X 1400 W liquid, sustained-clock effect
```

## Reading-depth guide (how much to load)
- **Just need a number/fact** (a peak, a cache size, a CU count): the target gen's `arch.md` cheat sheet
  or `peak_tables.md` is enough — don't load the whole subsystem.
- **Designing/tuning a kernel subsystem**: load the **`shared/` model** + the **gen-specific** doc for
  that subsystem (the pairing in the routing table). This is the common case.
- **Cross-generation reasoning / porting**: start from `cdna4_mi350/arch.md` (it enumerates the
  CDNA3→CDNA4 deltas), then the specific gen docs on both sides.
- **Authoring at the ISA level**: `<gen>/isa_notes.md` + `<gen>/matrix_core*.md`. The ISA guides are
  authoritative over any table here; `amd_matrix_instruction_calculator` covers CDNA1–CDNA3 but not
  CDNA4 [8], so for gfx950 use the CDNA4 ISA guide's MFMA tables and layouts [1].

## Cross-links out of this folder
Hardware facts are the substrate; kernel authoring and framework control live elsewhere. Operator/library
cards (e.g. `framework/aiter/`) and language folders (`languages/{hip,triton,flydsl,ck,asm}/`) cite these
hardware cards for the underlying constants. Each card ends with its own Sources: AMD's ISA guides, white
papers and datasheets, ROCm documentation and blogs, and the OCP MX specification.

## Sources
1. AMD, *CDNA4 Instruction Set Architecture Reference Guide*, 5 Aug 2025: §1.1 (wavefront of 64
   work-items), §2.2.1 (LDS: 160 kB, 64 banks), chapter 7 (MFMA outputs FP32/INT32).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
2. AMD, *AMD Instinct MI300 / CDNA3 Instruction Set Architecture Reference Guide*: §1.1 (wavefront of 64
   work-items), chapter 7.
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
3. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (MI16x16 vs
   MI32x32, ≥1024 workgroups, multiples of 8 XCDs, 512-byte stride hotspots, L2 per XCD, peaks).
   <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
4. AMD, *Introducing AMD CDNA 4 Architecture* (white paper).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
5. C. Ambati, T. Diep, *AMD MI300X GPU Performance Analysis*, arXiv:2510.27583.
   <https://arxiv.org/abs/2510.27583>
6. HIP documentation, *Low precision floating point types* (FNUZ on gfx94x, OCP on CDNA4).
   <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html>
7. ROCm Blogs, *Occupancy Math on the AMD MI355X GPU (CDNA4)* (VGPR file unchanged at 512 per SIMD).
   <https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html>
8. AMD Matrix Instruction Calculator (supported architectures: CDNA1–CDNA3, RDNA3, RDNA4).
   <https://github.com/ROCm/amd_matrix_instruction_calculator>
