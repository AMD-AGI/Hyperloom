---
title: AMD GPU hardware — knowledge map
authority: hardware-index
kind: index
scope: hardware
gens: [gfx950, gfx1151]
updated: 2026-09-02
---

# AMD GPU hardware — knowledge map

Entry index for `hardware/`. Backend-neutral facts about the metal: architecture, topology,
execution, memory, numeric formats and measurement constraints.

> **Convention (KernelForge standard):** a knowledge folder that contains an `INDEX.md` is navigated
> **through this file**—load it whole, then open only the cards matching the detected `gfx` target.

## Supported lanes are separate

| Lane | Product | Architecture | Native wave | Memory model |
|---|---|---|---:|---|
| `gfx950` | MI350X / MI355X | CDNA4 | 64 | 288 GB HBM3E / chiplet Instinct |
| `gfx1151` | Radeon 8060S / Strix Halo | RDNA3.5 | 32 | integrated UMA/shared LPDDR |

**Never combine constants across rows.** In particular:

- gfx950 MFMA, XCD, HBM, OCP-FP8/MX and partition-mode advice is not gfx1151 advice;
- gfx1151 WMMA, VOPD, WGP/CU-mode and UMA advice is not gfx950 advice;
- detect the live GFX target before loading a hardware card;
- framework support is narrower than ISA availability and must be qualified separately.

## gfx950 layout — MI350X / MI355X

| File | Subsystem |
|---|---|
| **`mi350_overview.md`** | **START HERE**—cheat sheet, peak tables, roofline ridges, topology and porting deltas |
| `mi350_execution.md` | wave64, SIMD/CU hierarchy, VGPR/AGPR file and occupancy |
| `mi350_matrix_core.md` | MFMA shapes/cycles, register layout and block-scaled MFMA |
| `mi350_dtypes.md` | OCP FP8, FP6/FP4, MXFP E8M0 scaling and accumulation |
| `mi350_lds.md` | 160 KiB, 64 banks, direct-to-LDS and transpose reads |
| `mi350_memory.md` | HBM3E, per-XCD L2, Infinity Cache and roofline |
| `mi350_chiplet.md` | 8 XCDs, locality/swizzle, stride cliff and SPX/DPX/CPX × NPS |
| `mi350_isa.md` | gfx950 target/toolchain, changed families and disassembly checklist |
| `mi350_clocks.md` | MI350X versus MI355X sustained-clock/measurement behavior |

### Common gfx950 constants

| | |
|---|---|
| 256 CU (8 XCD × 32) · 4 SIMD/CU · 1024 matrix cores | wave64 · 8 slots/SIMD → 32 waves/CU |
| 512 registers/SIMD, 16-granule · ≤256 AGPR, unified pool | LDS 160 KiB/CU, 64 banks, 256 B/clk |
| HBM3E 288 GB @ 8 TB/s · 256 MiB Infinity Cache · L2 per-XCD | FP16 2.5 PF · FP8 5 PF · FP6/FP4 10 PF |
| FP16 ridge ≈312 FLOP/byte | tuned GEMM sustains ~45–55% of peak |
| FP8 is OCP, not FNUZ | TF32 removed |

Use these only for a detected gfx950 target.

## gfx1151 layout — Radeon 8060S / Strix Halo

| File | Subsystem |
|---|---|
| **`gfx1151_overview.md`** | **START HERE**—evidence classes, live constants, UMA roofline and Instinct porting deltas |
| `gfx1151_topology.md` | 40 live CUs, 80 SIMD32s, WGP/CU hierarchy, one XCC and UMA placement |
| `gfx1151_execution.md` | native wave32, deliberate wave64, EXEC/VCC, register granules, occupancy and VOPD |
| `gfx1151_lds.md` | 128 KiB/WGP, two 64 KiB halves, 64 banks, CU/WGP modes and waits |
| `gfx1151_matrix_core.md` | 16×16×16 F16/BF16/IU8/IU4 WMMA, fragments and scheduling hazards |
| `gfx1151_dtypes.md` | architectural dtype boundary, integer algebra and software-format claim limits |
| `gfx1151_isa.md` | native compile target, VOPD/WMMA, waits, FLAT/GLOBAL and disassembly checklist |
| `gfx1151_memory.md` | UMA accounting, local bandwidth reference, bytes/launch roofline and co-tenancy |
| `gfx1151_clocks.md` | package DVFS, sysfs telemetry, thermal/power controls and paired measurement |

### Common gfx1151 constants

| | |
|---|---|
| native `gfx1151` · Radeon PCI `1002:1586` | wave32 native; wave64 supported deliberately |
| 40 live CUs · 80 SIMD32 · 2 SIMD/CU | 16 live wave slots/SIMD |
| 20 two-CU WGPs implied by the live CU count | 4 SIMD32/WGP |
| LDS 64 KiB/CU half · 128 KiB/WGP · 64 banks | one work-group may request ≤64 KiB |
| one live XCC / no Instinct XCD topology | integrated UMA/shared system memory |
| 256 GB/s node-theoretical memory reference | ~241 GB/s read / ~209 GB/s copy local probe |
| WMMA F16/BF16/IU8/IU4 | VOPD is wave32-only; CDNA MFMA is not the route |

The bandwidth values are local measurements/reference math, not universal architecture peaks.
There is intentionally no unverified peak-FLOP/ridge table for gfx1151.

## Portable golden rules

### gfx950

- wave64 everywhere;
- MFMA/SMFMAC/scaled MFMA, not RDNA WMMA;
- L2 is per-XCD; respect 8-XCD locality;
- FP8 is OCP and MX block scaling is native;
- use gfx950's 160 KiB LDS/resource model;
- quote achieved rather than peak.

### gfx1151

- compile natively for `gfx1151`; do not spoof another architecture;
- assume wave32 unless a code object proves deliberate wave64;
- use WMMA/VOP3P and VOPD rules, not CDNA MFMA rules;
- size grids from 40 live CUs and emitted resource metadata;
- keep each work-group ≤64 KiB LDS and derive 64-bank mappings;
- treat GTT/RSS as overlapping views of shared physical memory;
- separate host/launch, device, prefill, decode and shared-memory effects;
- do not call software FP8/MX/low-bit routes native ISA formats without proof.

## Problem → file

| Task / symptom | gfx950 | gfx1151 |
|---|---|---|
| Orient on the target | `mi350_overview.md` | `gfx1151_overview.md` |
| Topology/locality/grid | `mi350_chiplet.md` | `gfx1151_topology.md` |
| Low occupancy/register pressure | `mi350_execution.md` | `gfx1151_execution.md` |
| Matrix-kernel authoring | `mi350_matrix_core.md` | `gfx1151_matrix_core.md` |
| Numeric format/quant route | `mi350_dtypes.md` | `gfx1151_dtypes.md` |
| LDS conflict/tile capacity | `mi350_lds.md` | `gfx1151_lds.md` |
| Memory-bound/coalescing | `mi350_memory.md` | `gfx1151_memory.md` |
| ISA target/opcode/waits | `mi350_isa.md` | `gfx1151_isa.md` |
| Clock/thermal benchmark variance | `mi350_clocks.md` | `gfx1151_clocks.md` |
| Port an Instinct kernel to Strix | source card(s) above | start at `gfx1151_overview.md`, then topology/execution/matrix/LDS |

## Reading depth

- **One identity/number:** open the matching overview and verify its evidence class.
- **Designing a subsystem:** open overview plus that subsystem card.
- **Porting across gfx950 ↔ gfx1151:** read both overview cards and every affected subsystem; do not
  build one synthetic comparison table by copying constants.
- **ISA-level authoring:** read target ISA + execution + matrix/LDS card and inspect the emitted code.
- **Framework serving:** hardware cards are necessary but insufficient—also require framework/image,
  loader, route, correctness, quality and request evidence.

## Cross-links out of this folder

Hardware facts are the substrate. Decision methods live in `common_methodology/optimization/` and
measurement methods in `common_methodology/profiling/`. Language-specific authoring lives in
`languages/{hip,triton,gluon,flydsl,ck,asm}/`; apply those cards only where the selected language and
backend support the detected hardware.
