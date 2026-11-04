---
title: CDNA4 / MI350X / MI355X (gfx950) — architecture overview
kind: hardware
gens: [gfx950]
dtypes: [fp64, fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8]
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350X / MI355X (gfx950) — architecture overview

> Target: **AMD Instinct MI350X (air, 1000 W) / MI355X (liquid, 1400 W)**, CDNA4, ISA **gfx950**.
> Matrix cores in [matrix_core_blockscale.md](matrix_core_blockscale.md) and
> [fp4_fp6_microscaling.md](fp4_fp6_microscaling.md); memory in [memory.md](memory.md); clocks in
> [clocks_power.md](clocks_power.md); ISA in [isa_notes.md](isa_notes.md); peaks in
> [peak_tables.md](peak_tables.md).

## TL;DR
> CDNA4 is the AI-focused successor to CDNA3: **8 XCDs × 32 active CUs = 256 CUs** (16% fewer than
> MI300X, but **2× matrix throughput/CU**), **288 GB HBM3E @ 8 TB/s**, **160 KiB LDS** (2.5×) with
> 256 B/clk and **128-bit GLOBAL_LOAD_LDS**, new **FP6/FP4 + block-scaled MXFP** matrix cores, and
> **TF32 removed**. Peaks: FP16/BF16 **2.5 PF**, FP8 **5 PF**, FP6/FP4 **10 PF**.

## The one-screen cheat sheet
| Fact | Value | Why it matters |
|---|---|---|
| Wavefront | **64 lanes** | unchanged from CDNA3 |
| CUs (active) | **256** (8 XCD × 32) | 16% fewer than MI300X; grid still ≥1024 WGs |
| XCDs | **8** | per-XCD locality (L2 per-XCD) |
| CUs/XCD | **32 active** | a workgroup lives on one CU on one XCD |
| SIMD/CU | **4** | occupancy per-SIMD |
| Wave slots | 8/SIMD → 32/CU | hard cap |
| VGPR | 512 ×4 B/SIMD, 16-granule | + ≤256 AGPR |
| LDS | **160 KiB/CU**, **64 banks**, **256 B/clk** | 2.5× CDNA3; re-tune bank padding |
| GLOBAL_LOAD_LDS | **128-bit/lane** (1/2/4/12/16 DWORD) | direct g→LDS, bigger than CDNA3's 32-bit |
| L2 | per-XCD | not global |
| Infinity Cache | 256 MiB | device-shared |
| HBM3E | **288 GB**, **8 TB/s** | 8 × 36 GB (12-Hi) |
| Matrix peak | FP16/BF16 **2.5 PF**, FP8 **5 PF**, FP6/FP4 **10 PF** | 2× FP16/FP8 vs CDNA3 |
| TF32 | **removed** | emulate with BF16 or run FP32 |
| FP8 variant | **OCP** (E4M3FN/E5M2) | not FNUZ |
| Process | TSMC **N3P** (XCD) + N6 (IOD) | 185 B transistors |
| TDP | **1000 W** (MI350X) / **1400 W** (MI355X) | air vs liquid |
| Engine clock | up to ~2400 MHz (MI355X) | basis of peak math |

## Concepts

### Package topology (3D chiplet, N3P)
8 XCDs (TSMC **N3P**) hybrid-bonded on **2 I/O dies** (TSMC N6), 8 HBM3E stacks, 185 B transistors,
COWOS-S packaging. Each IOD connects 4 HBM3E stacks (36 GB, 12-Hi each → 288 GB total). Infinity Fabric
+ 256 MiB Infinity Cache provide the device-shared coherence layer; L2 remains **per-XCD**.
(Note: CDNA4 uses **2 IODs** vs CDNA3's 4; 4 XCDs sit on each IOD.)

### Compute Unit deltas vs CDNA3
- **32 active CUs/XCD** (down from 38) → 256 total. AMD's trade: fewer CUs but each Matrix Core does
  **4096 FP16 FLOPs/cycle** (2× CDNA3's effective rate) plus FP6/FP4/MXFP support.
- **LDS 160 KiB/CU**, **64 banks** (640 × 4 B entries), **256 B/clk** read BW (2× CDNA3). New
  **read-with-transpose `ds` loads** ease MFMA B-operand layout.
- **GLOBAL_LOAD_LDS up to 128 b/lane** (DWORD counts 1/2/4/12/16 vs CDNA3's 1/2/4) — direct
  global→LDS staging gets 4× wider.
- Same VGPR (512/SIMD) and AGPR (≤256) files; wave64.

### Matrix core (headline change)
Adds **FP6 (E2M3/E3M2)**, **FP4 (E2M1)**, and **block-scaled MXFP8/6/4** with **E8M0** per-32-element
scales via `v_mfma_scale_f32_*_f8f6f4`; new larger FP16/BF16 shapes (16×16×32, 32×32×16) and
f8f6f4 shapes (16×16×128, 32×32×64). **FP6 runs at the FP4 rate** (both 10 PF). **TF32 hardware
removed** — emulate via BF16 or run FP32. **FP8 is OCP** (E4M3FN/E5M2), not FNUZ. FP64 **matrix** rate
is halved vs CDNA3. Detail: [matrix_core_blockscale.md](matrix_core_blockscale.md),
[fp4_fp6_microscaling.md](fp4_fp6_microscaling.md).

### Partitioning
Same SPX/DPX/CPX × NPS family as CDNA3 (8 XCDs). The white paper highlights **CPX+NPS2** hosting up to
8 instances of a 70B model. The 288 GB/8 TB/s memory and 1000–1400 W envelope change the
density/throughput tradeoffs; mode rules (memory ≤ compute partitions) are unchanged. See
[../cdna3_mi300/xcd_chiplet.md](../cdna3_mi300/xcd_chiplet.md) for the partition mechanics.

## The levers
1. **≥1024 workgroups**, **8-multiple tiles** (still 8 XCDs).
2. **`mfma_16x16`** over 32×32.
3. **Exploit FP6/FP4 + MXFP** for weight-heavy layers (32× / 64× FP32) with a task-accuracy gate.
4. **Use OCP FP8** (re-cast from any FNUZ checkpoint).
5. **Re-tune LDS bank padding for 64 banks**; use **128-bit GLOBAL_LOAD_LDS** and read-transpose.
6. **Bigger LDS (160 KiB)** allows larger tiles / more buffering than MI300X.
7. **Drop TF32 code paths** (gone).

## Pitfalls
- **Porting MI300X kernels unchanged.** FNUZ→OCP FP8, 32→64 LDS banks, TF32 removed, fewer CUs.
- **Assuming MI300X's 38 CU/XCD.** It is 32 on CDNA4.
- **Quoting peak.** Sustained is well below; benchmark.

## Verify
- `rocminfo`/`amd-smi static` → gfx950, 256 CU, 288 GB.
- `rocprof-compute` for occupancy (160 KiB LDS limit), L2/L3, HBM BW, matrix utilization.
