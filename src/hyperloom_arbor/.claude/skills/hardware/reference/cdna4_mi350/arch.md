---
title: CDNA4 / MI350X / MI355X (gfx950) — architecture overview
kind: hardware
gens: [gfx950]
dtypes: [fp64, fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1, mxfp8, mxfp6, mxfp4, int8]
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350X / MI355X (gfx950) — architecture overview

> Target: **AMD Instinct MI350X (air, 1000 W) / MI355X (liquid, 1400 W)** [2], CDNA4, ISA **gfx950** [4].
> Matrix cores in [matrix_core_blockscale.md](matrix_core_blockscale.md) and
> [fp4_fp6_microscaling.md](fp4_fp6_microscaling.md); memory in [memory.md](memory.md); clocks in
> [clocks_power.md](clocks_power.md); ISA in [isa_notes.md](isa_notes.md); peaks in
> [peak_tables.md](peak_tables.md). Numbers in brackets are the sources at the end.

## TL;DR
> CDNA4 is the AI-focused successor to CDNA3: **8 XCDs × 32 active CUs = 256 CUs** [2] (16% fewer than
> MI300X's 304 [5], but **2× the matrix FLOPs per clock per CU** for 16-bit and smaller types [2]),
> **288 GB HBM3E @ 8 TB/s** [2], **160 KiB LDS** (2.5× CDNA3) with 256 B/clk [1][2] and direct global→LDS
> loads of up to **128 bits per lane** [1], new **FP6/FP4 + block-scaled MXFP** matrix instructions [1][2],
> and **TF32 removed** from hardware [2]. MI355X peaks: FP16/BF16 **2.5 PF**, FP8 **5 PF**, MXFP6/MXFP4
> **10 PF**; MI350X 2.3 / 4.6 / 9.2 PF [2][3].

## The one-screen cheat sheet
| Fact | Value | Why it matters |
|---|---|---|
| Wavefront | **64 lanes** [1] | unchanged from CDNA3 [9] |
| CUs (active) | **256** (8 XCD × 32) [2][4] | 16% fewer than MI300X's 304 [5]; AMD's guidance is still ≥1024 workgroups per grid [5] |
| XCDs | **8** [2] | per-XCD locality (L2 per XCD) [2] |
| CUs/XCD | **32 active** (of 36) [2] | a workgroup shares one CU's LDS, so it lives on one CU on one XCD [1] |
| SIMD/CU | **4** [5][7] | occupancy is counted per SIMD [7] |
| Wave slots | 8/SIMD → 32/CU [7] | hard cap |
| VGPR | 512 per lane per SIMD (512 KiB per CU) [4][7]; a wave holds ≤256 VGPR + ≤256 AccVGPR, allocated in groups of 8 [1] | unchanged from CDNA3 [7] |
| LDS | **160 KiB/CU**, **64 banks** [1], **256 B/clk** [2] | 2.5× CDNA3's capacity, 2× its banks and read bandwidth [2][9]; re-tune bank padding |
| GLOBAL_LOAD_LDS | **up to 128 bits per lane**: 1, 2, 4, 12 or 16 bytes (`_UBYTE` … `_DWORDX4`) [1] | direct g→LDS; CDNA3 stops at 32 bits [9] |
| L2 | 4 MB per XCD, 16-way [2] | not global |
| Infinity Cache | 256 MB [2] | device-shared, memory-side [2] |
| HBM3E | **288 GB**, **8 TB/s** [2] | 8 stacks × 36 GB (12-Hi) [2] |
| Matrix peak (MI355X) | FP16/BF16 **2.5 PF**, FP8 **5 PF**, MXFP6/MXFP4 **10 PF** [2] | 1.9× FP16/FP8 vs MI300X's 1.3 / 2.6 PF [2][5] |
| TF32 | **removed** [2] | emulated in software via BF16 [2]; or run FP32 |
| FP8 variant | **OCP** (E4M3FN/E5M2) [8][10] | not FNUZ |
| Process | TSMC **N3P** (XCD) + N6 (IOD) [2] | 185 B transistors [2] |
| TDP | **1000 W** (MI350X) / **1400 W** (MI355X) [2] | air vs liquid |
| Engine clock | peak 2,400 MHz (MI355X), 2,200 MHz (MI350X) [2] | basis of peak math |

## Concepts

### Package topology (3D chiplet, N3P)
8 XCDs (TSMC **N3P**) vertically stacked on **2 I/O dies** (TSMC N6), 8 stacks of 12-Hi HBM3E, 185 B
transistors [2][3]. Each IOD carries 4 XCDs and 4 HBM3E stacks: NPS2 splits the 288 GB into two 144 GB
pools, one per IOD (144 / 36 = 4 stacks) [2]. The Infinity Cache (256 MB) and the memory controllers sit in
the IODs and are device-shared; L2 remains **per-XCD** [2]. (CDNA4 uses **2 IODs** vs CDNA3's 4 [2].)

### Compute Unit deltas vs CDNA3
- **32 active CUs/XCD** (down from 38) → 256 total [2][5]. AMD's trade: fewer CUs, but each CU does
  **4,096 dense FP16/BF16 matrix FLOPs per clock** (2,048 on MI300X) and 8,192 FP8, plus FP6/FP4/MXFP [2].
- **LDS 160 KiB/CU**, **64 banks** of 640 entries × 4 B [1], **256 B/clk** read bandwidth (2× CDNA3) [2].
  New **read-with-transpose `ds` loads** (`DS_READ_B64_TR_B4/B8/B16`, `DS_READ_B96_TR_B6`) load a
  column-major A or row-major B operand for MFMA [1].
- **Direct global→LDS up to 128 b/lane**: `GLOBAL_LOAD_LDS_DWORDX3/X4` and `BUFFER_LOAD_DWORDX3/X4`
  with `LDS=1` join the byte, short and dword forms [1]; CDNA3 has only the forms up to a dword [9], so the
  widest load is 4× wider.
- Same register model: up to 512 VGPRs per wave, 256 architectural + 256 AccVGPR [1][9]; the VGPR file is
  unchanged at 512 per SIMD [7]; wave64.

### Matrix core (headline change)
Adds **FP6 (E2M3/E3M2)**, **FP4 (E2M1)**, and **block-scaled MXFP8/6/4** with one **E8M0** scale per 32
K-values via `v_mfma_scale_f32_*_f8f6f4` [1]; new larger FP16/BF16 shapes (16×16×32, 32×32×16) and
f8f6f4 shapes (16×16×128, 32×32×64) [1]. **MXFP6 runs at the MXFP4 rate** (both 10 PF on MI355X) [2].
**TF32 moved out of hardware** — emulated via BF16 [2], or run FP32. **FP8 is OCP** (E4M3FN/E5M2), not
FNUZ [8][10]. The FP64 **matrix** rate per CU is halved vs CDNA3 (256 → 128 FLOPs/clock/CU) [2]. Detail:
[matrix_core_blockscale.md](matrix_core_blockscale.md), [fp4_fp6_microscaling.md](fp4_fp6_microscaling.md).

### Partitioning
Compute splits along XCD lines into 1, 2, 4 or 8 partitions (SPX, DPX, QPX, CPX); memory is NPS1
(interleaved over all 8 stacks) or NPS2 (two 144 GB pools, one per IOD) [2][3]. CPX runs up to 8 instances
on one GPU for smaller inference models [2]. AMD names DPX + NPS2 the most efficient mode on CDNA4 (QPX +
NPS4 on CDNA3) [2][5]. See [../cdna3_mi300/xcd_chiplet.md](../cdna3_mi300/xcd_chiplet.md) for the
partition mechanics.

## The levers
1. **≥1024 workgroups** [5], and **multiples of 8** in the workgroup mapping (8 XCDs) [5].
2. **`mfma_16x16` vs 32×32:** AMD reports 16×16 ahead on MI300X; for MI350X it says to benchmark both [5].
3. **Exploit FP6/FP4 + MXFP** for weight-heavy layers (peaks 32× / 64× the FP32 matrix peak [2]) with a
   task-accuracy gate.
4. **Use OCP FP8** (re-cast from any FNUZ checkpoint: the encodings differ [8]).
5. **Re-tune LDS bank padding for 64 banks**; use **128-bit direct-to-LDS loads** and read-transpose
   (these follow from [1]).
6. **Bigger LDS (160 KiB)** allows larger tiles / more pipeline stages than MI300X [5].
7. **Drop TF32 code paths** (gone [2]).

## Pitfalls
- **Porting MI300X kernels unchanged.** FNUZ→OCP FP8, 32→64 LDS banks, TF32 removed, fewer CUs.
- **Assuming MI300X's 38 CU/XCD.** It is 32 on CDNA4 [2].
- **Quoting peak.** AMD's peaks are peak theoretical [2]; benchmark.

## Verify
- `rocminfo`/`amd-smi static` → gfx950, 256 CU, 288 GB.
- `rocprof-compute` for occupancy (160 KiB LDS limit), L2/L3, HBM BW, matrix utilization.

## Sources
1. AMD, *CDNA4 Instruction Set Architecture Reference Guide* (AMD Instinct MI350 Series), 5 Aug 2025:
   §1.1 (wavefront of 64 work-items), §2.2.1 and §11.1 (LDS: 160 kB, 64 banks × 640 entries × 4 B),
   §3.1 and §3.6.4 (VGPRs, AccVGPRs, allocation), §7.1.2 Table 28 (MFMA shapes), §7.2.1 (block scaling),
   §9.1.9 and §10 (loads to LDS), §11.4 (transpose loads).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
2. AMD, *Introducing AMD CDNA 4 Architecture* (white paper): chiplets, CU and LDS, Table 1 (FLOPs per
   clock per CU), memory, partitioning, Table 2 (MI350X and MI355X specifications).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
3. ROCm documentation, *AMD Instinct MI350 Series microarchitecture*.
   <https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html>
4. ROCm documentation, *AMD GPU specifications* (LLVM target, CUs, LDS, VGPR file per CU).
   <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
5. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (architecture
   comparison tables, 4 SIMDs per CU, workgroup and MFMA guidance, larger LDS tiles).
   <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
6. AMD, *AMD Instinct MI355X GPU* datasheet.
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf>
7. ROCm Blogs, *Occupancy Math on the AMD MI355X GPU (CDNA4)* (4 SIMDs, 8 waves per SIMD, 32 per CU,
   512-entry VGPR file per SIMD).
   <https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html>
8. ROCm Blogs, *Matrix Core Programming on AMD CDNA3 and CDNA4 architecture* (FP8 variants, MFMA
   instructions).
   <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html>
9. AMD, *AMD Instinct MI300 / CDNA3 Instruction Set Architecture Reference Guide*: §2.2.1 (LDS 64 kB,
   32 banks), §3.6.4, §9.1.9 (buffer loads to LDS up to a dword).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
10. HIP documentation, *Low precision floating point types* (OCP FP8 on CDNA4, FNUZ on gfx94x).
    <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html>
