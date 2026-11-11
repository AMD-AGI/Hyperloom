---
title: CDNA3 / MI300X (gfx942) — architecture overview
kind: hardware
gens: [gfx942]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8]
regimes: [both]
updated: 2026-11-07
---

# CDNA3 / MI300X (gfx942) — architecture overview

> Target: **AMD Instinct MI300X**, CDNA3, ISA **gfx942**; **MI300A** and **MI325X** are gfx942 too [3]. This
> is the orientation map; memory in [memory_hierarchy.md](memory_hierarchy.md), matrix cores in
> [matrix_core.md](matrix_core.md), chiplet topology in [xcd_chiplet.md](xcd_chiplet.md), occupancy in
> [occupancy.md](occupancy.md), peak tables in [peak_tables.md](peak_tables.md). Numbers in brackets
> point to the Sources at the end.

## TL;DR
> The MI300X is **8 XCD chiplets** × 38 active CUs = **304 CUs** [1][3], wave64 [3], 4 SIMD/CU [4],
> with a **per-XCD** 4 MiB L2 and a single device-shared **256 MiB Infinity Cache** [1][3].
> Launch **≥1024 workgroups**, use **multiples of 8** for tiles and grids, and prefer **`mfma_16x16`** [4].
> At 5.3 TB/s [2] the FP16 roofline ridge is ≈ 247 FLOP/byte (derived in [peak_tables.md](peak_tables.md)),
> so low-arithmetic-intensity kernels (decode, norms) are HBM-bound — optimize bytes, not FLOPs.

## The one-screen cheat sheet
| Fact | Value | Why it matters |
|---|---|---|
| Wavefront | **64 lanes** [3] | all divergence/shuffle/MFMA math is mod 64 |
| CUs (active) | **304** (8 XCD × 38) [1][3] | grid ≥ 1024 workgroups to fill + tail [4] |
| XCDs | **8** [1] | per-chiplet locality is not free |
| CUs/XCD | 40 physical / **38 active** [1] | a workgroup's waves all run on one CU [5] |
| SIMD/CU | **4** [4] | occupancy is per-SIMD |
| Wave slots | **8/SIMD → 32/CU** [6] | hard occupancy cap |
| Peak clock | **2100 MHz** [2] | basis of all peak math |
| VGPR | **512 per SIMD lane** (512 KiB VGPR file per CU) [3][4]; allocated in groups of 8 per the ISA [5], blocks of 16 in the workload guide's example [4] | occupancy limiter |
| AGPR | up to **256** of those 512 (MFMA accum) [5] | big accumulators without crushing occupancy |
| LDS | **64 KiB/CU**, 32 banks [5] | bank conflicts mod 32 |
| L1 vector | 32 KiB/CU, 128 B line [1] | coalesce to 128 B |
| L2 | **4 MiB per XCD** (not global) [1][3] | cross-XCD reuse misses to the Infinity Cache |
| Infinity Cache (L3/MALL) | **256 MiB**, 17.2 TB/s peak [1] | the only device-shared cache |
| HBM3 | **192 GB**, **5.3 TB/s** [2] | low-intensity kernels are HBM-bound |
| Matrix peak | FP16/BF16 **1307.4 TF**, FP8/INT8 **2614.9 TF** [2] | MFMA mandatory |
| Default partition | **SPX** [7] with NPS1 [4] | one logical 304-CU / 192 GB device |
| Max board power | **750 W** (MI300X) [2] / 1000 W (MI325X) [4] | thermal/clock budget |

## Concepts

### Package topology (3D-stacked chiplets)
The MI300X is **not monolithic**: 8 compute chiplets (XCD) vertically stacked on 4 I/O dies (IOD),
connected to 8 HBM3 stacks [1] (153 B transistors [4], 750 W [2]).
```
   HBM3  HBM3        HBM3  HBM3
   ┌──────────┐      ┌──────────┐   each XCD: 38 active CUs + 4 MiB L2
   │ XCD0 XCD1│      │ XCD2 XCD3│
   │  (IOD0)  │◄────►│  (IOD1)  │  ◄─ Infinity Fabric on-package
   ├──────────┤      ├──────────┤     256 MiB Infinity Cache spans IODs
   │  (IOD2)  │◄────►│  (IOD3)  │
   │ XCD4 XCD5│      │ XCD6 XCD7│
   └──────────┘      └──────────┘
   HBM3  HBM3        HBM3  HBM3
```
| Die | Count | Node | Contents |
|---|---|---|---|
| XCD | 8 | TSMC N5 [1] | 40 CUs (38 active), 4 ACEs, 4 MiB L2 [1], hardware scheduler [4] |
| IOD | 4 | TSMC N6 [1] | Infinity Cache, HBM3 interface, Infinity Fabric [1] |
| HBM3 stack | 8 | — | 24 GB each [1]; 8192-bit bus / 8 stacks × 5.2 GT/s ≈ 666 GB/s each (derived from [2]) → 192 GB / 5.3 TB/s aggregate [2] |

Each IOD sits under **2 XCDs** and fans out to **2 HBM stacks** [1][7]. There is **no device-wide L2**:
each XCD's L2 is coherent within that XCD, and the first device-shared level is the **256 MiB Infinity
Cache** on the IODs, whose snoop filter resolves most cross-XCD coherence requests [1]. Details:
[xcd_chiplet.md](xcd_chiplet.md).

### Compute Unit
Each CDNA3 CU: 4 SIMDs [4], 4 Matrix Cores (1216 matrix cores / 304 CUs [2]), 512 VGPRs per SIMD lane
of which up to 256 can be AGPRs [3][5], 64 KiB LDS (32 banks) [5], 32 KiB L1 vector (128 B line) [1];
the 16 KiB scalar cache and the 64 KiB instruction cache are each shared by 2 CUs [3]. Per-CU peak
basis in [peak_tables.md](peak_tables.md).

### Partitioning (SPX/DPX/QPX/CPX × NPS)
The 8-XCD design can be spatially partitioned via `amd-smi` [7][8]:
- **SPX** (default): 1 logical GPU, 8 XCDs, 304 CU, 192 GB [7].
- **DPX**: 2× (4 XCD / 152 CU / 96 GB) [7].
- **QPX**: 4× (2 XCD / 48 GB) [1][4].
- **CPX**: 8× (1 XCD / 38 CU / 24 GB) — many small jobs, inference density [7].
- Memory: **NPS1/NPS2/NPS4** [7]; memory partitions must **not exceed** compute partitions [1][7], so
  SPX+NPS4 is invalid.
- On a compute-bound FP16 GEMM, AMD measured **10–15% higher total throughput** in CPX/NPS1 and
  CPX/NPS4 than in SPX, with CPX/NPS4 running at higher clocks thanks to localized memory accesses
  [8]. Full table in [xcd_chiplet.md](xcd_chiplet.md).

## The levers (architecture-level)
1. **≥1024 workgroups** [4] (≈3.4 per CU on 304 CUs) to fill the device and hide tails.
2. **Multiples of 8** for tile and grid dims, for L2 efficiency across the 8 XCDs [4] →
   [memory_hierarchy.md](memory_hierarchy.md).
3. **`mfma_16x16`** over 32×32 [4] → [matrix_core.md](matrix_core.md).
4. **Push to FP8**: its peak is 16× the FP32 peak [1], where accuracy allows.
5. **Optimize bytes** on kernels left of the roofline ridge (decode, norms; follows from the ridge).
6. **Keep data-sharing workgroups on one XCD** [4]; for many-small-job density consider CPX/NPS4 [8].

## Pitfalls
- **Treating it as monolithic.** L2 is per-XCD [1]; cross-XCD reuse is not free.
- **Under-launching** (<1024 WGs) leaves XCDs idle [4].
- **Quoting peak.** One published study measured an average of ~45% of peak on square GEMMs across
  FP8/BF16/FP16 [9].

## Verify
- `rocminfo` / `amd-smi static` confirms gfx942, 304 CU, partition mode.
- `rocprof-compute` for occupancy, L2/L3 hit rates, HBM BW, XCD balance.

## Sources
1. AMD, *AMD CDNA 3 Architecture* white paper — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
2. AMD, *AMD Instinct MI300X Accelerator* data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
3. ROCm documentation, *AMD GPU specifications* (Instinct table) — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
4. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
5. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA reference guide; §2.2.1 LDS, §3.6.4 VGPR allocation, §4.3 work-groups) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
6. ROCm Compute Profiler documentation, *Pipeline descriptions* (VALU: 8 wavefront slots per SIMD, 32 per CU) — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.4.1/conceptual/pipeline-descriptions.html
7. AMD GPU driver documentation, *AMD Instinct MI300X GPU Partitioning Overview* — https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html
8. ROCm Blogs, *Deep dive into the MI300 compute and memory partition modes* — https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
9. C. Ambati, T. Diep, *AMD MI300X GPU Performance Analysis*, arXiv:2510.27583 — https://arxiv.org/abs/2510.27583
