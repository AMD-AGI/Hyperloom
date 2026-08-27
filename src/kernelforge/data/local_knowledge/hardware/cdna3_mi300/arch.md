---
title: CDNA3 / MI300X (gfx942) — architecture overview
kind: hardware
gens: [gfx942]
dtypes: [fp64, fp32, tf32, bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8]
regimes: [both]
updated: 2026-06-08
---

# CDNA3 / MI300X (gfx942) — architecture overview

> Target: **AMD Instinct MI300X**, CDNA3, ISA **gfx942** (also gfx940/941 for MI300A / early steppings;
> **MI325X** is also gfx942). This is the orientation map; memory in
> [memory_hierarchy.md](memory_hierarchy.md), matrix cores in [matrix_core.md](matrix_core.md), chiplet
> topology in [xcd_chiplet.md](xcd_chiplet.md), occupancy in [occupancy.md](occupancy.md), peak tables
> in [peak_tables.md](peak_tables.md).

## TL;DR
> The MI300X is **8 GPUs glued together**: 8 XCD chiplets × 38 active CUs = **304 CUs**, wave64,
> 4 SIMD/CU, with a **per-XCD** 4 MiB L2 and a single device-shared **256 MiB Infinity Cache**.
> Launch **≥1024 workgroups**, use **8-multiple tiles** and **`mfma_16x16`**, and remember most
> inference kernels are **HBM-bound** at 5.3 TB/s — optimize bytes, not FLOPs.

## The one-screen cheat sheet
| Fact | Value | Why it matters |
|---|---|---|
| Wavefront | **64 lanes** | all divergence/shuffle/MFMA math is mod 64 |
| CUs (active) | **304** (8 XCD × 38) | grid ≥ 1024 workgroups to fill + tail |
| XCDs | **8** | per-chiplet locality is not free |
| CUs/XCD | 40 physical / **38 active** | a workgroup lives on one CU on one XCD |
| SIMD/CU | **4** (SIMD64) | occupancy is per-SIMD |
| Wave slots | **8/SIMD → 32/CU** | hard occupancy cap |
| Peak clock | **2100 MHz** | basis of all peak math |
| VGPR | **512 ×4 B / SIMD**, 16-granule | #1 occupancy killer |
| AGPR | up to **256** (MFMA accum) | big accumulators without crushing occupancy |
| LDS | **64 KiB/CU**, 32 banks | bank conflicts mod 32 |
| L1 vector | 32 KiB/CU, 128 B line | coalesce to 128 B |
| L2 | **4 MiB per XCD** (not global) | cross-XCD reuse misses to L3 |
| Infinity Cache (L3/MALL) | **256 MiB**, ~218 ns, ~11.9 TB/s | the only device-shared cache |
| HBM3 | **192 GB**, **5.325 TB/s** | most inference kernels are HBM-bound |
| Matrix peak | FP16/BF16 **1307 TF**, FP8/INT8 **2615 TF** | MFMA mandatory |
| Default partition | **SPX + NPS1** | one logical 304-CU / 192 GB device |
| TDP | **750 W** (MI300X) / 1000 W (MI325X) | thermal/clock budget |

## Concepts

### Package topology (3.5D chiplet)
The MI300X is **not monolithic**: 8 compute chiplets (XCD) hybrid-bonded on 4 I/O dies (IOD),
surrounded by 8 HBM3 stacks (153 B transistors, 750 W).
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
| XCD | 8 | TSMC N5 | 40 CUs (38 active), 4 ACEs, 4 MiB L2, HW scheduler |
| IOD | 4 | TSMC N6 | HBM3 controllers, Infinity Cache slice, Infinity Fabric, PCIe5, XGMI |
| HBM3 stack | 8 | — | 24 GB each, ~662 GB/s → 192 GB / 5.325 TB/s aggregate |

Each IOD hosts **2 XCDs + 2 HBM stacks**. There is **no device-wide L2**; the first device-shared
level is the **256 MiB Infinity Cache** on the IODs. Cross-XCD sharing/atomics go through Fabric +
Infinity Cache (measured core-to-core atomic latency ~116–202 ns). Details:
[xcd_chiplet.md](xcd_chiplet.md).

### Compute Unit
Each CDNA3 CU: 4 × SIMD64, 4 Matrix Cores (one/SIMD), 512 VGPR + ≤256 AGPR per SIMD, 64 KiB LDS
(32 banks), 32 KiB L1 vector (128 B line), 16 KiB scalar/const. Packed FP16/INT16 run double-rate;
transcendentals 4 ops/SIMD/cycle. Per-CU peak basis in [peak_tables.md](peak_tables.md).

### Partitioning (SPX/DPX/CPX × NPS1/2/4)
The 8-XCD design can be spatially partitioned via `amd-smi`:
- **SPX** (default): 1 logical GPU, 8 XCDs, 304 CU, 192 GB.
- **DPX**: 2× (4 XCD / 152 CU / 96 GB).
- **CPX**: 8× (1 XCD / 38 CU / 24 GB) — many small jobs, inference density.
- Memory: **NPS1/2/4**; memory partitions must **not exceed** compute partitions (SPX+NPS4 invalid).
- In **CPX/NPS4** a kernel sees a 38-CU/24 GB GPU with XCD-local memory → higher effective BW/clocks
  (AMD reports +10–15% compute-bound GEMM vs SPX) because cross-XCD traffic is eliminated. Full table
  in [xcd_chiplet.md](xcd_chiplet.md).

## The levers (architecture-level)
1. **≥1024 workgroups** (≥~3.4/CU) to fill 304 CUs + hide tails.
2. **8-multiple tile dims** for even XCD spread + L2 reuse → [memory_hierarchy.md](memory_hierarchy.md).
3. **`mfma_16x16`** over 32×32 → [matrix_core.md](matrix_core.md).
4. **Push to FP8** for 16× over FP32 where accuracy allows.
5. **Optimize bytes** on decode/norm/attention kernels (HBM-bound).
6. **Keep working sets XCD-local**; for many-small-job density consider CPX/NPS4.

## Pitfalls
- **Treating it as monolithic.** L2 is per-XCD; cross-XCD reuse is not free.
- **Under-launching** (<1024 WGs) leaves XCDs idle.
- **Quoting peak.** ~45% of peak is the common sustained reality (arXiv 2510.27583).

## Verify
- `rocminfo` / `amd-smi static` confirms gfx942, 304 CU, partition mode.
- `rocprof-compute` for occupancy, L2/L3 hit rates, HBM BW, XCD balance.
