---
title: CDNA3 / MI300X (gfx942) — clocks, power & thermals
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# CDNA3 / MI300X (gfx942) — clocks, power & thermals

> Companion to [peak_tables.md](peak_tables.md) (all peak numbers use the clocks here) and
> [xcd_chiplet.md](xcd_chiplet.md) (per-XCD clock variance). Numbers in brackets point to the Sources
> at the end.

## TL;DR
> Peak engine clock **2100 MHz** [1][3], board power **750 W** (MI300X) [1] / **1000 W** (MI325X) [3].
> Peak FLOP math uses 2.1 GHz, but **the clock under sustained GEMM load is lower** (one study recorded
> 1083–1217 MHz [4]), and the **8 XCDs vary 3–10%** in clock [2] — so benchmark with warm repeats and
> never assume all CUs run at 2.1 GHz.

## Concepts

### Clock & power table
| Param | MI300X | MI325X |
|---|---|---|
| Arch / ISA | CDNA3 / gfx942 [2] | CDNA3 / gfx942 [2] |
| Peak engine clock | **2100 MHz** [1] | **2100 MHz** [3] |
| Board power | **750 W** [1] | **1000 W** [3] |
| Process | TSMC N5 (XCD) + N6 (IOD) [5] | N5 + N6 [2] |
| HBM | 192 GB HBM3, 5.3 TB/s [1] | 256 GB HBM3E, 6.0 TB/s [5] |
| Form factor | OAM [1] | OAM [2] |

The two share the **same compute** (304 CU @ 2.1 GHz, identical peak FLOPS) [5]; MI325X differs in
memory (more/faster HBM3E) and the higher 1000 W envelope [2][5].

### Clock dynamics that bite kernels
- **Peak ≠ sustained.** 2.1 GHz is the peak engine clock [1]. A published study recorded 1083–1217 MHz
  while running FP16/BF16/FP8 GEMMs on MI300X, one reason it measured ~45% of peak FLOPs on average
  [4]. Always compute achieved TFLOP/s from measured time, not from clock.
- **Per-XCD clock variance 3–10%**: typically XCD0 runs fastest and XCD7 slowest, and AMD recommends
  using the slowest XCD's average clock for efficiency calculations [2] (see
  [xcd_chiplet.md](xcd_chiplet.md)). It follows that device-wide-synchronized kernels run at the
  slowest XCD's pace; independent grids are affected only through load balance.

### Memory vs engine clock
- HBM bandwidth is set by the memory data rate: 8192-bit interface × 5.2 GT/s / 8 ≈ 5.32 TB/s, quoted
  as 5.3 TB/s [1]. That rate does not depend on the engine clock, so bandwidth-bound kernels gain
  nothing from engine-clock headroom, only from fewer bytes (derived).

## The levers
1. **Measure achieved TFLOP/s** from wall time; treat 2.1 GHz peak as a ceiling only [1][4].
2. **Warm up and take the median of repeats** to absorb run-to-run clock differences across XCDs [2].
3. **For compute-bound work**, watch clock, power and throttle status during the kernel
   (`amd-smi metric`).
4. **For bandwidth-bound work**, engine clock is irrelevant — cut bytes.
5. **CPX/NPS4** ran at higher compute clocks than SPX/NPS1 in AMD's GEMM test, thanks to localized
   memory accesses [6].

## Pitfalls
- **Using 2.1 GHz in efficiency claims** — overstates utilization; use the slowest XCD's clock [2] or
  measured time.
- **Assuming uniform clock across XCDs** — 3–10% spread [2].

## Verify
- `amd-smi metric --gpu <id>` (sclk, mclk, power, temperature, throttle status) during the kernel.
- `rocprof-compute` reports achieved vs theoretical at the measured clock.

## Sources
1. AMD, *AMD Instinct MI300X Accelerator* data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
2. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* (architecture tables; "Clock differences among XCDs") — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
3. AMD, *AMD Instinct MI325X* product page — https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
4. C. Ambati, T. Diep, *AMD MI300X GPU Performance Analysis*, arXiv:2510.27583 (Table 2, measured clock) — https://arxiv.org/abs/2510.27583
5. AMD, *AMD CDNA 3 Architecture* white paper (Table 1; memory section) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
6. ROCm Blogs, *Deep dive into the MI300 compute and memory partition modes* (GEMM benchmark) — https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
