---
title: CDNA4 / MI350X / MI355X (gfx950) — clocks, power & thermals
kind: hardware
gens: [gfx950]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA4 / MI350X / MI355X (gfx950) — clocks, power & thermals

> Peak math uses the clocks here; matrix peaks in [peak_tables.md](peak_tables.md). Per-XCD clock
> spread: see [../cdna3_mi300/xcd_chiplet.md](../cdna3_mi300/xcd_chiplet.md) and below. Numbers in
> brackets are the sources at the end.

## TL;DR
> Two SKUs of one chip differ in cooling, power and peak clock: **MI350X air-cooled, 1000 W, 2,200 MHz**;
> **MI355X direct liquid-cooled, 1400 W, 2,400 MHz** [1]. AMD says the MI355X's extra power lets it sustain
> higher performance over time, with less throttling under prolonged load [2]. N3P XCDs, 185 B transistors
> [1]. AMD's peaks are peak theoretical [1]: compute achieved FLOP/s from measured time.

## Concepts

### Clock & power table
| Param | MI350X | MI355X |
|---|---|---|
| Arch / ISA | CDNA4 / gfx950 [3] | CDNA4 / gfx950 [3] |
| Cooling | air [1] | direct liquid [1] |
| TDP | **1000 W** [1] | **1400 W** [1] |
| Peak engine clock | **2,200 MHz** [1] | **2,400 MHz** [1][2] |
| Process | TSMC **N3P** (XCD) + N6 (IOD) [1] | N3P + N6 [1] |
| Transistors | 185 B [1] | 185 B [1] |
| HBM | 288 GB HBM3E, 8 TB/s [1] | 288 GB HBM3E, 8 TB/s [1] |
| Server tray | 4 RU, UBB8 drop-in for MI325X platforms [1] | 2 RU, direct liquid cooling [1] |

Both have the **same compute** (256 CUs, 1,024 matrix cores) [1]. Their published peaks differ by the
clock ratio 2,200 / 2,400 (FP16 matrix 2.3 vs 2.5 PF, FP8 4.6 vs 5.0 PF) [1][4]; the MI355X's higher power
envelope also sustains higher performance under prolonged load [2].

### Clock dynamics that bite kernels
- **Peak ≠ sustained.** AMD's peaks are theoretical, at peak engine clock [1]. Always compute achieved
  TFLOP/s from wall time.
- **Per-XCD clock spread.** On MI300X, AMD documents a 3–10% clock spread among XCDs (XCD0 typically
  fastest, XCD7 slowest) and advises computing efficiency against the slowest XCD [5]. AMD's documents
  give no figure for MI350X/MI355X.
- **HBM bandwidth (8 TB/s)** comes from the memory interface: 8 Gbps HBM3E [1] over 8 stacks × 1024 bits
  [1] = 8.2 TB/s. Both SKUs publish the same 8 TB/s at different engine clocks [1], so bandwidth-bound
  kernels gain from fewer bytes, not from clock headroom (follows from those facts).

## The levers
1. **Measure achieved FLOP/s** from time; treat peak clock as a ceiling.
2. **Warm up + median-of-≥3** before quoting a number.
3. **Prefer MI355X (1400 W)** for compute-bound throughput: the higher peak clock [1] and AMD's sustained-load
   claim [2]; MI350X (1000 W) where air cooling is required [1].
4. **For bandwidth-bound work**, cut bytes — the engine clock does not set the bandwidth (above).

## Pitfalls
- **Using peak clock in efficiency claims** — overstates utilization.
- **Comparing MI350X vs MI355X by peak tables** — they differ only by the 2,200 vs 2,400 MHz peak clock
  [1]; under prolonged load the power envelope matters too [2].

## Verify
- `amd-smi metric --gpu <id>` (sclk/mclk/power/temp/throttle) during the kernel.
- `rocprof-compute` achieved vs theoretical at the measured clock.

## Sources
1. AMD, *Introducing AMD CDNA 4 Architecture* (white paper): products and cooling, memory (8 Gbps HBM3E,
   1024 bits × 8 stacks), Table 2 (MI350X and MI355X specifications).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
2. AMD, *AMD Instinct MI355X GPU* datasheet ("sustain higher performance over time, minimizing
   throttling"; 2.4 GHz peak engine clock).
   <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf>
3. ROCm documentation, *AMD GPU specifications*.
   <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
4. ROCm documentation, *AMD Instinct MI350 Series microarchitecture* (peak theoretical performance table).
   <https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html>
5. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* ("Clock
   differences among XCDs").
   <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
