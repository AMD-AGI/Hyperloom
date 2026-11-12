---
title: HBM & Infinity Fabric / Infinity Cache (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# HBM, Infinity Fabric & Infinity Cache

> Numbers in brackets point to the Sources at the end.

## TL;DR
> Kernels left of the roofline ridge (derived below) — decode GEMV, small-batch attention, norms — are
> **HBM-bandwidth-bound**, not FLOP-bound: optimize **bytes moved**, not FLOPs. The device-shared cache
> is the **256 MiB Infinity Cache (MALL/L3)** on the I/O dies; there is **no device-wide L2** (L2 is
> per-XCD on CDNA3/4) [6][9]. Cross-XCD and cross-GPU sharing goes over Infinity Fabric, so keep a
> kernel's working set local.

## Concepts

### HBM per generation
| Gen | Product | HBM | Capacity | Bandwidth | Bus | Notes |
|---|---|---|---|---|---|---|
| CDNA1 gfx908 | MI100 | HBM2 | 32 GB [1] | 1.23 TB/s [1] | 4096-bit [2] | 2.4 GT/s [1] |
| CDNA2 gfx90a | MI250X | HBM2e | 128 GB [3] | 3.2 TB/s [3] | 8192-bit [4] | 2 GCDs × 64 GB [3] |
| CDNA2 gfx90a | MI210 | HBM2e | 64 GB [3] | 1.6 TB/s [3] | 4096-bit [5] | single GCD [3] |
| CDNA3 gfx942 | MI300X | HBM3 | 192 GB [7] | 5.3 TB/s [7] | 8192-bit [7] | 8 stacks × 24 GB, 5.2 Gbps [6] |
| CDNA3 gfx942 | MI325X | HBM3E | 256 GB [6] | 6.0 TB/s [6] | 8192-bit [8] | 8 stacks × 32 GB, 6.0 Gbps [6]; same compute as MI300X [6] |
| CDNA4 gfx950 | MI350X/355X | HBM3E | **288 GB** [9] | **8.0 TB/s** [9] | — | 8 stacks × 36 GB (12-Hi), 8 Gbps [9] |

### The bandwidth ladder (CDNA3 MI300X)
| Level | Size | Scope | Bandwidth |
|---|---|---|---|
| LDS | 64 KiB [14] | workgroup/CU | 128 B/clk per CU [12] |
| L1 vector (TCP) | 32 KiB [6][11] | CU | — |
| L2 | 4 MiB [6][11] | **per XCD** | 2 KB/clk per XCD (16 channels × 128 B) [6] |
| Infinity Cache (MALL/L3) | **256 MiB** [6] | **device** | 17.2 TB/s peak [6] |
| HBM3 | 192 GB [7] | device | 5.3 TB/s [7] |

Cache line = **128 B** [6]. Infinity Cache stays 256 MiB on CDNA4 [9].

### Roofline ridge (why bytes win)
At MI300X 5.3 TB/s and 1307.4 FP16 TFLOP/s [7] the FP16 ridge is ≈ **247 FLOP/byte** (derived).
Decode-phase kernels (GEMV, small-batch attention, RMSNorm, RoPE, dequant) sit far left →
**bandwidth-bound**: fuse ops, cut bytes, and exploit Infinity Cache residency. Prefill GEMM with large
M sits right → compute-bound. On MI355X the ridge shifts (2.5 PF FP16 / 8 TB/s [9] ≈ 312 FLOP/byte,
derived), so even more kernels become bandwidth-bound relative to the bigger matrix core. See
[l2_xcd_swizzle.md](l2_xcd_swizzle.md).

### Infinity Fabric (the chiplet/GPU interconnect)
- **On-package (CDNA3/4):** Infinity Fabric ties the XCDs and I/O dies together [6][9]. Each XCD's L2
  is coherent within that XCD; the Infinity Cache is a memory-side cache that holds no dirty L2 data,
  and its snoop filter resolves most coherent requests between XCDs [6].
- **GCD↔GCD (CDNA2 MI250X):** up to 400 GB/s bidirectional between the two GCDs in one package
  [3] — each GCD appears to software as a **separate GPU** [3].
- **GPU↔GPU (inter-package):** MI300X has 7 Infinity Fabric links of 128 GB/s each, connecting 8 GPUs
  fully [7][15]. CDNA4 uses 4th-gen Infinity Fabric with 7 x16 links and 8 GPUs directly connected [9];
  AMD's sources give its peer-to-peer aggregate as **1,008 GB/s** (CDNA4 white paper Table 2) and as
  **1,075 GB/s** (the same paper's Figure 5, and the workload guide's 1,075.2 GB/s) [9][10].

## The levers
1. **Count bytes first.** For any memory-bound kernel, the model is `time ≈ bytes / HBM_BW`; minimize
   reads/writes (fuse, recompute cheap ops, quantize KV).
2. **Coalesce to aligned cache lines**, emit `global_load_dwordx4` (16 B/lane) [10][12]. See
   [l2_xcd_swizzle.md](l2_xcd_swizzle.md).
3. **Size hot read-only data (weights, KV blocks) to live in the 256 MiB Infinity Cache** — it is the
   one cache every XCD shares [6], and hits there save HBM traffic.
4. **Keep working sets XCD-local** (CDNA3/4): cross-XCD reuse misses the per-XCD L2 and falls to the
   Infinity Cache [6]; place data-sharing workgroups on one XCD [10].
5. **Cache-control bits**: on gfx942, vector memory instructions carry scope bits `sc0/sc1` and a
   non-temporal bit `nt` (`nt=1`: no temporal reuse expected) [14], for write-once or streaming data.

## Pitfalls
- **Treating MI250X as one GPU.** It is **two GCDs, each its own GPU**; cross-GCD traffic goes over the
  400 GB/s in-package link [3], not on-die bandwidth.
- **Assuming a global L2.** On CDNA3/4 L2 is per-XCD (4 MiB) [6][9]; the first device-shared level is
  the Infinity Cache.
- **Quoting HBM peak as achievable.** One published study measured ~81% of MI300X's 5.3 TB/s with
  BabelStream [13]; measure with a streaming microbench.

## Verify
- `rocprof-compute` memory chart: HBM BW utilization %, L2/L3 hit rates, bytes/kernel.
- `rocm-bandwidth-test` / a streaming-copy microbench for achievable HBM and Fabric BW.
- `rocm-smi --showmeminfo` / `amd-smi` for HBM capacity and partition layout.

## Sources
1. AMD, *AMD CDNA Architecture* white paper (MI100; L2 cache and memory) — https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf
2. AMD, *AMD Instinct MI100* product page — https://www.amd.com/en/products/accelerators/instinct/mi100.html
3. AMD, *AMD CDNA 2 Architecture* white paper (memory; in-package Infinity Fabric) — https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf
4. AMD, *AMD Instinct MI250X* product page — https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html
5. AMD, *AMD Instinct MI210* product page — https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html
6. AMD, *AMD CDNA 3 Architecture* white paper (memory section) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
7. AMD, *AMD Instinct MI300X Accelerator* data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
8. AMD, *AMD Instinct MI325X* product page — https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
9. AMD, *AMD CDNA 4 Architecture* white paper (Figure 5, memory section, Table 1, Table 2) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
10. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
11. ROCm documentation, *AMD GPU specifications* — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
12. HIP documentation, *Hardware implementation* (LDS bandwidth; coalescing) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
13. C. Ambati, T. Diep, *AMD MI300X GPU Performance Analysis*, arXiv:2510.27583 (§3) — https://arxiv.org/abs/2510.27583
14. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §2.2.1 LDS, §9.1.10 scope bits) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
15. ROCm documentation, *AMD Instinct MI300 Series microarchitecture* (node-level architecture) — https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html
