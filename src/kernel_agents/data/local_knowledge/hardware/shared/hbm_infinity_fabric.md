---
title: HBM & Infinity Fabric / Infinity Cache (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# HBM, Infinity Fabric & Infinity Cache

## TL;DR
> Most LLM-inference kernels are **HBM-bandwidth-bound**, not FLOP-bound — optimize **bytes moved**,
> not FLOPs. The device-shared cache is the **256 MiB Infinity Cache (MALL/L3)** on the I/O dies; there
> is **no device-wide L2** (L2 is per-XCD on CDNA3/4). Cross-XCD / cross-GPU sharing pays Infinity
> Fabric latency (~hundreds of ns), so keep a kernel's working set local.

## Concepts

### HBM per generation
| Gen | Product | HBM | Capacity | Bandwidth | Bus | Notes |
|---|---|---|---|---|---|---|
| CDNA1 gfx908 | MI100 | HBM2 | 32 GB | 1.23 TB/s | 4096-bit | 4 stacks |
| CDNA2 gfx90a | MI250X | HBM2e | 128 GB | 3.2 TB/s | — | 2 GCDs × 1.6 TB/s; capacity per OAM |
| CDNA2 gfx90a | MI210 | HBM2e | 64 GB | 1.6 TB/s | — | single GCD |
| CDNA3 gfx942 | MI300X | HBM3 | 192 GB | 5.325 TB/s | 8192-bit | 8 stacks × 24 GB, 5.2 Gbps |
| CDNA3 gfx942 | MI325X | HBM3E | 256 GB | 6.0 TB/s | — | same compute, more/faster mem |
| CDNA4 gfx950 | MI350X/355X | HBM3E | **288 GB** | **8.0 TB/s** | — | 8 stacks × 36 GB (12-Hi) |

### The bandwidth ladder (CDNA3 MI300X, representative)
| Level | Size | Scope | Latency | Bandwidth |
|---|---|---|---|---|
| LDS | 64 KiB | workgroup/CU | ~20–30 cyc | highest of any tested GPU |
| L1 vector (TCP) | 32 KiB | CU | tens of cyc | tens of TB/s |
| L2 | 4 MiB | **per XCD** | — | XCD-local |
| Infinity Cache (MALL/L3) | **256 MiB** | **device** | ~218 ns | ~11.9 TB/s (≈17 theoretical) |
| HBM3 | 192 GB | device | +~47 ns TLB miss | 5.325 TB/s |

Cache line = **128 B**. Page = **4 KiB** (use 2 MiB huge pages for >64 MB working sets to extend TLB
reach). Infinity Cache stays 256 MiB on CDNA4 too.

### Roofline ridge (why bytes win)
At MI300X 5.3 TB/s and 1307 FP16 TFLOP/s the FP16 ridge is ≈ **247 FLOP/byte**. Decode-phase kernels
(GEMV, small-batch attention, RMSNorm, RoPE, dequant) sit far left → **bandwidth-bound**: fuse ops,
cut bytes, and exploit Infinity Cache residency. Prefill GEMM with large M sits right → compute-bound.
On CDNA4 the ridge shifts (8 TB/s vs 2.5 PF FP16 → ≈ 312 FLOP/byte) so even more kernels become
bandwidth-bound relative to the bigger matrix core. See [l2_xcd_swizzle.md](l2_xcd_swizzle.md).

### Infinity Fabric (the chiplet/GPU interconnect)
- **On-package (CDNA3/4):** the XCDs and I/O dies are stitched by Infinity Fabric; the device-shared
  coherence point is the Infinity Cache. Measured global-atomic core-to-core latency on MI300X ranges
  **~116–202 ns** depending on whether the two workgroups land on the same or different XCD.
- **GCD↔GCD (CDNA2 MI250X):** 200 GB/s per direction (400 GB/s bidirectional) between the two GCDs in
  one OAM — the two GCDs are **separate GPU devices** to software.
- **GPU↔GPU (inter-package):** 7 Infinity Fabric links per MI300X for an 8-GPU all-to-all ring;
  CDNA4 platform = 4th-gen Infinity Fabric, **1075 GB/s** bidirectional aggregate per card, 8-GPU
  fully connected.

## The levers
1. **Count bytes first.** For any memory-bound kernel, the model is `time ≈ bytes / HBM_BW`; minimize
   reads/writes (fuse, recompute cheap ops, quantize KV).
2. **Coalesce to 128 B aligned**, emit `global_load_dwordx4` (16 B/lane) so each wave fills full cache
   lines. See [l2_xcd_swizzle.md](l2_xcd_swizzle.md).
3. **Size hot read-only data (weights, KV blocks) to live in the 256 MiB Infinity Cache** — it absorbs
   cross-XCD sharing and cuts HBM traffic.
4. **Keep working sets XCD-local** (CDNA3/4): cross-XCD reuse misses the per-XCD L2 and falls to L3 /
   Fabric. For device-wide reductions, stage through L3 and budget ~200 ns sync.
5. **Use huge pages** for large working sets to avoid the ~47 ns TLB-miss penalty per access.
6. **Cache-control flags** (`glc`/`slc`/`dlc`) bypass/stream caches for write-once data.

## Pitfalls
- **Treating MI250X as one GPU.** It is **two GCDs / two devices**; cross-GCD traffic is a 400 GB/s
  link, not on-die bandwidth.
- **Assuming a global L2.** On CDNA3/4 L2 is per-XCD (4 MiB); the first device-shared level is L3.
- **Quoting HBM peak as achievable.** Real sustained BW is below peak; measure with a streaming
  microbench.

## Verify
- `rocprof-compute` memory chart: HBM BW utilization %, L2/L3 hit rates, bytes/kernel.
- `rocm-bandwidth-test` / a streaming-copy microbench for achievable HBM and Fabric BW.
- `rocm-smi --showmeminfo` / `amd-smi` for HBM capacity and partition layout.
