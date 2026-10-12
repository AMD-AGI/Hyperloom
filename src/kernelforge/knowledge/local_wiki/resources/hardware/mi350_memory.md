---
title: MI350X — memory hierarchy, HBM3E, Infinity Cache, coalescing
kind: hardware
topic: memory
gens: [gfx950]
updated: 2026-08-28
---

# Memory hierarchy

**Most LLM-inference kernels are HBM-bandwidth-bound, not FLOP-bound.** Optimize **bytes moved**, not
FLOPs. The device-shared cache is the **256 MiB Infinity Cache**; there is **no device-wide L2**.

## The ladder

| Level | Capacity | Scope | Bandwidth |
|---|---|---|---|
| VGPR / AGPR | 512 / ≤256 × 4 B per SIMD | wave | register speed |
| LDS | **160 KiB/CU**, 64 banks | workgroup / CU | **256 B/clk**, ~20–30 cyc |
| L1 vector (TCP) | 32 KiB/CU, 128 B line | CU | tens of TB/s |
| **L2** | **per-XCD** | one XCD | XCD-local — a cross-XCD access is *not* an L2 hit |
| Infinity Cache (MALL/L3) | **256 MiB** | device | first device-shared level; ~hundreds of ns |
| HBM3E | **288 GB** (8 × 36 GB, 12-Hi) | device | **8.0 TB/s** peak |

Cache line = **128 B**. Page = **4 KiB** — use 2 MiB huge pages for working sets over ~64 MB to extend
TLB reach.

## Roofline ridge — why bytes win

At **8 TB/s** against **2.5 PF** FP16 the ridge is ≈ **312 FLOP/byte**.

| dtype | ridge |
|---|---|
| FP16 / BF16 | ≈ **312 FLOP/byte** |
| FP8 | ≈ 625 |
| FP6 / FP4 | ≈ 1250 |
| FP32 | ≈ 20 |

Decode-phase kernels (GEMV, small-batch attention, RMSNorm, RoPE, dequant) sit far left → **bandwidth-
bound**: fuse, cut bytes, exploit Infinity Cache residency. Prefill GEMM with large M sits right →
compute-bound.

The ridge is **higher than CDNA3's ≈247** because the matrix core doubled while bandwidth grew less.
**More kernels are bandwidth-bound here than on MI300X** — byte-cutting matters more, not less.

## Coalescing

One memory instruction issues **64 lane addresses**. The hardware merges lanes falling in the same
128 B cache line into one transaction.

- Widest single access is **`global_load_dwordx4`** = **128-bit / 16 B per lane**, requiring a
  **16-byte aligned** address. 16 B × 64 lanes = **1024 B**, exactly 8 cache lines — that is the target
  shape for every streaming access.
- Index so **lane `i` reads element `base + i`**; the innermost dimension runs along the wave.
- An **odd row stride breaks vectorization on every row** — a common silent regression when a tensor
  is sliced or a head-dim is not a power of two.
- `buffer_load` / `buffer_store` with a descriptor (V#) gives hardware bounds-checked OOB handling —
  cheaper than branchy guards in a tiled loop.
- Need the transposed order? Do the transpose **in LDS**, not with strided global reads
  (`mi350_lds.md`).

**Coalescing is not bank conflicts.** Coalescing is about *global* transactions across 64 lanes at
128 B granularity; bank conflicts are about *LDS* banks within a half-wave. Separate axes, separate
fixes.

## Infinity Fabric

- **On-package**: 8 XCDs and 2 I/O dies stitched by Infinity Fabric; the device-shared coherence point
  is the Infinity Cache. Cross-XCD atomics and device-wide reductions pay Fabric latency on the order
  of a couple hundred ns → `mi350_chiplet.md`.
- **Inter-package**: 4th-gen Infinity Fabric, **1075 GB/s** bidirectional aggregate per card, 8-GPU
  fully connected.

## What it means for kernels

1. **Count bytes first.** For any memory-bound kernel the model is `time ≈ bytes / HBM_BW`; minimize
   reads/writes (fuse, recompute cheap values, quantize KV).
2. **Coalesce to 128 B aligned**, emit `global_load_dwordx4` so each wave fills full cache lines.
3. **Size hot read-only data (weights, KV blocks) to live in the 256 MiB Infinity Cache** — it absorbs
   cross-XCD sharing and cuts HBM traffic.
4. **Keep working sets XCD-local** — cross-XCD reuse misses the per-XCD L2 and falls to L3/Fabric.
5. **Use huge pages** for large working sets.
6. **Cache-control flags** (`glc`/`slc`/`dlc`) to bypass or stream caches for write-once data.

## Pitfalls
- **Assuming a global L2** — it is per-XCD; the first shared level is the 256 MiB Infinity Cache.
- **Quoting HBM peak as achievable** — sustained is below 8.0 TB/s; measure with a streaming microbench.
- **Reusing an MI300X byte budget** — capacity 192 → 288 GB, bandwidth 5.3 → 8.0 TB/s, and the ridge
  moved from ≈247 to ≈312 FLOP/byte.
- **Wide loads in source, narrow in ISA** — the compiler could not prove 16 B alignment.
- **Over-wide loads on tiny tensors** — wasted tail lanes and predication overhead.
- **Assuming wave32** — the coalescing window is **64 lanes**.

## Verify
- `rocprof-compute` memory chart: HBM BW utilization %, L2/L3 hit rates, bytes/kernel, transaction
  efficiency (want near 128 B per transaction).
- ISA dump: count `global_load_dwordx4` vs `global_load_dword` in the hot loop — wide forms should
  dominate.
- `rocm-bandwidth-test` or a streaming-copy microbench for achievable HBM and Fabric BW.
- `rocm-smi --showmeminfo` / `amd-smi` for HBM capacity and partition layout.

## Related
`mi350_lds.md` (the level below) · `mi350_chiplet.md` (per-XCD L2 and locality) ·
`mi350_overview.md` (peaks and ridges) ·
`common_methodology/optimization/lever_coalescing.md`
