---
title: RDNA 3 (gfx1100) — LDS, caches and memory bandwidth
kind: hardware
topic: memory
gens: [gfx1100]
updated: 2026-11-16
---

# LDS, caches and memory bandwidth

## LDS

| | gfx1100 | Source |
|---|---|---|
| Size | **128 KiB per WGP** | ISA §1.2.2.1, ROCm GPU specs |
| Banks | 64 banks × 4 B (two sets of 32, one per pair of SIMD32s) | ISA LDS chapter |
| Max per work-group | **64 KiB** | ISA §3.3.4 |
| Allocation granule | 1 KiB | ISA §3.3.4 |

- A work-group that requests more than 64 KiB does not launch. CDNA kernels sized for MI355X's 160 KiB must be
  re-tiled, not just re-tuned.
- **WGP mode (default):** one contiguous LDS; two 64 KiB work-groups can share a WGP. **CU mode:** each
  work-group sees only its CU's 64 KiB half.
- LDS also caps occupancy: work-groups per WGP ≤ floor(128 KiB / LDS per work-group). A 48 KiB work-group
  leaves room for two per WGP, not three.
- Bank conflicts follow the usual rule: lanes hitting different addresses in the same 4-byte bank serialize.
  Pad rows or XOR-swizzle the column index as on CDNA, but size the swizzle for 32-lane waves.

## Cache ladder (W7900)

| Level | Size | Scope |
|---|---|---|
| L0 vector | 32 KiB | per CU |
| L0 scalar | 16 KiB | per CU |
| L1 | 256 KiB | per shader array |
| L2 | 6 MiB | whole GPU |
| Infinity Cache | 96 MiB | whole GPU, in front of GDDR6 |

(ROCm GPU hardware specifications, W7900 row.) The cache line size is not published in a primary source we
could find; do not hard-code one from memory — measure with a stride sweep if a decision depends on it.

## DRAM

- **GDDR6, 384-bit, 864 GB/s, 48 GB** (AMD product page). One die, so there is no XCD/chiplet L2 locality to
  manage and no CTA swizzle across dies.
- The ridge is ≈ 142 FLOP/byte for FP16 WMMA. Most inference kernels (decode GEMV, normalization, softmax,
  elementwise, small-batch attention) are bandwidth-bound here; optimize bytes moved first.
- Working sets under ~96 MiB can run above 864 GB/s from the Infinity Cache. When a benchmark's buffers are
  small enough to stay resident, its bandwidth reflects the cache, not DRAM; size the benchmark like the real
  workload before trusting a bandwidth-bound conclusion.
- Coalescing: a wave32 load of 4-byte elements covers 128 bytes; use 16-byte vector loads per lane
  (`global_load_b128`) to move 512 bytes per wave32 instruction.
