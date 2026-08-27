---
title: CDNA3 / MI300X (gfx942) — memory hierarchy & load/store
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA3 / MI300X (gfx942) — memory hierarchy & load/store

> Register/occupancy math is in [occupancy.md](occupancy.md); LDS bank rules in
> [../shared/memory_model_lds_bank.md](../shared/memory_model_lds_bank.md); HBM/Fabric in
> [../shared/hbm_infinity_fabric.md](../shared/hbm_infinity_fabric.md). This file is the MI300X ladder
> plus the load/store paths that keep MFMA fed.

## TL;DR
> The ladder: VGPR/AGPR → 64 KiB LDS → 32 KiB L1/CU → **4 MiB L2 per XCD** → **256 MiB Infinity Cache
> (device)** → 192 GB HBM3 @ 5.325 TB/s. There is **no global L2**. The biggest single GEMM win is
> **direct global→LDS** (`buffer_load ... lds`): on a reference kernel it freed ~100 VGPR/wave and
> moved 697 → 1113 TFLOP/s.

## Concepts

### The ladder (decision-driving numbers)
| Level | Capacity | Scope | Latency | Bandwidth | Alloc unit |
|---|---|---|---|---|---|
| VGPR | 512 ×4 B/SIMD | wave | register | — | 16 regs |
| AGPR | ≤256 ×4 B/SIMD | wave | register (MFMA) | — | — |
| LDS | **64 KiB/CU** (32 banks) | workgroup | ~20–30 cyc | highest tested | — |
| L1 vector (TCP) | 32 KiB/CU, 128 B line | CU | tens of cyc | tens of TB/s | — |
| L2 | **4 MiB/XCD** (16×256 KiB) | **per XCD** | — | XCD-local | — |
| Infinity Cache (MALL/L3) | **256 MiB** | **device** | ~218 ns | ~11.9 TB/s | — |
| HBM3 | **192 GB** | device | +~47 ns TLB miss | **5.325 TB/s** | 4 KiB page |

Cache line 128 B; page 4 KiB (huge pages for >64 MB working sets). FP16 roofline ridge ≈ 247 FLOP/byte
→ decode/norm/attention sit left of the ridge (HBM-bound).

### Global loads/stores: coalescing & paths
- Coalesce so a wave's 64 lanes touch a contiguous **128 B-aligned** region; emit
  `global_load_dwordx4` (16 B/lane) to fill full cache lines with fewer instructions.
- Three vector-memory paths: **`global_*`** (flat addressing, default for HIP pointers);
  **`buffer_*`** (descriptor V#, bounds-checked OOB → cheap guards in tiled GEMM, supports
  `glc/slc/dlc`); **`ds_*`** (LDS, prefer `ds_read_b128`/`ds_write_b128`).

### Direct global→LDS (the big win)
CDNA3 supports copying global memory straight into LDS, **bypassing VGPRs** (up to 32 b/lane):
```asm
buffer_load_dwordx4 ... lds        ; into LDS, no VGPR staging
global_load_lds_dwordx4 ...        ; flat variant
```
In Triton/Gluon this is `buffer_load_to_lds`. It removes the register round-trip and the copy index
math; measured: ~100 VGPR/wave freed → 697 → 1113 TFLOP/s on a reference GEMM (ROCm Triton guide).
Pair with **double buffering**: MFMA consumes LDS buffer 0 while async loads fill buffer 1.
(CDNA4 widens this to 128 b/lane — see [../cdna4_mi350/memory.md](../cdna4_mi350/memory.md).)

```cpp
// Double-buffered K-loop skeleton
load_global_to_lds(buf[0], A_k0, B_k0);             // prologue
for (int k = 1; k < K_tiles; ++k) {
    load_global_to_lds(buf[k & 1], A_k, B_k);       // prefetch next
    s_waitcnt(prev);                                // wait only for what MFMA needs
    mfma_accumulate(acc, buf[(k-1) & 1]);
}
mfma_accumulate(acc, buf[(K_tiles-1) & 1]);         // epilogue
```
Synchronization is **`s_waitcnt vmcnt/lgkmcnt`** (count-based) — wait only for the specific outstanding
loads, enabling deep overlap.

### Caches: exploit vs avoid
| Cache | Behavior | Optimization |
|---|---|---|
| L1 (32 KiB/CU) | write-through, 128 B line, per-CU | coalesce; reuse within a workgroup |
| L2 (4 MiB/XCD) | coalesces XCD traffic, **per-chiplet** | keep working set XCD-local; cross-XCD reuse misses to L3 |
| Infinity Cache (256 MiB) | device L3, ~218 ns, coherence point | size hot read-only data (weights, KV) to reside here |

### XCD locality & the 512 B Tagram hotspot
8-multiple tiles spread evenly across the 8 XCDs and reuse hits the same XCD's L2. A GEMM whose LD
byte stride is a **multiple of 512 B** (TN case) can hit the **L2 Tagram hotspot** — pad the LD off
512 B. Full treatment: [../shared/l2_xcd_swizzle.md](../shared/l2_xcd_swizzle.md).

## The levers
1. **`buffer_load_to_lds` + double-buffer** — top GEMM win (occupancy + no round-trip).
2. **`global_load_dwordx4`, 128 B-aligned** for full cache lines.
3. **`buffer_*`** for cheap bounds checks in tiled kernels.
4. **8-multiple tiles**; **pad LD off 512 B** (Tagram).
5. **Size hot data to the 256 MiB Infinity Cache**; keep reuse XCD-local.
6. **Huge pages** for big working sets (TLB reach).
7. `OPTIMIZE_EPILOGUE=1` to skip the epilogue LDS reblock → [../shared/l2_xcd_swizzle.md](../shared/l2_xcd_swizzle.md).

## Pitfalls
- **Register-staged copies** instead of `buffer_load_to_lds` waste VGPRs and occupancy.
- **Scalar/uncoalesced loads** crater HBM BW.
- **Assuming global L2** — cross-XCD reuse falls to L3.
- **512 B-stride TN GEMM** → Tagram cliff.

## Verify
- `rocprof-compute` memory chart: HBM BW %, L1/L2/L3 hit rates, bank-conflict rate, bytes/kernel.
- ISA dump for `.vgpr_count`/`.lds_size`; confirm `buffer_load ... lds` is emitted.
