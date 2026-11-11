---
title: CDNA3 / MI300X (gfx942) — memory hierarchy & load/store
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# CDNA3 / MI300X (gfx942) — memory hierarchy & load/store

> Register/occupancy math is in [occupancy.md](occupancy.md); LDS bank rules in
> [../shared/memory_model_lds_bank.md](../shared/memory_model_lds_bank.md); HBM/Fabric in
> [../shared/hbm_infinity_fabric.md](../shared/hbm_infinity_fabric.md). This file is the MI300X ladder
> plus the load/store paths that keep MFMA fed. Numbers in brackets point to the Sources at the end.

## TL;DR
> The ladder: VGPR/AGPR → 64 KiB LDS → 32 KiB L1/CU → **4 MiB L2 per XCD** → **256 MiB Infinity Cache
> (device)** → 192 GB HBM3 @ 5.3 TB/s [1][2][4]. There is **no global L2** [1]. **Direct global→LDS**
> (`buffer_load ... lds`) skips the VGPR round-trip; on gfx942 it moves at most 32 bits per lane [3].
> AMD's guide reports it saving ~100 VGPRs per wave and lifting its gfx950 reference GEMM from 697 to
> 1113 TFLOP/s [5].

## Concepts

### The ladder (decision-driving numbers)
| Level | Capacity | Scope | Bandwidth | Alloc unit |
|---|---|---|---|---|
| VGPR | 512 per SIMD lane (512 KiB file per CU) [2][5] | wave | — | 8 regs [3] (the workload guide uses 16 [5]) |
| AGPR | ≤256 of those 512 [3] | wave | — | — |
| LDS | **64 KiB/CU** (32 banks) [3] | workgroup | 128 B/clk per CU [6][5] | 512 B [3] |
| L1 vector | 32 KiB/CU, 128 B line [1] | CU | — | — |
| L2 | **4 MiB/XCD** (16 channels × 256 KiB, 16-way) [1] | **per XCD** | 128 B per channel per clock, 2 KB/clk per XCD [1] | — |
| Infinity Cache (MALL/L3) | **256 MiB**, 16-way [1] | **device** | 17.2 TB/s peak [1] | — |
| HBM3 | **192 GB** (8 × 24 GB) [1] | device | **5.3 TB/s** [4] | — |

Cache line 128 B [1]. FP16 roofline ridge ≈ 1307.4 TF / 5.3 TB/s ≈ 247 FLOP/byte (derived from [4])
→ decode/norm/attention kernels with lower arithmetic intensity sit left of the ridge (HBM-bound).

### Global loads/stores: coalescing & paths
- Coalesce so consecutive lanes touch consecutive addresses, aligned to cache-line boundaries [6];
  make sure loops emit `global_load_dwordx4` (16 B/lane) [5] to fill full 128 B lines with fewer
  instructions.
- Three vector-memory paths: **`global_*`** (FLAT-format global aperture); **`buffer_*`** (descriptor
  V#, range-checked: out-of-range reads return zero and writes are dropped → cheap guards in tiled
  GEMM; scope bits `sc0/sc1` and non-temporal `nt`) [3]; **`ds_*`** (LDS, prefer the `_b128` forms [5]).

### Direct global→LDS (the big win)
CDNA3 can load from a buffer or the global aperture straight into LDS, **bypassing VGPRs** [3]. The
gfx942 opcodes stop at a dword, so at most **32 bits per lane** [3]:
```asm
buffer_load_dword ... lds          ; MUBUF with LDS=1: into LDS, no VGPR staging
global_load_lds_dword ...          ; global-aperture variant
```
In Triton/Gluon this is `buffer_load_to_lds` [5]. AMD's guide: it removes a whole register-movement
phase, saves ~100 VGPRs per wave, and took its gfx950 Gluon reference GEMM from 697 to 1113 TFLOP/s
[5]. Pair with **double buffering**: MFMA consumes LDS buffer 0 while async loads fill buffer 1 [5].
(CDNA4 adds dwordx3/dwordx4, up to 128 b/lane [7] — see [../cdna4_mi350/memory.md](../cdna4_mi350/memory.md).)

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
Synchronization is **`s_waitcnt vmcnt/lgkmcnt`** (count-based) [3] — wait only for the specific
outstanding loads, enabling deep overlap [5].

### Caches: exploit vs avoid
| Cache | Behavior | Optimization |
|---|---|---|
| L1 (32 KiB/CU) | 128 B line, per-CU [1]; write-through to L2 [6] | coalesce; reuse within a workgroup |
| L2 (4 MiB/XCD) | writeback, write-allocate, coalesces the XCD's traffic, coherent within the XCD [1] | keep working set XCD-local; cross-XCD reuse misses to the Infinity Cache |
| Infinity Cache (256 MiB) | memory-side cache on the IODs; holds no dirty L2 data; its snoop filter covers the XCD L2s [1] | size hot read-only data (weights, KV) to reside here |

### XCD locality & the 512 B Tagram hotspot
Use multiples of 8 (the XCD count) for L2 efficiency, and put workgroups that share data on the same
XCD [5]. A GEMM whose matrix stride is a **multiple of 512 B** can cause **Tagram channel hotspotting**,
especially in the TN case; pad the stride (e.g. `lda = ldb = K + 128` for TN F16 when `K % 256 == 0`)
[5]. Full treatment: [../shared/l2_xcd_swizzle.md](../shared/l2_xcd_swizzle.md).

## The levers
1. **`buffer_load_to_lds` + double-buffer** — top GEMM win (occupancy + no round-trip) [5].
2. **`global_load_dwordx4`, cache-line aligned** for full cache lines [5][6].
3. **`buffer_*`** for cheap bounds checks in tiled kernels [3].
4. **Multiples of 8**; **pad strides off 512 B** (Tagram) [5].
5. **Size hot data to the 256 MiB Infinity Cache**; keep reuse XCD-local [1][5].
6. `OPTIMIZE_EPILOGUE=1` (Triton on ROCm) to skip the epilogue LDS round-trip [8] →
   [../shared/l2_xcd_swizzle.md](../shared/l2_xcd_swizzle.md).

## Pitfalls
- **Register-staged copies** instead of `buffer_load_to_lds` waste VGPRs and occupancy [5].
- **Scalar/uncoalesced loads** waste bandwidth [6].
- **Assuming global L2** — cross-XCD reuse falls to the Infinity Cache [1].
- **512 B-stride TN GEMM** → Tagram hotspot [5].

## Verify
- `rocprof-compute` memory chart: HBM BW %, L1/L2/L3 hit rates, bank-conflict rate, bytes/kernel.
- ISA dump for `.vgpr_count` and the LDS size; confirm `buffer_load ... lds` is emitted.

## Sources
1. AMD, *AMD CDNA 3 Architecture* white paper (compute unit and memory sections) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
2. ROCm documentation, *AMD GPU specifications* — https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
3. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §2.2.1 LDS, §3.6.4–3.6.5 VGPR and LDS allocation, §9.1.5 range checking, §9.1.9 buffer load to LDS, §9.1.10 scope bits, §12.15 global opcodes) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
4. AMD, *AMD Instinct MI300X Accelerator* data sheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
5. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
6. HIP documentation, *Hardware implementation* (LDS bandwidth, vector L1, coalescing) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
7. AMD, *AMD Instinct CDNA4 Instruction Set Architecture* (§9.1.9 buffer load to LDS) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
8. ROCm 6.1.1 documentation, *Optimizing Triton kernels* (`OPTIMIZE_EPILOGUE`) — https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
