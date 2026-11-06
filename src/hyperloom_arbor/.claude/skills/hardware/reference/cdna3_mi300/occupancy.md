---
title: CDNA3 / MI300X (gfx942) — occupancy math & tuning
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# CDNA3 / MI300X (gfx942) — occupancy math & tuning

> Cross-gen model in [../shared/wavefront_simd_vgpr_agpr.md](../shared/wavefront_simd_vgpr_agpr.md).
> This file is the MI300X numbers and worked examples.

## TL;DR
> Occupancy = resident waves/SIMD, capped at **8** (32/CU). It is the **minimum** of the VGPR limit
> (`floor(512/N)`), the LDS limit (`floor(65536/L)`), and the wave-slot cap. **Register pressure is
> the #1 killer**; attention/softmax are usually **LDS**-limited. HBM-bound kernels want ≥4 waves/CU;
> MFMA-bound GEMM often runs 1–2 wg/CU and hides latency with double-buffered LDS instead.

## Concepts

### Inputs
1. **N** = VGPRs/wave (ISA `.vgpr_count`, **rounded up to a multiple of 16**).
2. **L** = LDS bytes/workgroup.
3. **nW** = waves/workgroup = `ceil(threads_per_block / 64)`.

### Limits (MI300X)
```
occ_vgpr (waves/SIMD)    = floor(512 / N)            # cap 8
occ_lds  (workgroups/CU) = floor(65536 / L)          # 64 KiB LDS
wave_slots               = 8/SIMD = 32/CU            # hard cap
wg_from_vgpr             = floor(occ_vgpr * 4 / nW)  # 4 SIMDs/CU
wg_per_CU                = min(wg_from_vgpr, occ_lds, floor(32 / nW))
waves_per_CU             = wg_per_CU * nW
```

### Worked examples
**A — VGPR-limited GEMM.** N=176, threads=256 (nW=4), L=32 KiB.
```
occ_vgpr=floor(512/176)=2 ; wg_from_vgpr=floor(2*4/4)=2 ; occ_lds=floor(65536/32768)=2
wg_per_CU=min(2,2,8)=2 -> 8 waves/CU
```
Dropping N to 128 gives occ_vgpr=4 but `occ_lds=2` now binds → still 2 wg/CU. Reduce LDS to gain.

**B — LDS-limited attention/softmax.** N=64, threads=512 (nW=8), L=48 KiB.
```
occ_vgpr=floor(512/64)=8 ; wg_from_vgpr=floor(8*4/8)=4 ; occ_lds=floor(65536/49152)=1  <-- binds
wg_per_CU=min(4,1,4)=1 -> 8 waves/CU
```
Halving LDS → occ_lds=2 → 2 wg/CU → 16 waves/CU.

**C — fully occupied bandwidth kernel.** N=48, threads=256 (nW=4), L=8 KiB.
```
occ_vgpr=floor(512/48)=10->cap 8 ; wg_from_vgpr=floor(8*4/4)=8 ; occ_lds=floor(65536/8192)=8
wave-slot cap floor(32/4)=8 -> wg_per_CU=8 -> 32 waves/CU (max)
```

### The 16-VGPR granule trap
170 VGPRs rounds to 176. Watch tier boundaries at 64/80/96/128/168/256 — shaving a few VGPRs across a
boundary can jump an occupancy tier.

## The levers
1. **Cut VGPRs** (smaller tile, `waves_per_eu`, **AGPR escape hatch**
   `-mllvm -amdgpu-mfma-vgpr-form=false -mllvm -amdgpu-agpr-alloc=256`).
2. **Cut LDS** (less buffering, packing) when LDS binds (attention).
3. **`buffer_load_to_lds`** removes staging VGPRs — biggest GEMM occupancy win (see
   [memory_hierarchy.md](memory_hierarchy.md)).
4. **`__launch_bounds__(threads, waves_per_eu)`** to cap VGPRs to a target.
5. **≥4 waves/CU** for HBM-latency hiding; for MFMA GEMM accept 1–2 wg/CU + double-buffered LDS.

## Tuning checklist (from Omniperf symptoms)
| Symptom | Cause | Fix |
|---|---|---|
| Low waves/CU, high VGPR | register pressure | shrink tile, waves_per_eu, AGPR, buffer_load_to_lds |
| Low waves/CU, high LDS | LDS pressure | smaller tiles, less buffering, pack |
| LDS bank-conflict stalls | column access / stride mult of 32 | pad LD (+1/+4), swizzle, ds_*_b128 |
| Low HBM BW | uncoalesced/scalar loads | global_load_dwordx4, 128 B align |
| Memory-latency-bound | too few waves | raise to ≥4 waves/CU; prefetch |
| MFMA underfed | LDS read conflicts / no double-buffer | swizzled LDS, async copy + double buffer |

**Tuned FP16/FP8 GEMM target (MI300X):** 256×256 tile, 2-stage prefetch, **384–448 VGPR** budget,
≥4 waves/CU, `mfma_16x16`, `buffer_load_to_lds`, conflict-free swizzled LDS.

## Pitfalls
- **Maximizing occupancy blindly** — GEMM often wins at low occupancy with deep prefetch.
- **Ignoring the LDS limiter** for attention.
- **Forgetting 16-granule rounding** when budgeting VGPRs.

## Verify
- ISA `.vgpr_count`/`.lds_size`; on-box `occ.sh` (ROCm workload guide) → waves/CU.
- `rocprof-compute` occupancy panel: resident vs theoretical waves, which resource binds.
