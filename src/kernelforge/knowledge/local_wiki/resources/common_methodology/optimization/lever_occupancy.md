---
title: occupancy — register budget, wave slots, and the spill cliff
kind: lever
lever: occupancy
gens: [gfx950]
bottleneck: latency / occupancy-bound
updated: 2026-08-28
---

# Occupancy and the register budget

## Route here when
- `lever_bottleneck_class.md` said **latency/occupancy-bound** (both roofs far, high stall cycles).
- Profiler shows **< 4 waves/CU** on a memory-bound kernel.
- The ISA dump shows **scratch traffic** (`buffer_store`/`buffer_load` to scratch) — that is a spill,
  and it is a bug regardless of class.

**Skip this lever if** the kernel is compute-bound and already running 1–2 workgroups/CU with deep
prefetch. That is the *correct* operating point for MFMA GEMM — raising occupancy there costs you the
register budget the accumulators need. See "the counter-intuitive part" below.

## gfx950 constants

| Resource | Value |
|---|---|
| Registers per SIMD | **512 × 32-bit**, allocated in **16-register granules** |
| Split | ≤256 architected VGPR + ≤256 AGPR, **unified pool** (a wave flexes the split) |
| Wave slots | **8/SIMD → 32/CU** (hard cap) |
| SIMDs per CU | 4 |
| LDS | **160 KiB/CU**, allocated in **320-DWORD blocks** |
| Wavefront | **64 lanes** |

```
occ_vgpr (waves/SIMD)    = min(8, floor(512 / round_up(N,16)))   # N = VGPRs/wave
occ_lds  (workgroups/CU) = floor(163840 / L)                     # L = LDS bytes/workgroup
nW                       = ceil(threads_per_block / 64)
wg_per_CU                = min(floor(occ_vgpr * 4 / nW), occ_lds, floor(32 / nW))
waves_per_CU             = wg_per_CU * nW
```

| VGPR reserved | waves/SIMD |
|---:|---:|
| ≤ 64 | 8 (slot-capped) |
| 96 | 5 |
| 128 | 4 |
| 176 (e.g. 170 used) | **2** |
| 256 | 2 |
| 512 (256 VGPR + 256 AGPR) | 1 |

## The one thing that changed on gfx950: LDS almost never binds

The LDS denominator is **163840**, not 65536. At MI300X-era tile sizes the LDS term drops out of the
`min()` entirely, so **VGPR pressure is now nearly always the limiter**. Concretely: a 512-thread
attention kernel with 48 KiB/workgroup was pinned to 1 wg/CU on a 64 KiB part; here it gets 3.

Two consequences for how you tune:
1. **Do not carry an MI300X occupancy budget over.** Re-derive with 163840; tiles that were LDS-capped
   are register-capped here.
2. **Spend the LDS surplus on tiles and prefetch depth**, not on chasing more resident workgroups.
   160 KiB affords 3–4 pipeline stages at typical GEMM tile sizes.

## What to change, in order

### 1. Find the actual limiter before touching anything
Read `.vgpr_count` / `.agpr_count` / `.lds_size` from the ISA dump, or
`-Rpass-analysis=kernel-resource-usage`. Plug into the formula above. Do not guess which term binds.

### 2. Cut VGPRs (the primary lever on gfx950)
- **Watch the 16-granule rounding.** 170 used → 176 reserved. Tier boundaries sit at 64/80/96/128/168/256
  — shaving 2 registers across a boundary can jump a whole occupancy tier, and shaving 2 registers
  *within* a tier does nothing.
- `__launch_bounds__(threads, waves_per_eu)` (HIP) / `-mllvm -amdgpu-waves-per-eu=N` — hard-caps the
  register allocation so N waves fit. **Under-set it and you force spills.**
- Triton `waves_per_eu=N` inside a `triton.Config({...})` — a *hint*, not a guarantee; verify in the ISA.
- Shrink live state: recompute cheap values instead of holding them, narrow the `BLOCK_K` accumulation
  scope, hoist loop-invariants into SGPRs.

### 3. Move accumulators to AGPRs
MFMA can read/write its C tile from AGPRs, freeing the architected VGPR budget:
```
-mllvm -amdgpu-mfma-vgpr-form=false -mllvm -amdgpu-agpr-alloc=256
```
Cost is a `v_accvgpr_read_b32` per element in the epilogue (~5%). Not every C-tile layout is
AGPR-placeable — the matrix calculator's `--detail-instruction` reports ArchVGPR/AccVGPR eligibility.

### 4. Remove staging registers with 128-bit direct-to-LDS
`global_load_lds` / `buffer_load ... lds` at **12 or 16 DWORD** moves data global→LDS without passing
through VGPRs. This is the single biggest occupancy win for tiled GEMM, and gfx950 widened it 4× over
the previous generation. See `lever_prefetch.md`.

### 5. Only then raise the wave target
`≥4 waves/CU` is the rule of thumb for hiding HBM latency on memory-bound kernels. Compute-bound MFMA
kernels do **not** need it.

## The counter-intuitive part

MFMA latency is hidden by the **systolic pipeline depth and independent accumulator tiles**, not by
many resident waves. A GEMM holding a large accumulator tile in AGPRs is *inherently* low-occupancy and
that is correct. The decision rule:

> **2 waves/SIMD with zero spills beats 3 waves/SIMD that spill** — always, for GEMM-class kernels.

A spill turns a register access into scratch (global) memory traffic inside the inner loop. One spilled
hot value can cost more than the occupancy it buys.

## Verify

| Check | How | Pass |
|---|---|---|
| Register counts | ISA `.vgpr_count` / `.agpr_count` | matches your budget; below the tier boundary you targeted |
| **Zero spills** | grep the ISA for scratch `buffer_load`/`buffer_store` | none in the hot loop |
| Resident waves | `rocprof-compute` occupancy panel | matches the formula; panel says which resource binds |
| On-box quick check | `occ.sh` (ROCm workload guide) | VGPR/LDS → waves/CU |

## Expected magnitude
Memory-latency-bound kernels: going 2 → 4+ waves/CU typically recovers **10–40%**. Compute-bound GEMM:
usually **0%, sometimes negative**. Removing a spill from an inner loop: often **>20%** on its own.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Raised `waves_per_eu`, got slower | forced spills | check ISA for scratch; back off one tier |
| Cut registers, occupancy unchanged | shaved within a granule | target the next 16-boundary down |
| Occupancy fine, still stalled | not occupancy-bound | back to `lever_bottleneck_class.md` |
| Formula says 4 waves, profiler says 1 | LDS or wave-slot term binding, or the 320-DWORD LDS granule rounded `L` up | re-read `.lds_size`, recompute all three terms |
| Ported CUDA occupancy math | CDNA granularity is 16 VGPR, 8 slots/SIMD, wave64 | re-derive from the formula above |

## Deeper
`hardware/mi350_execution.md` (execution model + worked occupancy examples on the 160 KiB budget) ·
`lever_mfma_sched.md` (why GEMM wants low occupancy) · `lever_prefetch.md` (direct-to-LDS)
