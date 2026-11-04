---
title: Wavefront / SIMD / VGPR / AGPR execution model (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# Wavefront / SIMD / VGPR / AGPR execution model

## TL;DR
> CDNA is **wave64**: a wavefront = **64 work-items** in lockstep on one **SIMD64**; each CU has
> **4 SIMDs** (= 4 EUs = 4 Matrix Cores). Occupancy is per-SIMD and is gated by the **minimum** of
> VGPR limit, LDS limit, and the **8-waves/SIMD** hard cap. Register pressure is the #1 occupancy
> killer; **AGPRs** let matmul keep big accumulators without spending the VGPR budget.

## Concepts

### Wave64 (no wave32 on CDNA)
- A wavefront is 64 lanes. On a 16-wide SIMD the issue physically spans **4 cycles** (16 lanes ×4),
  but the programming model is "64 lanes, one instruction." Branch divergence uses the 64-bit `EXEC`
  mask; cross-lane ops (`ds_swizzle`, `v_permlane`, DPP, `__shfl`) operate over **64** lanes, so all
  shuffle/ballot/reduction math is **mod 64**, not mod 32 (a frequent CUDA-port bug).
- All CDNA generations (gfx908/90a/942/950) keep wave64. MFMA is a wave-level op — all 64 lanes
  cooperate on one `D=A·B+C`.

### CU → SIMD → wave slots
| Gen | SIMD/CU | Wave slots/SIMD | Wave slots/CU | Lanes/SIMD |
|---|---|---|---|---|
| CDNA1 gfx908 | 4 | 10 | 40 | 64 (SIMD16, 4-cycle) |
| CDNA2 gfx90a | 4 | 8 | 32 | 64 |
| CDNA3 gfx942 | 4 | 8 | 32 | 64 |
| CDNA4 gfx950 | 4 | 8 | 32 | 64 |

A **workgroup is dispatched to one CU** and never migrates; its waves stripe across that CU's 4 SIMDs.

### The register files: VGPR, SGPR, AGPR
| File | CDNA1 (gfx908) | CDNA2/3/4 (gfx90a/942/950) | Access |
|---|---|---|---|
| VGPR (architected) | 256 ×4 B / wave-slot region | **512 ×4 B per SIMD/EU** | all VALU |
| AGPR (accumulation) | up to **256** ×4 B | up to **256** ×4 B per SIMD | MFMA + `v_accvgpr_read/write_b32` only |
| SGPR (scalar) | ~800/CU (≤102/wave usable) | similar | scalar unit |

- **CDNA1 introduced AGPRs** alongside the first Matrix Core; CDNA2 doubled architected VGPRs to 512
  and unified the VGPR/AGPR pool so a wave can flex between them.
- **VGPRs allocate in blocks of 16.** A kernel reporting 170 VGPRs is rounded to **176** — this
  rounding alone can drop an occupancy tier. Watch boundaries at 64/80/96/128/168/256.
- **AGPRs** are the "escape hatch": park large FP32 matmul accumulators here so they don't consume the
  architected VGPR budget that limits occupancy. The compiler inserts `v_accvgpr_read_b32` in the
  epilogue before `global_store` (~5% cost).

### Occupancy math (per-SIMD)
```
occ_vgpr (waves/SIMD)    = floor(VGPR_per_SIMD / N)     # cap at wave-slot limit (8, or 10 on CDNA1)
                                                        # CDNA2-4: floor(512/N); CDNA1: floor(256/N)
occ_lds  (workgroups/CU) = floor(LDS_per_CU / L)        # 65536 (CDNA1-3) or 163840 (CDNA4)
nW                       = ceil(threads_per_block / 64) # waves per workgroup
wg_from_vgpr             = floor(occ_vgpr * 4 / nW)     # 4 SIMDs/CU
wg_per_CU                = min(wg_from_vgpr, occ_lds, floor(wave_slots_per_CU / nW))
waves_per_CU             = wg_per_CU * nW
```
Worked examples per generation live in each gen's `occupancy.md`.

## The levers
1. **Cut VGPRs to gain occupancy** — but watch the 16-granule rounding and the LDS limit, which often
   binds first for attention/softmax.
2. **`__launch_bounds__(threads, waves_per_eu)`** / `-mllvm -amdgpu-waves-per-eu=N` tells LLVM to cap
   VGPRs so N waves fit per EU.
3. **AGPR escape hatch** for GEMM: `-mllvm -amdgpu-mfma-vgpr-form=false -mllvm -amdgpu-agpr-alloc=256`.
4. **Direct global→LDS** (`buffer_load ... lds`) removes staging VGPRs — biggest single occupancy win
   for tiled GEMM (CDNA4 widens it to 128 b/lane). See
   [memory_model_lds_bank.md](memory_model_lds_bank.md).
5. **≥4 waves/CU** for HBM-latency hiding; MFMA-bound GEMM often runs 1–2 wg/CU and hides latency with
   double-buffered LDS instead of raw occupancy.

## Pitfalls
- **Assuming wave32.** All warp-width constants (shuffle masks, ballot, `__activemask`) must be 64.
- **The "512" double-meaning.** 512 = VGPRs **per EU** (occupancy math); the CU's combined vector
  register file is ~512 KiB across 4 SIMDs. Don't conflate count with bytes.
- **Ignoring AGPR eligibility.** Not every C-tile layout is AGPR-placeable; the calculator's
  `--detail-instruction` reports ArchVGPR/AccVGPR usability per MFMA.
- **CDNA1 256-VGPR ceiling.** gfx908 has half the architected VGPRs of later gens — port tile sizes
  down accordingly.

## Verify
- ISA dump `.vgpr_count` / `.agpr_count` / `.sgpr_count` / `.lds_size`, or
  `-Rpass-analysis=kernel-resource-usage`.
- `rocprof-compute` occupancy/wave panels: resident waves/CU vs theoretical, and which resource binds.
- The on-box `occ.sh` from the ROCm workload guide turns VGPR/LDS into waves/CU.
