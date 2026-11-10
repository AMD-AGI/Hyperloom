---
title: MI350X — execution model, registers, occupancy
kind: hardware
topic: execution
gens: [gfx950]
updated: 2026-08-28
---

# Execution model, registers and occupancy

Covers wave64, the SIMD/CU hierarchy, the VGPR/AGPR file, and the occupancy arithmetic with worked
examples. This is the card behind every "how many waves fit?" question.

## Wave64 — no wave32 on CDNA

A wavefront is **64 lanes**. On a 16-wide SIMD the issue physically spans 4 cycles, but the programming
model is "64 lanes, one instruction."

Everything derived from lane width is **mod 64**:
- Divergence uses the 64-bit `EXEC` mask.
- Cross-lane ops (`ds_swizzle`, `v_permlane`, DPP, `__shfl`) span 64 lanes.
- Ballot masks are `unsigned long long` with `__popcll`.
- Coalescing windows are 64 lanes wide.
- MFMA is a wave-level op — all 64 lanes cooperate on one `D = A·B + C`.

32-lane code ported from CUDA **runs correctly and uses half the machine.** It will not error.

## The hierarchy

| Level | Count | Note |
|---|---|---|
| Device | 256 active CUs | 8 XCD × 32 |
| CU | 4 SIMDs | = 4 EUs = 4 Matrix Cores |
| SIMD | 8 wave slots | → 32 waves/CU hard cap |
| Wave | 64 lanes | |

A **workgroup is dispatched to one CU** and never migrates; its waves stripe across that CU's 4 SIMDs.

## The register file

| File | Size | Access |
|---|---|---|
| VGPR (architected) | **512 × 4 B per SIMD** | all VALU |
| AGPR (accumulation) | up to **256 × 4 B per SIMD** | MFMA + `v_accvgpr_read/write_b32` only |
| SGPR (scalar) | ~800/CU, ≤102/wave usable | scalar unit |

- The VGPR/AGPR pool is **unified** — a wave flexes the split between them.
- **Allocation granule is 16 registers.** 170 used → **176 reserved**. Tier boundaries sit at
  64 / 80 / 96 / 128 / 168 / 256. Shaving registers *within* a tier changes nothing; shaving across one
  can jump a whole occupancy tier.
- **AGPRs are the escape hatch**: park large FP32 matmul accumulators there so they do not consume the
  architected budget that limits occupancy. Cost is a `v_accvgpr_read_b32` per element in the epilogue
  (~5%). Not every C-tile layout is AGPR-placeable — the matrix calculator's `--detail-instruction`
  reports ArchVGPR/AccVGPR eligibility.

## Occupancy arithmetic

```
occ_vgpr (waves/SIMD)    = min(8, floor(512 / round_up(N,16)))   # N = VGPRs/wave
occ_lds  (workgroups/CU) = floor(163840 / L)                     # L = LDS bytes/workgroup
nW                       = ceil(threads_per_block / 64)
wg_per_CU                = min(floor(occ_vgpr * 4 / nW), occ_lds, floor(32 / nW))
waves_per_CU             = wg_per_CU * nW
```

LDS allocates in **320-DWORD blocks** on gfx950, so a small `L` still rounds up.

| VGPR reserved | waves/SIMD |
|---:|---:|
| ≤ 64 | 8 (slot-capped) |
| 96 | 5 |
| 128 | 4 |
| 176 | **2** |
| 256 | 2 |
| 512 (256 VGPR + 256 AGPR) | 1 |

## The gfx950 change: LDS almost never binds

The LDS denominator is **163840**, not 65536. At MI300X-era tile sizes the LDS term drops out of the
`min()` entirely, so **VGPR pressure is now nearly always the limiter.**

### Worked examples

**A — VGPR-limited GEMM.** N=176, threads=256 (nW=4), L=32 KiB.
```
occ_vgpr = floor(512/176) = 2 ; wg_from_vgpr = floor(2*4/4) = 2 ; occ_lds = floor(163840/32768) = 5
wg_per_CU = min(2, 5, 8) = 2  ->  8 waves/CU        # VGPR binds; LDS has 2.5x headroom
```
Dropping N to 128: `occ_vgpr=4` → `wg_from_vgpr=4`, and `occ_lds=5` still does not bind →
**4 wg/CU = 16 waves/CU**. Cutting registers pays off directly here.

**B — attention with a big tile.** N=64, threads=512 (nW=8), L=48 KiB.
```
occ_vgpr = 8 ; wg_from_vgpr = floor(8*4/8) = 4 ; occ_lds = floor(163840/49152) = 3
slot cap = floor(32/8) = 4  ->  wg_per_CU = min(4,3,4) = 3  ->  24 waves/CU
```
On a 64 KiB-LDS part this was pinned to 1 wg/CU (8 waves). LDS only starts binding again above
**~53 KiB/workgroup** at this shape — spend the budget on bigger tiles or a third/fourth prefetch
stage instead of chasing occupancy.

**C — fully occupied bandwidth kernel.** N=48, threads=256 (nW=4), L=8 KiB.
```
occ_vgpr = 10 -> cap 8 ; wg_from_vgpr = 8 ; occ_lds = 20 ; slot cap = 8
wg_per_CU = 8  ->  32 waves/CU (maximum)
```

## What it means for kernels

1. **Cut VGPRs first** — the primary lever. Watch the 16-granule boundary.
2. **`__launch_bounds__(threads, waves_per_eu)`** / `-mllvm -amdgpu-waves-per-eu=N` hard-caps the
   allocation. Set below the real block size and you force spills.
3. **AGPR accumulators**: `-mllvm -amdgpu-mfma-vgpr-form=false -mllvm -amdgpu-agpr-alloc=256`.
4. **128-bit `global_load_lds`** removes staging VGPRs — the biggest tiled-GEMM occupancy win
   (`mi350_lds.md`).
5. **≥4 waves/CU** to hide HBM latency. MFMA-bound GEMM does **not** need it — 1–2 wg/CU with deep
   prefetch is the correct operating point.

> **2 waves/SIMD with zero spills beats 3 waves/SIMD that spill**, always, for GEMM-class kernels.
> A spill turns a register access into scratch memory traffic inside the inner loop.

## Pitfalls
- **Carrying an MI300X occupancy budget over** — the denominator is 163840, not 65536.
- **The "512" double meaning** — 512 VGPRs *per SIMD* (the occupancy math); the CU's combined vector
  register file is ~512 KiB across 4 SIMDs. Count vs bytes.
- **Forgetting AGPRs come out of the same pool** — a fat accumulator silently caps occupancy.
- **Raising `waves_per_eu` without reading the ISA** — it can force spills and lose more than it gains.
- **Assuming CUDA blocks/SM math** — granule is 16 VGPR, slots are 8/SIMD, wave is 64.

## Verify
- ISA `.vgpr_count` / `.agpr_count` / `.sgpr_count` / `.lds_size`, or
  `-Rpass-analysis=kernel-resource-usage`.
- **Grep the ISA for scratch `buffer_load`/`buffer_store`** — any spill in the hot loop is a bug.
- `rocprof-compute` occupancy panel: resident vs theoretical waves, and **which resource binds**.
- On-box `occ.sh` (ROCm workload guide) turns VGPR/LDS into waves/CU.

## Related
`mi350_overview.md` · `mi350_lds.md` (the LDS term) · `mi350_matrix_core.md` (why GEMM wants low
occupancy) · `common_methodology/optimization/lever_occupancy.md` (the tuning procedure)
