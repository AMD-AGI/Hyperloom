---
title: CK tuning knobs — ranked by impact on MI300X
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, mxfp4]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
  - https://github.com/ROCm/composable_kernel/issues/1727
---

# CK tuning knobs (MI300X, 304 CUs)

## TL;DR
The CK config space is large but a handful of knobs dominate. In priority order: **block tile** →
**K-loop tile** → **pipeline version/scheduler** → **MFMA tile** → **wave map** → **load vector width**.
Tune block tile + pipeline first; everything else is second-order. Decode (skinny-M) inverts several
defaults (split-K, Interwave, 16×16 MFMA). This file is the cross-cutting knob reference for both
[ck_classic.md](ck_classic.md) and [ck_tile.md](ck_tile.md).

## The levers (ranked)
| Knob | values | effect | priority |
|---|---|---|---|
| `MPerBlock × NPerBlock` | 256×256, 256×128, 128×128, 128×64, 64×64 | block tile; bigger = more reuse, fewer blocks (occupancy/tail trade-off) | ★★★★★ |
| `KPerBlock` | 32, 64, 128 | K-loop tile; bigger = better MFMA/load overlap, more LDS+VGPR | ★★★★ |
| `BlockGemmPipelineVersion` | v1/v2/**v3**/v4/v5 | hot-loop schedule depth | ★★★★ |
| `BlockGemmPipelineScheduler` | **Intrawave** / Interwave | overlap strategy | ★★★★ |
| `MPerXDL × NPerXDL` | **16×16**, 32×32 | MFMA tile; 16×16 often wins on MI300X (power) | ★★★ |
| `MXdlPerWave × NXdlPerWave` | 4×4, 4×2, 2×2 | waves-per-tile map; drives VGPR & occupancy | ★★★ |
| `AK1 / BK1` | 8 (bf16), 16 (fp8) | global-load vector width; **≥128 bit/load**; must match alignment | ★★★ |
| `GemmSpecialization` | Default / MNKPadding / MNPadding | pad guards for non-divisible shapes | ★★ |
| `KBatch` (split-K) | 1, 2, 4, 8 | atomic K split → fills CUs for **small-M decode** | ★★★ (decode) |
| CShuffle store vec | 8 (bf16) | coalesced C store width | ★★ |
| LDS swizzle | XOR (`make_xor_transform`) | kills LDS bank conflicts, no extra LDS | ★★ |

## Heuristics for LLM shapes (MI300X)
- **Prefill (large M):** 256×256×64, `MPerXDL=NPerXDL=32` (test 16×16), WaveMap 4×4, **v3 Intrawave**,
  `AK1=BK1=8`. → ~615 TFLOP/s @ 4096³ bf16 (Issue #1727).
- **Decode (M = batch ≪ N,K):** small M tile (16/32×256), **split-K (`KBatch≥2`)** to occupy CUs,
  **Interwave** often wins, **16×16 MFMA**. A 256×256 tile leaves most CUs idle at tiny M.
- **fp8 weight-only linear:** `*_b_scale` fp8 instance, `AK1/BK1=16`, `bpreshuffle` the static weight
  into MFMA register layout at load. `KPerBlock` can double (K-density doubles).
- **MoE:** `DeviceGroupedGemm*` (+ `mx`/`b_scale` for low-precision experts).
- Aim `ceil(M/MPerBlock)·ceil(N/NPerBlock) ≈ k·304` to avoid a wave-quantization tail.

## Pitfalls
- **Don't default to 32×32 MFMA** — 16×16×16 usually yields higher *achievable* FLOPs on MI300X because
  the 32×32 op draws more power and clocks lower (see ../asm_mfma/mfma_intrinsics.md).
- A pinned "winning instance" is **CK-build-specific** — re-sweep after any CK/ROCm bump.
- Bigger block tile is not free: VGPR/AGPR pressure → spills → throughput regresses to a smaller-tile
  class. Verify in disassembly, not just by config.
- Split-K writes via atomics → extra HBM traffic; only a win when it fills otherwise-idle CUs (decode).

## Verify
- Offline `ckProfiler` sweep at the exact LLM shape; record the top instance's TFLOP/s + GB/s.
- Re-measure after ROCm/CK upgrades; **append** the new number with date (don't overwrite).
- Cross-check vs hipBLASLt solidx and aiter tuned config at the same shape.

## Sources
- ROCm "Optimizing with Composable Kernel" (instance selection, profiler, knob guidance): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- A Block GEMM on MI300 (tile/occupancy, 304 CU, LDS): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- MI300X workload optimization (16×16 vs 32×32, 128-bit load, split-K): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Issue #1727 (615 TFLOP/s reference instance, v3/Intrawave): https://github.com/ROCm/composable_kernel/issues/1727
