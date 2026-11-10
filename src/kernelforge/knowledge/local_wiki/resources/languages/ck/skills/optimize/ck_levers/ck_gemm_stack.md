---
title: CK — the GEMM tile hierarchy, and the arithmetic that has to close
kind: language
lever: ck_gemm_stack
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
  - https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1_blockwise_gemm_xdlops__pipeline__v1__ab__scale_3_01_block_gemm_pipeline_scheduler_1f98d5cb27163c1a3364a8c8f61866821.html
---

# The GEMM tile hierarchy

## Route here when
- You are writing a template instantiation by hand and need to know what each parameter controls.
- `IsSupportedArgument()` came back `false` and you need to find which rule you broke.
- You made the tile bigger and it got slower.
- Bandwidth is roughly half of what the roofline says it should be.

**For which knob to sweep first**, go to `ck_tuning_knobs.md`. This card is the semantics and the
arithmetic, not the search order.

## The idea in one paragraph
A CK GEMM — classic `DeviceGemmXdlUniversal` or ck_tile `GemmPipeline`, the parameters are the same
family — decomposes the output into four nested levels: the block tile, the per-wave tile, the MFMA
instruction tile, and the per-lane load. Those four are not independent. Two exact equations and one
sizing target tie them together, and every "why won't this instantiate" question is one of the three.

## What the parameters are
The block-level pipeline template (`BlockwiseGemmXdlops_pipeline_vX`) takes:

```
BlockSize, ADataType, BDataType, ComputeDataType, AccDataType,
ATileDesc, BTileDesc, AMmaTileDesc, BMmaTileDesc,
ABlockTransferSrcScalarPerVector, BBlockTransferSrcScalarPerVector,
MPerBlock, NPerBlock, KPerBlock, MPerXDL, NPerXDL, MRepeat, NRepeat, KPack
```

The device-level template layers on `AK1` / `BK1` — the per-lane global-load width along K — plus the
CShuffle store parameters.

| Parameter | Controls | Typical, bf16 prefill |
|---|---|---|
| `BlockSize` | threads per block; divide by 64 for the wave count | 256 (4 waves) |
| `MPerBlock × NPerBlock` | the C tile one block owns | 256×256 |
| `KPerBlock` | how much K one loop iteration consumes | 64 |
| `MPerXDL × NPerXDL` | the MFMA instruction shape | 16×16 (measure 32×32 before assuming) |
| `MRepeat × NRepeat` (aka `MXdlPerWave × NXdlPerWave`) | MFMA tiles issued per wave | 4×4 |
| `AK1` / `BK1` | global-load vector width along K | 8 for bf16, 16 for fp8 |
| `KPack` | K elements packed into one MFMA operand | follow the MFMA's K |

## Rule 1 — the levels have to multiply out exactly
```
MPerBlock = MPerXDL × MRepeat × MWaves
NPerBlock = NPerXDL × NRepeat × NWaves
subject to   MWaves × NWaves × 64 = BlockSize
```

The `64` is the wavefront width and is not negotiable on CDNA. If you carried a configuration over from
a 32-lane architecture, this equation is where it fails.

## Rule 2 — `KPerBlock` follows the MFMA's K-density
`KPerBlock` must be a multiple of `AK1 × (the MFMA's K density)`. The practical consequence: switching
bf16 → fp8 doubles the K density, so `KPerBlock` can double too. A configuration ported from bf16 to
fp8 without touching `KPerBlock` leaves half the available K-depth on the table.

## Rule 3 — size the grid to the device, not to habit
```
ceil(M/MPerBlock) · ceil(N/NPerBlock)  ≈  k · 256
```

**256 is gfx950's CU count.** A grid laid out for MI300X's 304 does not merely miss the target — it
leaves a partial final wave in which most of the GPU sits idle while a handful of CUs finish. That tail
shows up as latency on a configuration whose steady-state throughput looks fine.

Query the CU count rather than hardcoding either number:
`hipGetDeviceProperties(...).multiProcessorCount`.

## The 128-bit floor on loads
Pick `AK1` / `BK1` so that each lane's global load is **at least 128 bits wide**:

| dtype | `AK1` | Why |
|---|---|---|
| bf16 | 8 | 8 × 16 bit = 128 bit → `buffer_load_dwordx4` |
| fp8 | 16 | 16 × 8 bit = 128 bit |

This is the highest-leverage load decision in the whole stack. **Below 128 bits you lose roughly half
your effective HBM bandwidth**, and the kernel will still be correct, so nothing tells you. Alignment
is a hard requirement too — pointers not aligned to the vector width cause the instance to be rejected
outright.

## A concrete instance, and how to read it
The bf16 4096³ RCR winner from Issue #1727, measured on MI300X:

```
BlockSize 256 · 256×256×64 · MPerXDL = NPerXDL = 32 · MRepeat = NRepeat = 4 (wave map 4×4)
AK1 = BK1 = 8 · Intrawave · v3 · PrefetchStages 2        →  615 TFLOP/s
```

The ck_tile spelling of the same thing is `GemmPipelineAgBgCrCompV3` at that tile, with
`UniversalGemmPipelineAgBgCrPolicy::GetWarpGemm()` choosing the WarpGemm.

**Do not copy those numbers onto gfx950.** Three things moved underneath them: the bf16 MFMA family is
now 16×16×32 and 32×32×16, the CU count is 256 rather than 304, and 160 KiB of LDS permits a deeper
pipeline than the config was designed around. What transfers is the *shape* of a good answer — a
256-thread block, a square-ish tile, ≥128-bit loads, two prefetch stages. The specific values need
re-measuring.

**Decode is a different regime.** Small M means the prefill answer is wrong in every dimension: shrink
`MPerBlock` (16 or 32 against N=256), turn on split-K (`KBatch ≥ 2`) so there is enough work to fill
256 CUs, switch to Interwave, and use the 16×16 MFMA.

## Verify
| Check | How | Pass condition |
|---|---|---|
| Throughput is competitive | `ckProfiler gemm <args>`, then the same shape through hipBLASLt | within reach of the library, or better |
| Loads are wide enough | disassemble the K-loop | `buffer_load_dwordx4` present |
| Loads overlap the math | same disassembly | `s_waitcnt lgkmcnt(1)` before `v_mfma` — not `lgkmcnt(0)` |
| Nothing spilled | same disassembly | no `scratch_` traffic, no run of `v_accvgpr` moves |

The disassembly answers three of these four. Get in the habit of reading it before changing parameters.

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| `IsSupportedArgument()` is false | M/N/K not divisible by the tile, or a pointer not aligned to `AK1`/`BK1` | add `GemmSpecialization::MNKPadding`, or fix the alignment |
| Roughly half the expected bandwidth | the per-lane load fell under 128 bits | raise `AK1`/`BK1` to hit 128 bit |
| **Throughput falls as the tile grows** | past the VGPR/AGPR budget: the compiler starts moving through `v_accvgpr` and spilling to `scratch_` (LLVM #131954) | shrink the tile. Confirm in the disassembly — the config alone will not show it |
| 32×32 slower than 16×16 | 32×32 holds 16 C registers per lane against 16×16's 4, **and** draws more power so clocks drop | default to 16×16; treat 32×32 as something to measure, not assume |
| Steady state fine, tail latency bad | grid is not near a multiple of 256 | resize the block tile |
| bf16 config moved to fp8, no gain | `KPerBlock` never updated for the doubled K density | raise it |

Full trap list, indexed by symptom: `ck_traps.md`.

## Sources
- Block GEMM on MI300 — tile sizing, LDS, pipeline stages:
  https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- WarpGemm, policy `GetWarpGemm`, `AK1`/`BK1` (ROCm blog, hands-on CK-Tile GEMM):
  https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- `BlockwiseGemmXdlops` pipeline template parameters:
  https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1_blockwise_gemm_xdlops__pipeline__v1__ab__scale_3_01_block_gemm_pipeline_scheduler_1f98d5cb27163c1a3364a8c8f61866821.html
- The 256×256×64 / v3 instance at 615 TFLOP/s on MI300X:
  https://github.com/ROCm/composable_kernel/issues/1727
- Large MFMA tiles producing `v_accvgpr` moves and spills:
  https://github.com/llvm/llvm-project/issues/131954
- 128-bit load guidance and MFMA shape selection:
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
