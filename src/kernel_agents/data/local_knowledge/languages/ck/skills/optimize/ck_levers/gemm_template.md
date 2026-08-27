---
title: CK GEMM template — the XDL block-GEMM parameters explained
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8, mxfp4]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
  - https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1_blockwise_gemm_xdlops__pipeline__v1__ab__scale_3_01_block_gemm_pipeline_scheduler_1f98d5cb27163c1a3364a8c8f61866821.html
---

# CK GEMM template parameters

## TL;DR
A CK GEMM (classic `DeviceGemmXdlUniversal` or ck_tile `GemmPipeline`) is defined by a stack of tile
sizes that must agree across four hierarchy levels (block → wave → MFMA → load). Get the four
load-width / MFMA-tile / block-tile / pipeline knobs right and you reach hipBLASLt-class throughput; get
them wrong and `IsSupportedArgument` returns false or you spill. This file explains each parameter and the
constraints between them. For the ranked priority list see [knobs.md](knobs.md).

## Core concepts — the parameter stack
The block-level pipeline template (`BlockwiseGemmXdlops_pipeline_vX`) exposes:
```
BlockSize, ADataType, BDataType, ComputeDataType, AccDataType,
ATileDesc, BTileDesc, AMmaTileDesc, BMmaTileDesc,
ABlockTransferSrcScalarPerVector, BBlockTransferSrcScalarPerVector,
MPerBlock, NPerBlock, KPerBlock, MPerXDL, NPerXDL, MRepeat, NRepeat, KPack
```
The device-level template adds `AK1`/`BK1` (the global-load vector width along K) and the CShuffle store
parameters.

| Param | meaning | typical (bf16 prefill) |
|---|---|---|
| `BlockSize` | threads/block (= 4 waves) | 256 |
| `MPerBlock × NPerBlock` | block C tile | 256×256 |
| `KPerBlock` | K-loop tile | 64 |
| `MPerXDL × NPerXDL` | the MFMA tile shape | 32×32 (or 16×16) |
| `MRepeat × NRepeat` (= `MXdlPerWave × NXdlPerWave`) | MFMA tiles **per wave** | 4×4 → 256×256/block |
| `AK1 / BK1` | global-load vector width along K | 8 (bf16), 16 (fp8) |
| `KPack` | K elements packed per MFMA operand | matches MFMA K |

### The 128-bit-per-load rule
`AK1`/`BK1` are chosen so each lane's global load is **≥128 bit** — for bf16 that is `AK1=8`
(8×16 bit = 128 bit → `buffer_load_dwordx4`); for fp8 `AK1=16` (16×8 bit = 128 bit). This is the single
most important load knob: a sub-128-bit load halves effective HBM bandwidth (see
../asm_mfma/mfma_intrinsics.md and AMD's tuning guide). Pointers must be
aligned to the vector width or `IsSupportedArgument` rejects the instance.

### Constraints between levels
- `MPerBlock = MPerXDL × MRepeat × MWaves`, `NPerBlock = NPerXDL × NRepeat × NWaves`
  (with `MWaves × NWaves × 64 = BlockSize`).
- `KPerBlock` must be a multiple of `AK1 × (MFMA K-density)`; fp8 doubles K-density so `KPerBlock` can
  double vs bf16.
- Block count `ceil(M/MPerBlock)·ceil(N/NPerBlock)` should be `≈ k·304` (MI300X CUs) to avoid a
  wave-quantization tail.

## The levers (worked instance)
A real winning bf16 4096³ RCR MI300X instance: `BlockSize 256`, `256×256×64`, `MPerXDL=NPerXDL=32`,
`MRepeat=NRepeat=4` (WaveMap 4×4), `AK1=BK1=8`, Intrawave, v3, PrefetchStages 2 → **615 TFLOP/s**
(Issue #1727). In ck_tile this is `GemmPipelineAgBgCrCompV3` with the same tile via
`UniversalGemmPipelineAgBgCrPolicy::GetWarpGemm()` selecting the 32×32×8 WarpGemm.

For decode (M = batch, tiny): shrink `MPerBlock` (16/32×256), use **split-K** (`KBatch≥2`) to fill CUs,
Interwave, 16×16 MFMA.

## Pitfalls
- **`mfma_16x16` often beats `32x32`** on MI300X even at large sizes (power-limited clock — see
  ../asm_mfma/mfma_intrinsics.md). Test both; don't default to 32×32.
- Sub-128-bit `AK1/BK1` silently halves bandwidth. Always size to ≥128 bit/load.
- Non-divisible M/N/K → add `GemmSpecialization::MNKPadding` (small perf cost) or `IsSupportedArgument`
  fails.
- Growing the tile past VGPR/AGPR headroom → spills / `v_accvgpr` spam → throughput collapses to a
  smaller-tile class (LLVM #131954). Check the disassembly.

## Verify
- `ckProfiler gemm <args>` reports each instance's TFLOP/s and GB/s — compare achieved vs the ~615
  TFLOP/s reference and vs hipBLASLt at the same shape.
- Disassemble: confirm `buffer_load_dwordx4` in the K-loop and `s_waitcnt lgkmcnt(1)` before `v_mfma`.

## Sources
- A Block GEMM on MI300 (tile sizing, LDS, 256×256/304 CU, pipeline stages): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- Hands-On with CK-Tile GEMM (WarpGemm, policy `GetWarpGemm`, AK1/BK1): https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- BlockwiseGemmXdlops pipeline template params (MPerBlock/NPerBlock/KPerBlock/MPerXDL/NPerXDL/KPack): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1_blockwise_gemm_xdlops__pipeline__v1__ab__scale_3_01_block_gemm_pipeline_scheduler_1f98d5cb27163c1a3364a8c8f61866821.html
- Issue #1727 (winning 256×256×64 / 32×32 / v3 instance @ 615 TFLOP/s): https://github.com/ROCm/composable_kernel/issues/1727
- 128-bit load / mfma_16x16 vs 32x32 guidance: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
