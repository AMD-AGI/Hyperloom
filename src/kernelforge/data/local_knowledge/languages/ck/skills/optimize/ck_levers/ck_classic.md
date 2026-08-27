---
title: Classic Composable Kernel — DeviceGemm* device-op model
kind: language
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, int8, mxfp4]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
  - https://github.com/ROCm/composable_kernel/issues/1727
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
---

# Classic CK (the `DeviceGemm*` model)

## TL;DR
Classic CK (`include/ck`) is a **tensor-coordinate-transform + tile** library: it describes data movement
as a *composition of `constexpr` coordinate transforms* on a tensor descriptor and lets the compiler fold
all index math into the load/store address — **no runtime index arithmetic** in a well-written CK kernel.
The public surface is the `DeviceGemm*` / `DeviceBatchedGemm*` / `DeviceGroupedGemm*` family plus the
XDL blockwise-GEMM pipelines v1–v5. It is hard to author but, for **dense square bf16/fp16 GEMM**, the
unified `DeviceGemmXdlUniversal` v3/Intrawave instance is often still the strongest baseline (615 TFLOP/s
@ 4096³ MI300X, Issue #1727 — ~1.7× faster than ck_tile at the same tile). For attention, classic
softmax-GEMM is legacy → use [ck_tile.md](ck_tile.md) FMHA. Target gfx942 (MI300X/A/325X); gfx950 noted.

> Repo: standalone `ROCm/composable_kernel` is **DEPRECATED** → `ROCm/rocm-libraries:projects/composablekernel`.

## Core concepts — the descriptor hierarchy
| Level | CK object | owns | typical MI300X size |
|---|---|---|---|
| **Grid** | `GridwiseGemm_xdl_cshuffle_v3` | whole C tensor | M×N |
| **Block** | `BlockwiseGemmXdlops_pipeline_vX` | `MPerBlock × NPerBlock` C tile | 256×256 |
| **Wave** | XDL warp tile | `MPerXDL × NPerXDL` ×(MRepeat×NRepeat) | 32×32 ×(4×4) |
| **Lane** | MFMA fragment | per-lane VGPR/AGPR fragment | e.g. 4 acc regs |

CK uses **256 threads = 4 waves**; `MXdlPerWave`/`NXdlPerWave` (= `MRepeat`/`NRepeat`) is how many MFMA
tiles *one wave* computes, not the wave count.

**Tensor descriptors** are `constexpr` objects built by composing transforms:
`make_naive_tensor_descriptor(lengths,strides)`, `make_unmerge_transform` (tile a dim),
`make_merge_transform` (flatten), `make_pass_through_transform`, `make_pad_transform` (alignment/OOB
guard), `make_xor_transform` (LDS swizzle). The crucial property: `desc.CalculateOffset({...})` compiles
to a handful of integer ops with the tile constants folded — the transform chain is *erased*. This is why
classic CK hits hipBLASLt-class throughput without per-shape hand asm.

**CShuffle epilogue:** the scattered MFMA accumulator layout is not coalescable; CShuffle stages C
through LDS into a vectorized layout before the elementwise op + global write. Knobs:
`CShuffleMXdlPerWavePerShuffle`, `CShuffleNXdlPerWavePerShuffle`,
`CShuffleBlockTransferScalarPerVector_NPerBlock` (store width, usually 8 for bf16).

## The levers
### The five-call lifecycle (uniform across all device-op families)
```cpp
using DeviceOp = ck::tensor_operation::device::DeviceGemmXdlUniversal<
    Row, Col, Row, BF16, BF16, BF16, F32, BF16, PassThrough, PassThrough, PassThrough,
    GemmDefault, 256, /*M,N,K PerBlock*/ 256,256,64, /*AK1,BK1*/ 8,8,
    /*MPerXDL,NPerXDL*/ 32,32, /*MXdlPerWave,NXdlPerWave*/ 4,4, /* ...transfer... */
    BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, BF16, BF16>;
auto op = DeviceOp{};
auto arg = op.MakeArgument(a,b,c, M,N,K, /*StrideA*/K,/*StrideB*/K,/*StrideC*/N, 1, PT{},PT{},PT{});
if(!op.IsSupportedArgument(arg)) throw ...;   // (!) capability gate — NEVER skip
auto inv = op.MakeInvoker();
float ms = inv.Run(arg, StreamConfig{stream, /*time_kernel*/true});
```
`IsSupportedArgument` checks M/N/K divisibility vs the tile, K vs `KPerBlock×KBatch`, pointer alignment
vs `AK1/BK1`, and layout/spec. **An instance forced past a `false` produces garbage.**

### Layout shorthand
`R`=row, `C`=col. **RCR** (A row, B col, C row) is the standard `Y=X·Wᵀ` linear layer (W stored N×K col)
and is the most-tuned layout in CK's instance DB. Also RRR, CRR.

### Pipelines (the hot K-loop scheduler), selected by two template params
- **Scheduler:** `Intrawave` (one wave's loads+MFMAs software-pipelined via `s_setprio` + sched barriers;
  compute-bound, MI300X prefill default) vs `Interwave` (hide latency by switching waves; memory-bound /
  skinny / low-occupancy / when Intrawave spills).
- **Version:** v1 (single buffer, lowest VGPR), v2, **v3** (2-stage prefetch, double-buffer LDS — the
  workhorse for large compute-bound GEMM), v4 (deeper ping-pong, huge-K), v5 (persistent/async-input).

Real winning instance for **bf16 4096³ RCR MI300X** (from CK profiler): `BlkTile 256×256×64`,
`WaveTile 32×32`, `WaveMap 4×4`, `VmemRead(AK1,BK1) 8×8`, Intrawave, v3, PrefetchStages 2
→ **0.223 ms, 615 TFLOP/s, 451 GB/s** (Issue #1727).

### Instance factory + sweep (this *is* ckProfiler / the framework CK fallback)
```cpp
std::vector<DeviceOpPtr> ops;
DeviceOperationInstanceFactory<DeviceGemm<Row,Col,Row,BF16,BF16,BF16,PT,PT,PT>>::GetInstances(ops);
for(auto& op : ops){ auto arg=op->MakeArgumentPointer(...);
    if(!op->IsSupportedArgument(arg.get())) continue;        // skip incompatible
    float ms=op->MakeInvokerPointer()->Run(arg.get(), StreamConfig{nullptr,true});
    if(ms<best){best=ms; winner=&op;} }
```
Run once offline, record the winning instance index, pin it for a fixed LLM shape. See
[knobs.md](knobs.md) for ranked tuning knobs.

### Families beyond plain GEMM
`DeviceBatchedGemmXdl` (batch stride), `DeviceGroupedGemm*` (variable-M MoE; arrays of ptrs/strides, the
CK path behind fused-MoE), `DeviceGemmMultipleD*` (bias/residual fuse). Low-precision: `*_fp8`,
`*_b_scale` (weight-only scale), `*_ab_scale`, `*_mx_gemm`/`mx_gemm_bpreshuffle` (mxfp8/mxfp4 block-scaled).

## Pitfalls
- **ck_tile is *not* automatically faster** for dense square GEMM — Issue #1727 (615 vs 359 TFLOP/s).
  Benchmark classic v3 first for that case.
- Skipping `IsSupportedArgument` → silent garbage. Always gate.
- `ckProfiler` may be missing in deployment images. Build it (`make -j ckProfiler`) on a dev box.
- gfx950 features (fp4/mxfp4, larger-K MFMA) are gated behind `DTYPES` build flags — they won't appear
  unless you enabled them at cmake time.

## Verify
- `ckProfiler gemm <args>` prints every instance's TFLOP/s; the top line is your pinned config.
- Cross-check the winning instance string against a hipBLASLt solidx at the same shape.
- Parity vs a reference GEMM (fp32 accumulate) before pinning.

## Sources
- A Block GEMM on MI300 (descriptor hierarchy, pipeline stages, 256×256 / 304 CU): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- `DeviceGemm` base struct (MakeArgument/IsSupportedArgument/MakeInvoker lifecycle): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
- `BlockwiseGemmXdlops_pipeline_v1_ab_scale` template params (Intrawave/Interwave, MPerXDL/NPerXDL/KPack): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1_blockwise_gemm_xdlops__pipeline__v1__ab__scale_3_01_block_gemm_pipeline_scheduler_1f98d5cb27163c1a3364a8c8f61866821.html
- Issue #1727 — ck_tile vs classic CK v3 (615 vs 359 TFLOP/s, winning instance string): https://github.com/ROCm/composable_kernel/issues/1727
- ROCm "Optimizing with Composable Kernel" how-to: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- Repo deprecation/move to ROCm/rocm-libraries: https://github.com/ROCm/composable_kernel (README banner)
