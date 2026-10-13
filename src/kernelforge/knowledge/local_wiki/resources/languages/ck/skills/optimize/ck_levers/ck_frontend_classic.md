---
title: CK — the classic DeviceGemm* front-end
kind: language
lever: ck_frontend_classic
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
  - https://github.com/ROCm/composable_kernel/issues/1727
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
---

# Classic CK — the `DeviceGemm*` model

## Route here when
- **Dense square bf16/fp16 GEMM.** This front-end is often still the strongest baseline for that case —
  see the measurement below before assuming ck_tile is newer-therefore-faster.
- You need `DeviceGroupedGemm*` (variable-M MoE) or `DeviceGemmMultipleD*` (bias/residual fusion).
- You are sweeping instances with `ckProfiler` and need to understand what it is sweeping.

**Go to `ck_frontend_tile.md` instead for** attention (FMHA), fused MoE, fusion-heavy kernels, or any
new low-precision work. Classic softmax-GEMM for attention is **legacy** — do not start there.

## The decision, with the number attached

| | Classic `DeviceGemmXdlUniversal` v3 | ck_tile `universal_gemm` |
|---|---|---|
| 4096³ bf16, same 256×256×64 tile | **0.223 ms, 615 TFLOP/s** | 0.382 ms, 359 TFLOP/s |
| Wins on | **dense square GEMM** | fusion, attention, MoE, low-precision |

Measured on MI300X (Issue #1727). The *ranking* is the durable part; re-measure the absolute numbers on
gfx950. **Benchmark classic v3 first for any dense path** — "ck_tile is the new one" is not a perf
argument.

> **Repo pin:** standalone `ROCm/composable_kernel` is **DEPRECATED** → development is in
> `ROCm/rocm-libraries` under `projects/composablekernel/`. The old `develop` branch is a read-only
> mirror.

## What classic CK actually is

A **tensor-coordinate-transform + tile** library. Data movement is described as a composition of
`constexpr` coordinate transforms on a tensor descriptor, and the compiler folds all index math into the
load/store address — **there is no runtime index arithmetic in a well-written CK kernel.**

Descriptors are built by composing transforms:

| Transform | Purpose |
|---|---|
| `make_naive_tensor_descriptor(lengths, strides)` | the base view |
| `make_unmerge_transform` | tile a dimension |
| `make_merge_transform` | flatten dimensions |
| `make_pass_through_transform` | identity |
| `make_pad_transform` | alignment / OOB guard |
| `make_xor_transform` | **LDS swizzle** — re-derive for 64 banks on gfx950 |

`desc.CalculateOffset({...})` compiles down to a handful of integer ops with the tile constants folded;
the transform chain is *erased*. That is why classic CK reaches hipBLASLt-class throughput without
per-shape hand assembly.

## The descriptor hierarchy

| Level | CK object | Owns | Typical |
|---|---|---|---|
| **Grid** | `GridwiseGemm_xdl_cshuffle_v3` | the whole C tensor | M×N |
| **Block** | `BlockwiseGemmXdlops_pipeline_vX` | `MPerBlock × NPerBlock` | 256×256 |
| **Wave** | XDL warp tile | `MPerXDL × NPerXDL` × (`MRepeat`×`NRepeat`) | 32×32 × (4×4) |
| **Lane** | MFMA fragment | per-lane VGPR/AGPR fragment | 4 or 16 acc regs |

CK uses **256 threads = 4 waves**. `MXdlPerWave` / `NXdlPerWave` (= `MRepeat`/`NRepeat`) is how many
MFMA tiles **one wave** computes — not the wave count. This trips people up constantly.

## The five-call lifecycle

Uniform across every device-op family:

```cpp
using DeviceOp = ck::tensor_operation::device::DeviceGemmXdlUniversal<
    Row, Col, Row, BF16, BF16, BF16, F32, BF16, PassThrough, PassThrough, PassThrough,
    GemmDefault, 256, /*M,N,K PerBlock*/ 256,256,64, /*AK1,BK1*/ 8,8,
    /*MPerXDL,NPerXDL*/ 32,32, /*MXdlPerWave,NXdlPerWave*/ 4,4, /* ...transfer... */
    BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, BF16, BF16>;

auto op  = DeviceOp{};
auto arg = op.MakeArgument(a,b,c, M,N,K, /*StrideA*/K, /*StrideB*/K, /*StrideC*/N, 1, PT{},PT{},PT{});
if (!op.IsSupportedArgument(arg)) throw ...;      // (!) capability gate — NEVER skip
auto inv = op.MakeInvoker();
float ms = inv.Run(arg, StreamConfig{stream, /*time_kernel*/true});
```

**`IsSupportedArgument` is a correctness gate, not a hint.** It checks M/N/K divisibility against the
tile, K against `KPerBlock × KBatch`, pointer alignment against `AK1`/`BK1`, and layout/spec. **An
instance forced past a `false` returns silent garbage — not an error.**

### Layout shorthand
`R` = row, `C` = col. **RCR** (A row, B col, C row) is the standard `Y = X·Wᵀ` linear layer with W
stored N×K column-major — the most-tuned layout in CK's instance DB. Also RRR, CRR.

## Pipelines — the hot K-loop scheduler

Two template parameters select it:

**Scheduler**
- **`Intrawave`** — one wave's loads and MFMAs software-pipelined via `s_setprio` + sched barriers.
  Compute-bound prefill default.
- **`Interwave`** — hide latency by switching waves. Memory-bound, skinny-M, low occupancy, or when
  Intrawave spills.

**Version**

| Version | Shape |
|---|---|
| v1 | single buffer, lowest VGPR |
| v2 | — |
| **v3** | **2-stage prefetch, double-buffered LDS — the workhorse for large compute-bound GEMM** |
| v4 | deeper ping-pong, huge K |
| v5 | persistent / async-input |

On gfx950's **160 KiB LDS** a deeper pipeline is cheaper than it was on a 64 KiB part — v4 is worth
testing where v3 used to be the ceiling.

## The instance sweep — this *is* `ckProfiler`

```cpp
std::vector<DeviceOpPtr> ops;
DeviceOperationInstanceFactory<DeviceGemm<Row,Col,Row,BF16,BF16,BF16,PT,PT,PT>>::GetInstances(ops);
for (auto& op : ops) {
    auto arg = op->MakeArgumentPointer(...);
    if (!op->IsSupportedArgument(arg.get())) continue;      // skip incompatible
    float ms = op->MakeInvokerPointer()->Run(arg.get(), StreamConfig{nullptr, true});
    if (ms < best) { best = ms; winner = &op; }
}
```

Run offline, record the winning instance index, pin it for a fixed LLM shape. **The pin is
build-specific** — re-sweep after any CK/ROCm bump (`ck_traps.md` §3).

## Families beyond plain GEMM

| Family | Use |
|---|---|
| `DeviceBatchedGemmXdl` | batch stride |
| `DeviceGroupedGemm*` | **variable-M MoE** — the CK path behind fused-MoE |
| `DeviceGemmMultipleD*` | bias / residual fusion |
| `*_fp8`, `*_b_scale`, `*_ab_scale` | low precision; weight-only scale |
| `*_mx_gemm`, `*_mx_gemm_bpreshuffle` | mxfp8 / mxfp4 block-scaled |

## Verify

| Check | How |
|---|---|
| Instance ranking | `ckProfiler gemm <args>` — top line is your pin |
| Cross-check | the same shape against a hipBLASLt solidx and the aiter tuned config |
| Correctness | fp32-accumulate reference parity **before** pinning |
| No spills | disassemble; `buffer_load_dwordx4` in the K-loop, no `scratch_` / `v_accvgpr` spam |

## Pitfalls
- **Assuming ck_tile is faster for dense square GEMM** — it is ~1.7× slower at 4096³ (Issue #1727).
- **Skipping `IsSupportedArgument`** — silent garbage.
- **`ckProfiler` missing in deployment images** — build it on a dev box, or fall back to aiter/Triton.
- **gfx950 fp4/mxfp4 gated behind `DTYPES` build flags** — they will not appear unless enabled at cmake
  time.
- **Confusing `MXdlPerWave` with wave count** — it is MFMA tiles per wave.

Full list: `ck_traps.md`.

## Sources
- A Block GEMM on MI300 (descriptor hierarchy, pipeline stages, tile sizing): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- `DeviceGemm` base struct (MakeArgument / IsSupportedArgument / MakeInvoker lifecycle): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
- `BlockwiseGemmXdlops_pipeline` template params (Intrawave/Interwave, MPerXDL/NPerXDL/KPack): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1_blockwise_gemm_xdlops__pipeline__v1__ab__scale_3_01_block_gemm_pipeline_scheduler_1f98d5cb27163c1a3364a8c8f61866821.html
- Issue #1727 — ck_tile vs classic v3, 615 vs 359 TFLOP/s @ MI300X, winning instance string: https://github.com/ROCm/composable_kernel/issues/1727
- ROCm "Optimizing with Composable Kernel": https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- Repo deprecation / move to ROCm/rocm-libraries: https://github.com/ROCm/composable_kernel
