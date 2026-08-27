---
title: conv2d — tuning
kind: operator_overview
operator: conv2d
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# conv2d — tuning

Medium-depth: vision/MIOpen-territory, light on the LLM path. The LLM-relevant conv is the 1D causal one
(causal_conv1d); conv2d matters only for VLM vision towers / diffusion / CNNs.

## The decision first
conv2d is **implicit-GEMM** → it runs on the matrix cores and behaves like a GEMM. So unlike
depthwise_conv (memory-bound), the GEMM levers **do** apply once you're in the XDL/implicit-GEMM
solver: tile to fill the CUs, mfma_16x16, NHWC. But you usually don't set these directly — **MIOpen picks
the solver**; you select the *mode* and seed FindDb. Drop to CK/HIP only when you need a solver/tile
MIOpen's DB doesn't have.

## MIOpen levers (the production path)
| lever | knob | note |
|---|---|---|
| **layout** | NHWC/NDHWC | the XDL implicit-GEMM + Winograd fast solvers + the fused path are channels-last; `to(memory_format=torch.channels_last)` |
| **solver mode** | `MIOPEN_FIND_MODE` = NORMAL/1 (benchmark all → best, slow start), FAST/2, **HYBRID/3 default**, DYNAMIC_HYBRID/5 | run NORMAL **once** to seed FindDb for production shapes |
| **immediate fallback** | `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK` (default ON, ~90% pick accuracy) | OFF → throughput index, can pick generic GEMM |
| **workspace** | reserve + reuse; a too-small workspace forces a slow algorithm | size once per conv config |
| **precompile** | `miopenConvolution*CompileSolution` | avoid first-call JIT stall |
| **logging** | `MIOPEN_ENABLE_LOGGING_CMD=1` | reveals the chosen solver + a `MIOpenDriver` replay |

Solver families: **implicit-GEMM (XDL/MFMA)** for large-channel dense conv, **Winograd** for 3×3 stride-1,
**direct** for small/odd, **FFT** for large kernels (rare). CK implicit-GEMM solvers are built **inline**
in MIOpen (CHANGELOG) — so "MIOpen conv2d" on the MFMA path *is* CK under the hood.

## CK levers (if you author the implicit-GEMM directly)
`DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3` — the same XDL template knobs as a GEMM:
`BlockSize`, `MPerBlock/NPerBlock/KPerBlock` (e.g. 256×256×64), `MPerXDL/NPerXDL` (32×32), `MXdlPerWave/
NXdlPerWave` (4×4), pipeline scheduler (Intrawave/Interwave) + version (v3 the workhorse), CShuffle store
width. Size so `ceil(M/MPerBlock)·ceil(N/NPerBlock) ≈ k·304` to fill MI300X (M=N·P·Q, N=K). The DL/WMMA
conv kernels have a *single* fixed template set (poor for many shapes) → prefer XDL on CDNA. **Always gate
`IsSupportedArgument`** (missing-instance crash otherwise). See
[ck.md](ck.md) and
`../../languages/composable_kernel/`.

## CDNA3 vs CDNA4
Same matrix-core story as GEMM: gfx942 LDS 64 KB/CU, gfx950 160 KB/CU (larger conv tiles / deeper
prefetch affordable on CDNA4); fp8/MXFP block-scaled MFMA + OCP fp8 on gfx950 (FNUZ on gfx942) for
quantized conv. mfma_16x16 preferred. NHWC on both.

## How to verify a tune helped
```bash
MIOPEN_ENABLE_LOGGING_CMD=1 python infer.py
MIOpenDriver conv -n N -c C -H H -W W -k K -y R -x S -p pad -u stride --layout NHWC -t 1 -V 1
# CK direct:
ckProfiler grouped_conv_fwd <layout/dtype/dims>   # prints every instance's TFLOP/s
```
Compare NHWC vs NCHW and HYBRID vs NORMAL-find; for the matrix-core path judge achieved vs ~45–55% of
peak (not peak). Seed FindDb with one NORMAL find for production shapes.

## Sources
- FIND_MODE, FindDb, AI fallback, immediate, precompile: https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- CK-inline implicit-GEMM solvers, NHWC fast path, repo move: https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
- CK XDL implicit-GEMM template (conv→GEMM transform, layouts, CShuffle): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
- mfma_16x16, ≥1024 grid, ~45–55% of peak, fill 304 CUs: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
