---
title: conv2d on Composable Kernel — SOTA card
kind: sota_card
operator: conv2d
backend: ck
gens: [gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
status: sota
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
  - https://github.com/ROCm/composable_kernel
  - https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
---

# conv2d × Composable Kernel

## TL;DR
CK provides the **XDL implicit-GEMM conv solvers** that MIOpen calls inline — and that you can call
directly when MIOpen's DB lacks a fast instance for your shape, or when you need a custom fused epilogue.
The forward op is `DeviceGroupedConvFwdMultipleABD` (1D/2D/3D via `NumDimSpatial`); the matrix-core
implementation is `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3`. CK maps the conv onto a GEMM
(`M=N·P·Q, N=K, K_gemm=C·R·S`) by coordinate transforms in `transform_conv_fwd_to_gemm.hpp` — **no
materialized im2col** — and runs the gridwise XDL GEMM. For most users this is consumed *through* MIOpen;
author CK directly only for an uncovered shape or a fusion MIOpen can't express.

## SOTA implementation(s)
| impl | source | gens/dtypes/layouts | measured perf | when best |
|---|---|---|---|---|
| `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3` | `ROCm/composable_kernel` (`device_grouped_conv_fwd_multiple_d_multiple_r_xdl_cshuffle.hpp`); instances under `library/.../grouped_conv2d_fwd/` | gfx90a/942/950; fp32/fp16/bf16/int8; layouts NHWGC/GKYXC/NHWGK, GNHWC/GKYXC/GNHWK, NGCHW/GKYXC/NGKHW | GEMM-class on large-channel conv; pick the fastest instance with ckProfiler. No on-box measurement. | uncovered shape / custom fused epilogue / direct control |
| `DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK` (DL) | same repo | non-CDNA (RDNA) mostly | single fixed template → poor for many sizes | RDNA / fallback, not CDNA |

Honest note: the **XDL** path is the CDNA one (many tunable instances). The **DL/WMMA** path has a single
fixed template (poor for many shapes) and is for non-CDNA GPUs — don't ship it on Instinct.

## Config space / knobs
Same XDL template knobs as a GEMM: `BlockSize`, `MPerBlock/NPerBlock/KPerBlock` (e.g. 256×256×64),
`MPerXDL/NPerXDL` (32×32), `MXdlPerWave/NXdlPerWave` (4×4), pipeline scheduler (Intrawave/Interwave) +
version (v3 workhorse), CShuffle store vector width, vector loads `AK1/BK1`. Size so
`ceil(M/MPerBlock)·ceil(N/NPerBlock) ≈ k·304` (M=N·P·Q, N=K) to fill MI300X. Pick with `ckProfiler` /
the instance factory. See `../../../languages/composable_kernel/`
and `../../../backends/composable_kernel_lib/`.

## Numerics / parity
fp32 accumulate in the XDL CShuffle (same as GEMM); same-math vs `F.conv2d`, `atol≈1e-2` bf16. fp8 dialect
must match arch (FNUZ gfx942 / OCP gfx950) — wrong dialect ≈ 2× off. See [numerics.md](numerics.md).

## Integration (how it gets used)
- **Via MIOpen** (the common path): the CK conv solvers are built inline; you don't call CK's C++ API —
  you ensure the packaged CK build has instances for your shapes and use `MIOpenDriver` to confirm.
- **Direct**: instantiate `DeviceGroupedConvFwd*`, `MakeArgument` → **`IsSupportedArgument` gate** →
  `MakeInvoker` → `Run`. No Python rebind seam; you compile a CK extension and call it from a custom op.

## Pitfalls & anti-patterns
- ⚠ **Missing instance coverage** — "does not support this GEMM problem" = no compiled instance satisfies
  `IsSupportedArgument` for your dims/strides/dtype/layout. Fix: rebuild including dtype/layout, pad to a
  covered multiple, or generate an instance. Skipping the gate → silent garbage.
- ⚠ Using the DL path on CDNA (single fixed template, poor) — use XDL.
- ⚠ Wrong layout family (NHWGC vs GNHWC vs NGCHW) → no matching instance.
- CK repo deprecated/moved → `ROCm/rocm-libraries:projects/composablekernel`; CK going header-only (build
  `ckProfiler` standalone if absent).

## How to verify
`ckProfiler grouped_conv_fwd <layout/dtype/dims>` prints every instance's TFLOP/s — top line is your
pinned config. Cross-check vs the MIOpen `MIOpenDriver --layout NHWC` solver at the same shape. Parity vs
`F.conv2d` (fp32 accumulate) before pinning.

## Alternatives / cross-links
[miopen.md](miopen.md) (consumes these solvers) · hip · [overview.md](overview.md) ·
languages: `../../../languages/composable_kernel/` (ck_classic,
ck_tile) · lib consumption: `../../../backends/composable_kernel_lib/`.

## Sources
- CK grouped-conv-fwd XDL CShuffle (conv→GEMM transform, layouts NHWGC/GNHWC/NGCHW, fused Ds): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
- CK repo (instances `library/.../grouped_conv2d_fwd/`, DL vs XDL, deprecation/move): https://github.com/ROCm/composable_kernel
- Implicit-GEMM convolution concept (im2col not materialized): https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
- Optimizing with Composable Kernel: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
