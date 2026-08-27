---
title: conv2d — overview
kind: operator_overview
operator: conv2d
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
  - https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
---

# conv2d  (dense spatial convolution via implicit GEMM)

## TL;DR
Standard dense 2D convolution (cross-channel, the ResNet/ViT-stem/diffusion workhorse). On AMD it is
**implicit-GEMM**: the conv is mapped onto a GEMM by coordinate transforms (no materialized im2col), so it
runs on the **matrix cores** and reaches GEMM-class throughput. The production backend is **MIOpen**
(which calls **Composable Kernel** XDL implicit-GEMM solvers inline for the MFMA path); CK can also be used
directly. The two performance facts: **NHWC layout** (the fast XDL/implicit-GEMM solvers + the fused path)
and **solver selection** (`MIOPEN_FIND_MODE` + FindDb). In an LLM stack conv2d is **idle** — it's the VLM
vision-tower / diffusion / CNN lever, not the language-model lever.

## Math contract
`Y[n,k,p,q] = Σ_{c,r,s} X[n,c, p·s_h+r·d_h, q·s_w+s·d_w] · W[k,c,r,s] (+ bias[k])`, stride `s`, dilation
`d`, padding, `groups` (groups=1 dense; groups=C → depthwise_conv). dtype: fp16/bf16/int8 in, **fp32
accumulate**, cast out.
- **Implicit-GEMM mapping**: the conv becomes `Y[N·P·Q, K] = Im2Col(X)[N·P·Q, C·R·S] · W[K, C·R·S]ᵀ`,
  where the `Im2Col` matrix is **never materialized** — CK/MIOpen fold the gather (the `p·s+r·d` index
  math) into the GEMM's A-tile load via coordinate transforms (`transform_conv_fwd_to_gemm`). So the GEMM
  dims are **M=N·P·Q, N=K, K_gemm=C·R·S**; forward maps to a K-contiguous layout (no bank conflict).
- Backward-data and backward-weight are the transposed implicit-GEMMs (training).

## Shape regimes
- Vision: `[N,C,H,W]`, 1×1 / 3×3 / 7×7 filters, stride 1/2, C and K in hundreds–thousands. 1×1 conv is a
  pure GEMM. 3×3 stride-1 is the Winograd/implicit-GEMM sweet spot.
- The dominant-time convs are the large-channel ones (deep ResNet stages, ViT patch-embed) — those are
  matrix-core-bound implicit-GEMM and behave like GEMM (tile to fill 304 CUs, mfma_16x16, NHWC).

## Where it matters (Amdahl)
LLM transformer: **0%** (no convs). VLM vision tower / diffusion U-Net / CNN: the convs are the bulk of
FLOPs, so the implicit-GEMM tiling + NHWC + solver choice is the whole game there. Optimize the LLM GEMMs/
attention first; this matters only for the vision/diffusion component.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| miopen | 🟢 sota (the production conv library; implicit-GEMM/Winograd/CK-inline; NHWC; fusion) | miopen |
| composable_kernel | 🟢 sota-authorable (the XDL implicit-GEMM solvers MIOpen calls; `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3`) | [ck.md](ck.md) |
| hip | 🟡 competitive (hand-written implicit-GEMM only for shapes the libraries miss) | hip |
| triton | 🟡 (Inductor max-autotune can lower conv to Triton when it beats MIOpen) | see depthwise_conv/backends/triton.md |
| cudnn/cutlass | ⚪ na (NVIDIA-only) | — |

## Fusion neighbors
conv + bias + activation (MIOpen fusion plan / CK CShuffle epilogue), conv + residual, batchnorm fold →
see [fusion.md](fusion.md).

## Numerics
fp32 accumulate; implicit-GEMM vs Winograd solver swaps parity-safe within band; int8 needs a quant gate
→ see [numerics.md](numerics.md).

## How to bench
`MIOpenDriver conv -n N -c C -H H -W W -k K -y R -x S -p pad -u stride --layout NHWC -t 1`; CK:
`ckProfiler grouped_conv_fwd ...`; PyTorch: `channels_last` + `MIOPEN_ENABLE_LOGGING_CMD=1`.

## Sources
- MIOpen find/immediate + NHWC + CK-inline solvers: https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html ; https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
- CK grouped-conv-fwd implicit-GEMM (XDL CShuffle, `transform_conv_fwd_to_gemm`, layouts): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
- Implicit-GEMM convolution concept (im2col not materialized): https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
