---
title: conv2d — fusion
kind: operator_overview
operator: conv2d
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
  - https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
---

# conv2d — fusion

Medium-depth (vision-territory). For the LLM-relevant 1D variant's fusion see causal_conv1d/fusion.

## The fusion that matters: conv + bias + activation (+ residual)
- **MIOpen fusion plan**: build a plan, add ops in **data-flow order** (conv → bias → activation),
  compile **once** (costly), reuse with different buffers. Execute with `miopenExecuteFusionPlan_v2` —
  **NHWC/NDHWC only**. This folds the bias+act into the conv's epilogue so its output tensor is written
  once, not re-read by separate kernels.
- **CK CShuffle epilogue**: the `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle` family fuses an
  elementwise op (bias/act/residual) into the CShuffle writeback — the implicit-GEMM analog of a fused
  GEMM epilogue. `DsLayout`/`DsDataType` carry extra fused tensors (e.g. residual). The XDL CShuffle path
  supports fused bias; the DL path is more limited.

## Conv → 1×1 / next conv
- A **1×1 conv is a pure GEMM** — it does not fuse into the preceding 3×3 conv (different boundness/tiling)
  but its own bias+act fuses into its CShuffle epilogue.
- conv→conv chains stay as separate kernels (each on its tuned tile); the cheap win is fusing the *small*
  epilogue ops, not merging two matrix-core convs.

## Batchnorm
- **Inference**: fold BN into conv weights/bias offline (graph rewrite) → BN is free. Do it at export.
- **Training**: MIOpen offers NHWC batchnorm (now default-NHWC) as a separate op; conv+BN training fusion
  is limited — the BN reduction needs the full conv output. Recent MIOpen added grouped-conv + activation
  and grouped-conv + bias + activation fusions plus NHWC BN.

## Fusion support matrix
| fusion | MIOpen fusion plan | CK CShuffle | HIP (authored) |
|---|---|---|---|
| + bias | yes (NHWC) | yes (epilogue) | yes |
| + activation | yes (NHWC, op-order) | yes (epilogue) | yes |
| + residual add | via plan ops | yes (`DsLayout`) | yes |
| + bias + act (grouped) | yes (recent) | yes | yes |
| batchnorm (inference) | fold offline | fold offline | fold offline |
| conv + 1×1 (next conv) | no | no | no |

## What does NOT fuse (and why)
- conv + the following matrix-core conv (both compute-bound, distinct tiling) — keep separate.
- NCHW input → **no fused path** (the fused executor + CShuffle fast path are NHWC); convert to
  channels-last first or lose the fusion entirely.

## Where fusion moves wall time
In a vision tower, fusing bias+act+residual into the conv epilogue removes extra full-tensor read/writes
of feature maps (often a meaningful fraction of a memory-heavy stage). The matrix-core conv math itself is
tuned independently (like a GEMM). On NCHW the layout transform can cost more than the fusion saves —
channels-last end-to-end is the prerequisite.

## Sources
- MIOpen fusion API (`_v2` NHWC-only, op order, compile-once; grouped-conv fusions, NHWC BN): https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/use-fusion-api.html ; https://github.com/ROCm/MIOpen/blob/develop/CHANGELOG.md
- CK CShuffle fused epilogue (bias/act/residual via Ds, XDL path): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
