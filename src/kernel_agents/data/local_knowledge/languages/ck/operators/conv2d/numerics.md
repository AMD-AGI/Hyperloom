---
title: conv2d — numerics
kind: operator_overview
operator: conv2d
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int8]
regimes: [both, training]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
  - https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
---

# conv2d — numerics

Medium-depth (vision-territory). For the LLM-relevant 1D variant see causal_conv1d/numerics.

## The contract
Implicit-GEMM conv accumulates in **fp32** for fp16/bf16 inputs (the MFMA C accumulator is fp32), cast to
output dtype on store — identical numerics to a dense GEMM of dims `M=N·P·Q, N=K, K_gemm=C·R·S`. Because
the reduction is over `C·R·S` (can be large for deep stages), accuracy follows GEMM rules: fp32 accumulate
keeps it tight, bf16 in/out floor is the 8-bit mantissa. Parity vs `F.conv2d` is **same-math**:
- bf16: `atol≈1e-2/rtol≈1e-2`; fp16 tighter; fp32 near-exact.

## Solver swaps are parity-safe (within band), not bit-identical
implicit-GEMM (XDL), Winograd, direct, and FFT solvers compute the **same math** with different rounding/
algorithm:
- **Winograd** (3×3 stride-1) does the multiply in a transformed domain → measurably larger error than
  direct/implicit-GEMM, but normally within an `atol≈1e-2` bf16 band. A parity failure *only* on 3×3
  stride-1 → suspect Winograd; loosen tolerance or force a direct/implicit-GEMM solver to isolate.
- **FFT** (large kernels) also carries transform error.
- implicit-GEMM/direct are the closest to a naive reference.
So a FindDb-driven solver change can shift the last bits run-to-run; pin the solver (seed FindDb) if you
need exact reproducibility.

## int8 / fp8 quant
- **int8 conv**: int32 accumulate, per-channel (per-K) scales, requantize on store → **task-accuracy
  gate**, not byte parity. Conv int8 is generally more forgiving than depthwise int8 (channel reduction
  averages scale error) but still validate end-task accuracy.
- **fp8 conv** (gfx942 FNUZ, gfx950 OCP + MXFP block-scaled): the fp8 *dialect* must match the arch —
  wrong dialect is off by ~2× (silent garbage), the same trap as fp8 GEMM/FMHA. Gate on task accuracy.

## Layout ≠ math
NHWC vs NCHW is layout only; same fp32-accumulate result within rounding. NHWC is a perf choice (fast
solvers + fusion). NHWC/NCHW disagreement beyond the band = descriptor/stride bug (MIOpen NHWC wiring
friction, issue #2001), not numerics.

## Verify
```python
y = conv(x_nhwc)                                  # MIOpen/CK via channels_last
y_ref = F.conv2d(x, W, bias, stride, pad, dilation, groups)
torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)   # bf16; loosen for Winograd 3x3s1
```
CK path: `IsSupportedArgument` must pass (a forced-false instance produces garbage, not an error).

## Sources
- fp32 accumulate for fp16/bf16 conv; solver families: https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
- Implicit-GEMM = GEMM of (N·P·Q, K, C·R·S); CShuffle fp32 accumulate: https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/device__grouped__conv__fwd__multiple__d__multiple__r__xdl__cshuffle_8hpp.html
- Implicit-GEMM convolution numerics concept: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
