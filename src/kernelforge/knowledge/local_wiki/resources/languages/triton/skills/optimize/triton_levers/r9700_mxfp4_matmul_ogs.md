<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# R9700 BF16 activation × packed-MXFP4 weight `matmul_ogs`

For GPT-OSS weight-only MXFP4, preserve BF16 activations, packed FP4 weights
and E8M0 scales shared by 32 weights. The serving path is
`triton_kernels.matmul_ogs`; an unquantized BF16 GEMM does not measure it.
Benchmark the routed up/down projections with their installed padding and
routing, preserve the correctness oracle, and measure candidates independently
in serving. The storage format alone does not establish a native scaled-FP4 ceiling.
