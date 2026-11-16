<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Radeon AI PRO R9700 (RDNA 4, gfx1201) hardware

This map covers the verified R9700 target. Use wave32 and WMMA, and check the
compiled gfx1201 ISA for the actual wave size and operand layout. Accumulators
use VGPRs; CDNA MFMA shapes and AGPR budgets do not apply. Memory is GDDR6.

Static wave32 allocation uses 1536 VGPRs per SIMD in blocks of 24, with up to
16 waves: 96 VGPRs keeps all wave slots. LDS is 128 KiB per WGP with a 64 KiB
per-work-group cap. These are resource limits, not measured occupancy.

For GPT-OSS weight-only MXFP4, preserve BF16 activations, packed FP4 weights
and E8M0 scales shared by 32 weights. The serving path is
`triton_kernels.matmul_ogs`; an unquantized BF16 GEMM does not measure it.
Check the installed routing, padding and compiled kernel, preserve the
correctness oracle, and measure candidates independently in serving.

rocprof-compute is unavailable on this target, so automated Analysis is
static-only. For manual profiling, start with rocprofv3 kernel tracing, list
the runtime's counters, use small supported groups and finite timeouts, and
retain raw traces. Missing counters and the native scaled-FP4 ceiling are unknown.

Sources: [AMD R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html),
[LLVM gfx12 features](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNFeatures.td),
[LLVM register allocation](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp).
