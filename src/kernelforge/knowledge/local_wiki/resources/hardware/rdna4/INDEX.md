<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Radeon AI PRO R9700 (RDNA 4, gfx1201) hardware

This map covers gfx1201 architecture facts, with R9700 as the reference board. Use wave32 and WMMA, and check the
compiled gfx1201 ISA for the actual wave size and operand layout. Accumulators
use VGPRs; memory is GDDR6.
Product identity, CU counts and board-specific peaks come from the selected
device; the gfx1201 ISA alone does not identify a board.

Static wave32 allocation uses 1536 VGPRs per SIMD in blocks of 24, with up to
16 waves: 96 VGPRs keeps all wave slots. LDS is 128 KiB per WGP with a 64 KiB
per-work-group cap. These are resource limits, not measured occupancy.
For a kernel using dynamic VGPR allocation, inspect its allocation block size
instead of applying this static allocation estimate.

Use rocprofv3 kernel tracing. List the runtime's counters, use small supported
groups and finite timeouts, and retain raw traces. The implementer receives
rocprofv3 self-profiling instructions; the profiled Analysis tier remains
static-only. Missing counters and the native scaled-FP4 ceiling are unknown.

Sources: [AMD R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html),
[LLVM gfx12 features](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNFeatures.td),
[LLVM register allocation](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp).
