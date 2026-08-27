---
instruction: hsa_aql_dispatch
category: runtime
architecture: gfx950
tags: [HSA, AQL, dispatch, kernel-launch, barrier, queue, hipModuleLaunchKernel]
---

# HSA AQL Direct Dispatch

## Overview

HSA AQL (Architected Queuing Language) provides direct hardware queue access for kernel dispatch, bypassing the HIP runtime's `hipModuleLaunchKernel`. On gfx950, this does NOT improve throughput versus HIP — the bottleneck is kernel execution, not dispatch latency.

## When to Use

Use HSA AQL dispatch only when:
- CPU-side dispatch is proven to be on the critical path (CPU can't keep up with GPU)
- You need fine-grained control over queue management (custom ordering, multi-queue)

Do NOT use to reduce dispatch overhead in typical workloads — HIP dispatch (~4us/call) is already hidden behind kernel execution.

## Key Findings (gfx950)

| Property | Value | Notes |
|----------|-------|-------|
| barrier=1 penalty | 2x execution time | 8.6ms vs 5.0ms for 292 kernels |
| Recommended fence scope | AGENT (acquire=1, release=1) | SYSTEM scope adds overhead |
| Queue wrapping | SIGSEGV on ROCm 7.2.0 | Use large queue or careful read_idx tracking |

## AQL Packet Structure

```c
hsa_kernel_dispatch_packet_t pkt = {
    .header = HSA_PACKET_TYPE_KERNEL_DISPATCH,  // bits[7:0]
    // barrier=0 (bit 8) — set to 1 only if you need full pipeline drain
    // acquire=1, release=1 (AGENT scope)
    .setup = dimensions,                         // 1D/2D/3D
    .workgroup_size_x = 256,
    .grid_size_x = total_threads,
    .kernel_object = kernel_code_handle,
    .kernarg_address = kernarg_ptr,
    .group_segment_size = lds_bytes,
    .private_segment_size = 0,
};
```

## barrier=1 vs barrier=0

- **barrier=0**: Kernels can overlap at SIMD boundaries. Use for independent kernels.
- **barrier=1**: Full pipeline drain between packets. Doubles execution time. Only use when the next kernel depends on ALL stores from the previous kernel being globally visible.

## Symbol Naming

Kernel symbols in `.hsaco` have `.kd` suffix for `hsa_executable_get_symbol_by_name`:
```c
hsa_executable_get_symbol_by_name(exec, "my_kernel.kd", &agent, &symbol);
```

## Fine-Grained Memory

For CPU→GPU coherent control blocks (avoiding `hipMemcpyAsync`), use HSA fine-grained system memory:
```c
hsa_amd_memory_pool_allocate(system_pool, size, 0, &ptr);
```
