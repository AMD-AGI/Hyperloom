---
title: "Validate an assembly candidate and its measurements"
kind: guide
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Validate an assembly candidate and its measurements

Use a protected driver and independent oracle. The
[FlyDSL and HIP runtime references](../../API_docs/runtime_api.md) describe the
supported loaders; [compilation and build](../../API_docs/compilation_and_build.md)
describes complete AMDHSA source. A Python file under `ops/flydsl/` can still
load a standalone HIP module; inspect its actual call path.

## Execution and correctness

| Boundary | Required evidence |
| --- | --- |
| Specialization and ABI | Match target/features, symbols, argument order/widths, device, dtype, shapes, strides, grid/block and static/dynamic LDS. Retain pointed-to tensors through execution. |
| Candidate identity | Rebuild after `.s` edits and create a fresh callable outside timing/capture. A deliberate assembler error must propagate. A no-op must produce a measured correctness failure on fresh outputs beyond compiler warmup; crashes, skips and timeouts cannot satisfy that probe. Restore and revalidate. |
| Independent reference | Derive expected results from original inputs. For quantized MoE, preserve weight/scale shuffle, packing and operand-rounding semantics; a synthetic layout must match the actual specialization. Validate source before candidate. |
| Coverage | Check changed pointers/inputs, partial tiles, zero/near-zero data, valid extreme values and dirty outputs. Verify input preservation and required padding. Zero reference norm needs an absolute-error check. |
| Streams and graphs | Forward the actual stream. Build/load before capture, retain module lifetime through graph use, reset atomic outputs on every invocation, and check changed-input replay on nondefault streams. Reject empty captures. |
| Numerical stability | Use the task's frozen contract and fresh source-before/candidate/source-after evidence. Oracle error and repeat-output error are separate; use synchronized CPU-owned snapshots. |

The minimal workflow example is `examples/flydsl2asm-vector-add/`.
The optional `src/kernelforge/tests/test_assembly_aiter_moe_gpu.py` exercises an
installed W4A16 kernel, argument rebinding and a deliberately wrong SiLU edit.
These validate integration; they are not performance-success cases or permission
to change a campaign's tolerance.

## Timing and model attribution

Keep warmup, stream, synchronization, capture mode, initialization scope, input
distribution and repetitions identical. Exclude compilation consistently and
check the graph actually executes the selected candidate. Test representative
cache reuse and eviction before accepting a cache-policy change.

Record source, unchanged roundtrip and candidate separately. A smaller
instruction count or a better equal-case kernel score does not establish model
throughput. Audit dispatch on all workers; a fused path may bypass the function
being replaced. Compare complete callers with equivalent fusion, intermediate
precision and runtime scalars. Measure TTFT and tail latency alongside throughput.

For E2E finalists, bracket the candidate with source runs under fixed requests
and sampling settings. Retain every run, report drift, and evaluate model quality
separately. Keep source revision, assembly digest, target/toolchain, specialization,
contract, raw measurements and dispatch evidence in campaign artifacts.

Neha/Evolve's [pinned AITER launcher](https://github.com/ROCm/aiter/blob/1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68/aiter/ops/flydsl/kernels/kimi_k3_attnres.py)
and [tests](https://github.com/ROCm/aiter/blob/1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68/op_tests/test_kimi_k3_attnres_asm.py)
provide standalone module-integration examples. Their author-reported speedups
are not Forge reproductions. Use the [debugging guide](../bottleneck/debug-assembly-kernel.md)
when integration or numerical controls fail.
