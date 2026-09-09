---
title: Integrating standalone assembly - lessons from Evolve AttnRes
kind: guide
gens: [gfx950]
status: source-inspected; experimental gfx950 harness validated
updated: 2026-09-09
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Validate a standalone assembly candidate's execution path

A fast `.s` file is useful to Forge only when the protected driver executes
that candidate through the correct ABI. Neha Prakriya's Evolve AttnRes
contribution supplies a concrete launcher and oracle to study. The checks
below are integration lessons derived from that code, not recovered Evolve
agent instructions or an already implemented general HIP adapter.

## Source evidence

Read [AITER PR #4863](https://github.com/ROCm/aiter/pull/4863), pinned at
`1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68`:

- [Launcher](https://github.com/ROCm/aiter/blob/1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68/aiter/ops/flydsl/kernels/kimi_k3_attnres.py):
  `ctypes` calls to `hipModuleLoadData`, `hipModuleGetFunction`, and
  `hipModuleLaunchKernel`.
- [Tests and Triton references](https://github.com/ROCm/aiter/blob/1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68/op_tests/test_kimi_k3_attnres_asm.py):
  correctness, repeated-call determinism, and timing wrappers.
- [Score source](https://github.com/ROCm/aiter/blob/1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68/hsa/gfx950/attnres/kimik3_attnres_score_gfx950.s)
  and [combine source](https://github.com/ROCm/aiter/blob/1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68/hsa/gfx950/attnres/kimik3_attnres_combine_gfx950.s):
  full target, symbols, descriptors, and kernarg metadata.

Although the Python file is under `aiter/ops/flydsl/`, its execution path loads
standalone code objects directly. It replaces Triton implementations and does
not use FlyDSL's compiled host launcher. Forge's `with_assembly` adapter needs
an actual supported FlyDSL compiled callable; it cannot accept this wrapper
as if it were one. The [assembler helper](../INDEX.md) can build the source,
but launcher integration is a separate step.

## Available Forge loader

`kernelforge.assembly.hip.HipKernel` provides fresh byte-based module loading,
explicit typed arguments (`ptr`, signed/unsigned 32/64-bit integers, FP32/FP64),
device binding, stream forwarding, error propagation, and explicit unloading.
The caller must match metadata and validate the tensor/launch contract below.
The Qwen3 example demonstrates this route; it does not make the pinned AttnRes
kernel safe without its separately described numerical/address corrections.

Create modules before capture, rebuild after source edits, and retain them
until captured graphs are retired. `close()` requires completed GPU work;
there is no destructor that could unexpectedly invalidate a graph.

## What must hold at the driver boundary

| Boundary | Observation in the pinned source | Requirement for a Forge candidate |
| --- | --- | --- |
| Specialization | gfx950 support guard and fixed Kimi-K3 dimensions; some tensor properties are asserted. | Check the actual architecture, BF16/FP32 types, device, shapes, strides, and launch geometry. The inner H dimension must have the layout the ISA assumes; shape checks alone do not prove this. Preserve the authorized input domain. |
| ABI and resources | Explicit ctypes argument packing, exported symbols, 1024-thread blocks, and shared-memory launch arguments. | Match argument order, widths, offsets, and pointer lifetime to metadata. Reconcile descriptor static LDS with additional dynamic shared memory instead of copying resource numbers blindly. |
| Candidate identity | `_load_fn` caches by `(co_path, kernel_name)` for the process lifetime. | Replacing bytes at the same path need not reload the module. Give each candidate a fresh object identity or isolated process; account for target and device/context as well as code content. |
| Address arithmetic | The pinned combine source has nine low-half pointer adds without a high-half carry; a local correction passed valid tensors spanning a 4-GiB boundary. | Audit every complete address calculation, not only argument packing. Test valid boundary-spanning tensor views and multiple independent allocations. Keep pointer-faulting candidates rejected. |
| Numerical conversion | The pinned combine source truncates FP32 to BF16; a nearest-rounding control substantially reduces error. | Compare against the driver's conversion contract and an independent oracle, not just a global SNR threshold. Record changes to numerical semantics separately from scheduling gains. |
| Stream | `_launch` passes `None` for the HIP stream. | Propagate the driver's actual stream. Verify nondefault-stream execution and graph capture before accepting the wrapper. Load/build before timing or capture. |
| Dispatch | Unsupported hardware or missing `.co` files cause the upstream suite to skip. | Record a skip as unevaluated. Verify the selected function is the assembly candidate, and use a disposable wrong-result edit to prove the unchanged oracle rejects it. |
| Output contract | Score writes nine columns of a 16-column tensor; references initialize outputs. | Preserve required initialization and in-place behavior. Test reused, dirty outputs and changed input tensors to detect stale results or pointer binding. |

An assembly build or launch failure must fail the attempt. A silent fallback
to the original kernel would measure a different implementation. Keep the
protected driver and oracle unchanged while integrating candidate-side code.

## Correctness and measurement evidence

The upstream tests use Triton references, seed 42 random inputs scaled by
0.1, score SNR greater than 35 dB, combine SNR greater than 40 dB, and repeated
bitwise equality. These document the author's test scope. They do not prove
arbitrary inputs, nondefault streams, graph replay, or other GPUs work.

Use the campaign's existing correctness gate. For broader supported inputs,
check multiple seeds, zero/near-zero vectors, and supported extreme finite
logits against an independent reference. The source's SNR helper returns
infinity when the reference norm is zero, regardless of candidate error;
zero-reference cases need a meaningful absolute-error check in the trusted
oracle. Never weaken or edit a protected driver to make a candidate pass;
report missing oracle coverage through the normal driver review process.

The reference wrappers explicitly synchronize while the assembly wrappers
do not. Inspect the timing helper and use the same warmup, synchronization,
stream, graph mode, initialization scope, and repetition protocol for both
sides before interpreting a speedup. Report kernel-only and end-to-end
measurements separately; do not infer model throughput from a kernel ratio.

Retain the source revision, candidate assembly digest, target/toolchain,
specialization, launch contract, correctness outcome, and per-case timing in
the campaign's normal artifacts. Independent gfx950 results are now recorded
in the [score](../cases/evolve_attnres_score_gfx950.md) and
[combine](../cases/evolve_attnres_combine_gfx950.md) cards. They do not reproduce
the published headline ratios: score regresses, while a locally corrected
combine has workload-dependent gains. The validation harness preserves the
upstream snapshots and lives outside the production adapter implementation.
