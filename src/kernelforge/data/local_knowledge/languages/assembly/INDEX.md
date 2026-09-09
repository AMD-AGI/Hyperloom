---
title: AMDGPU assembly workflow
kind: index
scope: languages/assembly
updated: 2026-09-09
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AMDGPU assembly workflow

Use assembly to test a specific compiler limitation: instruction scheduling,
register pressure, spills, barriers, or waits. Structural changes belong first
in FlyDSL, Triton/Gluon, or HIP; compile a fresh assembly baseline after each
such change. Read the actual GPU's ISA and memory-ordering documentation from
the hardware knowledge map before changing synchronization or register usage.

## Case knowledge: Neha / Evolve

Read these cards when the symptom matches. Paths are relative to this folder.
They distill Neha Prakriya's published Evolve-produced kernels, launcher, and
tests, with commit-pinned sources. They are not a copy of Evolve's agent skill
or search controller, which are not available in these sources.

| Symptom or decision | Read | Transferable lesson |
| --- | --- | --- |
| Two reductions over the same input; dependency-bound wave reduction | [AttnRes score](cases/evolve_attnres_score_gfx950.md) | Interleave independent DPP chains; finish wave partials through compact LDS. |
| Small softmax followed by a weighted sum; proposed load/barrier overlap | [AttnRes combine](cases/evolve_attnres_combine_gfx950.md) | Schedule independent exponentials; audit live registers before moving loads. |
| A standalone `.s`/`.co` looks fast but its execution path is unverified | [HIP module integration](guides/hip_module_validation.md) | Verify ABI, candidate identity, stream, oracle, and timing before accepting a result. |

These cases replace Triton kernels with handwritten gfx950 assembly. The
AITER launcher lives under `ops/flydsl/` but uses HIP module APIs directly;
it is not evidence of FlyDSL `CompiledFunction` compatibility. Their fixed
Kimi-K3 shape and author-reported timings do not establish performance on
other shapes, GPUs, MoE kernels, or Forge's FlyDSL adapter. The cards distinguish
source observations, reported results, and experiments still to run.

Independent MI355X reproduction is recorded in both cards: the tested score
assembly is slower than the Triton baseline, and combine requires address-carry
corrections before wider allocation testing. Corrected combine shows gains
that depend on cache/input reuse. The published headline ratios were not
reproduced; use the measured scope and numerical contract when choosing a case.

## Case knowledge: verified FlyDSL roundtrip

For AITER's INT4/BF16 MoE, read the
[W4A16 stage1 case](cases/aiter_w4a16_roundtrip_gfx950.md). It covers the actual
FlyDSL launcher adapter, weight/scale layouts, negative controls, and a packed
multiply experiment that passed correctness but produced no useful speedup.
This is Forge validation evidence, separate from the Evolve source cases.

## Case knowledge: Qwen3 model integration

The [Qwen3 Q/K normalization and RoPE case](cases/qwen3_qk_rope_gfx950.md)
connects a standalone handwritten kernel to an existing vLLM model through
Forge's explicit-ABI HIP loader. It covers graph/worker dispatch verification,
BF16 intermediate rounding, same-fusion attribution, and the distinction
between decode throughput and first-token latency. This is a separate
deterministic experiment, not a FlyDSL change or an agent-discovered KEEP.

## Source and toolchain

An editable AMDHSA assembly file contains `.amdgcn_target`, device symbols,
`.amdhsa_kernel` descriptors, and `.amdgpu_metadata`. Preserve the complete
file. `llvm-objdump -d` is useful for inspection but its instruction listing
alone is not a reassemblable source or a launch ABI description.

The embedded compiler in FlyDSL 0.2.0 and 0.2.4 can emit this file with `FLYDSL_DUMP_IR=1` and
`FLYDSL_DUMP_DIR=/path/to/attempt/dumps`. Run the original kernel in a fresh
process with a private `FLYDSL_RUNTIME_CACHE_DIR` so an old disk cache does not
bypass compilation. Find the matching `*_final_isa.s` under the device-symbol
directory. Dump one specialization per directory: shape, dtype, compile-time
constants, target features, and compiler options are part of its identity.
External LLVM mode can skip the ISA dump; do not substitute disassembly or a
different specialization when the compiler did not emit assembly.

Reassemble with the same ROCm LLVM toolchain and target ID as the compiler:

```bash
python -m kernelforge.assembly assemble \
  --source kernel.s --output build/kernel.hsaco \
  --gpu-target gfx950 --toolchain-dir /opt/rocm/llvm/bin
```

The target is an example. Copy the exact target ID from `.amdgcn_target`,
including `xnack`/`sramecc` features if present. The helper invokes `llvm-mc`
and `ld.lld`, retaining descriptors and metadata. It assembles the current
bytes each time and publishes the output only after both commands succeed.
Propagate errors; a previous output at the same path is not a new candidate.

## FlyDSL launcher adapter

The first adapter supports self-contained FlyDSL kernels using the
`CompiledFunction`/`CompiledArtifact` interfaces shipped in 0.2.0 and 0.2.4. It clones the
compiled host module and replaces one GPU code object while retaining the
original argument packing, device symbol, grid, block, shared-memory setup,
and stream. It rejects extern-linked kernels and multi-target objects. For
multiple GPU modules, pass the explicit `binary_name` to choose one.

```python
from pathlib import Path
import flydsl.compiler as flyc
from kernelforge.assembly.flydsl import with_assembly

# example_args contains all original positional arguments, including stream.
reference = flyc.compile(launch_fn, *example_args)
candidate = with_assembly(
    reference,
    Path(__file__).with_name("kernel.s"),
    gpu_target="gfx950",
    toolchain_dir=Path("/opt/rocm/llvm/bin"),
)
candidate(*example_args)
```

Construct the candidate once before timing or graph capture. The returned
callable takes positional arguments, just like `flyc.compile`'s result;
preserve any keyword-based public wrapper the driver already uses. Forward
the stream provided at every invocation. Both reference and candidate remain
independently usable; the adapter never modifies a global compiler hook or
FlyDSL cache entry. Rebuild the candidate after an assembly edit; an existing
callable retains its own previous code object.

Standalone kernels can use `kernelforge.assembly.hip.HipKernel` with explicit
argument types, launch geometry, and stream. It loads fresh code-object bytes
per instance, propagates HIP errors, and binds to the current device. It does
not infer an ABI from the original frontend; the candidate wrapper must verify
tensor shapes, strides, dtype, resource requirements, and metadata. Retain the
module while captured graphs can run and unload explicitly only after GPU work
and graph use finish. See the [HIP guide](guides/hip_module_validation.md).
Selecting Assembly expertise does not automatically make the FlyDSL adapter
work with Triton or HIP callables.

## Evidence and artifacts

1. Measure the original high-level implementation with the protected driver.
2. Run the unmodified assembly through the same callable contract. Require the
   complete correctness suite and timing parity before optimizing.
3. Make one instruction change and rerun correctness, graph-capture
   verification, per-case timing, and relevant counters. Keep the same inputs,
   dtype, tolerances, stream, launch dimensions, and measurement method.
4. Before trusting the route, use a disposable negative control that changes
   the output and confirm the unchanged oracle rejects it. Restore that edit.
5. Keep the launcher and `.s` in source control together. Use explicit
   `--commit-new-path` entries for files created during a campaign; pretracked
   files use the existing KEEP/REVERT path. Do not commit temporary `.o`,
   `.hsaco`, IR dumps, compiler caches, or benchmark logs.

Assembly does not guarantee a speedup. Report parity, regression, and variance
as measured. When the limiting factor is an algorithm or layout, return to
the high-level language, regenerate assembly, and repeat the parity check.
