---
title: AMDGPU assembly runtime API
kind: guide
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AMDGPU assembly runtime API

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

Triton and Gluon use `kernelforge.assembly.triton.with_assembly` with a
`CompiledKernel` returned by JIT `warmup`. It assembles `.s` into a separate
compiled kernel and preserves the original argument marshaller and metadata.
It does not modify Triton's JIT cache. The automatic `TritonAssembly` binding
wraps one bracket launch, validates the compilation identity, and launches only
the candidate. Never run the source as a warmup side effect: atomic outputs would
accumulate twice. Preserve constexprs, options, argument layout, shared-memory
declarations and the current stream. Resolve autotuning/heuristics before capture.

For standalone HIP source, `kernelforge.assembly.hip_source.compile_hip(source,
output, gpu_target="gfx950", flags=("-O3",))` builds a device code object while
retaining compiler ISA. Supply the original include paths and defines. Forge can
wrap this build call with `HIPAssembly`, verify its source/build identity and
rebuild the edited `.s` into the original loader's output path. Construct before
timing or graph capture. This does not extract arbitrary `.so` kernels or infer
C++ argument structs; the existing launcher must already specify that ABI.

Standalone kernels can use `kernelforge.assembly.hip.HipKernel` with explicit
argument types, launch geometry, and stream. It loads fresh code-object bytes
per instance, propagates HIP errors, and binds to the current device. It does
not infer an ABI from the original frontend; the candidate wrapper must verify
tensor shapes, strides, dtype, resource requirements, and metadata. Retain the
module while captured graphs can run and unload explicitly only after GPU work
and graph use finish. See the [HIP guide](../skills/profile/hip_module_validation.md).
Select the adapter for the actual compiler and launch boundary, even when the
operator belongs to AITER, CK or another library.
