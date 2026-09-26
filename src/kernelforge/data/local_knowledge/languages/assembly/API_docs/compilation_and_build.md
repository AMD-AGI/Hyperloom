---
title: AMDGPU assembly compilation and build
kind: guide
scope: languages/assembly
updated: 2026-09-16
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AMDGPU assembly compilation and build

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
