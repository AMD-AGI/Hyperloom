---
myst:
  html_meta:
    "description": "Develop AMDGPU assembly candidates in Forge while preserving FlyDSL's launcher ABI and correctness driver."
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Assembly candidates

Select `--kernel-backend assembly` to explore AMDGPU assembly from a FlyDSL,
Triton/Gluon, HIP, or existing assembly source in the ordinary `forge-loop`.
The backend supplies the high-level/assembly development workflow. The
executable helpers provide AMDHSA reassembly and a FlyDSL launcher adapter;
other frontends still need their own verified code-object loader.

Use the original Python launcher as `--kernel` and retain the existing driver:

```bash
kernelforge forge-loop \
  --workspace "$W" --kernel "$W/kernel.py" --driver "$W/driver.py" \
  --kernel-backend assembly --gpu-target gfx950 \
  --commit-new-path kernel.s --max-hours 1 --git-branch forge-assembly
```

The assembly file and any launcher edits are part of the candidate. Keep
them together through the loop's ordinary correctness, benchmark, KEEP,
rollback, and export path. A change to a global compiler cache is not a
reproducible candidate.

## FlyDSL to assembly

1. Compile one concrete specialization in a fresh process with
   `FLYDSL_DUMP_IR=1`, an attempt-local `FLYDSL_DUMP_DIR`, and a fresh
   `FLYDSL_RUNTIME_CACHE_DIR`. FlyDSL 0.2.4's embedded compiler writes
   `*_final_isa.s` under the device-symbol directory. External LLVM mode can
   skip this dump. Match the dump to its shape, dtype, target, and options.
2. Copy that complete file into the candidate workspace. It must include
   `.amdgcn_target`, AMDHSA descriptors, symbols, and metadata. An instruction
   listing from `llvm-objdump` alone does not contain a callable kernel ABI.
3. Compile the reference launcher and create an independent assembly variant:

   ```python
   from pathlib import Path
   import flydsl.compiler as flyc
   from kernelforge.assembly.flydsl import with_assembly

   reference = flyc.compile(launch_fn, *example_args)
   candidate = with_assembly(
       reference, Path(__file__).with_name("kernel.s"),
       gpu_target="gfx950", toolchain_dir=Path("/opt/rocm/llvm/bin"),
   )
   candidate(*example_args)
   ```

   Use the actual target ID, including any `xnack`/`sramecc` features. The
   adapter preserves the original host module and argument packing, including
   the caller's stream. It accepts the compiled-function interface shipped in
   FlyDSL 0.2.4 and self-contained, single-target GPU objects. Extern-linked
   kernels are rejected. Use `binary_name` to select among multiple modules.
4. Establish correctness and timing parity with the unedited assembly, then
   optimize one hypothesis at a time. Build/load outside timing and graph
   capture. Recreate the candidate after edits; existing callables keep their
   original code objects. Keep the driver's public wrapper and test all of
   its cases, including graph-capture verification.

Standalone reassembly is also available:

```bash
python -m kernelforge.assembly assemble \
  --source kernel.s --output build/kernel.hsaco \
  --gpu-target gfx950 --toolchain-dir /opt/rocm/llvm/bin
```

This invokes `llvm-mc` and `ld.lld` from the selected toolchain. It does not
infer kernel arguments or launch the output. Build failures raise an error;
callers must not load a previous output after failure.

## Moving between languages

Use the high-level source for algorithm, layout, tiling, or pipeline changes.
Use assembly when evidence points to emitted instructions, barriers, waits,
or register allocation. After a structural change, regenerate assembly and
recheck parity before applying instruction edits. FlyDSL campaigns also read
the assembly knowledge card, unless the task explicitly restricts languages.

The `forge-rewrite-by-flydsl` command retains its existing FlyDSL apply-back
contract. Assembly candidates use `forge-loop` and its regular artifacts.
No speedup is assumed: report correctness, per-case timings, variance, and
the original high-level baseline before deciding whether to keep a variant.
