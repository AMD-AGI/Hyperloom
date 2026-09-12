---
myst:
  html_meta:
    "description": "Port a kernel to verified AMDGPU assembly, then optimize only its .s in the existing Forge loop."
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Assembly campaigns

Use the existing `forge-loop --kernel-backend assembly`. The host runs a
correctness-only PORT phase before optimization; there is no separate ASM
rewrite command. The current scope is one Python Triton/FlyDSL entry point and
one complete AMDHSA `.s`, or an existing standalone ASM implementation.

```bash
kernelforge forge-loop \
  --workspace "$W" --kernel "$W/kernel.py" --driver "$W/driver.py" \
  --kernel-backend assembly --gpu-target gfx950 \
  --max-hours 1 --git-branch forge-assembly
```

## PORT: establish a callable, correct assembly implementation

The workspace must have its source, driver and independent reference committed.
The original driver must validate the original source and supply per-case
benchmarks. An oracle that simply imports the editable kernel as its reference
is insufficient; make the reference independent before starting a campaign.

The host saves the source under `forge_experiments/assembly_port/`, measures it,
and lets a correctness-only agent replace `kernel.py` with launch glue and
create sibling `kernel.s`. Use `--source-files path/to/selected.s` to select a
different single ASM path. Compiler output or an attributed handwritten seed
can supply the initial instructions. The agent preserves the public API and
implements the argument ABI, symbol, grid, block, LDS, device and stream through
`kernelforge.assembly.compiler.assemble` and `kernelforge.assembly.hip.HipKernel`.
All other tracked files and the saved original source are protected.

The host checks complete AMDHSA metadata and target, runs the unchanged driver
and the task's `config.yaml` acceptance suite when present, and deliberately
injects an assembler error. A fresh driver must fail with that error; restoring
the `.s` must pass again. This verifies the build path, not arbitrary Python
semantics: review the launcher and use an independent oracle and wrong-result
negative controls as well. Compilation and module loading belong outside timing
and graph capture; retain each loaded module while its graphs can run.

A correct port may be slower. The host benchmarks it over the complete source
case set and commits the launcher plus ASM before optimization. An existing
standalone launcher/ASM pair follows the same validation and can skip the LLM.
Failed preparation restores the original input files and does not start the
optimizer. PORT shares the campaign deadline and makes at most three attempts.

## OPTIMIZE: edit only the declared .s

Only the selected assembly file is editable. The launcher, original reference,
driver and other tracked files are frozen through the normal workspace and
session integrity guards, including when in-session performance gating is off.
ABI, specialization and launch geometry stay fixed. New files are not accepted.

The ordinary Forge loop performs correctness, repeated timing, KEEP/REVERT and
export. Source timings remain the pristine baseline, and the verified port is
the initial incumbent even if it is slower. The result includes `assembly_port`
with source/initial ASM timings and hashes; regular loop artifacts carry the
optimized ASM measurement. Do not attribute source-level tiling or fusion gains
to instruction edits. Algorithm/layout changes require a separate source
campaign followed by a fresh PORT.

The export base remains the original source commit, so exports include both
launcher and `.s`, including when no later ASM candidate is kept. Resume requires
the matching PORT record and unchanged launcher. Whole-implementation KB
warm-start patches and `--return-after-read-kb` are unavailable for assembly;
such patches could replace the verified launcher. The normal knowledge maps
remain available to the agent.

## Minimal example and validation

The [AttnRes score example](../../../examples/triton2asm-attnres/README.md)
ports one Triton operator using Neha's attributed ASM seed, then optimizes `.s`.
It includes an independent FP64 oracle, fixed cases and a HIP launcher. It is
specialized to gfx950 and makes no speedup claim. The optional GPU regression
covers real PORT, build-error and wrong-result rejection, streams, graph replay
and clean export replay:

```bash
pytest -q src/kernelforge/tests/test_assembly_hip_gpu.py
```

CPU tests cover phase transitions, frozen files, failed preparation, resume,
source-relative scoring and compiler/loader contracts. GPU tests require ROCm
PyTorch and compatible hardware; a skip is not GPU validation.

## Lower-level helpers

`python -m kernelforge.assembly assemble --source kernel.s --output build/kernel.hsaco
--gpu-target gfx950 --toolchain-dir /opt/rocm/llvm/bin` invokes `llvm-mc` and
`ld.lld`. Supply full source with descriptors/metadata, not instruction-only
`llvm-objdump` output. No arguments or launch geometry are inferred. Errors
propagate; a stale binary must never substitute for a failed build.

`HipKernel` accepts explicit `ptr`, `i32`, `u32`, `i64`, `u64`, `f32` and `f64`
arguments and an explicit HIP stream. Initialize the intended device before
loading and call `close()` only after synchronization and graph retirement.

The existing `kernelforge.assembly.flydsl.with_assembly` helper preserves a
FlyDSL 0.2.0/0.2.4 compiled-function launcher while replacing its GPU object.
It is a lower-level roundtrip adapter, separate from the standalone HIP PORT
contract above. Its optional regressions are `test_assembly_flydsl_gpu.py` and
`test_assembly_aiter_moe_gpu.py`. It does not make Triton/HIP callables compatible
with FlyDSL. `forge-rewrite-by-flydsl` retains its existing FlyDSL contract.
