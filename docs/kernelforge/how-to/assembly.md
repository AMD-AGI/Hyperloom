---
myst:
  html_meta:
    "description": "Optimize compiler-emitted AMDGPU assembly through the original kernel launcher."
---

<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Assembly campaigns

Use `forge-loop --kernel-backend assembly` to optimize the current implementation's
compiler-emitted instructions. There is no LLM PORT phase or separate rewrite
command. The host exports the assembly, connects rebuilding to the original
launcher and validates that path before the optimizer edits `.s`.

```bash
kernelforge forge-loop \
  --workspace "$W" --kernel "$W/kernel.py" --driver "$W/driver.py" \
  --kernel-backend assembly --gpu-target gfx950 \
  --max-hours 1 --git-branch forge-assembly
```

## Optional optimization after a source backend

Source optimization and instruction optimization address different decisions:

```text
original kernel -> source backend optimization -> best source implementation
                                                    |
                                  optional assembly campaign
                                                    |
                         compiler output -> rebuild/validate -> edit .s
                                                    |
                                  KEEP winner or retain best source
```

Start the second campaign from the first campaign's selected source commit in
a fresh workspace with the same independent driver. Its timings become the
assembly campaign's original baseline. The two campaigns have separate budgets
and records; Forge does not automatically run a second stage after every backend.
`forge-rewrite-by-flydsl` keeps its semantic translation PORT phase. Its selected
FlyDSL result can subsequently enter an assembly campaign if it meets the
adapter contract.

| Input | Current support |
| --- | --- |
| FlyDSL with one explicit `flydsl.compiler.compile(...)` call | Automatic capture and replacement, using the original compiled-function ABI. |
| Existing `.s` with a working launcher | Verify its rebuild path and optimize the selected `.s`. |
| Triton / Gluon | Automatic extraction and replacement adapter not implemented. |
| HIP / CK / AITER / hipBLASLt / fusion | Require an explicitly extracted kernel and matching launcher; no general library-binary replacement. An individual AITER FlyDSL kernel can meet the FlyDSL contract. |

That all these implementations eventually execute machine code does not make
their packaging, specialization, ABI or dispatch interchangeable. Only add a
backend adapter when it can export and replace the actual executed kernel.

## Programmatic preparation

Commit the source, driver, and independent reference before starting. For automatic
capture, the Python module must have one direct FlyDSL compile call using a module
import or an imported `compile` alias. Compile hints via subscripting, multiple
compile sites and multiple specializations are currently rejected. Extract a
minimal single-specialization entry rather than generating another algorithm.

The host measures and saves the original source, then mechanically wraps only
the compile call. It captures `*_final_isa.s` from FlyDSL's compiler using private
dump/cache directories. It does not generate instructions with an LLM. The final
binding rebuilds that file with ROCm LLVM and replaces the GPU object through
`kernelforge.assembly.flydsl.with_assembly`. The frontend definitions, host launch
logic, call arguments and stream forwarding remain in place. The binding belongs
at the existing compilation boundary, outside timing and graph capture.

The source IR fingerprint in `kernel.s.json` binds the assembly to one compiled
specialization, ignoring debug locations so the patch can be relocated. Different
specializations fail explicitly. The current adapter requires a self-contained,
single-target `gpu.binary`; extern links, post-load processors and explicit-module
artifacts are unsupported. FlyDSL's dump must be available; an external compiler
mode that does not emit ISA is rejected instead of substituting disassembly.

The host runs the unchanged driver's correctness suite and the task's canonical
`config.yaml` commands. It also inserts a deliberate assembler error, requires a
fresh driver to propagate it, and tests a no-op assembly negative control.
The driver must reject that no-op with a measured SNR or `allclose: False` result
and exit normally before restoration validates again. Timeouts, build/load errors,
unclassified assertions and missing metrics cannot prove that the candidate ran.
A driver that normally raises on output mismatch must report that comparison
as a metric for this check; other runtime errors must still propagate. FlyDSL
`compile(...)` executes the source once during compilation, so the driver must
exercise the returned candidate on fresh outputs beyond that warmup. Missing source,
failed builds and invalid candidates never fall back to the frontend. These checks
verify integration as well as numerical correctness; the original compiler's
correctness alone cannot prove the replacement was loaded.

`forge_experiments/assembly_preparation/result.json` records source and roundtrip
timings, source/launcher/assembly hashes, the binding manifest, and preparation
commit. Preparation failures restore the original files and Git state. Resume
requires a schema-3 record and unchanged frozen inputs, including tracked reference
helpers, compared with the preparation commit. Only the selected `.s` may change.
Older preparation records lack the measured execution proof and are not migrated; start a fresh
campaign from their selected source instead.

## Numerical acceptance

An assembly campaign requires `config.yaml` with both canonical commands and a
`numerical_validation` contract. It is enforced on the source, the compiler
roundtrip, and every candidate that would otherwise be kept. Missing contracts
or measurements block acceptance. Older preparation records without this
contract cannot resume; prepare a fresh campaign with the expanded driver.

The protected task configuration specifies coverage and error limits before
optimization. For an exact vector add, for example:

```yaml
numerical_validation:
  schema_version: 1
  repetitions: 5
  cases:
    random/output/eager:
      max_oracle_error: 0.0
      max_error_ratio: 1.0
      error_floor: 0.0
```

List every required input/seed, output and execution mode as a distinct case.
Include graph replay when the serving path uses it. Existing checks for indices,
padding, input preservation, changed inputs and supported streams still belong
in the authoritative correctness suite. The framework checks declared coverage;
it cannot infer missing workload cases from arbitrary Python or an assembly ABI.

The correctness commands must emit exactly one record through
`kernelforge.loop.numerical.emit_evidence(cases)`. Each case contains its `id`
and `source_before`, `candidate`, and `source_after` measurements. Use
`measure_outputs(callback, reference, repetitions=5)` for each measurement. The
callback runs the chosen implementation and returns the tensor to check; reset
accumulating or dirty output buffers as the operator requires. Compile and warm
up before sampling. The original source callable must bypass the ASM binding;
using the candidate as its own source defeats the independent reference.

The helper synchronizes GPU work and clones each output into separate CPU
storage. `oracle_errors` contains normalized RMS error against one fixed
reference; `repeat_errors` compares successive outputs with the same
normalization. An all-zero reference uses absolute L2 error. Source and candidate
must both be finite and meet `max_oracle_error`. In addition, candidate oracle
error and repeat error must each stay within
`max(max_source_error * max_error_ratio, error_floor)`, using the larger error
from the two bracketing source measurements. Repetitions must match the contract
and number at least three. A deterministic operator can require zero error;
an atomic reduction needs task-specific tolerances established from independent
source measurements. There is no universal repeat-output dB cutoff.

Forge binds the record to the fresh validation invocation and stores its
contract digest with the result. Preparation also records the configuration
digest, which must remain unchanged on resume. The optimizer cannot relax this
contract to save a faster candidate. Other backends can declare the same
contract through their canonical suite.

A candidate-to-oracle SNR of 31 dB can clear a 30 dB mathematical floor while
its repeated outputs agree at only 27 dB and the source agrees at 45 dB.
These are different comparisons. Repeating a scalar SNR test does not measure
this instability. The source-relative checks reject this pattern even when
the original scalar pre-filter passes.

Passing this kernel contract does not certify model quality. Evaluate E2E
finalists on fixed model inputs and quality tasks, and retain their source and
candidate revisions. A throughput-only experiment is not model-validated
acceptance.

## Assembly-only search and result selection

Only the declared `.s` is editable. The frontend, launcher, provenance manifest,
driver, ABI, launch geometry and specialization are frozen. To select an explicit
existing assembly path, use `--source-files path/to/kernel.s`. The build-error probe
must prove the driver actually consumes it. Whole-implementation KB warm starts
are disabled because they could replace the verified binding; knowledge maps
remain available to guide instruction edits.

The original source timings remain the scoring incumbent at 1.0x. The unmodified
roundtrip is diagnostic evidence, not a KEEP, even if noise makes its measurement
look faster. A candidate must pass the normal repeated-measurement KEEP rule and
also beat the original aggregate time. If none does, the result selects the
original commit (`selected_implementation: original`) and publishes no assembly
solution patch. The prepared search workspace remains available for resume.
With a KEEP, the cumulative patch includes the mechanical binding, manifest and
edited `.s`, so a fresh checkout can rebuild it without campaign caches.

Timing noise and integration overhead can make measured roundtrip time differ
from the source. Investigate substantial differences. Keeping the original as
an alternative prevents choosing a measured regression; it is not a guarantee
that every future run has a speedup. Kernel results also do not establish E2E gains.

## Minimal example and tests

The [FlyDSL vector-add example](../../../examples/flydsl2asm-vector-add/README.md)
has one specialization, an independent exact oracle, changed inputs, a nondefault
stream and graph replay. It demonstrates compiler capture and instruction edits,
without a handwritten seed or a performance claim.

```bash
pytest -q src/kernelforge/tests/test_assembly_campaign_gpu.py \
  src/kernelforge/tests/test_assembly_flydsl_gpu.py
```

These optional tests require ROCm PyTorch, FlyDSL and a compatible GPU. They verify
build-failure and wrong-result rejection and replay the complete patch in a clean
checkout. A skip is not GPU validation. CPU tests cover capture provenance,
specialization refusal, frozen files, rollback, result selection and resume.

## Lower-level helpers

`python -m kernelforge.assembly assemble --source kernel.s --output build/kernel.hsaco
--gpu-target gfx950 --toolchain-dir /opt/rocm/llvm/bin` invokes `llvm-mc` and
`ld.lld`. Supply complete AMDHSA source with descriptors and metadata, not an
instruction-only disassembly. Preserve target features such as `xnack`/`sramecc`.

`HipKernel` loads standalone code with explicit argument types, launch geometry
and HIP stream. `with_assembly` retains a FlyDSL compiled function's launcher and
call specification while creating an independently owned execution engine. Both
require compilation/loading before capture and module lifetime through graph
retirement. Neither infers a universal ABI for arbitrary frontend kernels.
