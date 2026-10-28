<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AveLang authoring and validation

## Applicability

These conventions come from the AveLang MoE history suite. Confirm APIs against the source being optimized. The compiler limitations below were observed with AveLang revision `f2c2496e227d838e0edbc6edf878d9799239af85` and its LLVM 23 toolchain; they are not claims about every AveLang version.

## JIT boundaries

- Every helper called from a `@avelang.jit` body must also be `@avelang.jit`. The compiler links helpers by their literal `def` names, so helpers reachable from one kernel need distinct names.
- Annotate every helper parameter, including scalar and tensor arguments. Unannotated parameters in helpers defined inside a `@cache` factory caused native AMDGPU compiler crashes in the pinned version.
- Capture layout sizes and other compile-time values as plain Python constants before defining the JIT function. For example, capture `SIZE = layout.workspace_bytes` before calling `al.make_layout((SIZE,), (1,))` in the body. Python attribute access inside a traced layout expression can fail resolution.
- `al.static_range` unrolls; `al.range` emits a runtime loop. Choose according to compile-time dimensions and code size. Raw-buffer `aux` operands must be compile-time constants.
- Size-one `al.make_local` / `al.make_shared` allocations and scalar views triggered codegen bugs in the pinned version; use at least two elements there.

## Validation

Preserve the supplied driver, references, cases and tolerances. A GPU graph captures kernel arguments and Python-side state: repeated invocations must derive epochs, tickets and counters on the device when those values change between replays. Check repeated graph replay as well as one-call correctness. Run the supplied correctness and benchmark commands during the editing session, inspect failures and repair the scoped sources before finishing. The host independently repeats correctness and timing before accepting the step.
