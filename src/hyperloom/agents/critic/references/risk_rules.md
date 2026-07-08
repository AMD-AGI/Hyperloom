# Risk Rules

Use these rules when deciding which review verdict to return.

## Blockers

Return `reject` or `needs_review` with `severity: "blocker"` when any of these
are true:

- No after-change benchmark is provided for a claimed performance win.
- Before/after benchmark parameters are not comparable.
- Accuracy gate is missing, failed, or waived without explanation.
- The patch changes one layer but the measured active path is not proven to use
  that change.
- The patch requires rebuild, reinstall, server restart, or cache clearing but
  the packet does not show that it happened.
- The change can only be rolled back manually through unclear steps.
- Robustness reports a related crash, hang, accuracy failure, or cache corruption
  that the packet does not address.
- The patch contains unrelated broad refactors that make the optimization effect
  impossible to isolate.

## Major Risks

Return `advise` or `needs_review` with `severity: "major"` when the patch may be
valid but needs more evidence before dispatch:

- Micro-benchmark improved but E2E throughput is missing or inconclusive.
- Benchmark shows a small gain inside expected noise and no repeated run is
  provided.
- Sweep covers only one operating point while the final config will be used more
  broadly.
- The patch edits generated or tuned files without explaining their provenance.
- Prior session history shows similar attempts regressed at adjacent shapes.
- The run uses partial logs that omit build, restart, or cache state.

## Minor Warnings

Use `advise` when the issue should be tracked but does not block dispatch:

- Follow-up sweep could cover more concurrency points.
- Report omits a minor environment field while all core comparability fields are
  present.
- The patch has a clear rollback but no convenience command.
- The final report should add a KB entry for a validated pitfall.

## Benchmark Validity Checklist

Core comparability fields:

- Model and quantization.
- GPU type and GPU count.
- Framework repository and commit or build identifier.
- ROCm, driver, and relevant runtime versions when available.
- Launch script and environment variables.
- TP, PP, DP, node count, and topology.
- Concurrency, batch size, ISL, OSL, dataset, warmup, sample count, and duration.
- Baseline and final `tok/s/GPU` absolute values.
- Timing of restart, rebuild, install, and cache clear steps.

If one of these fields is missing but the packet still proves comparability by
other means, downgrade to `major` or `minor` instead of `blocker`.

## Correctness Checklist

Required evidence depends on patch scope:

- Kernel/operator patch: correctness tests for affected shapes plus E2E accuracy.
- Dispatch patch: proof that the intended implementation is selected at runtime.
- Compiler/config patch: rebuild or cache invalidation evidence plus E2E accuracy.
- Communication patch: hang-free run at target topology plus E2E accuracy.
- Serving/runtime patch: benchmark and accuracy under the final launch config.

## KB Draft Risk Rules

Reject a KB candidate when:

- The lesson depends on unvalidated benchmark evidence.
- The result is contradicted by Robustness findings.
- The action is too specific to a temporary file path or one-off debug state.
- The entry repeats an existing lesson without new scope, evidence, or
  supersession.

Prefer `pitfall`, `benchmark_methodology`, or `crash_recovery` categories for
failed attempts that still teach a reusable lesson.
