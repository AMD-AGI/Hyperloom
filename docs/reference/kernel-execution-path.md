---
myst:
    html_meta:
        "description": "Understand Hyperloom kernel execution through GEAK or the phase-level KernelForge rewrite controller."
        "keywords": "Hyperloom, KernelForge, GEAK, kernel rewrite controller, TraceLens, collective, fusion, GEMM"
---

# Kernel execution path

Hyperloom owns kernel work at phase level. It does not select and dispatch one
Forge subprocess per kernel.

## Phase routes

At KERNEL entry, Hyperloom chooses one route:

- **GEAK** is the default whole-phase optimizer.
- **KernelForge** is selected when `KERNEL_OPT_BACKEND_ORDER=forge`.
- **Collective-only** mode runs the collective lane without GEAK or the rewrite
  controller.

The KernelForge route performs these steps:

1. Refresh profiling evidence when required.
2. Run independently gated fusion and collective lanes.
3. Write a versioned handoff containing candidates, workload context, repository
   roots, and trace evidence.
4. Launch `kernelforge.cli kernel-rewrite-controller` once with the handoff,
   output directory, and phase budget.
5. Read the controller's durable state and integrate every published patch
   through Hyperloom's existing E2E validation path.

The controller owns candidate scheduling, retries, and rewrite concurrency.
Hyperloom supplies one phase deadline and reclaims the controller's complete
subprocess tree if that deadline expires.

## Programmatic request handlers

The remaining inline kernel request kinds are:

- `trace_analyze` → `trace_analyze_handler`
- `run_gemm_tuning` → `run_gemm_tuning_handler`
- `run_collective` → `run_collective_handler`
- `integrate` / `apply_patch` → `integrate_handler`

`run_fusion_handler` is called directly by the KERNEL phase. Fusion and
collective remain independently gated and retain their own SharedState
accounting.

## Controller artifacts

For macro cycle `<n>`, Hyperloom writes and reads:

```text
kernel-agent/forge/cycle-<n>/
  handoff/
    handoff.json
    candidates.json
    workload.json
    repositories.json
    trace-evidence.json
  controller/
    state.json
  result/
    summary.md
    patches/
      <task-id>/
        change.patch
        report.md
```

Only patch directories containing both `change.patch` and `report.md` are
published for integration.

## Shared KernelForge submitters

The controller cutover does not retire shared submitter infrastructure.
`forge_submit.py` remains used by GEMM, fusion, collective, integrate, GEAK, and
`forge_collective` paths. Those paths must not be removed as part of per-kernel
dispatch cleanup.

## Multi-node behavior

Patch integration continues through `integrate_handler`. In multi-node mode,
apply/revert is distributed to the serving pods and a successful apply forces a
full server restart before the E2E benchmark.
