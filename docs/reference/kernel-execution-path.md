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
   in operator-directory order through Hyperloom's existing E2E validation path.

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

Each KERNEL entry gets its own attempt directory under its macro cycle, numbered
upward from `attempt-0`. The Controller refuses an output root it has already
initialized, and `macro_cycle` only advances in EXPLORE — which never runs under
`--no-framework-agent` — so a second entry needs a directory of its own rather
than the cycle's. For macro cycle `<n>`, attempt `<m>`, Hyperloom writes and reads:

```text
kernel-agent/forge/cycle-<n>/attempt-<m>/
  handoff/
    workload.md
    serving-context.md
    trace-evidence.md
  controller/
    state.json
  result/
    summary.md
    patches/
      <encoded-operator-id>/
        change.patch
        report.md
        publication.json
  integration/
    summary.json
    results/
      <sequence>.json
```

Only patch directories containing `change.patch`, `report.md`, and a schema-v2
`publication.json` are eligible for integration. The publication binds the
patch to its six-dimensional operator identity, source repository, kernel path,
Controller base commit, validated Forge commit, and the repo-relative paths the
patch changes. That last list is read from Git rather than from the optimizer's
manifest, because `source_files` is orientation for `forge-loop` rather than an
edit allowlist, so only the diff itself bounds what integration stages.

`controller/state.json` is rewritten as the campaign progresses — after analysis
and after every task reaches a terminal state — not only when the Controller
exits. Hyperloom raises `TimeoutExpired` without the child's streams when it hard
-kills the Controller, so a field that only landed in the terminal write was
unreadable on the ordinary end of a long campaign.

Filesystem directory names percent-encode the canonical ID separators
(`:` → `%3A`) because some shared filesystems reject colons. The canonical
operator ID remains unchanged inside `task.json` and `publication.json`.

Hyperloom verifies the publication against configured patch roots and the
Controller-entry Git baseline. It then applies patches by operator directory
name to the current HEAD. Each KEEP is committed immediately, so the next patch
stacks on the accepted source state. An apply conflict skips only that patch;
an E2E or accuracy failure restores the current HEAD and continues. Integration
results stay in the Hyperloom session and are not written back to KernelForge.

## Shared KernelForge submitters

The controller cutover does not retire shared submitter infrastructure.
`forge_submit.py` remains used by GEMM, fusion, collective, integrate, GEAK, and
`forge_collective` paths. Those paths must not be removed as part of per-kernel
dispatch cleanup.

## Multi-node behavior

Patch integration reuses `integrate_handler` for serving E2E and accuracy
grading after Git applies the source change. In multi-node mode, the configured
integration repository must be visible to the serving environment and
Hyperloom forces a full server restart before the E2E benchmark.
