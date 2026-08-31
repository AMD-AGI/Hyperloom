---
myst:
    html_meta:
        "description": "Understand the Hyperloom optimization loop: runtime contracts, phase order (PRELUDE through CLOSE), orchestration model, feedback loops, and session artifacts."
        "keywords": "Hyperloom, optimization loop, PRELUDE, FRAMEWORK_AGENT, KERNEL_AGENT, SWEEP, CLOSE, orchestration, session artifacts, AMD GPU, ROCm, LLM inference, PolicyGate, enablement, targeted build, escalation ladder, runnable gate"
---
# Hyperloom optimization loop

This topic describes the current Hyperloom agentic code optimizer loop from the
runtime contracts outward. It intentionally avoids retired action names
and old DFS demo mechanics; the live action catalog, phase allowlist,
PolicyGate, and session artifacts are the source of truth. This optimization
loop runs alongside the agentic kernel optimizer.

```{image} ../images/Hyperloom_optimization_loop.png
:alt: Hyperloom optimization loop: the phase chain PRELUDE, FRAMEWORK_AGENT, KERNEL_AGENT, SWEEP, and CLOSE, where SWEEP can cycle_reloop back to FRAMEWORK_AGENT while budget and leverage remain. Cross-cutting roles — Orchestration, Critic, Robustness, and PolicyGate — govern every write, which flows emit_intent to Critic review to accuracy gate to PolicyGate to runtime state.
:class: hl-lightbox-trigger
```

```{raw} html
<style>
  img.hl-lightbox-trigger { cursor: zoom-in; }
  dialog.hl-lightbox-overlay {
    max-width: 90vw; max-height: 90vh; padding: 0; border: none;
    background: transparent; overflow: visible;
  }
  dialog.hl-lightbox-overlay::backdrop { background: rgba(0,0,0,0.75); }
  dialog.hl-lightbox-overlay img { max-width: 90vw; max-height: 90vh; display: block; cursor: zoom-out; }
</style>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    if (window._hlLightboxInit) return;
    window._hlLightboxInit = true;
    document.querySelectorAll('img.hl-lightbox-trigger').forEach(function(img) {
      img.addEventListener('click', function() {
        var dialog = document.createElement('dialog');
        dialog.className = 'hl-lightbox-overlay';
        var clone = new Image();
        clone.src = img.src;
        clone.alt = img.alt;
        dialog.appendChild(clone);
        dialog.addEventListener('click', function() { dialog.close(); dialog.remove(); });
        document.body.appendChild(dialog);
        dialog.showModal();
      });
    });
  });
</script>
```

## Runtime contract

The optimizer is launched through `python -m hyperloom.inference_optimizer.cli optimize`. A run
must be able to:

- Create or resume a session directory,
- Write `manifest.json`, `state.json`, `storage/coordinator.db`, action
  run workspaces, reports, and `session_breakdown.json`,
- Route intents through the Orchestration, Critic, and Robustness LLM roles,
  and dispatch kernel work to programmatic Python handlers,
- Produce a final report and a dashboard-consumable breakdown.

Private helper names and internal prompt wording are not contracts. The
observable session artifacts and subprocess JSON bridges are.

## Phase order

The Coordinator advances through the live phase chain:

```text
PRELUDE -> FRAMEWORK_AGENT -> KERNEL_AGENT -> SWEEP -> CLOSE
```

Cyclic macro-cycling is always enabled. After SWEEP, the Coordinator can
`cycle_reloop` back to `FRAMEWORK_AGENT` for another pass while
budget and leverage remain. The effective minimum remaining budget to justify
opening a new cycle scales with session length (capped at the 3-hour absolute
floor), so shorter sessions can also reloop when they have proportionally
sufficient time left. The 24-hour threshold only selects long-run budget
accounting: short bounded runs keep charge-back phase budgeting, while long /
unbounded runs use the fixed per-cycle budget window.

Whether another cycle is feasible is surfaced as `cycle_reloop_feasible` in
the ``=== Phase ===`` block for the four middle phases.

`machine_state.PHASE_ALLOWED_ACTIONS` and `PolicyGate` enforce which
actions can run in each phase. Coordinator-owned actions such as
analysis refreshes and close sequencing might be enqueued internally even
when the LLM is not allowed to propose them.

## PRELUDE

PRELUDE establishes the session baseline:

1. `target_analysis` writes the target comparison artifact. If no
   external target GPU is configured, it writes a no-target marker rather
   than pretending target data exists.
2. `baseline` measures the starting throughput and records the benchmark
   invocation needed to reproduce it.
3. `roofline` or `profile` captures the first performance analysis.
   `roofline` is the preferred composite path when enabled; it wraps
   profiling, trace analysis, and `analysis.md` snapshot publication.

`model_class` is supplied by the launcher or derived once from model
metadata at boot. There is no separate live `classify` action.

## FRAMEWORK_AGENT — the optimisation phase

One phase, two levers worked in parallel:

- **Configuration** — the `explore` action runs server-argument and
  environment grids through the canonical `explore_search` ledger. Nothing on
  disk changes, so a revert is a non-composition.
- **Source and upstream** — `integrate_patch` lands every patch, and
  `patch_source` says where the diff came from: `specialist_authored` for one
  a specialist wrote, `upstream_pr` for a diff fetched from an upstream PR
  that a `candidate_discovery_specialist` found, ranked and judged. Both
  serialise on the `workspace_mutation` lane: one landing at a time,
  independent of how fast proposals arrive.

`specialist` serves both levers — investigation, patch authoring, and
candidate discovery are all dispatches of the one specialist action.

The phase advances to KERNEL_AGENT only when **both** levers are dry
(`optimize_no_more_leverage`). Either arm going quiet raises
`switch_bottleneck` so the next macro-cycle steers off this bottleneck,
without abandoning the lever that is still paying. It also exits when its
phase budget is spent, and at the absolute phase cap.

After each KEEP the runtime revalidates the full stack end to end, so the
reported `cumulative_gain_validated` always comes from a measurement taken
with every accepted change applied.

### Enablement escalation ladder

Enablement admission is controlled by `--enablement {off,launch,eval,all}`,
defaulting to `all`. `launch` admits the boot-failure lane, `eval` the
accuracy-failure lane, `all` both. With `off`, neither engages and a baseline
that keeps failing terminates the run with `stop_reason='baseline_failed'`
rather than opening an authoring loop.

When the admitted lane fires — a `(model, backend)` combination cannot launch, or
it launches but fails its accuracy eval (`accuracy_below_floor` /
`eval_runtime_failure`) — enablement repairs it along two axes. **Diagnosis (once):** work out which capability layer is
missing — read the failure signature, the model's `config.json` architecture,
the framework's supported-architecture registry and installed version, and
upstream (whether the capability already exists and in which version/PR). That
picks the entry rung. **Climb (as needed):** start at the lowest plausible rung
and go up only when the current rung cannot make it boot; after each cleared
boot failure, re-diagnose the new (deeper) failure and pick a rung again
(serial enablement — progress is stacked). A model whose architecture is
already supported but merely un-wired needs only the cheap top rungs; a
genuinely-new architecture climbs higher.

The full rendered methodology (the advisory "ladder book") is the canonical
text, built by `build_enablement_ladder_book` in
`hyperloom.agents.framework.enablement_ops` and injected into the enablement
authoring specialist's prompt. The rungs, in increasing complexity:

0. **Rung 0 — diagnose / capability-gap localization.** Read-only:
   classify the failure, read `config.json`, check the supported-arch registry
   and version, and look up upstream. Output: the missing layer and the entry
   rung.
1. **Rung 1 — serve-flag / config wire-up.** The architecture is supported and
   only a serve flag / env / tokenizer-mode / trivial registration alias is
   missing. No new code or dependencies.
2. **Rung 2 — in-tree source patch.** A unified diff against the installed
   source tree: register the arch, a small forward/config/tokenizer bridge, or
   backport a merged PR. Pure Python, no compile.
3. **Rung 3 — attempt-scoped runtime.** Acquire a runtime (a published
   wheel, an editable checkout at a ref, or a local source tree) into an
   isolated per-attempt venv. That venv is activated only through the
   per-variant YAML benchmark envs; the Coordinator never mutates its own
   process environment to point at an attempt runtime.
4. **Rung 4 — source localization.** Localize a merged-PR or vendored
   closure into the source tree. A change that touches compiled or
   build-backend files cannot be satisfied by a plain source edit, so it
   defers to Rung 5.
5. **Rung 5 — off-loop compiled build.** Perform a compiled-component build
   (AITER kernels, sgl-kernel, or vLLM-from-source). Builds run *off* the
   coordinator tick loop on a dedicated single-slot build lane: each build
   is spawned detached and reaped across later ticks against a wall-clock
   budget, so a long compile never blocks the loop. Auto-escalation into
   Rung 5 can be disabled with the `HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD`
   environment variable (see
   [Targeted builds (Rung 5)](../reference/environment-variables.md#targeted-builds-rung-5)).

### Runnable gate (earned KEEP)

A verified build does not KEEP on artifact verification alone. After a
build's artifacts verify, the Coordinator runs a launch probe: it boots the
actual model with the built runtime through the same runnable-decision gate
the authored-patch lane uses. Only a runtime that actually launches — and
passes minimal correctness — earns KEEP. For an eval-origin trigger the gate
additionally re-runs the accuracy eval against the captured contract and a
KEEP requires the accuracy to meet the floor; the KEEP is finalized only after
a genuine baseline revalidates it. Otherwise the build reverts, or, if the
boot advanced past the original failure to a new or deeper gap, the loop
advances to the next round to repair that gap.

### Discovery-driven build refs

Candidate PR refs discovered by the framework-agent feed directly into the
build: a PR reference becomes a checkout of that PR's head, not just a
released-tag autoselect. This makes support that exists only in an
unreleased PR or branch reachable. When a discovered PR ref drives the
build, the source PR URL is recorded as build provenance in the session
breakdown's `installed_versions` map (`source_pr_url`); see the
[`enablement` section](../reference/session-breakdown.md#enablement--admission-round-lifecycle-builds--attempt-runtimes).

### Novelty and crash safety

- **Novelty-ledger stall gate.** Repeated identical build attempts — same
  component, ref, GPU arch, and build command — are treated as a stall and
  revert. A novel attempt, or a time-based failure such as a timeout,
  advances instead, so the loop keeps making forward progress rather than
  looping on an identical failing build.
- **Crash / resume.** An in-flight build is tracked by a durable sentinel,
  so a crash or resume reclaims the running build or cleans up the orphaned
  one rather than leaking it.

## KERNEL_AGENT

The `KERNEL_AGENT` phase is the bridge to kernel-agent work. Orchestration might
send kernel requests, but the Coordinator owns the request handlers and safety
gates.

What the phase actually does depends on the kernel backend, and the branches
look very different from Orchestration's side:

- **Default (`geak`)**: entering the phase hands it to a single
  Coordinator-owned whole-pipeline GEAK e2e run, which then sets the
  `skip_to_sweep` escalate hint. When the run produces no win, `exit_normal_kernel`
  honours the hint immediately and the phase closes without Orchestration ever
  taking a turn in it.
- **On a GEAK win**, the Coordinator enqueues a same-harness revalidation
  rebench and marks `geak_pending.status = "awaiting_rebench"` with the task id.
  `kernel_work_pending` then reports `True`, and `exit_normal_kernel` refuses the
  `skip_to_sweep` handoff while work is pending — so KERNEL stays open, and
  Orchestration does tick until the rebench lands (or the revalidation turns out
  to be unavailable, which drops the pending slot and lets the exit through).
- **Forge (`KERNEL_OPT_BACKEND_ORDER=forge`)**: the phase runs the deterministic
  KERNEL-entry ladder — GEMM tuning, the fusion and collective lanes, then
  per-kernel `kernel_opt` — and Orchestration drives the remaining kernel work
  through the request channel.

See [Kernel optimization execution path](../reference/kernel-execution-path.md) for the
entry-hook branch order.

The phase allowlist (`machine_state.PHASE_ALLOWED_ACTIONS[KERNEL_AGENT]`)
admits these actions:

- `kernel_opt`
- `integrate`
- `gemm_tuning`
- `specialist`
- `roofline`
- `profile`
- `recover`

Within the kernel-agent request channel, the handler dispatches request kinds
such as `trace_analyze`, `run_optimization`, and `run_gemm_tuning`
(`request_handlers.py`); these are handler kinds, not phase actions.

Kernel-owned results are recorded separately from non-kernel action
attempts. A KEEP must be integrated before the run can proceed to final
reporting, and hot reusable kernels above the configured threshold must
be attempted or explicitly rejected before report can close the run.

## SWEEP

SWEEP checks whether the optimized stack still wins across workload
frontiers. The normal sweep action explores concurrency and ISL/OSL
points; `conc_sweep` can run a post-sweep concurrency ladder when
enabled.

Sweep results update `last_sweep` / `last_conc_sweep` and feed the final
report and breakdown.

## CLOSE

CLOSE drains the final artifacts:

1. `report` renders the operator-facing final report.
2. `session_breakdown` writes the downstream JSON contract.
3. The CLI finally-block writes a safety-net breakdown if the close
   sequencer did not already finish cleanly.

The close path must be idempotent because sessions can end through a
normal phase transition, a wall-clock deadline, an operator interrupt, or
a resumed run.

## Orchestration conversation model

The Orchestration role runs as a single persistent multi-turn
conversation that continues across ticks, rather than a fresh
stateless call each tick. The agent's plan and reasoning live in the
conversation, so reasoning continuity is preserved between ticks.

- **Delta prompts**: The first turn of a (re)started conversation gets a
  full state seed; later turns get only a delta (current phase, mission
  progress, time budget, and new inbox events). The agent pulls anything
  else it needs on demand using read-only context tools
  (`get_shared_state`, `get_gaps`, `get_warm_start`,
  `get_proposal_scores`, `get_intervention_mix`, `why_denied`,
  `show_analysis_md`, `get_inbox`) instead of receiving a full state
  dump every tick.
- **Checkpoint / compaction**: Periodically (phase boundaries and a
  tick/time/size cadence) the Coordinator asks the agent to summarize its
  working memory, persists it to `state.json`
  (`orchestration_memory`), then resets and re-seeds the conversation
  from that compacted memory so context stays bounded on long runs.
- **Resume**: On resume the conversation is rebuilt from
  `orchestration_memory` plus the authoritative `SharedState` facts —
  not by replaying a non-deterministic transcript.
- **Write path unchanged**: All write actions still flow through
  `emit_intent` → the Coordinator's intent handler, so Critic review,
  the accuracy gate, Robustness escalation, and PolicyGate's real
  invariants (path sandbox, resource leases, phase ordering, data
  dependencies, single-writer rules) apply exactly as before. Only the
  compensatory anti-amnesia guards (for example, the baseline same-fingerprint
  self-loop deny) were removed, since a conversational agent remembers
  its own prior attempts. Robustness additionally surfaces a
  conversation no-progress signal as an external circuit-breaker.

The other two roles (Critic, Robustness) remain reactive and stateless per tick.

## Feedback loops

The loop adapts through facts, not through retired score tables:

- `SharedState` carries current best, stack entries, phase history,
  action attempts, kernel attempts, framework-agent progress, and warnings.
- `RecipeKB` records durable lessons and pitfalls for future sessions.
- Critic verdicts gate risky patches and framework candidates.
- Robustness watches stalls, crashes, config-only loops, specialist
  storms, and recovery signals.
- PolicyGate blocks retired actions, wrong-phase actions, unsafe paths,
  and invalid envelopes before they mutate runtime state.

## What is retired

These names shouldn't appear as live positive instructions in prompts
or docs:

- `setup`
- `classify`
- `backends`
- `params`
- `validate_stack`
- `select_kernels`

They can remain only in migration readers, archived breakdown aliases,
or explicit rejection tests.

## Artifacts to inspect

For a finished or interrupted session, start with:

- `manifest.json`
- `state.json`
- `storage/coordinator.db`
- `runs/<action>/<task_id>/`
- `reports/`
- `session_breakdown.json`

For reports and dashboards, `session_breakdown.json` is the external
contract. Its producer code lives under
`src/hyperloom/inference_optimizer/breakdown/`,
and its consumer-facing shape is documented in
[`session_breakdown.json` integration in Hyperloom](../reference/session-breakdown.md).
