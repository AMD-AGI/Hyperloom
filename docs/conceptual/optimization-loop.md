---
myst:
    html_meta:
        "description": "Understand the Hyperloom optimization loop: runtime contracts, phase order (PRELUDE through CLOSE), orchestration model, feedback loops, and session artifacts."
        "keywords": "Hyperloom, optimization loop, PRELUDE, FRAMEWORK_AGENT, EXPLORE, KERNEL_AGENT, SWEEP, CLOSE, orchestration, session artifacts, AMD GPU, ROCm, LLM inference, PolicyGate"
---
# Hyperloom optimization loop

This topic describes the current Hyperloom agentic code optimizer loop from the
runtime contracts outward. It intentionally avoids retired action names
and old DFS demo mechanics; the live action catalogue, phase allowlist,
PolicyGate, and session artifacts are the source of truth. This optimization
loop runs alongside the agentic kernel optimizer.

![Hyperloom optimization loop: the phase chain PRELUDE, FRAMEWORK_AGENT, EXPLORE, KERNEL_AGENT, SWEEP, and CLOSE, where SWEEP can cycle_reloop back to FRAMEWORK_AGENT when the time budget exceeds 24 hours. Cross-cutting roles — Orchestration, Critic, Robustness, and PolicyGate — govern every write, which flows emit_intent to Critic review to accuracy gate to PolicyGate to runtime state.](../images/optimization-loop.svg)

## Runtime contract

The optimizer is launched through `python -m hyperloom.inference_optimizer.cli optimize`. A run
must be able to:

- Create or resume a session directory,
- Write `manifest.json`, `state.json`, `storage/coordinator.db`, action
  run workspaces, reports, and `session_breakdown.json`,
- Route intents through the Orchestration, Kernel, Critic, and
  Robustness roles,
- Produce a final report and a dashboard-consumable breakdown.

Private helper names and internal prompt wording are not contracts. The
observable session artifacts and subprocess JSON bridges are.

## Phase order

The Coordinator advances through the live phase chain:

```text
PRELUDE -> FRAMEWORK_AGENT -> EXPLORE -> KERNEL_AGENT -> SWEEP -> CLOSE
```

For a normal single-pass run (`--max-hours < 24`) the chain is traversed
once. Cyclic macro-cycling is enabled by default
(`INFERENCE_OPTIMIZER_CYCLIC_PHASES`): with a large or unbounded budget
(`--max-hours >= 24`), SWEEP can `cycle_reloop` back to `FRAMEWORK_AGENT` /
`EXPLORE` for another pass instead of closing.

`machine_state.PHASE_ALLOWED_ACTIONS` and `PolicyGate` enforce which
actions can run in each phase. Coordinator-owned actions such as
analysis refreshes and close sequencing may be enqueued internally even
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

## FRAMEWORK_AGENT

When enabled, the `FRAMEWORK_AGENT` phase (framework-PR enablement) is managed
by the Coordinator. It covers discovery/ranking/audit via `fa phase-discover`,
plus authoring-specialist dispatch (`framework_agent_authoring_enabled` is on by
default), enablement repair, and Critic review of each candidate — discovery is
one integration among several, not the only one.

For each candidate:

1. The framework-agent returns candidate metadata and diff information,
2. The Critic reviews the candidate before apply,
3. The framework PR executor applies, benchmarks, and either keeps or
   reverts the candidate,
4. Progress is recorded in `SharedState` and later surfaced in
   `session_breakdown.json`.

The LLM doesn't own a separate framework role in the current runtime.

## EXPLORE

EXPLORE searches configuration and source-patch levers through the
canonical `explore` ledger:

- `explore` runs server-argument and environment variants.
- `specialist` delegates targeted research or patch proposals. A single
  unified specialist covers single-domain, cross-domain (`scope=domains`),
  and free-form (`scope=freeform`) investigations via its dispatch dials
  (`scope` / `mode` / `bench` / `lane`).
- `integrate_patch` applies Critic-reviewed specialist patches and
  benchmarks them.

The old `backends` and `params` action names are compatibility aliases
for archived reporting only. New sessions write the merged
`explore_search` ledger.

After each KEEP, the runtime revalidates the full stack end to end so
the reported cumulative gain is not just a sum of per-round deltas.

## KERNEL_AGENT

The `KERNEL_AGENT` phase is the bridge to kernel-agent work. Orchestration may
send kernel requests, but the Coordinator owns the request handlers and safety
gates.

The phase allowlist (`machine_state.PHASE_ALLOWED_ACTIONS[KERNEL_AGENT]`)
admits these actions:

- `kernel_opt`
- `integrate`
- `deep_kernel_analysis`
- `operator_tuning`
- `vendor_kernel_config`
- `gemm_tuning`
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

The other three roles (Kernel, Critic, Robustness) remain reactive and
stateless per tick.

## Feedback loops

The loop adapts through facts, not through retired score tables:

- `SharedState` carries current best, stack entries, phase history,
  action attempts, kernel attempts, framework PR progress, and warnings.
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
contract. Its producer code lives under `inference_optimizer/breakdown/`,
and its consumer-facing shape is documented in
[`session_breakdown.json` integration in Hyperloom](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/reference/session-breakdown.md).
