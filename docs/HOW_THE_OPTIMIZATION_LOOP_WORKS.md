# How the Optimization Loop Works

This document describes the current Hyperloom optimizer loop from the
runtime contracts outward. It intentionally avoids retired action names
and old DFS demo mechanics; the live action catalogue, phase allowlist,
PolicyGate, and session artifacts are the source of truth.

## Runtime Contract

The optimizer is launched through `inference_optimizer optimize`. A run
must be able to:

- create or resume a session directory,
- write `manifest.json`, `state.json`, `storage/coordinator.db`, action
  run workspaces, reports, and `session_breakdown.json`,
- route intents through the Orchestration, Kernel, Critic, and
  Robustness roles,
- produce a final report and a dashboard-consumable breakdown.

Private helper names and internal prompt wording are not contracts. The
observable session artifacts and subprocess JSON bridges are.

## Phase Order

The Coordinator moves monotonically through the live phase chain:

```text
PRELUDE -> FRAMEWORK_PR -> EXPLORE -> KERNEL -> SWEEP -> CLOSE
```

`phase_state.PHASE_ALLOWED_ACTIONS` and `PolicyGate` enforce which
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

## FRAMEWORK_PR

When enabled, FRAMEWORK_PR is managed by the Coordinator. The only
protected framework-agent integration is discovery via `fa
phase-discover`.

For each candidate:

1. framework-agent returns candidate metadata and diff information,
2. Critic reviews the candidate before apply,
3. the framework PR executor applies, benchmarks, and either keeps or
   reverts the candidate,
4. progress is recorded in `SharedState` and later surfaced in
   `session_breakdown.json`.

The LLM does not own a separate framework role in the current runtime.

## EXPLORE

EXPLORE searches configuration and source-patch levers through the
canonical `explore` ledger:

- `explore` runs server-argument and environment variants.
- `specialist` delegates targeted research or patch proposals to
  specialist domains.
- `integrate_patch` applies Critic-reviewed specialist patches and
  benchmarks them.
- `dynamic_action` can dispatch bounded ad-hoc investigations when the
  policy contract allows it.

The old `backends` and `params` action names are compatibility aliases
for archived reporting only. New sessions write the merged
`explore_search` ledger.

After each KEEP, the runtime revalidates the full stack end to end so
the reported cumulative gain is not just a sum of per-round deltas.

## KERNEL

KERNEL phase is the bridge to kernel-agent work. Orchestration may send
kernel requests, but the Coordinator owns the request handlers and safety
gates.

Live request kinds are:

- `trace_analyze`,
- `run_gemm_tuning`,
- `run_optimization`,
- `integrate`,
- `apply_patch`.

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

## Orchestration Conversation Model

The Orchestration role runs as a **single persistent multi-turn
conversation** that continues across ticks, rather than a fresh
stateless call each tick. The agent's plan and reasoning live in the
conversation, so reasoning continuity is preserved between ticks.

- **Delta prompts.** The first turn of a (re)started conversation gets a
  full state seed; later turns get only a delta (current phase, mission
  progress, time budget, and new inbox events). The agent pulls anything
  else it needs on demand via read-only context tools
  (`get_shared_state`, `get_gaps`, `get_warm_start`,
  `get_proposal_scores`, `get_intervention_mix`, `why_denied`,
  `show_analysis_md`, `get_inbox`) instead of receiving a full state
  dump every tick.
- **Checkpoint / compaction.** Periodically (phase boundaries and a
  tick/time/size cadence) the Coordinator asks the agent to summarise its
  working memory, persists it to `state.json`
  (`orchestration_memory`), then resets and re-seeds the conversation
  from that compacted memory so context stays bounded on long runs.
- **Resume.** On resume the conversation is rebuilt from
  `orchestration_memory` plus the authoritative `SharedState` facts —
  not by replaying a non-deterministic transcript.
- **Write path unchanged.** All write actions still flow through
  `emit_intent` → the Coordinator's intent handler, so Critic review,
  the accuracy gate, Robustness escalation, and PolicyGate's real
  invariants (path sandbox, resource leases, phase ordering, data
  dependencies, single-writer rules) apply exactly as before. Only the
  compensatory anti-amnesia guards (e.g. the baseline same-fingerprint
  self-loop deny) were removed, since a conversational agent remembers
  its own prior attempts. Robustness additionally surfaces a
  conversation no-progress signal as an external circuit-breaker.

The other three roles (Kernel, Critic, Robustness) remain reactive and
stateless per tick.

## Feedback Loops

The loop adapts through facts, not through retired score tables:

- `SharedState` carries current best, stack entries, phase history,
  action attempts, kernel attempts, framework PR progress, and warnings.
- `RecipeKB` records durable lessons and pitfalls for future sessions.
- Critic verdicts gate risky patches and framework candidates.
- Robustness watches stalls, crashes, config-only loops, specialist
  storms, and recovery signals.
- PolicyGate blocks retired actions, wrong-phase actions, unsafe paths,
  and invalid envelopes before they mutate runtime state.

## What Is Retired

These names should not appear as live positive instructions in prompts
or docs:

- `setup`,
- `classify`,
- `backends`,
- `params`,
- `validate_stack`,
- `select_kernels`.

They may remain only in migration readers, archived breakdown aliases,
or explicit rejection tests.

## Artifacts To Inspect

For a finished or interrupted session, start with:

- `manifest.json`,
- `state.json`,
- `storage/coordinator.db`,
- `runs/<action>/<task_id>/`,
- `reports/`,
- `session_breakdown.json`.

For reports and dashboards, `session_breakdown.json` is the external
contract. Its producer code lives under `inference_optimizer/breakdown/`,
and its consumer-facing shape is documented in
`docs/INTEGRATION_SESSION_BREAKDOWN.md`.
