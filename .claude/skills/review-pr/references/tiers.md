# Backbone tier table

Consulted from Step 3 of [`../SKILL.md`](../SKILL.md). Tiers are assigned by blast radius and by how
a failure surfaces, not by file size or churn.

| Tier | File | Blast radius / failure mode |
|---|---|---|
| 1 | `inference_optimizer/cli/__init__.py` | Eagerly imports Coordinator, executors, `_workload_envs`, `ACTION_CATALOGUE`. Any ImportError in that chain kills the run before a session dir exists |
| 1 | `orchestrator/loop/coordinator.py` | Inherits its 24 collaborator mixins and builds the state they share. A name two mixins define resolves silently by MRO order; `loop/tests/test_coordinator_composition.py` catches it |
| 1 | `orchestrator/state/shared_state.py` | Sole writer of `state.json`, enforces `CORE_STATE_FIELDS`. A dropped field silently changes what every phase reads, and what `--resume-from` can interpret |
| 1 | `orchestrator/loop/writeback.py` | The one path turning a measurement into KEEP/REVERT + KB record. A win recorded as a regression, or nothing persisted and no error |
| 1 | `orchestrator/phases/machine_state.py` | Phase identifiers, ordering, budget redistribution, exit scan. Stall in a phase forever, or wrong budget math for all phases at once |
| 1 | `orchestrator/loop/dispatcher.py` | In-flight table, deadlines, cancellation. Actions never retire, GPU lanes never free, subprocesses leak past session end |
| 1 | `orchestrator/policy/gate.py` | The single chokepoint every Intent crosses before a side effect. A denied action executes, or everything is denied |
| 2 | `orchestrator/kernel/request_handlers.py` | `KERNEL_REQUEST_HANDLERS` dispatch table. A renamed kind dead-ends the lane; the agent waits out its budget |
| 2 | `actions/executors/baseline.py` | The anchor every gain % is measured against |
| 2 | `actions/executors/integrate_patch.py` | Patches live framework trees. A failed revert contaminates every later measurement |
| 2 | `actions/executors/_workload_envs.py` | Single source of the rendered Magpie YAML; `seal_server_argv` settles the arg string. Divergence invalidates the A/B while both runs look healthy |
| 2 | `actions/executors/_grid_runner.py` | Shared by explore and sweep |
| 2 | `orchestrator/phases/framework.py`, `phases/kernel.py` | One phase produces nothing or loops |
| 2 | `protocol/action_surfaces.py` | Consumed by the CLI catalogue, the gate, the phase machine and the dispatcher. An action here but not in the owning role's set is unroutable |
| 2 | `protocol/intent.py` | Severity/verdict frozensets deliberately duplicated from `agents/critic/runtime/intent_envelope.py`. Change one copy and the emitter produces intents the transport rejects |
| 2 | `common/llm_config.py`, `common/perf_metric.py` | Gateway env resolution and client construction for every role; `graded_axes_of` is what the phase handlers and writeback grade against |
| 2 | `breakdown/schema.py` | Typed shape of the persisted `session_breakdown.json`. Zero import fan-in, so nothing catches writer/reader drift |
| 2 | `orchestrator/framework/paths.py` | Three non-interchangeable resolvers. Patches land in the wrong tree, or the session optimizes a tree it is not measuring |
| 2 | `kernelforge/cli.py`, `kernelforge/config.py` | Dispatched as `python -m kernelforge.cli` — a subprocess contract |
| 3 | everything else | one phase handler, one executor helper, one agent tool, one KB view |

## Tiering a file not in the table

Applies to new files too.

```
Q1 — If this file raises at import time, does `inference_optimizer optimize`
     still reach the point of creating a session directory?
     (cli/__init__.py imports Coordinator, executors, _workload_envs,
     ACTION_CATALOGUE at module level — so most of orchestrator/ is in
     that chain.)                                     → NO  → Tier 1

Q1b — Is it reached not by import but by NAME at runtime — an entry in
     KERNEL_REQUEST_HANDLERS or the action catalogue? Name-resolved
     wiring fails mid-session, not at startup, so it is Tier 1 even
     though nothing imports it.
                                                      → YES → Tier 1

Q2 — Does it decide, persist, or read the number a KEEP/REVERT is made
     from — the baseline anchor, the graded comparison, state.json, or
     the writeback that records the outcome?
     → YES → Tier 1 (a wrong answer here is silent: the run finishes
        green and the conclusion is inverted)

Q3 — Is it a contract with more than one owner: a schema written to disk
     and read by a different module (state.json, session_breakdown.json,
     the session manifest), a constant set duplicated across a layer
     boundary (protocol/intent.py vs agents/critic), or a subprocess
     command line the orchestrator builds and another package parses
     (python -m kernelforge.cli, install.sh, the slurm launchers)?
     → YES → Tier 2 (both sides must move in the same PR; a one-sided
        change type-checks and imports fine)

Q4 — Does it touch a live framework tree, a GPU lane, or a server
     process — i.e. does its failure survive the round that caused it?
     → YES → Tier 2 minimum, regardless of size

Otherwise → Tier 3 (one phase handler, one executor helper, one agent
tool, one KB view).
```

Q1b is the one that bites here. `KERNEL_REQUEST_HANDLERS` and the action catalogue route by string,
so a renamed kind or action passes every import check, passes lint, passes collection, and fails only
hours into a session when that request arrives. Grep the string, not the symbol.
