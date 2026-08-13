# Open findings from the kernel action cut-off

Evidence collected while removing the three unimplemented kernel actions and
the action-metadata yaml layer. Nothing here is fixed by that work — each item
is recorded so the decision to fix it stays separate from the decision to
delete dead code.

## 1. The KERNEL_AGENT phase gate is bypassed for every real request kind

`src/hyperloom/orchestrator/policy/gate.py:1148-1152`

```python
# R1 phase_incompatible: treat REQUEST kind as the action name for kernel_agent-owned + coordinator-internal kinds.
if (
    target == "kernel_agent" and gated_kind in KERNEL_AGENT_OWNED_ACTIONS
) or gated_kind in COORDINATOR_INTERNAL_ACTIONS:
    self._validate_phase_action(role, gated_kind, intent_kind="request")
```

The lookup tests a **request kind** against a set of **action names**.

- `KERNEL_AGENT_OWNED_ACTIONS` = `{kernel_opt, integrate, gemm_tuning}`
- registered request kinds = `{trace_analyze, run_gemm_tuning, run_optimization, integrate, apply_patch}`
  (`orchestrator/kernel/request_handlers.py` dispatch table)

The two sets intersect only at `integrate` (and `apply_patch` via
`KERNEL_REQUEST_KIND_ALIASES`). So `run_optimization`, `run_gemm_tuning` and
`trace_analyze` — the kinds every real kernel request actually carries — skip
`_validate_phase_action` entirely and are never checked against
`PHASE_ALLOWED_ACTIONS`.

Before the cut-off the gate did fire for `vendor_kernel_config` /
`operator_tuning` / `deep_kernel_analysis`, because those names were in the
owned set: the phase gate was live **only for the three actions that had no
implementation**, and dead for the three that do.

Reproduce: emit `request{target_agent='kernel_agent', kind='run_optimization'}`
from a phase other than KERNEL_AGENT and observe that no `policy_denied`
lands.

Suggested fix: resolve the request kind back to its action name through
`KERNEL_ACTION_REQUEST_KINDS`
(`inference_optimizer/protocol/action_surfaces.py`) and gate on that. Not done
here because it changes what PolicyGate rejects, which is a behaviour change
rather than a removal.

## 2. Three modules with no production caller — deletion or wiring?

Each looks dead to a static call-graph, but each also sits on a path the
product claims to support, so the "delete" reading and the "never got wired"
reading are both consistent with the evidence. They need a runtime probe, not
a grep, before anything is removed.

### `agents/kernel/tools/geak_prompt_patcher.py` (146 lines)

Only importer is `agents/kernel/tests/test_kernel_optimization_verification.py`
(`common/io.py:22` merely names one of its functions in a docstring). The tool
patches GEAK's bundled prompt YAML — and GEAK is the **default** kernel backend
(`request_handlers.py:290`, `_DEFAULT_KERNEL_PHASE_BACKEND_ORDER = ("geak",)`),
so a patcher for the default path that nothing invokes reads more like a
missing call site than like dead code.

### `orchestrator/roles/robustness_pulse.py` (181 lines)

`_resolve_session_dir` (`:55-62`) reads `ROBUSTNESS_AGENT_SESSION_DIR` then
`SESSION_DIR`. The CLI exports neither — it sets
`INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`. The pulse therefore resolves `None`
and returns early on every production run, even though `_grid_runner.py:35`
imports it and `:1245` calls it after every explore variant. Either the env
names are stale or the whole variant-boundary pulse has been inert since the
env was renamed.

### `_grid_variant_filter.apply_compatibility_filter` (~120 lines)

`orchestrator/actions/executors/explore.py:947-950` says so in its own comment:

```python
# `apply_compatibility_filter` is only exercised in tests; the live
# explore dispatch path assembles `grid` here from LLM/specialist
# proposals (params.grid) or the programmatic seed and never re-runs it.
```

Three test files exercise it. Whether the live path *should* be filtering
variants for workload compatibility is a product question.
