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

### `agents/kernel/tools/parallel_e2e_runner.py` (538 lines)

No in-repo caller: the only importer is
`agents/kernel/tests/test_parallel_e2e_env.py` (67 lines, covering
`load_env_file` alias derivation), and `test_kernel_agent_live.py` only names
it in a comment. But it is a standalone CLI (`main()` at `:268`, `__main__`
guard at `:537`) and `pyproject.toml:261` lists it under coverage-omit as a
subprocess driver -- i.e. it is *designed* to be launched from outside the
package, which no grep can rule out. The preceding PR also repaired a flag it
passes (`--test-harness-path` -> `--benchmark-file`), so it was treated as
live very recently. Needs an operator answer, not a call-graph.

### `robustness/signals/stall.py:28` -- `"kernel_agent"` in `_TRACKED_AGENTS`

Not dead, but misfiring. `intent_router.py:718` really does post bus messages
with `from_agent="kernel_agent"`, so the detector has timestamps to compare
against -- it just treats "no kernel request answered for 300s" as a stall,
which is the normal state through EXPLORE and through a long GEAK phase. The
16h Mixtral-8x7B run collected three such alerts (`reports/final.md`:
`agent kernel_agent silent for 309s / 619s / 312s`). Since kernel requests are
answered synchronously inside the tick that raises them, there is no stall
this signal can detect that an orchestration stall would not. Removing it
changes what Robustness reports, so it belongs in a behaviour change rather
than a deletion.

### The remaining 16 `SharedState` forwarding shims (~170 lines)

`shared_state.py` keeps 16 one-to-four-line methods that forward to
`orchestrator/kernel/_kernel_decisions`. Collapsing them was scoped as the
last cut, then measured and dropped: they carry **45 production call sites and
~205 test call sites**, and removing them means every one of those ~40 modules
imports the private `_kernel_decisions` and threads `state` through by hand.
That trades 170 lines of facade for a wider import surface onto a private
module — a relocation, not a removal.

Three shims with no reader at all (`_kernel_ids_in_optimization_stack`,
`_source_files_in_optimization_stack`, `_kernel_trace_impact_pct`) were
removed; their implementations stay because `_kernel_decisions` calls them
internally. `kernel_opt_attempts_count` looked dead to a `.name(` grep but is a
`@property` read by `render.py:588` and `robustness/signals/progress.py:217`.

### `_grid_variant_filter.apply_compatibility_filter` (~120 lines)

`orchestrator/actions/executors/explore.py:947-950` says so in its own comment:

```python
# `apply_compatibility_filter` is only exercised in tests; the live
# explore dispatch path assembles `grid` here from LLM/specialist
# proposals (params.grid) or the programmatic seed and never re-runs it.
```

Three test files exercise it. Whether the live path *should* be filtering
variants for workload compatibility is a product question.
