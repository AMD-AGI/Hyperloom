# adjustment.md — Deviations from enablement-refactor-2.plan.md

This file records every place where the implementation diverges from the source plan,
with the reason. Append a new entry at each step.

## Execution order (plan §Landing sequence)

The plan states steps 1–5 are independent and step 8 is last.
The correct order, derived from actual import edges, is:

  1 → [2 → 8 → 6b] and [3 → 7 → guard], 4, 5 independently; 6a before 6b.

Reason:
- Step 2 is a prerequisite for step 8: `adapters.py` has one module-level
  first-party import — `agents.framework.enablement` (line 15). After step 2
  that becomes `common.failure_signature`, eliminating the only cross-boundary
  dep that would have prevented moving the six runtime modules cleanly.
- Step 8 is a prerequisite for step 6b: seven of the 26 misplaced test files
  (`test_build_actions.py`, `test_build_utils.py`, `test_stack_actions.py`,
  `test_targeted_build.py`, `test_targeted_build_recipes.py`,
  `test_framework_adapters.py`, `test_localization.py`) test exactly the modules
  step 8 relocates; doing 6 before 8 would force two moves.

## Step 1 — path ownership

### Extra literal: `stacks/` root
The plan lists 6 literal sites to replace. There is a seventh:
`actions/executors/integrate_patch.py:1996-1997` assembles
`<session>/enablement/stacks/<fw>/<task_id>`. Added `enablement_stacks_dir()` to
`session_paths.py` and replaced that literal.

### Segment-name constants added
`session_package.py:49,64,66` are glob patterns, not `Path` objects, so a `Path`
helper cannot replace them directly. A `BRINGUP_SEGMENT = "bringup"` and
`ENABLEMENT_SEGMENT = "enablement"` constant is exported from `session_paths.py`
so all four callers derive from the same source.

### `attempt_root` back-write: pre-generated task_id approach
The plan does not specify which of two strategies to use. The implementation
pre-generates a UUID task_id in `enqueue_targeted_build`, derives `attempt_root`
from it, fills `action.attempt_root`, and passes `task_id=` to
`create_or_return_existing` (that parameter already exists at `task_registry.py:240`).
This deletes the fallback branch in `enablement/build.py:400-404` and the
`TargetedBuildExecutor._attempt_root` static method without changing on-disk layout.
No `params` update-after-create is needed, and the idempotency key is unaffected
(`build_novelty_key` does not include `attempt_root`).

## Step 2 — classifier relocation

### Failure-kind constant count: 15, not 14
The plan says "14 failure-kind constant values in the classifier". The actual count is
15: `EVAL_RUNTIME_FAILURE = "eval_runtime_failure"` was not listed.
All 15 string values are preserved verbatim (they are persisted in session breakdowns).

### `test_common_import_lint.py` not modified
The plan says "replace the stale `*_agent` entries in `test_common_import_lint.py`".
Those five entries (`framework_agent`, `kernel_agent`, `critic_agent`,
`robustness_agent`, `quantization_agent`) are in `_FORBIDDEN_TOP_LEVEL`; the actual
guard fires on the `hyperloom.* where parts[1] != "common"` branch (lines 58-62),
which does not depend on those names. Modifying them in step 2 would add churn with
no correctness benefit. They will be cleaned up if and when the `agents/` directory
rename that made them stale is addressed.

## Step 3 — mandate relocation

### Not a purely mechanical move
The plan calls this move "mechanical". It is not, for one reason: `enablement_ops.py`
has module-level imports of `agents.framework.keywords` and `agents.framework.repo_map`
(lines 31-32). Those two modules stay in `agents/framework/`; they are legitimately
shared agent-side (`explorer.py`, `sources/__init__.py`, `sources/github.py` all import
`keywords`). After the move, `enablement/mandate.py` has module-level imports of
`agents.framework.*`, which is the legal orchestrator→agents direction and is fine.
The entry is recorded because the plan's "mechanical" characterisation was wrong.

### Function-local import and try/except deleted
`_resolve_actual_root_hints` previously guarded `from hyperloom.orchestrator.framework.paths import ...`
in a `try/except Exception: pass` (lines 220-225, 244-245) to tolerate the case where
the module is loaded outside the orchestrator. Now that the module lives inside
`orchestrator/enablement/`, the import becomes a top-level relative import and the
guard is unnecessary — removing it is correct, not optional.

## Step 5 — delete `framework/client.py`

### `DISCOVER_FAILURE_RETRY_LIMIT` destination
The plan proposes moving `DISCOVER_FAILURE_RETRY_LIMIT` to `framework/artifacts.py`.
`artifacts.py` contains `candidate_key` and `summarize_candidate_outcomes` — candidate
outcome classification — which is unrelated to a network retry limit. The constant's
only two production readers are `phases/framework.py:1107` and `phases/framework.py:1878`.
It is inlined as a module-level constant in `phases/framework.py` instead.

## Step 8 — runtime relocation

### `adapter_parsers.py` instead of splitting `adapters.py`
The plan proposes splitting `adapters.py` into introspection and acquisition halves,
requiring a shared third module. Exploration showed this is a refactor, not a move:
`argv_parser_source` and `provision` live on the same concrete classes, sharing
`_VenvProvisionMixin`, `BaseAdapter`, and the `RunFn`/`_default_run` injection point.

The actual coupling is much thinner: `argv_parser_source` has exactly one caller in
the entire repo (`bringup/argv_preflight.py:387`), and all four implementations are
bare `return "<hardcoded string>"` with zero internal calls to the acquisition side.

Solution: extract those four strings into a new `framework/adapter_parsers.py`
(~30 lines, one `{framework: source_str}` dict and a `parser_source_for()` function).
`argv_preflight.py` calls `parser_source_for()` instead of `get_adapter().argv_parser_source()`,
importing from `framework.adapter_parsers` only. `adapters.py` moves intact to
`enablement/runtime/adapters.py` and the `argv_parser_source` methods are removed from
the four classes. No class hierarchy is split; no shared base module is needed.

### `integrate_patch.py` import lines changed
The plan states "Do not touch the enablement code in `integrate_patch.py`". That
prohibition covers its enablement *logic* (107 enablement-keyword mentions). Moving the
six runtime modules requires updating 7 import lines in that file (`get_adapter` three
times at :1990/:3162/:3570, `stack_actions` at :1991-ish, `localization` at :2105).
These are mechanical path updates with no logic change.

### Subprocess entry-point: derived from `__name__`
`build_lifecycle.py:59-65` hard-codes the dotted module path
`"hyperloom.orchestrator.framework.targeted_build"` in the `python -m` argv. After the
move that string would be wrong, and the error would only appear at subprocess spawn
time. The string is replaced by `targeted_build.__name__` (a local import then
`module.__name__`), making the entry point self-describing.

## Step 6 — test relocation

### conftest scope: new `orchestrator/conftest.py` required
The plan says "relocation needs no configuration change". That is true for `testpaths`
(the `src/**/tests` glob already covers new directories) but not for conftest scope.
`inference_optimizer/tests/conftest.py` has an `autouse` fixture
`_isolate_session_layout_env` that clears three env vars. Tests moved out of that
directory silently lose it. A new `src/hyperloom/orchestrator/conftest.py` is
introduced, scoped to orchestrator tests only (blast radius: 7 existing dirs +
new ones created in this step). Three fixtures are hoisted: `_isolate_session_layout_env`,
`launch_backend`, `virtual_clock`.

### Shared helpers: `orchestrator/tests/_helpers.py`
Plain functions (`init_git_repo`, `git_commit_all`, `patch_integrate_patch_roots`,
`variant_result`) are imported via `from .conftest import X`, which works because
`inference_optimizer/tests/` has `__init__.py`. A moved file without that package
context gets an `ImportError` at collection. These functions are extracted to
`src/hyperloom/orchestrator/tests/_helpers.py` (a proper importable module).
Seven files that still live in `inference_optimizer/tests/` have their imports updated
to `from hyperloom.orchestrator.tests._helpers import ...`.

### Misplaced test count: 26, not 18
The plan estimates 18 misplaced orchestrator tests. Actual count by import analysis: 26.

### Four boundary files stay in place
`test_build_lifecycle.py`, `test_enablement_coordinator_wiring_unit.py`,
`test_bringup_round_scenario.py`, `test_enablement_breakdown.py` import
`inference_optimizer.protocol` / `session` / `breakdown` surfaces substantively;
they belong where they are.

## Guard — import direction rule scope

The plan proposes a full `agents/* → orchestrator/*` direction rule. That rule hits
20 sites on day one (8 production in `agents/kernel/tools/`, 12 in agent tests).
Landing it would require "fix 1 violation + write 20 allowlist entries". The rule is
scoped to `agents/framework/* → orchestrator/*` instead. After step 3 there are zero
violations in that narrower scope, meaning zero allowlist entries are needed.

## Step 7 — `__init__.py` export surface

The plan says "export data types and pure functions, not the four mixin classes".
After inspection: `lane.py`/`build.py`/`revalidation.py` have no public non-collaborator
module-level names; `params.py` has only `ENABLEMENT_PARAMS_BUDGET_SEC` with zero
external importers. The only name consumed cross-package after step 3 is `build_mandate`
(`specialist_prompt_builder.py:2334`), plus `build_search_plan` and `score_enablement_title`
consumed from `enablement/params.py` (which becomes an intra-package relative import).
`__all__` contains `build_mandate`, `build_search_plan`, `score_enablement_title`.
Exporting names with no external consumer would add non-necessary code.
