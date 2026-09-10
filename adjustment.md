# Adjustments

Where the enablement relocation departed from `enablement-refactor-2.plan.md`,
and why. Everything else in the plan landed as written.

## Order

The plan calls steps 1-5 independent and step 8 last. Two edges force an order:

- Sinking the classifier to `common/` must precede the runtime move.
  `adapters.py`'s only module-level first-party import was
  `agents.framework.enablement`; until that became `common.failure_signature`
  the six runtime modules could not move without carrying the agents package
  with them.
- The runtime move must precede the test relocation. Seven of the 26 misplaced
  test files exercise exactly the modules the runtime move relocates, so the
  other order moves them twice.

## No shims

The plan is silent on whether the old import paths survive. They do not: every
consumer was repointed and the old modules deleted, so `git ls-files` carries
one spelling per module and the guard treats any old path as a live regression
rather than a deprecation.

## Path ownership covers a segment the plan missed

The plan lists six literals. There is a seventh root, `<session>/enablement/stacks/`,
assembled in `integrate_patch.py`; `enablement_stacks_dir()` owns it.

`session_package.py`'s three references are glob patterns, not paths, so a `Path`
helper cannot replace them. `BRINGUP_SEGMENT` and `ENABLEMENT_SEGMENT` are
exported for that reason.

## attempt_root is minted at enqueue, not re-derived

The plan says the fallback in `enablement/build.py` "disappears entirely" without
saying how. `attempt_root` derives from the task_id, which did not exist until
the insert returned. `enqueue_targeted_build` now mints the id itself and passes
it to `create_or_return_existing`, so the params carry the path they name and
both re-derive sites are gone. On-disk layout is unchanged and the idempotency
key is unaffected -- `build_novelty_key` does not read `attempt_root`.

## adapter_parsers instead of splitting adapters.py

The plan proposes splitting `adapters.py` into introspection and acquisition
halves over a shared third module. That is a refactor, not a move: the two
concerns share `BaseAdapter`, `_VenvProvisionMixin` and the `RunFn` injection
point.

The actual coupling is one method. `argv_parser_source` has a single caller and
all four implementations are a bare `return "<source string>"` with no calls into
the acquisition side. Lifting those strings into `framework/adapter_parsers.py`
lets `bringup` import a lookup table instead of the adapter registry, after
which `adapters.py` moves intact.

## DISCOVER_FAILURE_RETRY_LIMIT goes to phases/framework.py

The plan sends it to `framework/artifacts.py`, which classifies candidate
outcomes. A discovery retry bound is unrelated; its only two readers are in
`phases/framework.py`, so it lands beside them and one module is deleted rather
than replaced.

## The test relocation needed a fixture home first

The plan says relocation needs no configuration change. True of `testpaths`, not
of conftest scope: a test moved out of `inference_optimizer/tests/` silently
loses that package's autouse `_isolate_session_layout_env`.

`orchestrator/tests/_fixtures.py` holds the fixtures both packages need and each
`conftest.py` imports them, which is what registers them. `_helpers.py` holds the
plain functions, which need a real module because pytest shares only conftest
fixtures across packages. Neither lives in a common ancestor `conftest.py`, whose
scope would be every test package in the repo.

`test_argv_refusal_round.py` did not move: it imports fixtures from
`test_bringup_round_scenario.py`, one of four files that genuinely depend on
`inference_optimizer.protocol` and stay.

## Guard scoped to agents/framework

The plan proposes an `agents/* -> orchestrator/*` direction rule. Repo-wide it
matches twenty existing sites, nearly all in `agents/kernel/tools`, so it would
land as one fix plus twenty exemptions. Scoped to `agents/framework` it lands
with an empty allowlist and still covers the edge it was written for.

## Counts corrected

- 15 failure-kind constants, not 14 (`EVAL_RUNTIME_FAILURE` was uncounted).
- 26 misplaced test files by import analysis, not 18.
- `test_common_import_lint.py` needed no change; its stale `*_agent` entries are
  inert, and the rule that fires is the `hyperloom.*` branch.

## Two guards that would have failed silently

`build_lifecycle._driver_command` spawned `python -m` against a hard-coded module
path, and `test_git_foreign_checkout_callers` reads a module's source by path to
scan for unguarded git calls. Both are string-typed and fail long after a rename.
The first now derives from the module's `__name__`; the second was repointed.
