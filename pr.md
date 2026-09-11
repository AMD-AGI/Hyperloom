# enablement-refactor-2 + ENABLEMENT as the sixth phase

119 files, +1120 / −658 lines. Nine commits, two conceptual halves.

---

## Part 1 — module relocation (commits 1–8)

Implements `enablement-refactor-2.plan.md`. Collapses enablement's six homes
into three owners, deletes the repo's only reverse dependency edge, and moves
tests next to the code they exercise.

### Commits

**`5660a0c85` — give session_paths sole ownership of the bringup/enablement segments**

Eight path literals assembled the same two segment strings independently.
`session_paths.py` gains `bringup_dir()`, `enablement_builds_dir()`,
`enablement_stacks_dir()`, and the segment constants `BRINGUP_SEGMENT` /
`ENABLEMENT_SEGMENT` (for callers that need glob strings rather than `Path`
objects).  Also carries the `attempt_root` fix: `enqueue_targeted_build` now
pre-generates the task ID so `attempt_root` is filled in params before the row
is written, eliminating the re-derive fallbacks in `enablement/build.py` and
`targeted_build_executor.py`.

**`df787662e` — sink the failure classifier from the agent package into common**

`agents/framework/enablement.py` had no production consumer inside the agent
runtime — all 22 call sites were in the orchestrator. Living in `agents/` made
the orchestrator import agent Python in-process, rendering the subprocess JSON
contract decorative. `common/failure_signature.py` is the only non-inverting
home (`bringup` and `framework.adapters` both need it at module level).

**`085d31ceb` — move the mandate builder and round artifacts into enablement**

`enablement_ops.py` held the repo's only reverse dependency edge: a
function-local import of `orchestrator.framework.paths` inside
`_resolve_actual_root_hints`, wrapped in a bare `except Exception: pass` to hide
the cycle. As `enablement/mandate.py` that becomes a top-level relative import
and the guard disappears. `phases/_enablement_artifacts.py` (private module of
`phases/` with one production consumer in the enablement lane) becomes
`enablement/artifacts.py`.

**`409d5a648` — delete framework/client.py**

Sixteen-line re-export of two names. `repo_url_for_framework` now comes from
`agents.framework.repo_map` directly. `DISCOVER_FAILURE_RETRY_LIMIT` moves to
`phases/framework.py` beside its only two readers; the module count drops by one.

**`356c769e3` — move runtime acquisition into enablement/runtime**

`bringup/argv_preflight` called `get_adapter(name).argv_parser_source()` to
judge a server argv, which pinned the whole adapter registry — venv creation,
pip installs, ROCm probes — at the bringup layer. The coupling was four bare
string methods with a single caller. They are extracted into
`framework/adapter_parsers.py` (~65 lines); `bringup` imports a lookup function
and nothing else. With that edge removed, `adapters`, `stack_actions`,
`localization`, `build_actions`, `build_utils`, and `targeted_build` move intact
to `enablement/runtime/`. `framework/` is left with `paths.py`, `artifacts.py`,
and `adapter_parsers.py`. Two silent-failure risks fixed in this commit:
`build_lifecycle._driver_command` was spawning `python -m` against a hard-coded
path (now derived from `targeted_build.__name__`), and
`test_git_foreign_checkout_callers` was scanning a 33-line shim.

**`561b00270` — move orchestrator tests next to the code they exercise**

26 test files under `inference_optimizer/tests/` imported only
`hyperloom.orchestrator.*`. Moving them required a conftest hoist first:
`orchestrator/conftest.py` takes the three fixtures (including the autouse
session-layout isolation), and `orchestrator/tests/_fixtures.py` holds the
shared class so both conftest files import it rather than duplicating.
`orchestrator/tests/_helpers.py` holds four plain helper functions that tests
imported as `from .conftest import X` — these need a real module since pytest
does not inject conftest names across packages. Four files stay in
`inference_optimizer/tests/` because they genuinely depend on
`inference_optimizer.protocol` or `session`.

**`bc318b92c` — draw the enablement export surface and guard the relocation**

`enablement/__init__.py` (previously four lines, no `__all__`) exports
`build_mandate`, `build_search_plan`, and `score_enablement_title` — the three
names with cross-package consumers. The four collaborator classes are withheld:
they are resolved by `Coordinator._COLLAB_MODULES` via dotted strings, exposing
them would invite instantiation outside the coordinator.

The relocation guard (`test_enablement_relocation_completeness.py`) follows
`kernelforge/tests/test_rename_completeness.py`: `git ls-files` sweep, justified
allowlist, self-validating "every entry must exempt something" test, plus an
importability check for all ten canonical module paths and a spawn-path check for
`_driver_command`. The direction rule (`agents/framework/* → orchestrator/*`)
lands with an empty allowlist since the commit that removed the last violation
precedes it.

**`3e70587e4` — drop the defensive code and change-narration the relocation carried in**

Three fallbacks that could not fire, four fixture definitions duplicated across
two conftest files, seven docstrings narrating the move rather than the module,
and one dead allowlist entry in the guard. Net −288 lines. Also: `_fixtures.py`
extracted so both `orchestrator/conftest.py` and
`inference_optimizer/tests/conftest.py` import rather than copy; three
three-strike tests set `enablement_mode="off"` to remain honest about the
fast-fail path they are asserting.

---

## Part 2 — ENABLEMENT as the sixth phase (commit 9)

Implements `enablement-rearrange.plan.md`, first PR of three.

**`e139a3408` — make ENABLEMENT the sixth phase of the state machine**

The phase chain is now:
```
PRELUDE → ENABLEMENT → FRAMEWORK_AGENT → KERNEL_AGENT → SWEEP → CLOSE
```

ENABLEMENT is entered from PRELUDE when `enablement_admitted` is true and
`baseline_failure_streak >= 1`. A run whose baseline boots on the first attempt
never enters the phase — `streak == 0` so PRELUDE goes straight to
`_post_prelude_target()`.

`compute_next_phase` gains three keyword arguments — `enablement_enabled`,
`enablement_stalled` (from `RoundStore.consecutive_stalled()`), and
`enablement_in_flight` — passed by the already-async `_advance_phase_if_needed`
in `machine.py`. `machine_state.py` gains no new first-party imports.

**Normal exit** requires all three conjuncts: `baseline_tput > 0`,
`not validation_pending`, and no queued/running enablement `targeted_build` or
`integrate_patch`. The third conjunct is what makes the close-guard deletion
sound: `_maybe_rearm_authored_lane` routes on the result's lane field, not the
current phase, so a build outliving the round would reopen `validation_pending`
inside `FRAMEWORK_AGENT` without it.

**Terminal exits**: `server_argv_invalid`, `environment_fault` (written via
`stop_reason` by the lane's terminal helpers, routed to CLOSE by
`_global_terminal`), and `enablement_attempts_exhausted` (from
`consecutive_stalled >= ENABLEMENT_MAX_ATTEMPTS` in both the lane and the phase
branch of `compute_next_phase`).

**Deletions** — net reduction in mechanism:

- `enablement_close_guard_active()`, `MAX_SKIP_TO_CLOSE_SUPPRESSIONS`, and the
  `skip_to_close_suppressions` field on `EnablementRound` are all gone. The
  `from_dict` method filters unknown keys, so old `state.json` files load cleanly
  with no migration.
- The intent_router suppression block (including the `enablement_skip_to_close_suppressed`
  observation) is deleted.
- The dead stop reason `"enablement_stalled"` is removed from `STOP_REASON_VOCAB`
  (no production writer existed).
- `ENABLEMENT_MAX_ATTEMPTS` moves from `coordinator.py` to `machine_state.py`,
  deleting the import inversion through `lane.py`.

**Bounds change**: both the three-strike gate (`baseline_failure_streak >= 3`)
and the combined backstop (`_BASELINE_MAX_TOTAL_FAILURES = 3`) in `writeback.py`
are suppressed while `phase == ENABLEMENT`. Without this, the 8-round cap is
unreachable because three failures terminate the run first.

**Budget**: ENABLEMENT 5%, FRAMEWORK_AGENT 40%→38%, KERNEL_AGENT 50%→47%.
Sum stays 1.0; `work >= 0.8` and `overhead <= 0.1` both hold. No budget exit
for ENABLEMENT (same as PRELUDE).

**Surfaces updated**: `_PHASE_ORIENTATION` (Critic gains an ENABLEMENT entry),
`_BASELINE_RECOVERY_PHASES` (rules F1/F2 render in ENABLEMENT too),
`orchestration.md` (phase-goal section, roofline tag),
`v6.py` `phase_map`, `attribution.py` `phase_buckets`, `render.py`
`cycle_reloop_feasible` set, CLI `--phase-budget-enablement-pct` flag,
`failure_recovery.md` phase tag, and all external docs (README, loop diagram
SVG, `optimization-loop.md`, `how-to/optimize.md`, `environment-variables.md`,
`SKILL.md`).

**Tests**: constant assertions updated (`PHASE_NAMES` tuple, budget identity/sum,
`_PHASE_ORIENTATION`, CLI redistribution figures, `PHASE_GOAL_BLOCKS`, codex
phase list). Three three-strike tests set `enablement_mode="off"`. Six
close-guard and suppression-counter tests deleted; the one asserting the guard
active in SWEEP encodes a state the drain exit condition now makes unreachable.

---

## Follow-up commits (this branch, continuation)

Four additional commits close the remaining items found during a post-merge sweep.

**`378b78ae5` — repoint `_environment_fault_is_terminal` → `_check_environment_terminal` in test**

The lane helper was renamed in an earlier commit; seven call sites in
`test_environment_fault_round.py` still used the old name, producing an
`AttributeError` at module collection time that silently dropped all 13 tests.

**`cb7ec4a9c` — cover ENABLEMENT entry and exit predicate in `compute_next_phase`**

Neither `enablement_entered` nor `enablement_done` appeared in any test, and
`enablement_enabled=True` was never passed. Five focused cases now cover: routing
from PRELUDE on a failure streak, skipping the branch when the gate is closed,
normal exit once tput is set and work is drained, hold while a task is in flight,
and hold while `validation_pending` is set.

**`a35c7bdef` — render enablement section in the Markdown session report**

`EnablementBreakdown` reached the JSON but not the Markdown report. The new
`_renderers/enablement.py` surfaces admission status, round outcomes, a bounded
rounds table, and a build-attempts table. Skipped automatically when the
section is `{}` so sessions without enablement are unaffected.

**`13c703ee0` — correct stale phase enumerations after ENABLEMENT insertion**

- `redistribute_budget_pct` docstring: ENABLEMENT is excluded alongside PRELUDE/CLOSE.
- `_post_prelude_target` docstring: called on ENABLEMENT exit too.
- `backfill_langfuse.py` comment: adds ENABLEMENT to the phase-span list.
- `session-breakdown.md`: notes that the Markdown report now renders the section.

**What comes next**

- `AgentBucket` and `AGENT_BY_PHASE` deliberately omit ENABLEMENT. The lever-based
  attribution path (`LEVER_ENABLEMENT → framework_agent`) is reached before the
  phase fallback, so a phase entry would be dead weight.
- `enablement-next.plan.md` asks whether enablement becomes a real agent. That
  decision governs where `classify_failure` and `FailureSignature` live long-term.
