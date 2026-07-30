# Delete gates that never fired, and show the planner what it was blind to

15 commits, two halves. The first removes dispatch-time gates that production
never reaches or that deny what the runtime already handles gracefully. The
second closes the observability holes that removal exposed: the orchestration
model dispatches work it cannot see, and was being told a SIGKILLed specialist
succeeded.

## Result

| | Added | Removed | Net |
|---|---:|---:|---:|
| Source | +802 | −530 | **+272** |
| Tests | +357 | −448 | **−91** |
| **Total** | **+1,159** | **−978** | **+181** |

42 files. `policy/gate.py` alone is +72/−277. The source net is positive
because the second half adds a pull tool, a prompt block, an intent and a
bidirectional file channel; the first half is pure deletion.

**Regressions: zero.** The failure set is byte-identical to a stashed baseline
of the same tree on the same machine — 17 failures (15 under
`inference_optimizer/tests`, 2 under `agents/robustness/tests`), all from shell
environment leakage (`ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` / a gateway key)
and one venv-discovery test. Each batch was separately baseline-diffed before
any failure was attributed to the change, and CI re-establishes this
independently.

## Why the gates went

Each removal below was verified unreachable or non-load-bearing at the source,
not inferred from a grep.

**The free-form red-line scan** matched destructive shell in
`params.task_description`. It scanned that one field, while `notes`,
`research_hints`, `arch_notes` and `gap_symptom` reach the identical specialist
system prompt unscanned — the same text passes by moving fields. After dispatch
nothing enforces it: the subprocess runs under `bypassPermissions` with `Bash`,
`Write` and `Edit`, and no PreToolUse hook exists. Its own comment said it was
not a security boundary. Its only reliable effect was costing the planner a
tick for prose that *describes* a destructive command rather than running one.

**The `specialist_done` validator (R3)** never saw a real payload. Production
specialists write `specialist_done.json`; the dispatcher reads it and calls
`_record_specialist_result` directly, bypassing `_handle_intent` and therefore
`validate_intent`. It was also redundant: `runner._finalize` re-stamps
`gap_canonical_id` and `domain` from the dispatch params and defaults
`proposal_set`, `empty` and `summary`, so every field it checked was already
guaranteed by code the specialist cannot influence.

**Five scope/tag denials became observations.** `resolve_specialist_profile` is
documented never to raise — it re-infers the scope from the tag count — and
`SpecialistRunner` synthesizes a well-formed empty result for an anchor it
cannot resolve. Denying these converted a graceful degradation into a lost
tick.

**Dead or self-limiting checks** went with them: the wave shape checks
(non-list, non-dict entry, empty description) are all re-checked in
`_fan_out_specialist_wave` and `intent_router`; a negative `max_turns` yields
an empty turn range; the `gpu_count` type check is re-parsed by the dispatcher
with the same default.

**Task-registry surface with no counterpart**: `failed -> running` (auto-retry
creates a *new* row under an `-autoretryN` key), `needs_manual_review` (no
production writer), and `Task.attempts` (no production reader — the retry cap
is driven by `params["_auto_retry_attempt"]`).

### What was deliberately kept

- **`max_turns` upper bound.** The in-process backend's turn loop has no
  wall-clock check, so this is the only bound on that path.
- **Wave size cap (16).** `research_lane` capacity bounds concurrency, not
  total spend. One turn can authorize N `claude` subprocesses that outlive the
  turn that asked for them.
- **`params` must be a dict.** Without it the `AttributeError` fires *inside*
  `validate_intent`, escapes the `PolicyDenied`-only handler, and aborts the
  turn's remaining intents while incrementing the emergency-stop crash count.
- **`gpu_count <= 0`.** On the default single-node Ray path this does not
  livelock — `try_acquire_ray_observation` succeeds and the dispatcher writes
  `ROCR_VISIBLE_DEVICES=100000` into the specialist, which then measures
  garbage and reports success.

## The blindness this exposed

A specialist SIGKILLed at its wall-clock cap rendered **byte-identically to a
fully successful one**. Three things combined: the executor never raises, so
the failure rides inside the result envelope with `SubAgentResult.error` left
`None`; the envelope reports `runner_status`, which was absent from the
renderer's status key list; and the reaper's error text was folded into bare
constants. The planner read `kind='specialist' state='succeeded'` and nothing
else — and could not reach the reason even by pulling, because
`get_recent_outcomes` re-renders through the same formatter.

Fixed by adding `runner_status` to the status keys, falling back to the nested
error, surfacing the audit notes (a run whose patches were all dropped as
ungrounded no longer reads as a plain success), and keeping the reaper's
verbatim text after the classifier token so the elapsed and threshold numbers
survive. A checkpoint salvaged after an infra failure now reports `partial`
rather than `succeeded`, and is classified non-retryable so a retry cannot
discard what was rescued.

Three further holes closed:

- **Silent abandonment.** Both auto-retry bail-outs returned without emitting
  anything, so the planner saw retries 1..N−1 and nothing for the give-up. It
  now broadcasts `specialist_auto_retry_exhausted`.
- **No view of running work.** `get_recent_outcomes` queries terminal events
  only, and no prompt section carried task rows, so a dispatched task was
  invisible between `task_queued` and its `delegated_result` — hours, for a GPU
  specialist. `get_running_tasks` joins the running rows against the lease and
  GPU-lease tables: elapsed seconds, domain and gap, lease TTL and remaining
  time, held lanes, leased GPU ids, and heartbeat age from the same files the
  reap loop polls.
- **Unjudgeable GPU requests.** The prompt asked the planner to reason about
  `gpu_count` relative to serving TP without rendering the TP or either pool
  size. The `=== Resource pools ===` block reports the numbers PolicyGate
  actually admits against, including that the serving-disjoint pool is empty
  whenever serving owns every card.

## New capabilities

**`kill_task` from orchestration.** The restriction to Robustness was a role
partition, not a safety argument. This required fixing a bug first: killing a
running task destroyed its result, because the cancel makes the row terminal
while the executor is still running, so its closing transition raised
`IllegalTransition`, escaped `run_task`, and was dropped by the reap loop — no
`delegated_result`, no bookkeeping, for up to hours of GPU work. `scope` stays
task-only.

**`extend_lease`.** Nothing ever renewed a lease: `heartbeat` existed,
CAS-protected, with zero callers, so `lease_ttl_sec` was fixed at enqueue and a
task that legitimately outran its registry default was failed out from under
live work. The intent refreshes the task TTL, its lane rows and its GPU rows
together, preserving `kill <= gpu_lease TTL <= gpu_research_lane TTL`.

**A two-way specialist channel.** Previously fire-and-forget: parent writes
`prompt.md`, spawns, reads `specialist_done.json` after death. A specialist
that discovered its mandate was wrong an hour in could only burn the remaining
budget.

- *Uplink*: the reap loop already stat'd the workspace every 5s but read
  `specialist_done.partial.json` only after the process died, as salvage. It
  now parses each rewrite while the specialist is alive and republishes it as a
  `specialist_progress` observation.
- *Downlink*: a `send_message` addressed to `specialist:<task_id>` is appended
  to `inbox.json` in the specialist's workspace, which the prompt tells it to
  read between steps. The reaper ignores that file, so a message steers a live
  run instead of ending it — the missing half, since `residual_questions`
  previously had no answer path.

## Bugs found and fixed during self-review

The final two commits are the result of reviewing the first thirteen. They are
worth reading as a unit, because four of the six are defects the
instrumentation work itself introduced.

### The prompt that triggered this review

This prompt is general — it is worth running at the end of any batch of work,
fanned out across several review lenses in parallel:

> For the commits starting at `<base-commit>`, reflect on three questions:
> 1. Is the change complete — are the docstrings updated too?
> 2. Is the change redundant? Do not over-commit fallback code; I want logic
>    only.
> 3. Are the comments too long? Functional comments only.
>
> Keep the whole change concise and precise.

The three questions catch three different classes of defect, and all three
landed here:

- **"Is it complete"** surfaced two hard breakages — the missing
  `_PAYLOAD_REQUIRED` entry for `EXTEND_LEASE`, and a contract test that was
  already failing. Both were invisible because `agents/robustness/tests` is a
  separate test tree and only `inference_optimizer/tests` had been run. The
  point of this question is that **when behaviour changes, every docstring,
  comment, prompt string and mirrored enum describing it has to change too** —
  each one missed is the entry point for the next bug.
- **"Is it redundant"** removed six `try/except` blocks that cannot fire, an
  unread `-> bool` return, a hand-rolled `needs_gpu` coercion the repo already
  provides as `coerce_needs_gpu`, and a fabricated `Lease` that only worked
  because the callee happened to read two of its fields.
- **"Are the comments too long"** deleted roughly 20 lines narrating history
  and rationale ("this used to…", "we changed this because…"). That belongs in
  the commit message, not the code.

One lesson: **self-review needs an adversarial second pass, not a single
sweep.** Across the eight agents used here, the reviewers and the verifiers
contradicted each other on several findings, which were then settled against
the code — and the overturned ones ran both ways: some "delete this" calls were
load-bearing, and some "this is fine" calls were not.

1. **`EXTEND_LEASE` had no `_PAYLOAD_REQUIRED` entry.** `validate_envelope`
   uses a bare subscript, so it raised `KeyError` instead of
   `IntentValidationError`. The backends catch only the latter, so the first
   `extend_lease` emitted would have been swallowed as an SDK stream failure,
   **discarding every intent already collected in that turn.** Also mirrored
   into the robustness envelope, whose contract test diffs the two enums and
   was failing.
2. **`_transition_resilient` swallowed `IllegalTransition` everywhere**,
   including `queued -> running` where that rejection *is* the double-spawn
   guard. Now opt-in per call site; only the three terminal transitions
   tolerate it.
3. **`extend_lease` restamped `updated_at`**, which forgave the elapsed time on
   top of `extra_sec` and reset the elapsed-seconds readings that both the
   health block and `get_running_tasks` derive from it.
4. **`extend_lease` did not refresh `gpu_leases`**, so the GPU reaper could
   still free cards under a live specialist — the exact failure the intent
   exists to prevent.
5. **The specialist inbox was written to the workspace** while the prompt
   advertises the worktree. Steering never reached a specialist that had one,
   which is the production case. The test passed only because its environment
   had no worktree.
6. **`get_running_tasks` reported an arbitrary lane's expiry** for a multi-lane
   task; it now reports the soonest, which is when reclaim starts.

## Scaffolding removed in the same pass

The review also stripped guards that cannot fire or that a caller already
provides: a third-layer `_rows` closure over a read already wrapped twice;
`session_dir is None` checks on a non-Optional field that is never reassigned;
a `runs_dir` guard unreachable for a literal action and a uuid task id; a
`try/except` around three lines of arithmetic; and a resource-pools guard
sitting beside a larger unguarded renderer. A bare `except Exception` in
`_handle_extend_lease` was narrowed to the two registry errors a bad `task_id`
actually produces — it was reporting infra failures back to the planner as its
own mistake. A hand-built `Lease` that only worked because `heartbeat` happened
to read two of its fields was replaced with `heartbeat_by_task`.

One correctness fix came out of this: the `Resource pools` block read
`serving_tp` from `shared_state.tp` while the pool size beside it came from
`_serving_tp_for_policy`, which also honours the `TP` env — the block could
report `serving_tp=0` next to a carve of four cards.

## Review guidance

**Read the deletions for reachability claims, not diff size.** Every removed
gate is justified above by a specific unreachable path; the useful question is
whether that path is genuinely unreachable in *your* configuration. The
Ray-vs-non-Ray split matters here: on the default single-node Ray path the
physical GPU mutex is Ray's `num_gpus` / `serving_slot`, which is
process-scoped; with `INFERENCE_OPTIMIZER_RAY_EXEC=0`, on multi-node, or under
pytest, the SQLite lanes are the only mutex and are clock-scoped.

**For the new intents, check the wiring is complete rather than the logic.**
An intent must land in `IntentType`, `_PAYLOAD_REQUIRED`, the role intent sets,
the PolicyGate dispatch, the IntentRouter dispatch table, the `emit_intent`
tool description, the `claude.py` enum, and the robustness envelope mirror.
Missing one of those is how finding 1 above shipped.

**The two-way channel is the highest-risk addition.** Its downlink writes into
a directory the specialist reads concurrently; the tmp+rename swap is
load-bearing. Note that the path had to match `worktree or workspace` to work
at all, which no test caught.

**Behavior worth a release note:** orchestration can now cancel a running task,
and the specialist prompt gained an `inbox.json` contract. Neither changes an
existing interface, but both change what an operator watching a session will
see.
