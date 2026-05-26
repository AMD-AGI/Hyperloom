# assess_remaining_gaps — session_steward dispatch

IR-7 (honest self-stop). Dispatches the
``session_steward_specialist`` domain to read the full session state
and return an exit recommendation. **You normally do NOT need to
propose this action** — the Coordinator auto-enqueues it the moment
the EXPLORE plateau judge fires. Propose it manually only when you
believe EXPLORE is exhausted before the plateau judge says so (e.g.
five consecutive REVERTs against the same root cause but the
``params_no_promote_streak`` hasn't yet hit its threshold).

## When to propose (LLM)

All of the following must hold before you propose:

- ``phase == 'EXPLORE'``
- ``len(optimization_stack) >= 3`` (we have *something* to assess)
- Either of:
  - You see 3+ consecutive REVERTs in
    ``explore_search.rejected[-3:]`` against the same root cause /
    domain, **and** the Coordinator's plateau judge has NOT triggered
    yet (otherwise it would have enqueued the steward internally).
  - You see 5+ consecutive ``stack_unstable`` rows where the
    underlying root cause is unclear — at that point the variants
    you're proposing are genuinely lossy and another round is
    unlikely to land KEEPs.

Throttle: PolicyGate ``assess_remaining_gaps_throttle`` denies the
delegate when ``last_remaining_gaps_assessment.ts`` is fresher than
``INFERENCE_OPTIMIZER_ASSESSMENT_MIN_INTERVAL_SEC`` (default 1800s).
Coordinator-internal enqueues bypass this rule.

## Output protocol

The steward emits a single ``specialist_done`` intent whose payload
contains, in addition to the standard fields:

- ``recommendation`` ∈ ``{continue_explore, advance_to_kernel, stop_session}``
- ``next_gap_canonical_id`` (required iff ``recommendation='continue_explore'``)
- ``remaining_potential_pct_estimate`` (float)
- ``rationale`` (≤ 2000 chars; quoted verbatim in the final report)

The Coordinator routes:

- ``stop_session`` → ``stop_reason='no_more_leverage'`` (immediate
  CLOSE on the next tick)
- ``advance_to_kernel`` → ``pending_escalate_hint='skip_to_kernel'``
  (next ``compute_next_phase`` advances to KERNEL or SWEEP per
  ``kernel_enabled``)
- ``continue_explore`` → appends ``next_gap_canonical_id`` to
  ``gaps[]``, resets ``params_no_promote_streak``, sets
  ``steward_continuation_used=True``. Only one continuation per
  session; a second invocation that returns ``continue_explore`` is
  coerced to ``advance_to_kernel``.

## Iron rules

- IR-7: the steward gates the EXPLORE→KERNEL transition softly; the
  HARD IR-6 force-exit still wins when wall-clock budget drops below
  the threshold, regardless of any steward verdict.
- The LLM may propose this action only with ``params.reason``
  populated; missing reason → PolicyGate denial.
