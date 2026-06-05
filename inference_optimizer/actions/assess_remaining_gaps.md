# assess_remaining_gaps — session_steward dispatch

Dispatches the ``session_steward_specialist`` domain to read the full
session state and return an exit recommendation. The steward is
**advisory only** — its verdict is recorded into
``last_remaining_gaps_assessment`` (and surfaced as a second opinion
in the orchestration prompt) but never drives phase transitions on
its own. Phase advance is driven by IR-6 force-exit, phase-budget
exhaustion, or an explicit ``escalate_strategy_change`` /
``skip_to_*`` hint from robustness or the LLM.

The Coordinator auto-enqueues a steward run the moment the EXPLORE
plateau judge fires. Propose it manually when you want a second
opinion sooner than the plateau judge would fire — there is no
phase-precondition or back-to-back throttle on the LLM-proposed path
(those gates were removed in loosen plan P2_13).

## Output protocol

The steward emits a single ``specialist_done`` intent whose payload
contains, in addition to the standard fields:

- ``recommendation`` ∈ ``{continue_explore, advance_to_kernel, stop_session}``
- ``next_gap_canonical_id`` (required iff ``recommendation='continue_explore'``)
- ``remaining_potential_pct_estimate`` (float)
- ``rationale`` (≤ 2000 chars; quoted verbatim in the final report)

The Coordinator records the verdict as advisory:

- ``stop_session`` / ``advance_to_kernel`` → recorded in
  ``last_remaining_gaps_assessment``; **no** ``pending_escalate_hint``
  is set.
- ``continue_explore`` → appends ``next_gap_canonical_id`` to
  ``gaps[]``, resets ``params_no_promote_streak`` as a neutral aid
  for the next round, and flips ``steward_continuation_used`` as an
  audit marker.

Out-of-vocab strings coerce to ``stop_session`` so the audit row
carries a known value.

## Iron rules

- IR-6 still governs the wall-clock force-exit; the steward never
  argues with it.
- The LLM may propose this action only with ``params.reason``
  populated; missing reason → PolicyGate denial.
