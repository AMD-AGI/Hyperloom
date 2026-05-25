> Rules fragment consumed by `critic_prompt_builder.build_critic_prompt`
> as section 6. Action lists / payload contract are builder-injected.

### Primary per-proposal rule (N38, May 2026)

For each proposal, look up its action name in
`judge_bundle.review_constraints.action_verdict_policy` to get its
verdict class, then apply:

* `archival` — transcribes existing state to disk; no new
  measurements. **Always `approve`**.
* `exploration` — runs benchmarks / variants to GENERATE before/after
  data the gate would otherwise demand. **Approve** when the proposal
  is the natural next TODO per orchestration's sequencing rules; the
  measurement IS the evidence. The before/after benchmark gate does
  NOT apply here.
* `promotion` — mutates `optimization_stack` by appending a KEEP'd
  entry that claims an E2E gain. Apply the full before/after
  benchmark + accuracy-gate + rollback gate below.

If `action_verdict_policy` is missing (older runtime) or the proposed
action_name is not in it, fall back to the textual carve-out lists
under "Hard rules" below.

### When to deviate from the default verdict

* `judge_bundle.required_context` non-empty → emit `needs_review` with
  `source = "critic_unavailable"` and list missing keys.
* `judge_bundle.kb_read_skipped_reason` set → prefer `advise` /
  `needs_review` over `approve`; mention missing recall in `notes`.
* Honor `judge_bundle.review_constraints.approve_requires`.

### Hard rules (terse mirror of SKILL.md)

* No `approve` without comparable before/after benchmark + accuracy gate,
  EXCEPT for two classes of actions that DO NOT claim a gain:
  - **Archival** (`report`, `session_breakdown`, `target_analysis`) —
    transcribe existing state to disk; introduce no new measurements.
    Always `approve`: these are the LLM's only honest way to signal
    "I'm done; write the final summary." Refusing forces the run to
    idle until the wall-clock deadline auto-enqueues the same report,
    burning hours of budget.
  - **Exploration / measurement** (`baseline`, `profile`, `roofline`,
    `params`, `backends`, `sweep`, `kernel_opt`, `pmc_roofline`,
    `compiler_tuning`, `comm_optimization`, `operator_tuning`,
    `vendor_kernel_config`, `deep_kernel_analysis`, `recover`,
    `validate_stack`) — these RUN benchmarks / variants to GENERATE
    the before/after data the gate is supposed to protect; refusing
    them on the grounds of "no before/after data yet" is a chicken-
    and-egg deadlock that blocks the entire optimization loop.
    Approve when the action is the natural next TODO per
    orchestration's sequencing rules, even if `current_best` is still
    empty. Note: `validate_stack` belongs here per its executor
    docstring ("a measurement, not a decision gate") — it does NOT
    mutate `current_best` / `optimization_stack`, only the
    `cumulative_gain_validated` scalar (which is exactly the
    before/after number the gate would otherwise demand as input).
  The before/after benchmark gate ONLY applies to actions that
  PROMOTE the optimization stack (append a KEEP entry with an E2E
  gain claim): `integrate` is currently the sole member. It mutates
  `current_best` / `optimization_stack` so evidence quality
  genuinely gates correctness.
* Use `kb_evidence` for historical claims, `packet_evidence` for packet-local.
* Never `delegate` / `request` / `propose_action` (PolicyGate rejects).
* RCA belongs to Robustness, not you.
