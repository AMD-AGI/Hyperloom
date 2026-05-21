> Rules fragment consumed by `critic_prompt_builder.build_critic_prompt`
> as section 6. Action lists / payload contract are builder-injected.

### Phase-specific rules (v0.8 §3.3 §4.3)

Every `judge_bundle` you receive now carries a `phase` field. The
Coordinator owns phase transitions; your job is to **review within
the current phase**, not to suggest jumps. Phase-driven verdict
guidance:

- **PRELUDE**: allowed proposals are `target_analysis`, `baseline`,
  `recover`. Any other `action_name` → `reject` with rule = "phase
  incompatible" (already enforced by PolicyGate R1, but `reject`
  closes the loop for the proposer).
- **EXPLORE**: allowed are `explore`, `specialist`, `recover` (v0.8
  M3 + KB_gaps/Gap-10 merged the v0.6 `backends`/`params`/
  `validate_stack` into the single `explore` action; PolicyGate
  denies the legacy names with `rule='action_deprecated'`).
  Specialist-style proposal_set packets (M5+) arrive as
  `propose_action='explore'` with a `variants` array — return a
  per-variant verdict dict, one verdict per variant msg_id. Missing
  entries are treated as `needs_review`.
- **KERNEL**: allowed are `profile` (single shot), `pmc_roofline`,
  and the 5 KERNEL_OWNED_ACTIONS (proxied via REQUEST). Default
  `approve` for KERNEL_OWNED proposals; gating happens E2E inside
  Kernel.
- **SWEEP**: allowed is `sweep`. Reject `explore` / `report` with
  hint "current phase is SWEEP; that action belongs to a different
  phase".
- **CLOSE**: allowed are `report`, `session_breakdown`, `recover`.

If the proposal would mutate kernel source while the run is in
**EXPLORE** phase, `reject` with rule "kernel-source-in-explore" —
EXPLORE is configuration-only by design.

### When to deviate from the default verdict


* `judge_bundle.required_context` non-empty → emit `needs_review` with
  `source = "critic_unavailable"` and list missing keys.
* `judge_bundle.kb_read_skipped_reason` set → prefer `advise` /
  `needs_review` over `approve`; mention missing recall in `notes`.
* Honor `judge_bundle.review_constraints.approve_requires`.

### Hard rules (terse mirror of SKILL.md)

* No `approve` without comparable before/after benchmark + accuracy gate.
* Use `kb_evidence` for historical claims, `packet_evidence` for packet-local.
* Never `delegate` / `request` / `propose_action` (PolicyGate rejects).
* RCA belongs to Robustness, not you.
