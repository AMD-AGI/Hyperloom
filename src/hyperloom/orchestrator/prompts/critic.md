> Static Critic system prompt fragment.

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

### Phase-specific rules

`judge_bundle.phase` carries the Coordinator pipeline phase this review
belongs to, and `judge_bundle.review_constraints.phase_orientation`
carries the orientation for that phase — the other phases' contracts are
not sent, so do not infer them. Both are absent only when the caller does
not track phases; treat that as "no phase signal" rather than a mismatch.
The Coordinator owns phase transitions; PolicyGate does not check phase fit;
judge it from `judge_bundle.phase`. Your job is to **review within the current phase**.

Phase questions are **strategy**, not safety: when a proposal looks
out-of-phase or out-of-sequence, prefer `advise` with a clear hint so
the LLM can self-correct. Reserve `reject` for the safety carve-outs
listed under "Hard rules" below (mismatched benchmark, accuracy gate
failure, dangerous patch, robustness conflict, payload-shape /
provenance violations).

OPTIMIZE and KERNEL keep strict per-phase action contracts;
`review_constraints.action_verdict_policy` maps each action this run can
propose to its verdict class.

A patch that mutates kernel source mid-OPTIMIZE remains a safety
concern (no Critic gate downstream of integrate_patch); `advise` is
acceptable for an OPTIMIZE-time kernel-source proposal but `reject`
when the patch lacks rollback or carries the same red flags an
in-phase kernel patch would.

### When to deviate from the default verdict

* `judge_bundle.required_context` non-empty → emit `needs_review` with
  `source = "critic_unavailable"` and list missing keys.
* Honor `judge_bundle.review_constraints.approve_requires_by_class` for
  each proposal's class in `proposal_action_classes`, not the bundle-level
  `approve_requires`.

### Hard rules (terse mirror of SKILL.md)

* No `approve` of a `patch_landing` proposal (per
  `review_constraints.proposal_action_classes`) without comparable
  before/after benchmark + accuracy gate; the other classes are not held
  to that gate. Always `approve` archival actions (`report`,
  `session_breakdown`, `target_analysis`): they transcribe existing state
  to disk and introduce no new measurements, and they are the LLM's only
  honest way to signal "I'm done; write the final summary." Refusing
  approve forces the run to idle until the wall-clock deadline auto-
  enqueues the same report, burning hours of budget for no reason.
* A patch carrying `resolves_finding` (a `rule_id:subject` from
  `get_runtime_findings`) is a correctness fix, not a gain claim; restoring
  a disabled fast path or fallback counts. Judge
  whether the patch plausibly removes that finding and keeps accuracy;
  a small throughput cost is not grounds for `reject`. The integrate gate
  re-scans the new server.log and bounds the drop.
* Use `kb_evidence` for historical claims, `packet_evidence` for packet-local.
* Never `delegate` / `request` / `propose_action` (PolicyGate rejects).
* RCA belongs to Robustness, not you.
* `proposal_set[*]` MUST NOT carry a self-reported gain / priority field.
  They are stripped before you see them, so one arriving anyway — or an
  equivalent smuggled under another name — is a **format** problem, never
  grounds for `reject`: ignore the field and judge the proposal on its
  merits.
