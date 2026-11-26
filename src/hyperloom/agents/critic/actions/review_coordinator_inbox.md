# Review Coordinator Inbox

Use this action for every Critic turn.

The judge bundle contains:

- `context` — model / framework / precision / workload / etc.
- `required_context` — non-empty when critical keys (`model`,
  `framework`) are still missing; if so, see *Special cases* below.
- `proposals` — only proposals not yet reviewed in this session.
- `review_constraints` — allowed verdicts, action-class-scoped approve
  checklists (`approve_requires_by_class`), and a `{msg_id: class}`
  map in `proposal_action_classes` so you can apply the right per-
  proposal bar in a heterogeneous batch.

## Reason

For each proposal in `proposals`:

1. Look up its class via
   `review_constraints.proposal_action_classes[<msg_id>]` →
   `patch_landing` / `enablement_landing` / `evidence_producer` /
   `framework_op`.
2. Apply the **per-class** Approve Standard from `SKILL.md`. The
   bundle-level `review_constraints.approve_requires` is the strictest
   class present in the batch (a fallback for callers that do not look
   at `proposal_action_classes`); always prefer the per-proposal class
   when emitting verdicts.
3. Grade risks and confidence as `SKILL.md` describes: the blocker and
   major lists under the `patch_landing` Approve Standard, and *Severity
   and Confidence*.

Default behavior summary:

- `evidence_producer` proposal + empty / silent KB priors → **approve**
  (the action's whole purpose is to produce the missing benchmarks).
- `evidence_producer` proposal + a contradicting KB prior (e.g. variant
  tried 3x and failed) → `reject` with the prior cited in `kb_evidence`.
- `patch_landing` proposal without comparable benchmark + accuracy
  gate → `needs_review` (or `reject` if the packet itself shows a
  regression).
- `enablement_landing` proposal (enablement `integrate_patch`) →
  **approve** on the structural bar; do NOT block on
  a missing throughput before/after or a restated rollback plan (rollback is
  automatic). Boot-origin has no baseline yet; eval-origin booted but missed the
  accuracy floor. Either way the downstream runnable-decision gate REVERTs any
  patch that fails to boot, and for eval-origin additionally re-runs the accuracy
  eval and REVERTs a patch that still misses the floor.
- `framework_op` proposal → approve by default; only block when
  structurally malformed.

Special cases:

- `judge_bundle.required_context` is non-empty → emit `needs_review`
  for every proposal with `source = "critic_unavailable"` and list the
  missing keys in `notes`.
- `proposals` is empty → emit nothing; the runtime will fall back to an
  idle intent.

The concrete `advise` text may be written either inline on
`review_verdicts[].advice_text` or as a separate `advice[].body_md`
entry targeting the same `target_proposal_msg_id` — both are accepted.
At commit time the runtime merges any matching `advice[]` bodies into
the verdict's `advice_text` (the verdict's own `advice_text` first, then
the `advice[]` bodies, joined with a blank line), so the
`review_verdict` payload always carries the full advisory. The separate
`advice[]` entries are still broadcast to the Coordinator inbox; the two
are complementary, not mutually exclusive.
