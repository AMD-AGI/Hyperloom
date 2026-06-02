# Review Coordinator Inbox

Use this action when `request.kind == "coordinator_inbox"`. The host has
forwarded the Coordinator's per-turn prompt verbatim (see
[references/coordinator_protocol.md](../references/coordinator_protocol.md)).

## Step 1 — Prepare

```bash
python -m runtime.cli prepare-review \
  --request "$CRITIC_WORKDIR/request.json" \
  --out "$CRITIC_WORKDIR/judge_bundle.json"
```

`judge_bundle.json` will contain:

- `merged_context` — model / framework / precision / workload / etc.
  combined with session memory.
- `missing_context` — keys that fell back to `unknown` (informational).
- `required_context` — non-empty when critical keys (`model`,
  `framework`) are still missing; if so, see Step 2.
- `proposals` — only proposals not yet reviewed in this session.
- `kb_priors_by_proposal` — array per `msg_id` (cache-aware).
- `review_constraints` — allowed verdicts, action-class-scoped approve
  checklists (`approve_requires_by_class`), and a `{msg_id: class}`
  map in `proposal_action_classes` so you can apply the right per-
  proposal bar in a heterogeneous batch.

## Step 2 — Reason

For each proposal in `proposals`:

1. Look up its class via
   `review_constraints.proposal_action_classes[<msg_id>]` →
   `patch_landing` / `evidence_producer` / `framework_op`.
2. Apply the **per-class** Approve Standard from `SKILL.md`. The
   bundle-level `review_constraints.approve_requires` is the strictest
   class present in the batch (a fallback for callers that do not look
   at `proposal_action_classes`); always prefer the per-proposal class
   when emitting verdicts.
3. Cross-reference with
   [references/risk_rules.md](../references/risk_rules.md) for
   blocker / major / minor categorisation,
   [references/verdict_schema.md](../references/verdict_schema.md) for
   per-verdict required fields, and any
   `kb_priors_by_proposal[<msg_id>]` returned in Step 1.

Default behavior summary:

- `evidence_producer` proposal + empty / silent KB priors → **approve**
  (the action's whole purpose is to produce the missing benchmarks).
- `evidence_producer` proposal + a contradicting KB prior (e.g. variant
  tried 3x and failed) → `reject` with the prior cited in `kb_evidence`.
- `patch_landing` proposal without comparable benchmark + accuracy
  gate → `needs_review` (or `reject` if the packet itself shows a
  regression).
- `framework_op` proposal → approve by default; only block when
  structurally malformed.

Special cases:

- `judge_bundle.required_context` is non-empty → emit `needs_review`
  for every proposal with `source = "critic_unavailable"` and list the
  missing keys in `notes`.
- `proposals` is empty → emit nothing; the runtime will fall back to a
  heartbeat in Step 3.

Write the result to `$CRITIC_WORKDIR/review.json`:

```json
{
  "review_verdicts": [
    {
      "target_proposal_msg_id": "abc1",
      "verdict": "approve",
      "source": "critic",
      "reasoning": "Active dispatch path proven; benchmark within tolerance; rollback documented.",
      "confidence": "high",
      "predicted_gain_pct": 4.2,
      "kb_evidence": ["kb_xxx"],
      "packet_evidence": ["benchmark.after.gain_pct", "accuracy_gate.status"],
      "risks": [],
      "required_evidence": [],
      "notes": [],
      "persist_to_kb": true,
      "topic": "kernel-opt-active-dispatch"
    }
  ],
  "advice": [
    {
      "target_proposal_msg_id": "abc1",
      "body_md": "Re-run the sweep at higher concurrency before promotion."
    }
  ]
}
```

`persist_to_kb` (optional) flags a verdict whose lesson is reusable and
should be upserted into KB. `topic` should be the slug-friendly handle
(the runtime calls `slugify_safe`).

## Step 3 — Commit

```bash
python -m runtime.cli commit-review \
  --request "$CRITIC_WORKDIR/request.json" \
  --review "$CRITIC_WORKDIR/review.json" \
  --out "$CRITIC_WORKDIR/emit.json"
```

`emit.json["intent_envelope"]` is the canonical reply for the host.

## Failure modes

- **Schema validation error** — runtime exits with code 2; the host
  should treat it as a Critic outage and fall back to a heartbeat
  envelope.
- **KB write dead-lettered** — `kb_writes[*].result.status` is
  `dead_lettered`. The verdict is still emitted; cron will replay later.
- **Missing critical context** — produce `needs_review` verdicts and
  set `source = "critic_unavailable"`.
