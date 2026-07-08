# Review Dialogue-Style Decision

Use this action when `request.kind == "critic_decision_request"`. The
caller is typically a Codex-based A2A chat server, not the Hyperloom
Coordinator.

## Step 1 — Prepare

```bash
python -m runtime.cli prepare-review \
  --request "$CRITIC_WORKDIR/request.json" \
  --out "$CRITIC_WORKDIR/judge_bundle.json"
```

The bundle contains:

- `messages` — the dialogue so far (oldest first).
- `decision` — the proposed decision summary the host wants you to
  judge.
- `merged_context` — context with session memory filled in.
- `required_context` — non-empty when critical keys are missing; the
  bundle's `kb_read_skipped_reason` will be `missing_critical_context`.
- `kb_priors_for_decision` — KB priors looked up by decision topic.

## Step 2 — Reason

Apply the rules in `SKILL.md`. Pick one verdict from
`{adopt, reject, revise, needs_info}` and write
`$CRITIC_WORKDIR/review.json`:

```json
{
  "verdict": "adopt",
  "confidence": "high",
  "reason": "Session evidence shows the change keeps tput at +4% with no accuracy regression.",
  "recommendation": "Proceed; clear compiled cache before final benchmark.",
  "basis": "mixed",
  "kb_evidence": [
    {
      "id": "kb_xxx",
      "kind": "pitfall",
      "slug": "cache-clear-required-after-dispatch-change"
    }
  ],
  "session_evidence": ["benchmark.after.gain_pct", "accuracy_gate.status"],
  "required_context": [],
  "notes": [],
  "topic": "adopt-patch-x"
}
```

When `judge_bundle.required_context` is non-empty:

- `verdict` = `needs_info`.
- `required_context` = the same list copied from the bundle.
- `basis` = `insufficient_context`.

## Step 3 — Commit

```bash
python -m runtime.cli commit-review \
  --request "$CRITIC_WORKDIR/request.json" \
  --review "$CRITIC_WORKDIR/review.json" \
  --out "$CRITIC_WORKDIR/emit.json"
```

The output is `emit.json["critic_decision_review"]`. The runtime also
writes a KB row when:

- the verdict is `adopt` / `reject` / `revise`,
- `confidence` is at least `medium`, and
- the topic is slugifiable.

KB writes are best-effort: if they fail, `emit.json["kb_writes"]` will
contain `status="dead_lettered"` and the decision review still goes
out.

## Common pitfalls

- Skipping `confidence` — defaults to `medium` but you should pick
  explicitly.
- Returning `adopt` without `session_evidence` or `kb_evidence` —
  refuses to write the supporting KB lesson.
- Bypassing `commit-review` — the dialogue caller never sees a verdict
  unless `commit-review` produced `emit.json`.
