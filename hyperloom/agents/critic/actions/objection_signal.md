# Objection Signal (Devil's Advocate)

Use this action when `request.kind == "objection_signal"`. The host
wants non-blocking advice about a decision that is already underway —
not a fresh verdict.

## Step 1 — Prepare

```bash
python -m runtime.cli prepare-review \
  --request "$CRITIC_WORKDIR/request.json" \
  --out "$CRITIC_WORKDIR/judge_bundle.json"
```

The bundle is identical to the dialogue-style one. Use
`kb_priors_for_decision` and `messages` to look for missed risks or
historical contradictions.

## Step 2 — Reason

Compose a short markdown advice body. Keep it under ~600 tokens; the
host injects this into the next prompt for Orchestration.

Write `$CRITIC_WORKDIR/review.json`:

```json
{
  "verdict": "advise",
  "advice": [
    {
      "target_proposal_msg_id": "abc1",
      "body_md": "Heads-up: kb_xxx records that this kernel rewrite needs an explicit cache clear; rerun the final benchmark after the patch."
    }
  ]
}
```

When in doubt, return one `send_message{topic="advice"}` plus
`verdict = "advise"`. The runtime will package it into the intent
envelope without blocking dispatch.

## Step 3 — Commit

```bash
python -m runtime.cli commit-review \
  --request "$CRITIC_WORKDIR/request.json" \
  --review "$CRITIC_WORKDIR/review.json" \
  --out "$CRITIC_WORKDIR/emit.json"
```

The host should treat the resulting envelope as informational only.
