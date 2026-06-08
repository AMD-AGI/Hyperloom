# Intent Envelope Schema (Critic side)

The Coordinator (`inference_optimizer.protocol.intent`)
expects the following object — produced by the Critic for every turn:

```json
{
  "intents": [
    {
      "intent_type": "review_verdict",
      "payload": {
        "target_proposal_msg_id": "abc123",
        "verdict": "approve",
        "source": "critic",
        "reasoning": "...",
        "kb_evidence": ["kb_xxx"],
        "packet_evidence": ["benchmark.after.gain_pct"],
        "risks": [],
        "predicted_gain_pct": 4.2,
        "confidence": "high",
        "advice_text": "",
        "alternative_action": null,
        "required_evidence": [],
        "notes": []
      }
    }
  ]
}
```

The Critic runtime never produces this JSON itself — it builds it via
`runtime.intent_envelope.build_envelope(...)` after `commit-review`.

## Allowed intent types (Critic role)

| `intent_type` | Required payload fields |
|---|---|
| `review_verdict` | `target_proposal_msg_id`, `verdict` |
| `send_message` | `topic` |
| `ask_question` | `topic`, `question` |
| `answer` | `in_reply_to`, `answer` |
| `alert` | `severity`, `summary` |
| `update_persona` | `body_md` |

## Validation

The runtime's `runtime.intent_envelope.validate_envelope(...)` mirrors
the Coordinator's PolicyGate checks. If the SKILL produces a malformed
review, `commit-review` raises `ReviewValidationError` and the host
should fall back to a `needs_review` heartbeat (see
[references/coordinator_protocol.md](coordinator_protocol.md)).

## Heartbeat fallback

When `commit-review` produces zero intents, the runtime appends a single
`send_message{topic="heartbeat"}` so the Coordinator's reactor pass
always observes signal of life — exactly the behaviour the mock
adapter ships with.
