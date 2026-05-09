# Coordinator Inbox Protocol

The Coordinator (`inference_optimizer.orchestrator.coordinator.Coordinator`)
hands every reactor a prompt with the layout below. Critic must consume
this format (or the equivalent payload that an A2A host preserves
verbatim into `request.raw_prompt`).

```text
=== Shared session state ===
session_id=<sess_id>   model=<model>   framework=<framework>   ...

=== Knowledge base hints ===          <-- only for orchestration role
<free text bullet list>

=== Inbox for critic (newest last) ===
  seq=<int> msg_id=<hex> from=<agent> topic=<topic> payload=<python repr dict>
  ...
```

The runtime parser (`runtime.inbox_parser.parse_inbox_prompt`) returns:

* `agent_name` — should be `critic` for our turns.
* `shared_state` — `key=value` tokens from the header.
* `inbox` — every parsed row.
* `proposals` — `topic=proposal` rows whose payload parsed as a dict.
* `kb_hints_text` — preserved (we do not read it for Critic turns).

## Per-tick contract (matches `MockCriticBackend`)

For every `topic=proposal` row in the inbox the Critic must emit one
`review_verdict` intent. If no proposal is present, the runtime emits a
`send_message{topic=heartbeat}` so the Coordinator never times out on an
empty envelope.

The intent envelope shape is:

```json
{
  "intents": [
    {
      "intent_type": "review_verdict",
      "payload": {
        "target_proposal_msg_id": "<hex>",
        "verdict": "approve|reject|redirect|advise|needs_review",
        "source": "critic",
        "reasoning": "...",
        "kb_evidence": [],
        "packet_evidence": [],
        "risks": [],
        "predicted_gain_pct": null,
        "confidence": "medium"
      }
    }
  ]
}
```

## Allowed values

| Field | Values | Source |
|---|---|---|
| `intent_type` (Critic) | `review_verdict`, `send_message`, `update_persona`, `ask_question`, `answer`, `alert` | `inference_optimizer.orchestrator.agent_role._CRITIC_INTENTS` |
| `verdict` | `approve`, `reject`, `redirect`, `advise`, `needs_review` | `policy.REVIEW_VERDICTS` |
| `source` | `critic`, `mock`, `timeout`, `critic_unavailable` | `references/verdict_schema.md` |

## Source mapping

| Critic situation | `source` |
|---|---|
| Normal LLM-driven verdict | `critic` |
| Runtime returned heartbeat or `needs_review` due to missing context | `critic_unavailable` |
| Forced timeout fallback | `timeout` |
| Mock backend (tests / dry-run) | `mock` |

## Idempotency

The runtime keeps a per-session `reviewed_msg_ids.json` and filters
already-decided proposals out of the prepare-review bundle. Do not
re-emit a verdict for the same `target_proposal_msg_id`; use a
`send_message{topic=advice}` instead if you need to amend a prior call.

## Common pitfalls

* Do not include `intent_envelope` outside the JSON object — Codex's
  `validated_json_output` must be exactly the envelope shape.
* Do not synthesise extra `intent_type` values; PolicyGate denies
  anything outside the Critic intent set.
* Do not rewrite `payload.target_proposal_msg_id` — the Coordinator
  matches it against `pending_proposals` literally.
