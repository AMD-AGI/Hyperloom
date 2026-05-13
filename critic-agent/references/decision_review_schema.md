# Decision Review Schema (dialogue-driven hosts)

For non-Coordinator hosts (e.g. an A2A chat server fronting Codex), the
Critic accepts a richer dialogue-style request. The first call carries
the full context; later calls may be incremental.

## Request: `critic_decision_request`

```json
{
  "kind": "critic_decision_request",
  "session_id": "sess_123",
  "decision_id": "dec_456",
  "messages": [
    {
      "role": "coordinator",
      "content": "We plan to adopt patch X because it improves throughput by 3%."
    },
    {
      "role": "kernel-agent",
      "content": "Patch X changes the dispatch path and requires cache clear."
    }
  ],
  "context": {
    "model": "deepseek-r1-0528-fp8",
    "framework": "sglang",
    "precision": "fp8",
    "workload": "decode",
    "scale": "8xMI300",
    "objective": "throughput"
  },
  "decision": {
    "summary": "Adopt patch X",
    "owner": "kernel-agent",
    "target": "dispatch path optimization"
  }
}
```

`context` is only required on the first call. Later calls may omit it;
the runtime fills missing fields from session memory.

## Response: `critic_decision_review`

```json
{
  "kind": "critic_decision_review",
  "session_id": "sess_123",
  "decision_id": "dec_456",
  "verdict": "adopt",
  "confidence": "high",
  "reason": "Session evidence supports the patch and no KB pitfall contradicts it.",
  "recommendation": "Proceed, but rerun the final benchmark after clearing compiled cache.",
  "basis": "mixed",
  "kb_evidence": [
    {
      "id": "kb_xxx",
      "kind": "pitfall",
      "slug": "cache-clear-required-after-dispatch-change",
      "summary": "Stale compiled cache hides regressions."
    }
  ],
  "session_evidence": ["benchmark.after.gain_pct", "accuracy_gate.status"],
  "required_context": [],
  "notes": []
}
```

### Allowed values

| Field | Values |
|---|---|
| `verdict` | `adopt`, `reject`, `revise`, `needs_info` |
| `confidence` | `high`, `medium`, `low` |
| `basis` | `kb`, `llm`, `mixed`, `session`, `insufficient_context` |

### Verdict semantics

| Verdict | Meaning |
|---|---|
| `adopt` | Proceed with the decision. |
| `reject` | Do not adopt. Provide a `reason` and either `kb_evidence` or `session_evidence`. |
| `revise` | Direction is sound but needs adjustments before adoption. |
| `needs_info` | Cannot decide because critical context is missing. Always populate `required_context`. |

## When KB writes happen

`commit-review` translates the dialogue verdict into a KB write trigger
when the lesson is reusable:

| Dialogue verdict | KB trigger | KB kind |
|---|---|---|
| `adopt` | `critic_verdict_approve` | `technique` |
| `reject` | `critic_verdict_reject` | `pitfall` |
| `revise` | `critic_verdict_redirect` | `pitfall` |
| `needs_info` | `critic_verdict_needs_review` | `pitfall` (informational) |

Writes are skipped silently when:

- `KB_WRITE_ENABLED=false`,
- `model` or `framework` is unknown,
- the verdict has no slugifiable `topic`, or
- `confidence == "low"` and there is no measurement evidence.
