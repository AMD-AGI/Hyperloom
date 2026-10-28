# Verdict Schema

Critic returns structured JSON. Do not wrap the response in markdown unless the
caller explicitly asks for markdown.

## Review Verdict Schema

```json
{
  "kind": "review_verdict",
  "target_proposal_msg_id": "msg_123",
  "verdict": "reject",
  "source": "critic",
  "confidence": "high",
  "reasoning": "The after-run changed max_concurrency, so the reported gain is not comparable with the baseline.",
  "predicted_gain_pct": null,
  "kb_evidence": [],
  "packet_evidence": [
    "benchmark.baseline.max_concurrency",
    "benchmark.after.max_concurrency"
  ],
  "risks": [
    {
      "type": "benchmark_invalid",
      "severity": "blocker",
      "reason": "Baseline used max_concurrency=128 while the after run used max_concurrency=256.",
      "required_fix": "Rerun before and after benchmarks with matched concurrency, ISL/OSL, sample count, and launch parameters.",
      "evidence_ref": "benchmark.after.max_concurrency"
    }
  ],
  "required_evidence": [
    "matched_benchmark"
  ],
  "alternative_action": null,
  "advice_text": "",
  "notes": [],
  "failure_reason_code": ""
}
```

Allowed values:

- `kind`: `review_verdict`
- `verdict`: `approve`, `reject`, `redirect`, `advise`, or `needs_review`
- `source`: `critic`, `mock`, `timeout`, or `critic_unavailable`
- `confidence`: `high`, `medium`, or `low`
- `risks[].severity`: `blocker`, `major`, or `minor`

Verdict rules:

- `approve`: dispatch may proceed. Include `predicted_gain_pct` when the
  proposal claims performance impact.
- `reject`: dispatch must not proceed. Include `kb_evidence` for historical or
  policy-backed rejection and `packet_evidence` for packet-local facts.
- `redirect`: include `alternative_action` with a registered action owned by the
  same agent family.
- `advise`: dispatch may proceed. Include non-empty `advice_text`. The
  text may also be supplied via a matching `advice[]` entry in a
  `coordinator_inbox` review; the commit adapter merges any `advice[]`
  bodies for the same `target_proposal_msg_id` into this `advice_text`
  (verdict's own text first, then the `advice[]` bodies, joined with a
  blank line).
- `needs_review`: dispatch must not proceed. Use for high-risk mock, timeout,
  unavailable, or insufficient-evidence cases.

`failure_reason_code` names the review rule a non-`approve` verdict rests on,
copied verbatim from the `failure_reason_code` of the matching rule in
`judge_bundle.review_constraints` (the quantitative-claim rule, a cross-domain
rule, or a safety guard). Leave it empty when the verdict rests on your own
judgement rather than a rule handed to you in the bundle. Several of those
rules declare `advise` as their `failure_verdict` because rejecting on them
costs the round every proposal in the set; naming the rule is how the
Coordinator can tell such a verdict apart from a substantive rejection, so a
verdict that cites a rule must carry its code rather than only mentioning it in
`reasoning`.

Approve example:

```json
{
  "kind": "review_verdict",
  "target_proposal_msg_id": "msg_123",
  "verdict": "approve",
  "source": "critic",
  "confidence": "high",
  "reasoning": "Controlled A/B evidence shows a 4.2% tok/s gain with passing accuracy and active-path proof.",
  "predicted_gain_pct": 4.2,
  "kb_evidence": [
    "kb:qwen3/attn-dispatch-best-config"
  ],
  "packet_evidence": [
    "benchmark.after.gain_pct",
    "accuracy_gate.status",
    "dispatch_evidence.active_path_proven"
  ],
  "risks": [],
  "required_evidence": [],
  "alternative_action": null,
  "advice_text": "",
  "notes": [
    "Rollback is clear."
  ]
}
```
