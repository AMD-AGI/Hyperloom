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
  "notes": []
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
- `advise`: dispatch may proceed. Include non-empty `advice_text`.
- `needs_review`: dispatch must not proceed. Use for high-risk mock, timeout,
  unavailable, or insufficient-evidence cases.

Approve example:

```json
{
  "kind": "review_verdict",
  "target_proposal_msg_id": "msg_123",
  "verdict": "approve",
  "source": "critic",
  "confidence": "high",
  "reasoning": "Controlled A/B evidence shows a 4.2% tok/s/GPU gain with passing accuracy and active-path proof.",
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

## KB Draft Schema

```json
{
  "kind": "kb_draft",
  "kb_drafts": [
    {
      "category": "kernel_optimization",
      "action": "Patched the active fused attention kernel for Qwen3-14B on MI355X.",
      "lesson": "The fused attention rewrite produced an E2E gain only when the dispatch path and shape-specific tuning config were updated together.",
      "model_family": "qwen",
      "model": "Qwen3-14B",
      "gpu": "MI355X",
      "framework": "SGLang",
      "tags": [
        "attention",
        "dispatch",
        "best_config"
      ],
      "result": {
        "status": "KEEP",
        "gain_pct": 4.2,
        "baseline_tput_per_gpu": 1200.0,
        "final_tput_per_gpu": 1250.4,
        "accuracy": "pass"
      },
      "confidence": 0.9,
      "source": "final_report",
      "context": "Controlled A/B with matched launch parameters and passing accuracy gate.",
      "supersedes": null,
      "validated_date": "",
      "strategy_tested": [
        "kernel_rewrite",
        "dispatch_update",
        "shape_config_update"
      ]
    }
  ],
  "rejected_candidates": [
    {
      "source_section": "Backend exploration",
      "reason": "No controlled benchmark evidence was reported."
    }
  ],
  "notes": []
}
```

Allowed values:

- `kind`: `kb_draft`
- `kb_drafts[].category`: one of the categories in `actions/draft_kb.md`
- `kb_drafts[].confidence`: number from `0.0` to `1.0`

## Combined Request

If the caller requests both outputs in one response, return:

```json
{
  "kind": "critic_response",
  "review_verdict": {
    "kind": "review_verdict",
    "target_proposal_msg_id": "msg_123",
    "verdict": "approve",
    "source": "critic",
    "confidence": "high",
    "reasoning": "Approved.",
    "predicted_gain_pct": 2.5,
    "kb_evidence": [],
    "packet_evidence": [],
    "risks": [],
    "required_evidence": [],
    "alternative_action": null,
    "advice_text": "",
    "notes": []
  },
  "kb_draft": {
    "kind": "kb_draft",
    "kb_drafts": [],
    "rejected_candidates": [],
    "notes": []
  }
}
```
