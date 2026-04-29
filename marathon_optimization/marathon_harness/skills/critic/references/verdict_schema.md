# Verdict Schema

Critic returns structured JSON. Do not wrap the response in markdown unless the
caller explicitly asks for markdown.

## Patch Vote Schema

```json
{
  "kind": "patch_vote",
  "approval": false,
  "confidence": "high",
  "summary": "The patch cannot be approved because benchmark parameters do not match the baseline.",
  "objections": [
    {
      "type": "benchmark_invalid",
      "severity": "blocker",
      "reason": "After-change benchmark omitted max_concurrency, so it is not comparable with baseline.",
      "required_fix": "Rerun before/after benchmarks with matched launch parameters, concurrency, ISL/OSL, warmup, and sample count.",
      "evidence_ref": "benchmark.after"
    }
  ],
  "required_evidence": [
    "matched_benchmark",
    "accuracy_gate"
  ],
  "warnings": [],
  "notes": []
}
```

Allowed values:

- `kind`: `patch_vote`
- `approval`: `true` or `false`
- `confidence`: `high`, `medium`, or `low`
- `objections[].severity`: `blocker`, `major`, or `minor`

Approval example:

```json
{
  "kind": "patch_vote",
  "approval": true,
  "confidence": "high",
  "summary": "The patch is approved: controlled benchmark shows a 4.2% tok/s/GPU gain with passing accuracy and a clear rollback path.",
  "objections": [],
  "required_evidence": [],
  "warnings": [
    {
      "type": "followup_sweep",
      "severity": "minor",
      "reason": "The final sweep covers the target concurrency but not the next lower operating point.",
      "required_fix": "Optional: include the lower concurrency point in the next run.",
      "evidence_ref": "sweep.summary"
    }
  ],
  "notes": [
    "Active dispatch path is proven by profile evidence."
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
  "patch_vote": {
    "kind": "patch_vote",
    "approval": true,
    "confidence": "high",
    "summary": "Approved.",
    "objections": [],
    "required_evidence": [],
    "warnings": [],
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
