# Knowledge Base — schema + curation

## Layout

```
<session_dir>/kb/
├── entries.jsonl        # individual lessons (per action / model)
├── insights.jsonl       # marathon Sage 6h synthesis records
├── conflicts.jsonl      # auto-detected keep-vs-revert collisions
└── SCHEMA.md            # this file
```

Each line in `entries.jsonl` / `insights.jsonl` is **one** JSON record;
no multi-line objects.

## `entries.jsonl` schema

```json
{
  "category": "model_class_lesson | kernel_opt_lesson | process_mgmt_trap | …",
  "user_id": "default",
  "model": "deepseek-ai/DeepSeek-V3-0324",
  "model_family": "deepseek",
  "action": "backends",
  "lesson": "vllm beat sglang by 9% for MLA models",
  "tags": ["mla", "moe", "backend_choice"],
  "gain": 9.0,
  "status": "keep | revert | fail | observation",
  "ts": 1719450000.0
}
```

Cold-start gating — DESIGN §6.2 ADR-21:

- The first time a session sees a `model_family`, the conductor only
  *writes*; it does not *read*.
- `KnowledgeBase.is_warm_start_eligible(family)` returns ``True`` once
  ≥1 entry exists for that family under the same `user_id`.

## `insights.jsonl` schema (marathon-only)

```json
{
  "kind": "cross_run_synthesis",
  "ts": 1719450000.0,
  "samples": 134,
  "by_family": {
    "llama":     { "count": 45, "kept_count": 18, "mean_gain": 6.4 },
    "deepseek":  { "count": 60, "kept_count": 22, "mean_gain": 5.2 },
    "mixtral":   { "count": 29, "kept_count": 11, "mean_gain": 4.8 }
  }
}
```

## `conflicts.jsonl` schema

```json
{
  "kind": "kb_conflict",
  "ts": 1719450000.0,
  "reason": "deepseek/backends: keep@gain=9 vs revert@gain=0",
  "entry_a": { ... entries.jsonl row ... },
  "entry_b": { ... entries.jsonl row ... }
}
```

## CLI

Append a lesson:

```bash
python -m inference_optimizer.kb.kb_ingest \
  --kb-dir <session>/kb \
  --category model_class_lesson \
  --model deepseek-V3 \
  --action backends \
  --lesson "vllm beat sglang for MLA" \
  --tags '["mla","moe"]' \
  --gain 9.0 \
  --status keep
```

Recall top-k for a query:

```bash
python -m inference_optimizer.kb.kb_query \
  "MLA backend" \
  --kb-dir <session>/kb \
  --top-k 5 \
  --compact
```

Supported flags: `--compact`, `--json`, `--top-k N`.

## Recommended categories

- `model_class_lesson` — model-class specific tips ("kv-cache fp8 hurts
  Mixtral by 3‑8%")
- `kernel_opt_lesson` — kernel optimisation outcomes
- `process_mgmt_trap` — operational gotchas (e.g. "always unset PROFILE
  before bench")
- `objective_calibration` — observed gain ranges per action / family
- `synthesised_insight` — Sage cross-run summary record (only in
  `insights.jsonl`)
