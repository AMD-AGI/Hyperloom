# KB JSONL schemas

> See IMPLEMENTATION-CHECKLIST.md Phase 5 §5.31‒5.32.

## `entries.jsonl` — per-action lesson

One JSON object per line:

```json
{
  "ts": "2026-04-27T15:42:00Z",
  "user_id": "default",
  "session_id": "abc123def456",
  "category": "model_class_lesson|kernel_opt_lesson|process_mgmt_trap|backend_choice|param_tuning|crash_recovery",
  "model": "gpt-oss-20b",
  "model_family": "dense",
  "action": "backends",
  "lesson": "vllm beat sglang by 9% on dense gpt-oss-20b",
  "tags": ["dense", "backend_choice"],
  "gain_pct": 9.0,
  "status": "keep|revert|fail"
}
```

Required fields: `ts`, `user_id`, `category`, `model`, `action`, `lesson`, `status`.

## `insights.jsonl` — Sage cross-run synthesis

```json
{
  "ts": "2026-04-27T18:00:00Z",
  "user_id": "default",
  "model_family": "moe_mla",
  "scope": "last_5_runs",
  "headline": "kernel-opt rarely improves moe_mla beyond 2%; deprioritize",
  "supporting_entries": ["entry_id_or_ts1", "..."]
}
```

## `conflicts.jsonl`

```json
{
  "ts": "2026-04-27T18:05:00Z",
  "entry_a_ref": "<ts of entry A>",
  "entry_b_ref": "<ts of entry B>",
  "reason": "entry A reports KEEP for backends:vllm on moe_mla, entry B reports REVERT"
}
```

## TODO

- [ ] Embed schema as JSONSchema and validate on ingest
- [ ] Per `user_id × model_family` partitioning helpers
