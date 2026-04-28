# Action: `profile` (STUB)

> Family: **analysis** · All modes · accuracy_risk=0.0.

Capture a torch / sglang trace, write the canonical
`filtered-TP-0.trace.json.gz`, and return analysis pointers.

## Output schema

```json
{
  "trace_path": "results/profile_<ts>/filtered-TP-0.trace.json.gz",
  "top_kernels": [{"name": "...", "pct": 0.34}, ...],
  "kernel_dispatch": "aiter_dominated|triton_dominated|mixed",
  "decode_stage_pct": 0.71
}
```

## TODO (IMPL-CHECKLIST §4.26)

- [ ] Set `PROFILE=1` and `SGLANG_TORCH_PROFILER_DIR` then unset on completion
- [ ] Use `process_management.pick_filtered_trace`
- [ ] Optional CSV summary for quick-mode reports
