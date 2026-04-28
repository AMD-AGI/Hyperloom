# Action: `target-analysis` (STUB)

> Family: **prep** · All modes · accuracy_risk=0.0.

If the user supplied `TARGET_DIR` (a baseline-comparison objective), inspect
the target directory: read its config, baseline metrics, kernel manifest.
Otherwise this action is a no-op stub returning `target_present: false`.

## Output schema

```json
{
  "target_present": true,
  "target_baseline_tput": 4500.0,
  "target_kernels_referenced": ["fused_moe", "rmsnorm"],
  "comparable_metrics": ["tok_per_s_per_gpu", "p95_latency"]
}
```

## TODO (IMPL-CHECKLIST §4.24)

- [ ] Define folder layout for TARGET_DIR
- [ ] Failure: TARGET_DIR exists but malformed → state.set_stopping("target_dir_invalid")
