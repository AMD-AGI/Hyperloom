# Action: `sweep` (STUB)

> Family: **shallow** · All modes · accuracy_risk=0.0.

Combine top-N param settings with top-N backend candidates and re-bench.
Used in quick mode to extract the last few percent before `report`.

## Output schema

```json
{
  "best_combination": {
    "backend": "vllm",
    "params": {"--attention-backend": "fa3", "--max-running-requests": 256}
  },
  "delta_vs_baseline_pct": 14.2
}
```

## TODO (IMPL-CHECKLIST §4.29)

- [ ] Cap combinations: top-3 backend × top-3 params = 9 runs maximum
