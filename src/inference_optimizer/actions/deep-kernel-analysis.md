# Action: `deep-kernel-analysis` (STUB)

> Family: **deep_kernel** · marathon-only · accuracy_risk=0.0.

Multi-hour kernel-dispatch deep dive: re-profile, classify dispatch
patterns, hunt for bottlenecks beyond the obvious top-3 kernels.

## Output schema

```json
{
  "promising_kernel_targets": ["fused_moe_topk", "rmsnorm_int8", ...],
  "dispatch_pattern": "aiter_dominated|triton_dominated|mixed",
  "register_pressure_hot_spots": ["..."]
}
```

## TODO (IMPL-CHECKLIST §4.33)

- [ ] Cross-reference with `kb` for prior patterns
- [ ] Output is read-only — no workspace mutation
