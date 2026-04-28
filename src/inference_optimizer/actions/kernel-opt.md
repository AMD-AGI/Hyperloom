# Action: `kernel-opt` (STUB)

> Family: **deep_kernel** · guided + marathon · accuracy_risk=0.05‒0.15.

Submit candidate kernels in PARALLEL via GEAK MCP (IR-1). Spawns
Codex+GEAK sub-agents through `SubAgentRunner`. Iterates `OOB_ROUND_ITERATIONS = 3`
rounds before handing off to `integrate`.

## Output schema

```json
{
  "candidates": [
    {"kernel_id": "...", "delta_pct": 7.2, "verdict": "keep"},
    {"kernel_id": "...", "delta_pct": -1.1, "verdict": "discard"}
  ],
  "winner_kernel_id": "...",
  "ready_for_integrate": true
}
```

## TODO (IMPL-CHECKLIST §4.31)

- [ ] IR-1 enforcement test: refuse single-candidate submissions
- [ ] IR-2 enforcement: source files untouched until GEAK finishes
- [ ] Use `KERNEL_OPT_BACKENDS` const + `kernel_opt_image()`
- [ ] Discard policy: `GEAK_CONSECUTIVE_DISCARDS=5` triggers stop
