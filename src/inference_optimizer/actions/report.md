# Action: `report` (STUB)

> Family: **shallow** · All modes · accuracy_risk=0.0.

Final summary report. Aggregates baseline → final, lists kept actions,
produces a markdown writeup under `results/report_<ts>.md` and a small
machine-readable summary at `results/report_<ts>.json`.

## Output schema

```json
{
  "report_md": "results/report_<ts>.md",
  "report_json": "results/report_<ts>.json",
  "final_tput": 5230.0,
  "cumulative_gain_pct": 26.9,
  "stop_reason": "target_reached"
}
```

## TODO (IMPL-CHECKLIST §4.30)

- [ ] Pull KEEP / REVERT decisions from SharedState.last_decisions
- [ ] Pull token-budget figures from TokenBudgetMeter
- [ ] Optionally seed `kb.ingest` lessons from this run
