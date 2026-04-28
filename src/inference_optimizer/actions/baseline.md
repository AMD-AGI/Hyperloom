# Action: `baseline` (STUB)

> Family: **prep** · All modes · accuracy_risk=0.0.

Run the canonical benchmark via `scripts/run_baseline.sh` to fix
`baseline_tput` (and `baseline_accuracy` if `accuracy_gate.requires_gate`
is in scope for any future action).

## Output schema

```json
{
  "tok_per_s_per_gpu": 4123.0,
  "p50_latency_ms": 91.2,
  "p95_latency_ms": 145.0,
  "results_dir": "results/baseline_<ts>",
  "baseline_accuracy_gsm8k": 0.812
}
```

## TODO (IMPL-CHECKLIST §4.25)

- [ ] Wire `process_management.enforce_run_baseline_sh("baseline")`
- [ ] Capture artefacts under `results/baseline_<ts>/`
- [ ] Optionally trigger an accuracy_gate baseline (depending on flags)
