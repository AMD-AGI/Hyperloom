# Action: `backends` (STUB)

> Family: **shallow** · All modes · accuracy_risk=0.10.

Try alternative serving backends (sglang vs vllm), each through
`scripts/run_baseline.sh`, and pick the winner by `tok/s/GPU` *and*
accuracy gate (because backend swap can change numerics).

## Output schema

```json
{
  "candidates": [
    {"backend": "sglang", "tok_per_s_per_gpu": 4123, "verdict": "keep"},
    {"backend": "vllm",  "tok_per_s_per_gpu": 4520, "verdict": "keep"}
  ],
  "winner": "vllm",
  "delta_pct": 9.6
}
```

## TODO (IMPL-CHECKLIST §4.27)

- [ ] vllm flag translator (`process_management.vllm_flag_translator`)
- [ ] Call `accuracy_gate.run_gsm8k` after every candidate KEEP
