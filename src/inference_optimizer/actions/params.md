# Action: `params` (STUB)

> Family: **shallow** · All modes · accuracy_risk=0.0 (default) / 0.30 (fp8 kv-cache, quantization, etc.).

Sweep server-side parameters: `--max-running-requests`, attention backend,
chunked prefill, kv-cache dtype, etc. Some sub-flags are quantization-style
and trigger the high-risk accuracy gate variant.

## Output schema

```json
{
  "winning_params": {"--attention-backend": "fa3", "--enable-mla": true},
  "delta_pct": 6.4,
  "high_risk_flags_used": ["--kv-cache-fp8"]
}
```

## TODO (IMPL-CHECKLIST §4.28)

- [ ] Sub-flag risk classifier; switch `accuracy_risk` to 0.30 when fp8 / quant detected
- [ ] Avoid clobbering user-supplied TP value (`assert_user_tp_respected`)
