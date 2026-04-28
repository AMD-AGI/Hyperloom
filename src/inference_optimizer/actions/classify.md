# Action: `classify` (STUB)

> Family: **prep** · All modes · accuracy_risk=0.0.

Determine model class (dense / moe+mla / moe+swa / moe+mla+nsa) so the
scheduler can apply the correct prior matrix (`score_priors.py`).

## Output schema

```json
{
  "model_class": "moe_mla",
  "evidence": ["config.json contains 'mla'", "expert_count=32"],
  "confidence": 0.85
}
```

## TODO (IMPL-CHECKLIST §4.23)

- [ ] Read model config.json + tokenizer + safetensors header
- [ ] Pattern table: deepseek/mla → moe_mla, mixtral → moe_swa, kimi → moe_mla_nsa, gpt-oss/llama/qwen-dense → dense
