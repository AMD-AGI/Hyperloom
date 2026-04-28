# Action: `setup` (STUB)

> Family: **prep** · All modes · accuracy_risk=0.0 · See DESIGN §12.1.

## Goal

Ensure the workspace is in a known state before optimization begins:
- model artifacts present at `MODEL_PATH`
- venv on PATH (`/opt/venv/bin`)
- baseline servers killed (IR-4 / IR-5)
- `results/` and `findings/` writable
- `kb/` directory bootstrapped (cold start: empty entries.jsonl)

## Output schema

```json
{
  "model_path": "...",
  "model_name": "...",
  "model_class": "dense|moe_mla|moe_swa|moe_mla_nsa|unknown",
  "venv_active": true,
  "gpu_count": 8,
  "free_gpus": 8
}
```

## TODO (IMPL-CHECKLIST §4.22)

- [ ] Concrete prompt body migrated from sprint
- [ ] Tool-use plan: Read MODEL_PATH dir, run `process_management.safe_kill_server`, `nvidia-smi`/`rocm-smi`
- [ ] Failure-mode list (missing model, occupied GPU, etc.)
