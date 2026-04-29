# setup — session bootstrap

**Family**: `prep` · **Cost**: ~1‑3 min · **Risk**: minimal

Verify the environment, MODEL_PATH, MAX_HOURS, GPU visibility (without
running anything heavy), and ensure the session directory tree is laid out:

```
<session_dir>/
  storage/conductor.db
  events/  cursors/  tasks/
  personas/  findings/  results/  checkpoints/
  kb/entries.jsonl  kb/insights.jsonl
```

Side‑effects (declared in `_meta/setup.yaml`):

- `reads_env` — only reads, never writes outside the session directory.

Output intents (Executor):

- `update_state` with `current_action="setup"` then `current_action=null`
- `send_message` topic=`event` summarizing GPU visibility / MODEL_PATH check

Done‑condition:

- All required env vars are present (MODEL_PATH, MAX_HOURS, optional
  TARGET_*).
- `nvidia-smi` / `rocm-smi` returned ≥1 GPU OR the run is in mock mode.
