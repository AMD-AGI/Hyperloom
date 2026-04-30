# run_optimization — Run GEAK + OOB on selected kernels

**Trigger**: `topic="request"` envelope with `payload.kind="run_optimization"`.

## Inputs (request payload)

| field | required | example |
|---|:-:|---|
| `params.selected_kernels` | yes | `[{"name": "...", "source_path": "..."}, ...]` |
| `params.backends` | no (default `["geak", "codex"]`) | `["geak", "codex"]` or `["geak"]` |
| `params.prompt_file` | no — only OOB needs | path to optimization prompt |

## Procedure

1. Validate every `selected_kernels[].source_path` exists. Filter out missing.
2. For each kernel × backend pair, fan out **in parallel** (IR-1, soft):
   - GEAK: `bash $AGENT_PKG_DIR/scripts/run_geak.sh <source_path>`
   - OOB:  `bash $AGENT_PKG_DIR/scripts/run_oob.sh <agent> <source_path> <prompt_file>`
3. Wait for all to finish. Each script writes a log under
   `$SESSION_DIR/results/<task_id>/<backend>_<kernel_stem>.log`.
4. Parse each log for the GEAK / OOB output kernel path + predicted gain.
5. Aggregate winners (rc=0 + parse succeeded) and losers.

### Parallelization notes

- IR-1 (soft): submit candidates in parallel. The `run_geak.sh` /
  `run_oob.sh` scripts are blocking per-call, so you should issue them
  in a single `bash` block with `&` background + `wait`, or open
  multiple Bash tool calls in parallel within one Claude turn.
- If a long Ray job blocks your turn past the SDK timeout, the launcher
  restarts you. On restart, check `$SESSION_DIR/results/<task_id>/` for
  partially-completed logs and resume parsing rather than re-submitting.

## Output (RESPONSE payload)

```json
{
  "intent_type": "response",
  "payload": {
    "in_reply_to": "<request msg_id>",
    "kind": "optimization_done",
    "status": "succeeded",  // or "failed" if zero candidates produced
    "result": {
      "patches": [
        {
          "candidate_id": "geak_triton_red_42",
          "backend": "geak",
          "source_kernel": "/tmp/torchinductor_root/abc/123.py",
          "patch_path": "/path/to/optimized_kernel.py",
          "best_config_path": "/path/to/best_config.json",
          "predicted_gain_pct": 8.0,
          "log_path": "$SESSION_DIR/results/<task_id>/geak_123.log"
        },
        ...
      ],
      "failed": [
        {"backend": "codex", "kernel": "...", "reason": "rc=1, see log"}
      ],
      "n_succeeded": 3,
      "n_failed": 2,
      "backends": ["geak", "codex"]
    }
  }
}
```

## Failure modes

| Symptom | Recovery |
|---|---|
| Zero candidates produced (every fan-out failed) | `response{status=failed, result.reason="no_candidates"}` |
| Ray cluster unreachable | `response{status=failed, result.reason="ray_unavailable"}` — executor will likely emit `alert` |
| One backend down (geak ok, codex broken) | `response{status=succeeded, result.partial=True}` — return what you have |

## Soft rules

- **IR-1** (recommended): submit in parallel — don't loop sequentially.
- **IR-2** (recommended): do not Edit any kernel source between
  selection and submission. GEAK rewrites the file; we don't.
- **IR-7** (recommended): do not modify GEAK MCP config. The
  `run_geak.sh` script already handles tracing-headers exception.

## Discipline

- Always include `predicted_gain_pct` per candidate even if you have to
  guess (use 5.0 as a conservative default). Brier scoring requires it.
- Cite the log path in every `patches[]` and `failed[]` entry — it's
  what executor / watchdog use to debug if a patch later regresses.
