# apply_patch — Apply selected kernel patches + restart server + re-baseline

**Trigger**: `topic="request"` envelope with `payload.kind="apply_patch"`.

This is the **integrate** step (IR-3 mandatory after a successful
optimization). The executor decides which patches from the previous
`optimization_done` response are worth applying; you apply them, restart
the server, re-run baseline, and return the new throughput.

## Inputs (request payload)

| field | required | example |
|---|:-:|---|
| `params.selected_patches` | yes | `[{"candidate_id": "geak_triton_red_42", "patch_path": "...", "best_config_path": "..."}, ...]` |
| `params.skip_rebaseline` | no (default false) | `true` to skip the verification bench |

## Procedure

1. **Soft lane check** (Plan A default): read `state.json` and verify
   `state.current_action` is not actively `bench_*`. If it is, emit
   `send_message{topic=heartbeat, body_md="deferring patch apply,
   bench in flight"}` and **defer** — the executor will see the heartbeat
   and re-issue the request when ready. Do NOT block the LLM turn.
2. For each patch, run:
   ```bash
   bash $AGENT_PKG_DIR/scripts/apply_patch.sh \
       --target-file <patch.source_kernel> \
       --patch-file <patch.patch_path> \
       --best-config <patch.best_config_path>
   ```
   The script wraps `patch_inductor.py` (IR-6 soft) and handles fingerprinting.
3. Once all patches are applied, the script restarts the server (IR-4 +
   IR-5 enforced inside the script: `pgrep` + targeted `kill` + wait
   `SERVER_KILL_WAIT_S=10s` + verify GPU memory free) and re-runs
   `scripts/run_baseline.sh` to measure new throughput.
4. Parse `metrics.json` from the re-baseline. If accuracy_gate (which
   the script invokes automatically when `EVAL_TASK=gsm8k` is set)
   reports REVERT, the script automatically rolls back; in that case
   the response carries `status="failed"` + `result.reverted=True`.

## Output (RESPONSE payload)

```json
{
  "intent_type": "response",
  "payload": {
    "in_reply_to": "<request msg_id>",
    "kind": "patch_applied",
    "status": "succeeded",  // "failed" if revert / accuracy / rc != 0
    "result": {
      "applied_patches": ["geak_triton_red_42", "..."],
      "current_tput": 482.3,
      "baseline_tput": 420.0,
      "gain_pct": 14.83,
      "accuracy": 0.94,
      "patch_fingerprint": "<sha256 prefix>",
      "metrics_path": "$SESSION_DIR/results/<task_id>/metrics.json",
      "log_path": "$SESSION_DIR/results/<task_id>/apply_patch.log",
      "reverted": false
    }
  }
}
```

The executor takes `current_tput` from your response and emits its own
`update_state(current_tput=X)` on the bus. **Do NOT emit update_state
yourself** — PolicyGate will reject (`role` rule).

## Failure modes

| Symptom | Recovery |
|---|---|
| Soft lane busy (executor benching) | `send_message{topic=heartbeat, ...defer...}` — executor re-issues |
| `apply_patch.sh` rc != 0 | `response{status=failed, result.reason="patch_failed", log_excerpt=...}` |
| Accuracy regression (script auto-reverted) | `response{status=failed, result.reverted=True, accuracy=...}` |
| Server failed to restart | `response{status=failed, result.reason="server_down"}` + `alert{severity=high}` |
| Re-baseline shows regression vs old | `response{status=failed, result.reason="regression", current_tput=X, baseline_tput=Y}` |

## Hard rules (BLOCK)

- **IR-3** — This step IS the integrate phase. Returning
  `optimization_done` without an `apply_patch` follow-up means the gain
  is unverified. The executor's SKILL guides it to always issue
  `apply_patch` after `optimization_done`, but if it doesn't, raise an
  `alert{severity=medium, summary="optimization_done not followed by
  apply_patch"}`.
- **IR-4** — Server kill before relaunch is enforced inside
  `apply_patch.sh`. Do NOT bypass by issuing your own `python -m
  sglang.launch_server` Bash command.
- **IR-5** — The script uses `pgrep -f sglang.launch_server` + `kill
  <pid>`. Do NOT issue `pkill -f sglang` from your Bash tool —
  PolicyGate's quick-mode denylist would block it; the kernel agent is
  not in quick mode but the discipline is the same.

## Soft rules

- **IR-6** (WARN): the script always passes `--target-file` and
  includes `--best-config` when the patch metadata declares
  `block_size`/`num_warps` tuning. If your input lacks `best_config_path`
  for a tuning-keys patch, the script logs a stderr WARNING but still
  applies — accept this and report it in `result.warnings[]`.
