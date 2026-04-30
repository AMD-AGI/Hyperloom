# Retry After Dedup — How to Pivot

**Trigger**: Inbox shows a `topic=event kind=delegate_dedup_to_terminal`
envelope. Example payload:

```json
{
  "kind": "delegate_dedup_to_terminal",
  "task_id": "<uuid>",
  "action_name": "profile",
  "task_state": "failed",
  "hint": "action 'profile' already ran and ended in state='failed'; the dispatcher will NOT re-run it. Pick a different action_name (e.g. bench_runner, param_sweep_run, kernel_opt) or change params to break idempotency."
}
```

## Why this exists

The conductor dedups `delegate(action_name, params)` pairs by their
SHA-256 idempotency hash. If you re-emit the **exact same** delegate
after the first attempt reached a terminal state (`succeeded` /
`failed` / `safely_failed` / `needs_manual_review`), the dispatcher
**will not re-run it** — the task row is already terminal. You'll
just get this dedup event back, every time, forever.

Two consecutive failed attempts with the same `(action_name, params)`
key is a **hard stop signal**: pivot, don't loop.

## Recovery menu

Pick ONE of these for the next turn — never two delegates at once.

### Option 1 — Different `action_name` (preferred)

Look at the live "Available actions" table in your prompt. Good
recovery picks when something fails:

| What failed | Try instead |
|---|---|
| `profile` returned `kind=profile_skipped` | `bench_runner` (re-measure) or `kernel_opt` (use the prior trace if any) |
| `baseline` failed (rc != 0) | `bench_runner` against an existing server, or `ask_question` to Sage about model env |
| `kernel_opt` no candidates | `param_sweep_run` (fall back to scheduling params) |
| `param_sweep_run` flat | `backends` (try a different attention/GEMM backend) |
| `bench_runner` regression | DO NOT immediately try the same again; emit `alert` if marathon, else move to `report` |

### Option 2 — Same action, different `params`

Changing any field in `params` flips the idempotency hash. Examples:

```json
// Original (terminal): delegate(profile, params={})
// Pivot:               delegate(profile, params={"warmup_iters": 10})
// Pivot:               delegate(param_sweep_run, params={"conc_grid": [16, 32]})
```

Don't add nonsense fields just to bypass dedup — pick params that
actually change executor behaviour, or pick a different `action_name`.

### Option 3 — Stop and report

If you've burned ≥ 3 dedup events in the last ~10 inbox messages, you
are stuck. Emit:

```json
{
  "intents": [
    {
      "intent_type": "alert",
      "payload": {
        "severity": "medium",
        "summary": "Executor stuck after 3 dedup events; cannot find a productive next action.",
        "detail": "Last dedup'd actions: profile, profile, param_sweep_run. Recommend manual review or graceful stop."
      }
    },
    {
      "intent_type": "delegate",
      "payload": {
        "action_name": "report",
        "params": {},
        "predicted_gain_pct": 0.0,
        "reason": "stop and dump current state for human review"
      }
    }
  ]
}
```

`report` is on the quick-mode allow-list and accepted in all modes; it
terminates the session by writing the final summary.

## DON'T do these

- DON'T treat the dedup event as a transient error to retry past.
- DON'T re-emit the dedup'd intent with a trivially-different param
  like `{"_retry": 1}` — you're lying to the idempotency layer.
- DON'T spam `alert` events without picking a real recovery action;
  Watchdog will see your alert but it can't choose your next delegate.
