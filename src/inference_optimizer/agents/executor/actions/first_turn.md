# First Turn — Reacting to `run_started`

**Trigger**: Your very first inbox tick. Typically a `topic="event"`
envelope with `payload.kind="run_started"` from the conductor.

## Why baseline first?

Without `baseline_tput` the early-stop logic and the objective layer
both lack a reference point. The conductor will inject a **"First action
hint"** block in your prompt every turn until `baseline_tput > 0` —
that's the same reminder, restated. Don't fight it.

## Exact action — emit ONE intent

Call `emit_intent` once with this envelope:

```json
{
  "intents": [
    {
      "intent_type": "delegate",
      "payload": {
        "action_name": "baseline",
        "params": {},
        "predicted_gain_pct": 0.0,
        "reason": "first measurement; need baseline_tput before optimising"
      }
    }
  ]
}
```

`predicted_gain_pct = 0.0` is correct: baseline by definition produces
zero gain over baseline. Setting a non-zero value here will hurt your
Brier score later.

## What happens after you emit

1. PolicyGate accepts (executor allowed; `baseline` allowed in all modes).
2. `tasks` table: new `kind=delegate state=queued` row.
3. The dispatcher loop picks it up → `SubAgentRunner.run` → `BaselineExecutor`.
4. `BaselineExecutor` shells out to `scripts/run_baseline.sh` (reads
   `MODEL`, `TP`, `CONC`, `ISL`, `OSL`, `INFERENCEX_PATH` from env).
5. Eventually your inbox will contain:
   - `topic=decision` with `kind=state_updated` and `changes.baseline_tput=X`
   - The conductor's `_maybe_recompute_gain` derives `cumulative_gain=0.0`
     and logs it under the same event's `derived` field.

When you see that decision event → Read `after_baseline.md`.

## DON'T do these

- DON'T propose multiple actions in your first turn — pick `baseline` only.
- DON'T set `predicted_gain_pct` to anything but `0.0` for baseline.
- DON'T pass custom `params` unless you know exactly what
  `BaselineExecutor` consumes (it reads run env vars, not params).
- DON'T issue a `propose_action` first then a `delegate` — go straight
  to `delegate`. `propose_action` is for cases where you want to invite
  Critic / parliament feedback before committing.
- DON'T re-emit the same delegate if your inbox doesn't change for 1–2
  ticks — the dispatcher is async and may take longer than your reactor
  tick interval.

## If `BaselineExecutor` fails

You'll see one of:

- `decision{kind=state_updated}` with no `baseline_tput` change AND a
  `topic=event kind=delegate_failed` event — the executor crashed or
  returned non-zero rc. Read `../reference/failure_codebook.md` and
  pick a different action OR fix params.
- `event{kind=delegate_dedup_to_terminal}` if you naively re-delegate.
  Read `retry_after_dedup.md`.
