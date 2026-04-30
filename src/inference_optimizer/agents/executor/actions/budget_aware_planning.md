# Budget-Aware Planning

**Trigger**: For your candidate next action `A`,
`state.time_left_minutes < A.cost_p75 × 1.25`. The "Available actions"
table the conductor injects every turn shows `cost_p75` per action.

## Why the buffer matters

The scheduler uses `depth_gate = (cost_p75 ≤ time_left × 0.8)` and zeros
out the score of an action that won't finish. If you `delegate` it
anyway, two things go wrong:

1. PolicyGate / scheduler still allows it (delegate is decoupled from
   scheduler.score) — but the action will run out of wall-clock and
   either time out (`needs_manual_review`) or be killed by the
   `time_exhausted` early-stop.
2. You burn budget that could have funded a cheaper action with smaller
   but real gain.

## Decision recipes

### Time left ≥ cost_p75 × 1.25

Plenty of headroom. Just delegate `A` normally.

### Time left between cost_p75 × 0.8 and × 1.25

Risky but possible. Two options:

- Delegate `A` with `predicted_gain_pct` lowered to reflect the risk.
- Pick a smaller action `B` with `cost_p75 ≤ time_left × 0.5`, even if
  its expected gain is smaller. Examples:
  - Instead of `kernel_opt` (cost_p75 ≈ 60min), try `param_sweep_run`
    (cost_p75 ≈ 8min) or `backends` (cost_p75 ≈ 12min).
  - Instead of `comm_optimization` (cost_p75 ≈ 30min), try
    `compiler_tuning` (cost_p75 ≈ 25min) or `params` (cost_p75 ≈ 8min).

### Time left < cost_p75 × 0.8

Hard skip. Pick a smaller action OR proceed to `report` to terminate
gracefully:

```json
{
  "intent_type": "delegate",
  "payload": {
    "action_name": "report",
    "params": {},
    "predicted_gain_pct": 0.0,
    "reason": "time_left=12m < cost_p75 of any productive action; emit final report"
  }
}
```

## Cheap actions that ALWAYS fit (≤ 5min cost_p75)

When in doubt, fall back to one of these:

- `bench_runner` — re-measure current state (~3min)
- `report` — final summary (terminates run gracefully)
- `setup` / `classify` / `target_analysis` — cheap prep, only useful
  early but always safe to run

## DON'T do these

- DON'T attempt a 60-min `kernel_opt` with 30 min on the clock.
- DON'T issue a delegate hoping the conductor will halt it cleanly mid-run
  (it won't — it'll either time out or run past `time_exhausted`).
- DON'T burn the last 10 minutes on `report` if you've never run anything
  else; the report will just say "no action taken". Better to skip
  `report` and let `time_exhausted` early-stop fire on its own.
