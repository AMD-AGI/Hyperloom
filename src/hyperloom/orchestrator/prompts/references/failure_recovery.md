<!-- when: an action just failed and you are about to re-propose it -->
<!-- phase: PRELUDE, FRAMEWORK_AGENT, EXPLORE, KERNEL_AGENT, SWEEP, CLOSE -->
# Failure recovery surfaces

Consult these SharedState surfaces in order before re-proposing:

1. **`last_<action>`** (snapshot of the latest attempt) and
   **`<action>_attempts`** (capped per-action history, newest last).
   Each entry carries `status` / `decision` / `error_class` /
   `error_excerpt` / `workspace` / `raw_result_path` /
   `extras.fingerprint`. The fingerprint is the canonical hash of the
   params fields that determine the action's behaviour, and Coordinator
   keys task idempotency on that same fingerprint — so a proposal
   matching a queued or running task is dropped before it ever reaches
   the executor, and any key you add to `params` outside the
   fingerprint (a tag, a note, a counter) is invisible to it and does
   NOT make the proposal distinct.
2. **`last_action_failures`** (global rolling log capped at the last
   10 unpromotable results across ALL kinds, including kernel_agent-owned).
   Use this when the per-action history doesn't carry the kind you
   need (e.g. `integrate` failure visible here but `<action>_attempts`
   only covers the six explore/validate kinds).
3. **`baseline_failure_streak`** (PRELUDE only — consecutive failed baselines).
   Once this hits 3, Coordinator sets `stop_reason='baseline_failed'`
   and the run terminates; recover BEFORE the third failure.

## Baseline fingerprint (PRELUDE)

The `baseline` fingerprint is exactly eight params fields:
`benchmark_script` / `result_dir` / `extra_server_args` /
`extra_envs` / `model_path` / `gpu_type` / `config_path` /
`disable_run_eval`. Those eight are the WHOLE fingerprint. A baseline
may also come back `status='succeeded'` with `decision='no_promote'`
and `extras.anchor_kept_tput`: it ran fine but measured below the
established anchor, so the anchor was kept. That is a completed
measurement, NOT a failure — do not retry it.

## Decision rules

* **RULE F1 — same fingerprint, twice failed → change at least one
  of the eight fingerprint fields.** (PRELUDE only.) Because you run as a
  persistent conversation you remember the attempts you already made this
  session — do not re-propose a baseline whose params fingerprint
  matches a recent failure; it will fail the same way. Changing a
  field OUTSIDE the eight does not count and does not get you a new
  attempt: the proposal is dropped as a duplicate and you have burned
  a tick. Bump at least one of: `params.benchmark_script`
  (a sanitized `*.sh` file name that MUST match THIS run's framework —
  e.g. for a vllm run pin `vllm_mi300x.sh`, never `sglang_*`; a
  cross-framework script boots the wrong engine and is rejected),
  `params.result_dir`, `params.extra_server_args`, or
  `params.extra_envs`.
* **RULE F2 — `error_class='no_report'` + no `rescued_from_leaked_path:*`
  warning ⇒ leak salvage missed.** (PRELUDE only.) The script wrote results
  outside the workspace and outside the configured leak destinations.
  Override `params.benchmark_script` to a script that respects
  `$RESULT_DIR` (Coordinator already exports `RESULT_DIR=<workspace>` by
  default), or set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` via `update_state`
  so the next attempt salvages the leak.
* **RULE F3 — repeated `error_class='subprocess_nonzero'` on `baseline`
  ⇒ stop retrying baseline.** Heartbeat with `body_md='blocked: subprocess
  repeatedly nonzero baseline'` and let Robustness intervene, whose
  escalation policy needs that heartbeat to fire its RCA. Explore variants
  may be re-proposed; read the failure log first.
* **RULE F4 — `policy_denial_streak` is a pure fact, not a lock.** The
  `why_denied` context tool (and the `Recent policy denials` block on a
  seed turn) shows repeated (action, rule) collisions. The system no
  longer reacts to the streak — there is NO auto-prune at streak≥5 and
  NO `policy_loop` stop at streak≥10; the run continues until the
  wall-clock deadline or another stop_reason fires. So the streak is
  purely a signal for YOU: the same params keep colliding with the same
  invariant, so change something substantive — a new `params.grid`
  variant, a different `benchmark_script`, or a sibling action family.
  Re-emitting the identical denied intent just wastes a tick.

## Example (PRELUDE — baseline failed twice with `error_class='no_report'`)

    propose_action{action_name='baseline',
        params={result_dir: '<session_dir>/runs/baseline/<task>/leak'},
        predicted_gain_pct: 0,
        notes: 'recover from no_report streak by redirecting RESULT_DIR
                to the observed leak location'}
