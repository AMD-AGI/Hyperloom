<!-- when: an action just failed and you are about to re-propose it -->
<!-- phase: PRELUDE, ENABLEMENT, FRAMEWORK_AGENT, KERNEL_AGENT, SWEEP, CLOSE -->
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
   30 unpromotable results across ALL kinds, including kernel_agent-owned).
   Use this when the per-action history doesn't carry the kind you
   need (e.g. `integrate` failure visible here but `<action>_attempts`
   only covers the six explore/validate kinds).
   Each entry now carries `failure_id` when the failure has a structured
   evidence packet; use
   `mcp__inference_optimizer_context__get_failure(failure_id)` to retrieve
   the full packet and `Read` to open the raw crash log.
3. **Structured failure evidence** — explore variant crashes write a
   full evidence packet to `failures[]` (accessible via
   `mcp__inference_optimizer_context__get_variant_failures` /
   `mcp__inference_optimizer_context__get_failure`), mirrored to
   `<session_dir>/reports/failures/<failure_id>.json` so a packet
   evicted from `failures[]` is still readable with `Read`.  The packet
   carries: `failure_id`, `stage` (warmup or decision), `error_class`,
   `error_excerpt` (tail of the error blob), `server_log_path`,
   `workspace`, and the variant knobs that produced the failure.
   Inbox failure lines and gap attempt rows carry the `failure_id` for
   quick cross-reference.  Full investigation path:
   `failure_id` → `mcp__inference_optimizer_context__get_failure` →
   `Read(server_log_path)`.
4. **`baseline_failure_streak`** (PRELUDE only — consecutive failed baselines).
   Once this hits 3, Coordinator sets `stop_reason='baseline_failed'`
   and the run terminates; recover BEFORE the third failure.
   A retry is also only booted while PRELUDE still has time outside the
   optimization reserve: past that line the executor refuses the retry
   unlaunched and the run closes on `stop_reason='prelude_budget_exhausted'`,
   which can happen on the first failure of the leg. Only this leg's failures
   are priced that way — a resumed session still gets one baseline on its new
   budget. The time you spend before asking for a retry is part of what pays
   for it.

## Baseline fingerprint (PRELUDE)

The `baseline` fingerprint is exactly seven params fields:
`benchmark_script` / `result_dir` / `extra_server_args` /
`extra_envs` / `model_path` / `gpu_type` / `config_path`.
Those seven are the WHOLE fingerprint. A baseline
may also come back `status='succeeded'` with `decision='no_promote'`
and `extras.anchor_kept_tput`: it ran fine but measured below the
established anchor, so the anchor was kept. That is a completed
measurement, NOT a failure — do not retry it.

## Decision rules

* **RULE F1 — same fingerprint, twice failed → change at least one
  of the seven fingerprint fields.** (PRELUDE only.) Because you run as a
  persistent conversation you remember the attempts you already made this
  session — do not re-propose a baseline whose params fingerprint
  matches a recent failure; it will fail the same way. Changing a
  field OUTSIDE the seven does not count and does not get you a new
  attempt: the proposal is dropped as a duplicate and you have burned
  a tick. Change the launch: `params.extra_server_args` or
  `params.extra_envs`.
* **RULE F2 — repeated `error_class='subprocess_nonzero'` or `'client_failed'`
  on `baseline` ⇒ stop retrying baseline.** Each class means the benchmark
  exited nonzero for a cause Hyperloom did not recognise — `client_failed`
  names the benchmark client as the process that exited, so another identical
  attempt buys nothing.
  `send_message` with `body_md='blocked: subprocess repeatedly nonzero
  baseline'`, quoting the failure excerpt rather than a diagnosis. Explore
  variants may be re-proposed; read the failure log first.
* **RULE F3 — `policy_denial_streak` is a pure fact, not a lock.** The
  `mcp__inference_optimizer_context__why_denied` context tool (and the
  `Recent policy denials` block in the state projection) shows repeated
  (action, rule) collisions. The system no longer reacts to the streak —
  there is NO auto-prune at streak≥5 and NO `policy_loop` stop at
  streak≥10; the run continues until the wall-clock deadline or another
  stop_reason fires. So the streak is purely a signal for YOU: the same
  params keep colliding with the same invariant, so change something
  substantive — a new `params.grid` variant or a sibling action family.
  Re-emitting the identical denied intent just wastes a tick.
* **RULE F4 — `error_class='engine_execute_timeout'` is not a boot failure.**
  The server was serving when one vLLM model step outran
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` and the engine shut down
  (`server_died` is the same death without that timeout). The same launch
  stalls the same way, and a roofline that failed so is not re-run on the
  recipe until a KEEP changes it. Raise `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`
  in `params.extra_envs` only when the step is genuinely that slow;
  otherwise change what the step runs.

## Example (PRELUDE — baseline failed twice with `error_class='server_init_dead'`)

    propose_action{action_name='baseline',
        params={extra_server_args: '--mem-fraction-static 0.8'},
        predicted_gain_pct: 0,
        notes: 'server_log_path shows an OOM during KV-cache allocation;
                lower the static memory fraction'}
