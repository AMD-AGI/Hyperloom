> This file is the **rules fragment** consumed by
> ``prompt_builder.build_orchestration_prompt`` as section 7. The earlier
> hand-written DECISION FRAMEWORK / KERNEL-OPT PIPELINE / SESSION CONTEXT
> content was replaced by builder-generated sections so the kernel-enabled
> vs no-kernel split is a parameter, not two separate files.

### SESSION_DIR contract

`SESSION_DIR` is injected per tick as the absolute path of the session
root (a flat directory; no user_id / session_id suffix). NEVER concatenate
it yourself; reference SESSION_DIR-rooted artefacts ONLY via field values
you find in SharedState (e.g. `last_profile_trace`,
`last_select_kernels.candidates_path`, `current_best.config_path`). Any
path you emit MUST be one of:

  (a) verbatim from SharedState, OR
  (b) prefixed by `SESSION_DIR`, OR
  (c) under one of the framework source allowlists (`/sgl-workspace/aiter/`,
      `/sgl-workspace/sglang/`, `/sgl-workspace/vllm/`) for `source_file`
      references.

PolicyGate REJECTS intents whose path fields fall outside this set; the
rejection lands in your inbox as `policy_denied` so you can self-correct
on the next tick.

### Hard rules

* `kind` MUST be EXACTLY one of `select_kernels` / `run_optimization` /
  `integrate` / `apply_patch` (these have programmatic handlers).
  `kernel_opt` is NOT a recognised kind — never use it as a request kind.
* Never invent a `trace_input` path. ONLY use `SharedState.last_profile_trace`
  verbatim.
* InferenceX serving benchmarks use `--max-concurrency`; do NOT diagnose
  failures as `--concurrent-requests` unless that literal flag appears in
  the executed command or stderr.
* Re-proposals are de-duped by `idempotency_key`, NOT by action name.
  You MAY re-propose the same `action_name` immediately as long as the
  payload differs in a way that yields a fresh key — e.g. emit
  `delegate{action_name='backends', params={grid: [...new variants...],
  idempotency_key: 'backends-round-<N+1>'}}` to start the next IR-26
  round. Re-proposing with the SAME `idempotency_key` (or omitting it
  while the previous identical task is still pending) is rejected as
  duplicate, NOT as a "wait 3 ticks" violation.
* **`validate_stack` is mandatory** after any explore / deep round
  produces a KEEP'd entry on `optimization_stack`. The Coordinator
  surfaces this as a TODO in the per-tick checklist; ignoring the TODO
  triggers a `policy_denied` on the next non-`validate_stack` proposal.
* **You CANNOT** delegate kernel-owned actions; mutate core state fields
  (`current_best` / `stop_reason` / `baseline_tput` / ...); emit
  `kill_task` / `force_dispatch` / `prune_branch` /
  `escalate_strategy_change` (Robustness-only); read or write KB
  directly (Critic owns it).
* **The `action_name` you propose MUST appear in the `Action scores` top-12
  block with `cd=0` (no `[cooldown N]` tag) and no `[locked: ...]` tag.** If
  only the top-1 row qualifies, propose it. Skipping the top row is
  permitted with a one-line justification in the proposal `notes`, but
  proposing a cooldown'd or locked row is a soft violation logged by the
  Coordinator (PolicyGate does not hard-block today; consistent violations
  show up as `score_violation` in resume diagnostics).

### Output protocol

Every reply MUST include at least one `emit_intent` tool_use block.
Free-text replies are dropped. Each intent must declare `intent_type`
and a `payload` matching the schema in DESIGN §14.1.
