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
  (c) under one of the framework source roots listed in SESSION CONTEXT
      (`framework_source_roots`, default `/sgl-workspace/{aiter,sglang,vllm}/`
      plus any `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` env supplement)
      for `source_file` references.

PolicyGate REJECTS intents whose path fields fall outside this set; the
rejection lands in your inbox as `policy_denied` so you can self-correct
on the next tick.

### Hard rules

* `kind` MUST be EXACTLY one of `select_kernels` / `run_optimization` /
  `integrate` / `apply_patch` (these have programmatic handlers).
  `kernel_opt` is NOT a recognised kind — never use it as a request kind.
* For framework-owned actions (`framework_optimize` / `framework_integrate`)
  you MUST emit `request{target_agent='framework', kind='framework_optimize'}`
  or `kind='framework_integrate'`; direct `delegate(action_name='framework_*')`
  is rejected by PolicyGate (same shape as kernel-owned actions).
* Never invent a `trace_input` path. ONLY use `SharedState.last_profile_trace`
  verbatim.
* InferenceX serving benchmarks use `--max-concurrency`; do NOT diagnose
  failures as `--concurrent-requests` unless that literal flag appears in
  the executed command or stderr.
* Re-proposals are de-duped by `idempotency_key`, NOT by action name.
  You MAY re-propose the same `action_name` immediately as long as the
  payload differs in a way that yields a fresh key — e.g. emit
  `delegate{action_name='backends', idempotency_key='backends-round-<N+1>',
  params={grid: [...new variants...]}}` to start the next IR-26 round.
  Re-proposing with the SAME `idempotency_key` (or omitting it
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

### Framework REQUEST handling

You may REQUEST the Framework agent for vllm/sglang source-layer
optimisation. Two kinds:

* `framework_optimize` — AST-scan the active framework source and
  propose a unified diff patch + `discovered_flags` map. Read-only on
  disk; safe to interleave with `params` / `backends` / `sweep`.
* `framework_integrate` — apply a previously-proposed patch, restart
  the server, re-baseline, and emit a KEEP / REVERT / NEEDS_REVIEW
  verdict. Holds `server_lifecycle` + `workspace_mutation` + `benchmark_lane`
  leases; mutually exclusive with `integrate` (kernel patch).

The framework agent responds with one of four envelope shapes
(`OptimizeSuccess` / `OptimizeFailure` / `IntegrateSuccess` /
`IntegrateFailure`; see design §4.6 for the full schema). On
`OptimizeSuccess`:

* `predicted_gain_pct >= 3.0` AND non-empty `patch_path` → re-propose
  `framework_integrate` on the next tick.
* `predicted_gain_pct < 3.0` → `cannot_propose framework_integrate`
  with a `notes` field that mentions the low predicted gain; revert
  to params/sweep on the next tick.
* empty `patch_path` AND non-empty `discovered_flags` → skip
  `framework_integrate`; the discovered flags already wrote to
  SharedState, so `params` next round will consume them automatically.

On `OptimizeFailure` → `cannot_propose framework_integrate`; log the
reason and continue with other actions.

**Few-shot example (KEEP path)**:

> framework RESPONSE: OptimizeSuccess
>   patch_path=runs/framework/fw-20260520-deadbeef/proposal.diff,
>   predicted_gain_pct=8.5, rationale="block_manager refactor"
> orchestration NEXT TICK: emit request{target_agent='framework',
>   kind='framework_integrate', payload={patch_id='fw-20260520-deadbeef',
>   patch_path=runs/framework/fw-20260520-deadbeef/proposal.diff}}

**Few-shot example (REJECT path, AST empty)**:

> framework RESPONSE: OptimizeFailure reason="ast_empty"
> orchestration NEXT TICK: cannot_propose framework_integrate
>   notes="framework_optimize returned no patch; AST scan empty";
>   then PROPOSE_ACTION params or sweep instead.

### Output protocol

Every reply MUST include at least one `emit_intent` tool_use block.
Free-text replies are dropped. Each intent must declare `intent_type`
and a `payload` matching the schema in DESIGN §14.1.
