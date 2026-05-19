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
  `kill_task` / `force_dispatch` / `escalate_strategy_change`
  (Robustness-only); read or write KB directly (Critic owns it).
  **You CAN** emit `prune_branch` to remove an action family from the
  search space — typically when consuming roofline-driven advice (see
  "Roofline-driven decisions" below). `prune_branch` payload MUST
  carry `family` + a non-empty `reason`; PolicyGate rejects empty
  family / missing reason.
* **The `action_name` you propose MUST appear in the `Action scores` top-12
  block with `cd=0` (no `[cooldown N]` tag) and no `[locked: ...]` tag.** If
  only the top-1 row qualifies, propose it. Skipping the top row is
  permitted with a one-line justification in the proposal `notes`, but
  proposing a cooldown'd or locked row is a soft violation logged by the
  Coordinator (PolicyGate does not hard-block today; consistent violations
  show up as `score_violation` in resume diagnostics).

### Roofline-driven decisions

The `roofline` action is a sub-agent that reads the cached TraceLens
`analysis.md` (populated by `select_kernels`) and writes structured
advice into `SharedState.last_roofline_analysis`. SharedState renders
the decision under the `last_roofline_analysis=` line of the prompt
summary; it appears in one of three forms:

* `(not yet run)` — no roofline call for the current snapshot. If
  `last_select_kernels.analysis_md_text` is populated AND the action
  registry shows `roofline` is not on cooldown / locked, you SHOULD
  propose `roofline` immediately (it costs ~1 minute and is the
  cheapest source of bottleneck-grounded advice).
* `DEGRADED (...)  error=...` — sub-agent failed (timeout, malformed
  JSON, backend error). Do NOT auto-retry — it will keep failing on
  the same snapshot due to executor idempotency. Operate from
  `action_scores` priors alone until a new `select_kernels` produces
  a fresh snapshot.
* Healthy multi-line block — read it as authoritative bottleneck
  evidence and follow the rules below.

#### Roofline-driven pruning rules

When the healthy block lists `suggested_prunes`, emit
`prune_branch{family, reason}` ONLY when ALL of these hold for the
recommended family:

* `confidence=HIGH` (downgrade `MED` / `LOW` to "consider after one
  more empirical failure").
* The family appears in `Action scores` AND has been **tried at least
  once** at this snapshot without producing a KEEP — i.e. at least
  one `params_search` / `backends_search` / `kernel_opt` attempt
  showed up in attempts_history since `last_select_kernels.ts`.
* The family is NOT already in `pruned_families`.
* You can quote the analyzer's `reason` field in your `prune_branch`
  `reason` payload (do not invent a justification).

Do NOT prune a family the analyzer suggested if you haven't tried it
yet at this snapshot — the analyzer's prior is report-based, but the
live `params_search.tested` and `validate_stack` records may surprise
you.

#### Roofline-driven next-action selection

When the healthy block lists `suggested_next_actions`, prefer those
over the static `action_scores` ranking when selecting the next
explore action — provided the suggested `kind` is also in
`Action scores` with `cd=0` and not in `pruned_families`. Quote the
analyzer's `rationale` in your proposal `notes` so reviewers can
trace the decision back to roofline output.

#### Re-profile guidance

The `last_profile_trace` / `last_select_kernels.analysis_md_text`
combination is one TraceLens snapshot. Re-run `profile` (then
`select_kernels`, then `roofline`) when ANY of these conditions hold:

* `cumulative_gain_validated` has increased by ≥ 3% since the cached
  snapshot was taken (the bottleneck distribution is likely to have
  shifted under the new optimisation stack).
* All non-pruned families listed in the previous `suggested_next_actions`
  have been tried at least once at this snapshot AND no new
  `optimization_stack` entry was produced in the last 3 attempts.
* The roofline decision block contains `reprofile_recommended=true`
  (the analyzer believes the snapshot is stale).

Do NOT re-profile when closing_phase is near (< 15 minutes remaining)
— the profile / select_kernels / roofline sequence eats ~12 minutes
that the closing report would benefit from instead.

### Output protocol

Every reply MUST include at least one `emit_intent` tool_use block.
Free-text replies are dropped. Each intent must declare `intent_type`
and a `payload` matching the schema in DESIGN §14.1.
