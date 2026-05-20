> This file is the **rules fragment** consumed by
> ``prompt_builder.build_orchestration_prompt`` as section 7. The earlier
> hand-written DECISION FRAMEWORK / KERNEL-OPT PIPELINE / SESSION CONTEXT
> content was replaced by builder-generated sections so the kernel-enabled
> vs no-kernel split is a parameter, not two separate files.

### Phase awareness (v0.8 §3.2 / §3.3)

The Coordinator owns a strict 5-phase pipeline:

    PRELUDE → EXPLORE → KERNEL → SWEEP → CLOSE

It enters PRELUDE at session start and advances **only forward** when
phase-specific exit conditions fire (see KB_design §3.2). Your job
**within a phase** is to drive that phase to its exit condition; **you
do NOT decide when to transition** — the Coordinator does. You may
strongly recommend a jump via `escalate_strategy_change` (Robustness
forwards it), but Orchestration cannot emit `escalate_strategy_change`
directly.

Every tick the per-tick prompt includes a `=== Phase ===` block with:

  - `phase=<PHASE>` — your current phase.
  - `allowed_actions=[…]` — the only actions you may `propose_action`
    / `delegate` / `request` this tick. PolicyGate **rule R1
    (phase_incompatible)** rejects anything outside this set; the
    rejection lands in your inbox as a `policy_denied` event with the
    exact hint string `"you are in phase=…"`.
  - `elapsed_sec / budget_remaining_sec` — how much wall-clock this
    phase has already burned vs its budget (KB_design §3.8 §5.3).

Per-phase intent map (v0.8 M3 + KB_gaps/Gap-10: legacy
`backends`/`params`/`validate_stack` are removed; PolicyGate denies
them with `rule='action_deprecated'` and the canonical replacement
is the merged `explore` action):

  - **PRELUDE**: `target_analysis`, `baseline`, `recover` only. Drive
    `baseline_tput > 0` so the Coordinator can advance to EXPLORE. Do
    NOT propose `profile` / `kernel_opt` / explore-family actions
    here — they will all be denied.
  - **EXPLORE**: `explore`, `specialist`, `recover`. `profile` /
    `kernel_opt` / `sweep` / `report` are **denied**. Goal: stack
    KEEPs onto `optimization_stack` until the plateau judge fires or
    the budget cap hits. The `explore` action runs its per-KEEP
    stack rebench inline, so the v0.6 standalone `validate_stack`
    step is gone.
  - **KERNEL**: `profile` (single shot at phase entry), `pmc_roofline`,
    the 5 KERNEL_OWNED_ACTIONS via REQUEST, and `recover`. Goal:
    integrate KEEP'd kernel patches; the Coordinator exits to SWEEP
    when a REVERT streak builds or the budget cap hits.
  - **SWEEP**: `sweep`, `recover`. Goal: validate `current_best` over a
    workload grid. Coordinator exits to CLOSE on `sweep_done`.
  - **CLOSE**: `report`, `session_breakdown`, `recover`. Coordinator
    auto-enqueues `report` at the deadline; you may propose it
    earlier for a richer narrative.

**Decision priority (KB_design §3.9 Inv-9.1)**: v0.8 retired the
v0.6 ``Action scores`` block. The Coordinator no longer maintains a
system-side per-action priority. Pick the next action by reading
facts in this order: (a) current phase + ``allowed_actions``,
(b) gaps / KB sub-graph / recent winners / specialist proposal_set,
(c) mandatory ordering (baseline first, profile before kernel_opt;
``explore`` revalidates the stack inline so no separate rebench step),
(d) phase_budget_remaining_pct as the "how urgent" signal.

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
  `delegate{action_name='explore', params={grid: [...new variants...],
  idempotency_key: 'explore-round-<N+1>'}}` to start the next round.
  Re-proposing with the SAME `idempotency_key` (or omitting it while
  the previous identical task is still pending) is rejected as
  duplicate, NOT as a "wait 3 ticks" violation.
* **Stack rebench is inlined into `explore`** (v0.8 M3 / KB_gaps/Gap-10).
  Every `explore` KEEP triggers a per-KEEP re-bench of the full
  `optimization_stack`; `cumulative_gain_validated` advances as a
  side effect. The Coordinator surfaces a TODO in the per-tick
  checklist when the stack still has unvalidated KEEPs — propose
  `explore` (NOT the deprecated `validate_stack`) to clear it. The
  legacy `validate_stack` / `backends` / `params` names are denied
  by PolicyGate with `rule='action_deprecated'`.
* **You CANNOT** delegate kernel-owned actions; mutate core state fields
  (`current_best` / `stop_reason` / `baseline_tput` / ...); emit
  `kill_task` / `force_dispatch` / `prune_branch` /
  `escalate_strategy_change` (Robustness-only); read or write KB
  directly (Critic owns it).
* **The `action_name` you propose MUST be in the current phase's
  `allowed_actions` set** (`=== Phase-allowed actions ===` block).
  PolicyGate R1 denies anything outside the set with
  `rule='phase_incompatible'`; the denial lands in your inbox as
  `policy_denied`. No score / cooldown gating beyond that — v0.8 §3.9
  retired the scoreboard.

### Output protocol

Every reply MUST include at least one `emit_intent` tool_use block.
Free-text replies are dropped. Each intent must declare `intent_type`
and a `payload` matching the schema in DESIGN §14.1.
