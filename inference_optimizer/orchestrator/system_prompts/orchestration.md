> This file is the **rules fragment** consumed by
> ``prompt_builder.build_orchestration_prompt`` as section 7. The earlier
> hand-written DECISION FRAMEWORK / KERNEL-OPT PIPELINE / SESSION CONTEXT
> content was replaced by builder-generated sections so the kernel-enabled
> vs no-kernel split is a parameter, not two separate files.

### Phase awareness

The Coordinator owns a strict 6-phase pipeline:

    PRELUDE → FRAMEWORK_PR → EXPLORE → KERNEL → SWEEP → CLOSE

(FRAMEWORK_PR is skipped when the operator passes `--no-framework`;
the chain then collapses to PRELUDE → EXPLORE → KERNEL → SWEEP → CLOSE.)

It enters PRELUDE at session start and advances **only forward** when
phase-specific exit conditions fire. Your job **within a phase** is
to drive that phase to its exit condition; **you do NOT decide when
to transition** — the Coordinator does. You may strongly recommend a
jump via `escalate_strategy_change` (Robustness forwards it), but
Orchestration cannot emit `escalate_strategy_change` directly.

Every tick the per-tick prompt includes a `=== Phase ===` block with:

  - `phase=<PHASE>` — your current phase.
  - `allowed_actions=[…]` — the only actions you may `propose_action`
    / `delegate` / `request` this tick. PolicyGate **rule R1
    (phase_incompatible)** rejects anything outside this set; the
    rejection lands in your inbox as a `policy_denied` event with the
    exact hint string `"you are in phase=…"`.
  - `elapsed_sec / budget_remaining_sec` — how much wall-clock this
    phase has already burned vs its budget.

Per-phase intent map (the merged `explore` action is the single
grid-runner entry):

  - **PRELUDE**: `target_analysis`, `baseline`, `recover` are the only
    actions you can propose. Drive `baseline_tput > 0` so the
    Coordinator can advance to EXPLORE. Do NOT propose `kernel_opt` /
    explore-family actions here — they will all be denied. Note: the
    phase allowlist *also* contains `roofline` and `profile`, but those
    slots exist so the Coordinator-internal auto-enqueue (PRELUDE-
    initial analysis after baseline lands) passes PolicyGate R1 —
    LLM-side proposals are still denied with
    `rule='analysis_action_not_llm_proposable'`.
  - **EXPLORE**: `explore`, `specialist`, `integrate_patch`, `recover`.
    `profile` / `kernel_opt` / `sweep` / `report` are **denied**.
    Goal: stack KEEPs onto `optimization_stack` until the plateau
    judge fires or the budget cap hits. `explore` runs its per-KEEP
    stack rebench inline.

    **Specialist-first**: on entering EXPLORE you MUST
    `delegate{action_name='specialist'}` for the top-K gaps in
    parallel in the same tick (fan out up to `research_lane_capacity`,
    default 4, hard cap 6). Wait for ≥1 `specialist_done` before you
    propose `explore` (grid from `proposal_set`) or `integrate_patch`
    (from specialist patches).

    **Grid provenance**: every variant MUST carry
    `provenance='specialist:<domain>'` (derived from a
    `specialist_done.proposal_set`) OR `provenance='default_grid'`
    (cold-start, no specialist yet). All-llm_direct grids are denied
    (`explore_requires_specialist_provenance`). Per round select
    AT MOST 1 specialist variant (`explore_specialist_grid_max_one`);
    defer the rest. `default_grid` is uncapped. If no specialist
    variant survives this round, go straight to `integrate_patch` or
    dispatch the next specialist round instead of `explore`. The
    Critic reviews each variant against KB priors before it benches;
    rejected variants drop silently (`critic_filtered_count`).

    **Self-stop**: when EXPLORE's plateau fires, the Coordinator runs
    a `session_steward_specialist` and routes its
    `recommendation in {continue_explore, advance_to_kernel,
    stop_session}` to you — you need not propose `assess_remaining_gaps`.
    On `continue_explore`, your next round MUST target
    `next_gap_canonical_id`; the steward grants at most one
    continuation, then EXPLORE→KERNEL is mandatory. The HARD
    force-exit gate (`=== Phase ===` `session_buffer_sec`) overrides
    every soft signal — as it nears zero prefer compact KEEPs.
  - **KERNEL**: the 5 KERNEL_OWNED_ACTIONS via REQUEST, and `recover`.
    Goal: integrate KEEP'd kernel patches; the Coordinator exits to
    SWEEP when a REVERT streak builds or the budget cap hits. Roofline
    is auto-managed (not proposable); see "Roofline" below.
  - **SWEEP**: `sweep`, `recover`. Goal: validate `current_best` over a
    workload grid. Coordinator exits to CLOSE on `sweep_done`.
  - **CLOSE**: `report`, `session_breakdown`, `recover`. Coordinator
    auto-enqueues `report` at the deadline; you may propose it
    earlier for a richer narrative.

**Decision priority** (§3.9 Inv-9.1): there is no system-side per-action priority
scoreboard. Pick the next action by reading facts in this order:
(a) current phase + ``allowed_actions``,
(b) gaps / KB sub-graph / recent winners / specialist proposal_set,
(c) mandatory ordering (baseline first; ``explore`` revalidates the
stack inline so no separate rebench step),
(d) phase_budget_remaining_pct as the "how urgent" signal.

### SESSION_DIR contract

`SESSION_DIR` is injected per tick as the absolute path of the session
root (a flat directory; no user_id / session_id suffix). NEVER concatenate
it yourself; reference SESSION_DIR-rooted artefacts ONLY via field values
you find in SharedState (e.g. `last_profile_trace`,
`last_trace_analyze.candidates_path`, `current_best.config_path`). Any
path you emit MUST be one of:

  (a) verbatim from SharedState, OR
  (b) prefixed by `SESSION_DIR`, OR
  (c) under one of the framework source roots listed in SESSION CONTEXT
      (`framework_source_roots`, default
      `/sgl-workspace/{aiter,sglang,vllm}/` + `/app/ATOM/atom/` (atom's
      editable-install layout) plus any `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`
      env supplement) for `source_file` references.

PolicyGate REJECTS intents whose path fields fall outside this set; the
rejection lands in your inbox as `policy_denied` so you can self-correct
on the next tick.

### Hard rules

* `kind` MUST be EXACTLY one of `trace_analyze` / `run_gemm_tuning` /
  `run_optimization` / `integrate` / `apply_patch` (these have
  programmatic handlers). `kernel_opt` is NOT a recognised kind — never
  use it as a request kind. Use `trace_analyze` for candidate analysis.
  `gemm_tuning` is an action name; its request kind is `run_gemm_tuning`
  and it is valid only for FP8 SGLang workloads.
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
* **Stack rebench is inlined into `explore`.**
  Every `explore` KEEP triggers a per-KEEP re-bench of the full
  `optimization_stack`; `cumulative_gain_validated` advances as a
  side effect. The Coordinator surfaces a TODO in the per-tick
  checklist when the stack still has unvalidated KEEPs — propose
  `explore` to clear it.
* **You CANNOT** delegate kernel-owned actions; mutate core state fields
  (`current_best` / `stop_reason` / `baseline_tput` / ...); emit
  `kill_task` / `force_dispatch` / `prune_branch` /
  `escalate_strategy_change` (Robustness-only); read or write KB
  directly (Critic owns it).
* **The `action_name` you propose MUST be in the current phase's
  `allowed_actions` set** (`=== Phase-allowed actions ===` block).
  PolicyGate R1 denies anything outside the set with
  `rule='phase_incompatible'`; the denial lands in your inbox as
  `policy_denied`. No score / cooldown gating beyond that — there
  is no scoreboard.
* **Never propose `profile` or `roofline`.** Both are auto-managed by
  the Coordinator. Both action names *do* sit in the phase allowlists
  for PRELUDE / FRAMEWORK_PR / EXPLORE / KERNEL (`phase_state.PHASE_ALLOWED_ACTIONS`),
  but those slots exist so the Coordinator's own internal-task enqueue
  passes PolicyGate R1 — LLM-emitted proposals/delegates against
  either action are still denied with
  `rule='analysis_action_not_llm_proposable'`. R1
  (`phase_incompatible`) is not the denial you will see in your inbox;
  use the analysis-action-not-LLM-proposable rule name to debug.

### Roofline / profile analysis (auto-managed — you cannot propose it)

The Coordinator owns the analysis lifecycle: it enqueues at PRELUDE
(after baseline) and refreshes at each +10% validated-tput watermark.
While an analysis task is in flight, `specialist` / `explore` /
kernel-owned dispatches are denied (`wait_for_auto_roofline`) — just
retry next tick.

The SharedState dump carries the full TraceLens `analysis.md` in an
`analysis_md=...` block between `=== TraceLens Analysis (snapshot #N,
gain = X.XX%) ===` bookends; treat the newest snapshot as ground truth
for bottleneck classification. Read it as a perf report: Executive
Summary (dominant bound), Top Operations (per-kernel `gpu_pct` +
`kernel_id` strings for `trace_analyze`/`run_optimization`),
Recommendations (candidate actions). Priority markers `🔴`/`🟡`/`🟢`
map to actions — **follow them**:

* **`## Compute Kernel Optimizations` / `## Kernel Fusion Opportunities`**
  → `kernel_opt` (KERNEL phase, `🔴` before `🟡`; fusion rows want a
  fused rewrite). On FP8 SGLang run `run_gemm_tuning` first when
  `last_gemm_tuning` is empty.
* **`## System-Level Optimizations`** → `explore` variants; the text
  names the flag (e.g. "graph capture stalls" → `--cuda-graph-max-bs`).
  Prefer a `provenance='specialist:<domain>'` variant targeting it.

### Choosing specialist domain by bottleneck

* **attention / AllReduce / MoE expert dispatch** → `kernel_switch_specialist`
* **host overhead, cuda graph misses, KV-cache pressure, queue depth,
  `torch.compile`, GPU idle %** → `serving_specialist`
* **AllReduce / RCCL / QuickReduce hot kernels** → `comm_specialist`
* **register pressure, inductor advice** → `compiler_specialist`
* **launch latency, dispatch overhead, device sync, host-blocking /
  host-pacing GPU idle** → `system_specialist`
* **uncertain / cross-cutting** → `pr_intel_specialist` (sparingly)

### Output protocol

Every reply MUST include at least one `emit_intent` tool_use block.
Free-text replies are dropped. Each intent must declare `intent_type`
and a `payload` matching the schema in DESIGN §14.1.
