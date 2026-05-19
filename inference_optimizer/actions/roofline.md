# `roofline` Action — Playbook

## What this action does

`roofline` invokes a dedicated **roofline-analyzer sub-agent** that
reads the TraceLens `analysis.md` cached on
`SharedState.last_select_kernels.analysis_md_text` and produces a
structured decision dict so you (the main Orchestration LLM) can:

1. **`PRUNE_BRANCH`** action families that the report shows have no
   remaining ceiling (e.g. `kernel_opt` when compute is saturated and
   no `reusable_native_kernel` appears in Top Operations).
2. **`PROPOSE_ACTION`** the families and specific flags the report
   highlights as high-ceiling for the current dominant bottleneck.
3. Decide whether to **re-profile** (when the bottleneck distribution
   has likely shifted after recent optimisations).

The sub-agent does **not** auto-emit any intent — it only fills in
`SharedState.last_roofline_analysis` with structured suggestions. The
main Orchestration LLM still has to read the rendered "Roofline
Decision" prompt section (see C5) and emit the actual `PRUNE_BRANCH`
/ `PROPOSE_ACTION` intents.

## When to propose this action

Propose `roofline` immediately after a successful `select_kernels`,
specifically when **all** of these hold:

* `last_select_kernels.analysis_md_text != ""` (the TraceLens report
  is cached — sequence_denial enforces this).
* `last_roofline_analysis.snapshot_id !=
  last_select_kernels.roofline_snapshot_id` (no roofline analysis yet
  exists for the current snapshot — the executor short-circuits as
  `idempotency_hit=True` if you propose anyway, so re-proposing
  against the same snapshot is wasteful but harmless).
* You haven't just re-profiled and run `roofline` already in this
  tick — wait until the cached analysis is from the new snapshot.

The typical session sequence is therefore:

    baseline → profile → select_kernels → roofline → (optimization loop) →
    profile → select_kernels → roofline → (optimization loop) → report

## What the executor returns

A dict with the following keys (consumed by
`SharedState.record_roofline_analysis`, see C2 for the full schema):

* `primary_bottleneck` — one of `comm` / `compute` / `memory` /
  `latency` / `idle` / `unknown`
* `bottleneck_distribution` — per-category fraction
* `suggested_prunes` — list of `{family, reason, confidence}`
* `suggested_next_actions` — list of `{kind, rationale, priority}`
* `reprofile_recommended` + `reprofile_reason`
* `raw_llm_response` (truncated, forensic only)

When the sub-agent backend fails (timeout, malformed JSON, etc.) the
executor still returns `status="succeeded"` with `degraded=true` and
a safe fallback (`primary_bottleneck="unknown"`, empty suggestions)
so the prompt renderer (C5) can show "roofline analysis
unavailable for snapshot #N" rather than blocking the optimisation
loop.

## What to do with the output

Read the **Roofline Decision** section that C5 renders into your
prompt on subsequent ticks. Then:

* For each `suggested_prunes` entry with `confidence="high"` and
  whose `family` has had at least one failed attempt at this
  snapshot, emit `PRUNE_BRANCH` (you have this intent permission per
  C3).
* For each `suggested_next_actions` entry with `priority="high"`,
  prefer it over the static `action_scores` ranking when proposing
  the next `params` / `backends` / `comm_optimization` etc.
* If `reprofile_recommended=true`, emit `PROPOSE_ACTION{profile}` to
  refresh the snapshot.

Do **not** PRUNE_BRANCH a family the analyzer suggested if you
haven't tried it at this snapshot yet — the analyzer's prior is
report-based, but the live `params_search.tested` and validate-stack
records may surprise you.

## Cost / runtime

`cost_minutes_p50=1.0`, `cost_minutes_p75=2.0`. Sub-agent makes a
single LLM call (~15 KB input from `analysis.md`, ~1 KB JSON output)
and parses the response. No GPU work; no server restart.

## Sequence dependencies

* `baseline` and `profile` must have already run (inherited from the
  standard sequence gate).
* `select_kernels` must have populated `analysis_md_text` —
  `sequence_denial` rejects the propose otherwise (rule
  `execution_order`, hint asks you to run `select_kernels` first).
