# `roofline` Action — Playbook

## What this is

`roofline` is a **composite action** (macro / pipeline). Its executor
internally invokes two atomic sub-steps in order:

1. **`profile`** — reuses `ProfileExecutor` to run Magpie + torch
   profiler against the running server and produce a `trace_dir`
   (and downstream `last_profile_trace`).
2. **`trace_analyze`** — invokes the kernel-agent `tracelens_analysis.py`
   tool (the request handler renamed from `select_kernels` in N1)
   which runs TraceLens internally (`trace_split` →
   `kernel_candidates` → write `analysis.md` + `summary.json`).

After both sub-steps succeed, `SharedState` carries:

* `last_profile_trace` — path to the merged trace JSON
* `last_trace_analyze.analysis_md_path` — path to TraceLens's
  human-readable Markdown report
* `last_trace_analyze.analysis_md_text` — full text of the report
  (cached for prompt injection)
* `last_trace_analyze.roofline_snapshot_id` — monotonically increasing
  counter so consumers (prompt renderer, re-profile guidance) can
  detect a fresh snapshot

The executor **does not invoke any LLM** and **does not produce a
structured `RooflineAnalysis` dict**. The Orchestration LLM consumes
the raw `analysis.md` from the prompt and makes decisions directly —
this is the TraceLens-team-mandated "no second interpretation"
contract (see design/roofline-v2.md §6.2).

## When to propose

Propose `roofline` (via `propose_action{action_name='roofline'}` or
`delegate{action_name='roofline'}`):

1. **Once after `baseline`** — to produce the very first TraceLens
   snapshot so `backends` / `params` / `kernel_opt` /
   `comm_optimization` can run (these four are gated on the cache
   by `_sequence_denial_for_action`).
2. **Whenever `cumulative_gain_validated_pct` has increased by ≥ 3%
   since the snapshot was taken** — bottleneck distribution likely
   shifted and the next round of decisions should be grounded in a
   refreshed `analysis.md`.
3. **When the previous snapshot's recommended next actions have all
   been tried with no new gain** — the report's signal is exhausted
   for this configuration, refresh.

Do **not** propose `roofline` when:

* `closing_phase` is near (< 15 minutes remaining) — the ~10 minute
  profile + trace_analyze cost would eat the closing window.
* You just emit `roofline` in the previous tick (idempotency: the
  Coordinator's task dedup will short-circuit a repeated propose
  with the same gain bucket; see §6.4 idempotency_key in the design
  doc).

## Failure semantics

Either sub-step failing causes the whole `roofline` task to fail:

* `profile` failure → `roofline.status=failed` with
  `error_class=profile_failed`; no `last_profile_trace` update.
* `trace_analyze` failure (after profile succeeded) →
  `roofline.status=failed` with `error_class=trace_analyze_failed`;
  `last_profile_trace` **is** updated (profile artifact is still
  valuable) but no `last_trace_analyze` cache.

There is no fallback — the executor returns a real failure and the
main Orchestration LLM should propose `roofline` again (or `profile`
+ `trace_analyze` separately as escape hatch) to retry.

## Cost / runtime

* `cost_minutes_p50=8` / `cost_minutes_p75=15` — dominated by profile
  (Magpie + torch profiler overhead) + trace_analyze (TraceLens
  subprocess); `trace_split` inside TraceLens adds < 30s.
* `requires_lanes=[profile_lane]` — same lane as `profile` so we
  don't run two profile-class tasks concurrently against the server.

## Why not just propose profile + select_kernels manually?

Pre-v2 the Orchestration LLM had to emit `propose_action{profile}` →
wait for completion → emit `request{kind="trace_analyze"}` → wait for
completion. That's 2 ticks of LLM-managed sequencing where a single
forgotten step strands `last_profile_trace` without a cache (or
worse, a cache against a stale trace). `roofline` removes the
sequencing burden from the LLM and guarantees the atomic snapshot.
