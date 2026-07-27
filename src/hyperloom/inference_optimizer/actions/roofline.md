# `roofline` Action — Reference

## What this is

`roofline` is a **Coordinator-internal composite action**. Its executor
internally invokes two atomic sub-steps in order:

1. **`profile`** — reuses `ProfileExecutor` to run Magpie + torch
   profiler against the running server and produce a `trace_dir`
   (and downstream `last_profile_trace`).
2. **`trace_analyze`** — invokes the kernel-agent `tracelens_analysis.py`
   tool which runs TraceLens internally (`trace_split` →
   `kernel_candidates` → write `analysis.md` + `summary.json`).

After both sub-steps succeed, `SharedState` carries:

* `last_profile_trace` — path to the merged trace JSON
* `last_trace_analyze.steady_state_trace` — the steady-state chunk selected by
  TraceLens after its normal mode selection / retry policy
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

## Who enqueues it

**The Coordinator does — never the LLM.** A roofline (or its
`--no-enable-roofline` profile-only alternative) is auto-enqueued for:

1. **PRELUDE bootstrap** — once after `baseline` lands, to seed the
   first `analysis.md`.
2. **+10% validated-gain watermark crossing** — whenever
   `cur_tput / last_roofline_tput >= 1.10`, where
   `cur_tput = baseline_tput * (1 + cumulative_gain_validated/100)`.
   Compound: 10% → 21% → 33% triggers.
3. **KERNEL runtime refresh** — before kernel work when validated throughput or
   the normalized model/workload/backend runtime context differs from the last
   profile.

Block-FP8 GEMM shape collection may invoke the same `RooflineExecutor` as an
isolated fallback when no matching steady-state trace exists. That nested run
uses a deep-copied `SharedState` and does not persist lifecycle/profile fields
to the parent session; only its selected steady-state trace is consumed.

While the Coordinator-enqueued analysis task is in flight, downstream
dispatches are no longer blocked: actions keep running against the
current `analysis.md` snapshot and resource conflicts (concurrent
profile / kernel work on the same GPU) are serialised by the lane /
GPU lease rather than a policy deny.

## PolicyGate denies LLM proposals

`propose_action{action_name='roofline'|'profile'}` and
`delegate{action_name='roofline'|'profile'}` are denied at PolicyGate
with `rule='phase_incompatible'`: both names are Coordinator-managed
and absent from `PHASE_LLM_PROPOSABLE_ACTIONS`, so the single LLM-
facing R1 phase rule rejects any LLM-side proposal. To run a profile
instead of a full roofline, the operator launches with
`--no-enable-roofline` (the Coordinator then auto-enqueues a `profile`
task in PRELUDE and at every watermark crossing); there is no
LLM-driven path.

## Failure semantics

Either sub-step failing causes the whole `roofline` task to fail:

* `profile` failure → `roofline.status=failed` with
  `error_class=profile_failed`; no `last_profile_trace` update.
* `trace_analyze` failure (after profile succeeded) →
  `roofline.status=failed` with `error_class=trace_analyze_failed`;
  `last_profile_trace` **is** updated (profile artifact is still
  valuable) but no `last_trace_analyze` cache.

Profile startup retries use the existing bounded Roofline retry policy.
TraceLens may also re-analyze the same trace once with an alternate
steady-state mode when its first selected chunk is missing, empty, or
low-quality. If those bounded retries fail, the roofline fails and does not
fall back to the raw, non-steady trace. A failed roofline does not block
downstream dispatch; later Coordinator refresh conditions may enqueue a fresh
analysis task.

## Cost / runtime

* `cost_minutes_p50=8` / `cost_minutes_p75=15` — dominated by profile
  (Magpie + torch profiler overhead) + trace_analyze (TraceLens
  subprocess); `trace_split` inside TraceLens adds < 30s.
* `requires_lanes=[profile_lane]` — same lane as `profile` so we
  don't run two profile-class tasks concurrently against the server.
