# Coordinator Internals (launcher does not drive these)

Reference for the optimizer's internal phases and contracts. The launcher only
needs to know these exist; it never proposes or executes them. Read this when
debugging optimizer behavior or interpreting EXPLORE / FRAMEWORK_AGENT / KERNEL
decisions.

## IR-4 / IR-6 / IR-7 — EXPLORE phase contracts

These govern the optimizer's EXPLORE phase, not the launcher; the full contract
lives in `orchestrator/system_prompts/orchestration.md`. In brief:

- **IR-4 — EXPLORE is specialist-informed**: prefer specialist- or
  research-backed variants when available, but `llm_direct`, `default_grid`,
  `specialist:<domain-or-tag>`, and `dynamic` provenance values are all accepted
  audit labels when phase and sequence gates pass. Specialist- and
  dynamic-sourced variants are not grid-size capped; per-round breadth is
  bounded by the `research_lane` / GPU pool leases (the `research_lane` scales
  with the `2 × visible GPU count` ceiling). Specialists author patches into an
  isolated worktree; `integrate_patch` does the actual `git apply` +
  throughput/accuracy gate after Critic review. GPU specialists are on by
  default at whole-machine capacity (WS2); launch with
  `--gpu-specialist-capacity 0` (or
  `INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY=0`) to disable them. When
  enabled, Orchestration may dispatch
  `delegate{action_name='specialist', params={needs_gpu: true, gpu_count: ...}}`.
  GPU specialists serialize against serving through `gpu_research_lane` and
  exclusively own their leased cards: they may start/stop their own servers
  (any port that is not the production serving port 8888), profile, autotune,
  and run real benchmark loops. The one invariant is that they must not touch
  the production serving process, its cards, or port 8888.
- **IR-6 HARD force-exit**: EXPLORE exits the moment wall-clock remaining <
  `--explore-force-exit-hours-remaining` (default 3.0 h) OR phase budget <
  `--explore-force-exit-budget-pct` (default 20%). Non-negotiable — leaves
  buffer for KERNEL_AGENT → SWEEP → CLOSE + report.
- **Plateau advisory**: EXPLORE / KERNEL_AGENT / FRAMEWORK plateau signals are
  computed every tick and rendered as advisory in the orchestration prompt. They
  do NOT drive phase advance — the LLM may emit
  `escalate_strategy_change{hint='skip_to_kernel'/'skip_to_sweep'/'skip_to_close'}`
  when it judges further effort unproductive. IR-6 force-exit and the per-phase
  budget remain the only hard advance gates.

## FRAMEWORK_AGENT phase

Inserted between PRELUDE and EXPLORE (`--no-framework-agent` opts out). The
Coordinator owns the loop end-to-end — the LLM never proposes the `framework`
action. It discovers a candidate batch **once** via `fa phase-discover`; each
exploration then processes exactly **one** candidate, with the agent ranking the
still-available candidates and choosing the one most likely to raise throughput
(LLM ranker; deterministic discovery-order fallback on any failure). The chosen
candidate is Critic-gated, then `git apply`d against the live
framework_source_roots and benchmarked; KEEP commits to the live tree (next
candidate stacks on top), REVERT does `git reset --hard`. Exits on low budget
(<0.6 × max_hours), plateau (3 consecutive benchmarked candidate tests with no
KEEP), or an empty
discovery batch. Resume skips completed candidates by idempotency key. The
launcher only chooses whether the phase runs (`--no-framework-agent`).

## Retired modules and rules (do not re-introduce)

The live runtime uses `actions/_meta/*.yaml`, `_grid_runner.py`, and the unified
specialist-informed `explore` flow. Do not recreate the retired `backends` /
`params` / `validate_stack` / scoring modules.

Rules that look reasonable but break the current flow:

- **No `framework first-explore priority` rule** in
  `system_prompts/orchestration.md` — conflicts with the EXPLORE
  specialist-informed flow. Framework-agent runs in the dedicated
  **FRAMEWORK** phase before EXPLORE; the LLM never proposes the
  `framework` action — it is Coordinator-managed and absent from
  `PHASE_LLM_PROPOSABLE_ACTIONS`, so PolicyGate R1 denies any LLM-side propose /
  delegate with `rule='phase_incompatible'`. Use `--no-framework-agent` to skip the
  phase entirely.
- **`kernel_opt` sequencing** is no longer gated by an explore-minimum check
  (the `explore_attempts_minimum_before_kernel_opt` rule was retired in
  loosen_plan P1_06). KERNEL_AGENT phase may propose `kernel_opt` directly; the
  `trace_analyze → run_optimization` data dependency (P2_11 handler-level check)
  and the reusable `kernel_id` validation still keep the inputs valid.

## SGLang Parameter Search

Serving-parameter search runs through the `explore` action (the legacy `params`
/ `backends` actions were merged into it); candidates are written via
`EXTRA_SGLANG_ARGS` / `benchmark.envs`. This is internal to the optimizer — the
launcher does not drive it. Useful InferenceX-derived candidate families a
specialist may surface: `--disable-radix-cache`, `--max-running-requests`,
`--tokenizer-worker-num`, `--stream-interval`, and ROCm/TileLang envs
(`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`,
`SGLANG_HACK_FLASHMLA_BACKEND=tilelang`). Speculative decoding
(`SGLANG_ENABLE_SPEC_V2` / `--speculative-*`) is model-specific — only where a
draft/MTP path exists, benchmarked with chat-formatted prompts.

### Per-Run Asset Override (advanced)

To override shipped configs without editing them, materialize a per-run asset
root and pass `--asset-root`. `mkdir -p "$ASSET_ROOT/scripts/configs"`,
`ln -sfn` `actions/` / `kernel_opt/` / `orchestrator/` and the two
`scripts/ab_torch_compile_*.py` from `$REPO_ROOT/src/hyperloom/inference_optimizer/`, then
copy + edit the relevant `baseline_*.yaml` / `profile_*.yaml`. Reach for this
only when `_workload_envs.materialize_config_with_envs` defaults don't fit
(e.g. per-yaml `profiler.torch_profiler.enabled`); otherwise `--model` /
`--gpu-type` overrides are enough.

## Expected Flow

The optimizer should:

1. Establish or reuse `baseline_tput`.
2. **Coordinator** auto-enqueues an analysis task at the end of PRELUDE (after
   baseline) and at each validated-tput watermark (`current_tput /
   last_roofline_tput >= 1.10`; compound). Default is `roofline` (profile +
   trace_analyze + analysis.md); `--no-enable-roofline` switches to plain
   `profile`. The LLM cannot propose either — both names are Coordinator-managed
   and absent from `PHASE_LLM_PROPOSABLE_ACTIONS`, so PolicyGate R1 returns
   `rule='phase_incompatible'`. Concurrent GPU work is serialised by the lane /
   GPU lease rather than a policy deny, so explore / kernel dispatches keep
   flowing while analysis refreshes. Each analysis also stamps a decode roofline
   ceiling (`orchestrator/roofline_ceiling.py`) for the report's
   `## Roofline Comparison` section.
3. Run `trace_analyze` once per trace/config and cache the result in
   `last_trace_analyze`.
4. Pick only `reusable_native_kernel_ids` for `run_optimization`.
5. Require compile + correctness + microbench/E2E evidence before KEEP.
6. Use `explore_search` to test parameters incrementally and remember rejected
   candidates across resume. The ledger keys entries by **content fingerprint**
   (a sha1 hash of sorted `extra_server_args` + sorted `extra_envs`), so
   renaming an already-tested variant does not bypass dedup — LLM-supplied
   `params.grid` is filtered through the same ledger as the default seed grid.
7. Use `optimization_stack` so backend + params + kernel changes do not
   overwrite each other.
8. Use `sweep` to understand workload-specific results beyond the smoke
   workload.
