# Coordinator Internals (launcher does not drive these)

Reference for the optimizer's internal phases and contracts. The launcher only
needs to know these exist; it never proposes or executes them. Read this when
debugging optimizer behavior or interpreting FRAMEWORK_AGENT / KERNEL
decisions.

## FRAMEWORK_AGENT — the optimisation phase

One phase carries two arms: a **configuration** arm (server-arg / env grids,
sourced by specialist fan-out) and a **source** arm (upstream PRs and
specialist-authored patches). Rotation between them is the arms' own plateau
judgement, not a wall-clock split, and the phase leaves only when both report
dry. `--no-framework-agent` skips it entirely.

The source arm's supply is the `candidate_discovery_specialist`: it finds
upstream work, judges each entry against the live source, and returns a batch
already ordered and routed (`direct_framework` -> apply the diff,
`author_via_specialist` -> rewrite it against live source). The pump takes that
order as given. When discovery comes back empty its full retry budget, the
local-exploration arm authors against profile evidence instead; when that is
off too, the source arm reports itself dry.

`integrate_patch` lands every diff, config-only or source, with
`patch_source` naming where it came from. KEEP commits to the live tree (the
next candidate stacks on top); REVERT does `git reset --hard`.

## IR-4 — optimisation-phase contracts

These govern the optimizer, not the launcher; the full contract lives in
`orchestrator/prompts/orchestration.md`. In brief:

- **IR-4 — exploration is specialist-informed**: prefer specialist- or
  research-backed variants when available, but `llm_direct`, `default_grid`,
  `specialist:<domain-or-tag>`, and `dynamic` provenance values are all accepted
  audit labels when phase and sequence gates pass. Specialist- and
  dynamic-sourced variants are not grid-size capped; per-round breadth is
  bounded by the `research_lane` / GPU pool leases (the `research_lane` scales
  with the `2 × visible GPU count` ceiling). Specialists author patches into an
  isolated worktree; `integrate_patch` does the actual `git apply` +
  throughput/accuracy gate after Critic review. GPU specialists are on by
  default at whole-machine capacity (WS2); launch with
  `--gpu-specialist-capacity 0` to disable them, or pass an explicit positive
  capacity to clamp the pool. The legacy
  `INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY` env is ignored by the CLI
  default resolver. When enabled, Orchestration may dispatch
  `delegate{action_name='specialist', params={needs_gpu: true, gpu_count: ...}}`.
  GPU specialists serialize against serving through `gpu_research_lane` and
  exclusively own their leased cards: they may start/stop their own servers
  (any port that is not the production serving port 8888), profile, autotune,
  and run real benchmark loops. The one invariant is that they must not touch
  the production serving process, its cards, or port 8888.
- **Plateau advisory**: both arms' signals and KERNEL_AGENT's are computed every
  tick and rendered as advisory in the orchestration prompt. One arm dry is not
  a plateau; the prompt says so. They do NOT drive phase advance — the LLM may
  emit
  `escalate_strategy_change{hint='skip_to_kernel'/'skip_to_sweep'/'skip_to_close'}`
  when it judges further effort unproductive. The per-phase budget and the
  absolute cap remain the only hard advance gates.

## Retired modules and rules (do not re-introduce)

The live runtime uses `protocol/action_surfaces.ACTION_CATALOGUE`,
`_grid_runner.py`, and the unified specialist-informed `explore` flow. Do not
recreate the retired `backends` / `params` / `validate_stack` / scoring
modules, nor the `actions/_meta/*.yaml` catalogue and its loader.

Rules that look reasonable but break the current flow:

- **No `framework first` ordering rule** in `prompts/orchestration.md` — the
  two arms share one phase and rotate on their own plateau judgement, so a
  fixed priority between them conflicts with the specialist-informed flow.
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
root and `export INFERENCE_OPTIMIZER_ASSET_ROOT="$ASSET_ROOT"` (env var, not a
CLI flag; `asset_root()` raises `AssetRootNotFound` if the dir is missing).
`mkdir -p "$ASSET_ROOT/assets/configs"`, `ln -sfn` `actions/` from
`$REPO_ROOT/src/hyperloom/inference_optimizer/` and `orchestrator/` from
`$REPO_ROOT/src/hyperloom/`, then
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
   ceiling (`orchestrator/kernel/roofline_ceiling.py`) for the report's
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
