# SharedState Field Classification

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

Classify `SharedState` (`inference_optimizer/orchestrator/shared_state.py`) fields into the categories defined by Step 05, so later field removal is evidence-based and resume/breakdown safe.

This is a cleanup guide, not a second source of truth. Runtime source remains the dataclass and its load/migration path.

## Compatibility Decision Context

Per `01-baseline-contracts/03-compatibility-decisions.MD`:

- Old session resume: KEEP for this cleanup round.
- `extra_sglang_args` read alias: KEEP read-only; block new writers.

Therefore this step does NOT delete migration code or compat fields. It classifies fields and records deletion conditions.

## Core Runtime Fields

Required for the live end-to-end optimization loop:

- identity: `schema_version`, `session_id`, `claw_session_id`, `sandbox_user_id`
- workload: `model_name`, `model_path`, `model_class`, `model_arch`, `model_architectures`, `model_type`, `framework`, `framework_version`, `gpu_type`, `tp`, `ep`, `precision`, `conc`, `isl`, `osl`, `max_model_len`
- mode flags: `kernel_enabled`, `explore_enabled`, `framework_phase_enabled`, `enable_roofline`, `continue_kernel_after_gemm`, `conc_sweep_enabled`
- baseline / best: `baseline_tput`, `baseline_accuracy`, `baseline_runtime_sec`, `current_best`
- gain: `cumulative_gain`, `cumulative_gain_validated`, `cumulative_gain_validated_ts`, `cumulative_gain_validated_stack_len`, `optimization_stack`, `gain_per_stack_entry`
- phase: `phase`, `phase_started_ts`, `phase_started_unix`, `phase_history`, `phase_budget_pct`
- loop control: `current_action`, `crash_count`, `tick`, `max_minutes`, `start_ts`, `stop_reason`, `target_gap_pct`, `target_summary`
- close sequence: `closing_phase`, `closing_started_unix`, `closing_report_task_id`, `close_sequence_done`
- escalation plumbing: `pending_escalate_hint`, `last_consumed_escalate_hint`, `last_consumed_escalate_hint_ts`, `plateau_overrides`

These must not be removed without a behavior change decision.

## Action Ledger Fields

Per-action attempt/result history for live actions:

- explore: `explore_search`, `explore_attempts`, `last_explore`
- baseline/profile: `baseline_attempts`, `profile_attempts`, `last_baseline`, `last_profile`, `last_profile_trace`, `last_profile_status`, `last_profile_args`
- sweep: `sweep_attempts`, `last_sweep`, `last_conc_sweep`
- roofline: `roofline_attempts`, `roofline_snapshots`, `last_roofline`, `last_roofline_tput`, `last_trace_analyze`, `roofline_snapshot_id`, `roofline_failure_streak`, `roofline_baseline_gain_at_snapshot`
- kernel: `kernel_opt_attempts`, `last_kernel_opt`, `kernel_integrate_attempts`, `rejected_kernel_patches`, `rejected_kernel_ids`, `gemm_tuning_attempts`, `last_gemm_tuning`
- framework PR: `framework_pr_phase_progress`, `framework_pr_batches`, `framework_pr_phase_done`, `framework_pr_discover_failures`, `framework_pr_max_candidates`, `framework_pr_critic_decisions`
- specialist: `specialist_rounds`, `specialist_domain_empty_streak`, `last_specialist`, `specialist_patch_verdicts`, `explore_specialist_dispatched_count`
- dynamic action: `dynamic_actions`, `dynamic_action_round_count`
- steward / gaps: `gaps`, `last_remaining_gaps_assessment`, `remaining_gaps_assessments`, `steward_continuation_used`, `steward_infra_failures_by_round`, `depth_tracker`, `depth_gate_thresholds`
- research scout: `research_scout_runs`, `research_scout_seen_pr_ids`, `research_scout_last_round`, `research_scout_enabled`, `research_scout_interval`
- intervention mix: `intervention_mix`, `consecutive_config_only_rounds`
- discovered flags: `discovered_flags`, `discovered_flags_error`

Cleanup note: avoid keeping a separate ledger per historical implementation. `explore_search` is the canonical merged ledger.

## Contract Export / Reporting Fields

Mainly consumed by breakdown / report:

- `stack_fingerprint_meta`, `baseline_workload_extra`, `baseline_config_path`
- `policy_denial_history`, `policy_denial_streak`
- `last_action_failures`, `last_tick_exception`
- `pruned_families`

Cleanup note: prefer writing reporting-only detail into task results / reports rather than growing `state.json`.

## Knowledge / Warm-Start Fields

- `cortex_session_id`, `cortex_session_summary`
- `warm_start_recipe`, `warm_start_pitfalls`, `warm_start_lessons`, `warm_start_ts`
- `warm_replay_attempted`, `warm_replay_outcome`, `warm_history_injected`

These feed warm start and KB sediment; see Step 06.

## Compat Fields (read-only this round)

Retained only to migrate / fall back for old sessions:

- `params_winner_history` — folded into `explore_search` at load.
- `backend_winners_history` — folded into `explore_search` at load and
  still rendered read-only for old prompt summaries.
- `synergy_attempted` — folded into `explore_search` at load.
- `params_no_promote_streak` — read-only fallback for the legacy M2 plateau proxy when `explore_search` is empty.

Deletion condition: remove once old-session resume is dropped and the legacy plateau proxy is retired.

## Migration / Drop Sites (centralized)

- `_PHASE4_LEGACY_KEY_RENAMES` + `_migrate_legacy_extra_sglang_args_keys`: rename `extra_sglang_args` -> `extra_server_args` on load.
  - Deletion condition: remove once pre-rename state.json resume is unsupported.
- `from_dict._legacy_drop_fields`: drop retired scoreboard fields (`action_scores`, `score_violation`, `cooldown_until_tick`, `locked_reason`, `ucb_bonus`, `aging_bonus`, `score_mult`, `effective_score`, `last_action_score_snapshot`) and `last_select_kernels`.
  - Deletion condition: remove once pre-v0.8 scoreboard / select_kernels resume is unsupported.
- `from_dict` schema migration (`schema_version` v1 -> latest) and fact-layer integrity check.
  - Deletion condition: remove the v1 migration branch once the minimum supported state schema is raised.

Both rename-map and drop-list sites now carry an inline deletion-condition comment.

## Dead Field Candidates

No field was confirmed dead enough to remove safely in this round.

Reason:

- `SharedState` is read across coordinator, executors, prompt rendering, and breakdown collectors.
- Proving a field is unread/unwritten/unexported requires a wider read+write audit than this low-risk batch.

Future Step 05 batch should:

- build a field read/write/export map,
- distinguish runtime reads from reporting reads,
- stop new writes before removing old reads.

## Canonical New-Session Schema (intent)

A new session should only write canonical fields. Specifically:

- never write `extra_sglang_args` / `candidate_extra_sglang_args` (canonical: `extra_server_args` / `candidate_extra_server_args`),
- never write retired scoreboard fields,
- never write `last_select_kernels`.

Existing writer sites already follow this; the static guard in `tests/test_no_legacy_writer_sites.py` keeps the alias surface narrow.

Follow-up applied:

- `push_backend_winners_round()` was removed because it had no live
  runtime caller. New sessions should update `explore_search`, not the
  legacy `backend_winners_history` field.
