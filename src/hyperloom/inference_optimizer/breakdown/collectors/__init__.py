# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` / ``state`` /
``manifest`` returning its schema section (see :mod:`..schema`). Collectors never
mutate state, fabricate values, or raise — failures are recorded in ``warnings``
and the section returns a best-effort partial.

Split into cohesive submodules (shared foundation helpers in :mod:`._common`);
this package re-exports the full namespace so every importer and monkeypatch
site keeps working.
"""

from __future__ import annotations

from ._common import (
    _load_json_safe as _load_json_safe,
    _load_jsonl_safe as _load_jsonl_safe,
    _to_float as _to_float,
    _to_int as _to_int,
    _rel as _rel,
    _benchmark_report_metrics as _benchmark_report_metrics,
    _benchmark_report_candidates as _benchmark_report_candidates,
    _latest_benchmark_report as _latest_benchmark_report,
    _find_benchmark_report as _find_benchmark_report,
    _resolve_under_session as _resolve_under_session,
    _safe_get as _safe_get,
    _parse_iso_unix as _parse_iso_unix,
    _load_optimization_journal as _load_optimization_journal,
    _scan_profile_reports as _scan_profile_reports,
)
from .sessions import (
    log as log,
    _ENV_ALLOWLIST_EXACT as _ENV_ALLOWLIST_EXACT,
    _ENV_ALLOWLIST_PREFIXES as _ENV_ALLOWLIST_PREFIXES,
    _ENV_DENY_PATTERN as _ENV_DENY_PATTERN,
    _filter_envs as _filter_envs,
    _FRAMEWORK_ARGS_NON_DEFAULT_RE as _FRAMEWORK_ARGS_NON_DEFAULT_RE,
    _FRAMEWORK_ARGS_HEADER_RE as _FRAMEWORK_ARGS_HEADER_RE,
    _FRAMEWORK_ARGS_NAMESPACE_RE as _FRAMEWORK_ARGS_NAMESPACE_RE,
    _FRAMEWORK_ARGS_LAUNCH_RE as _FRAMEWORK_ARGS_LAUNCH_RE,
    _LOG_PREFIX_RE as _LOG_PREFIX_RE,
    _LOG_TIMESTAMP_RE as _LOG_TIMESTAMP_RE,
    _PYTHON_CMD_PREFIXES as _PYTHON_CMD_PREFIXES,
    _SERVER_LOG_MAX_BYTES as _SERVER_LOG_MAX_BYTES,
    _strip_log_prefix as _strip_log_prefix,
    _starts_with_python_prefix as _starts_with_python_prefix,
    _load_yaml_dict_safe as _load_yaml_dict_safe,
    _yaml_cmd_from_dict as _yaml_cmd_from_dict,
    _yaml_benchmark_synthesis as _yaml_benchmark_synthesis,
    _extract_framework_args as _extract_framework_args,
    _read_invocation_envs as _read_invocation_envs,
    _detect_image_for_session as _detect_image_for_session,
    _close_phase_stop_reason as _close_phase_stop_reason,
    _should_use_close_stop_reason as _should_use_close_stop_reason,
    _collect_recovery as _collect_recovery,
    collect_session as collect_session,
    collect_session_meta as collect_session_meta,
    session_elapsed_minutes as session_elapsed_minutes,
    collect_workload as collect_workload,
    collect_model_info as collect_model_info,
    collect_baseline as collect_baseline,
    _reconstruct_baseline_attempts as _reconstruct_baseline_attempts,
    collect_final as collect_final,
    collect_enablement as collect_enablement,
    _runtime_summary as _runtime_summary,
    _find_latest_validate_stack_report as _find_latest_validate_stack_report,
    _find_current_best_report as _find_current_best_report,
    _find_stack_top_report as _find_stack_top_report,
    _find_matching_action_report as _find_matching_action_report,
    _build_final_invocation as _build_final_invocation,
)
from .timeline import (
    _AUDIT_ACTIONS as _AUDIT_ACTIONS,
    _journal_entry_to_event as _journal_entry_to_event,
    TimelineDedup as TimelineDedup,
    collect_phase_timeline as collect_phase_timeline,
    _capability_for_action as _capability_for_action,
    collect_capability_summary as collect_capability_summary,
    geak_route_evidence as geak_route_evidence,
    _specialist_capability_row as _specialist_capability_row,
    _empty_by_specialist_capability as _empty_by_specialist_capability,
    _SPECIALIST_DOMAIN_KEYS as _SPECIALIST_DOMAIN_KEYS,
    collect_phase_segments as collect_phase_segments,
)
from .kernels import (
    _kernel_agent_run_dirs as _kernel_agent_run_dirs,
    _parse_invocation_attempt as _parse_invocation_attempt,
    _stamp_kernel_level_decisions as _stamp_kernel_level_decisions,
    _shape_kernel_metadata as _shape_kernel_metadata,
    _infer_run_dir_kernel_id as _infer_run_dir_kernel_id,
    collect_kernel_invocations as collect_kernel_invocations,
    _read_kernel_candidates as _read_kernel_candidates,
    _index_invocations_by_kernel as _index_invocations_by_kernel,
    _collect_detected_kernels as _collect_detected_kernels,
    _collect_recommended_kernels as _collect_recommended_kernels,
    _collect_optimized_kernels as _collect_optimized_kernels,
    _collect_adopted_kernels as _collect_adopted_kernels,
    _collect_rejected_kernels as _collect_rejected_kernels,
    collect_kernel_lifecycle as collect_kernel_lifecycle,
    _KERNEL_OPT_SUMMARY_REL_PATH as _KERNEL_OPT_SUMMARY_REL_PATH,
    collect_kernel_optimization_summary as collect_kernel_optimization_summary,
    _CONC_SWEEP_SUMMARY_REL_PATH as _CONC_SWEEP_SUMMARY_REL_PATH,
    collect_conc_sweep_summary as collect_conc_sweep_summary,
    collect_optimization_stack as collect_optimization_stack,
    _normalize_optimization_stack_entry as _normalize_optimization_stack_entry,
    _resolve_gemm_engine as _resolve_gemm_engine,
    collect_gemm_tuning as collect_gemm_tuning,
    collect_collective as collect_collective,
    collect_source_files as collect_source_files,
)
from .roofline import (
    collect_roofline as collect_roofline,
    _KERNEL_ROOFLINE_REL_PATH as _KERNEL_ROOFLINE_REL_PATH,
    DEFAULT_ROOFLINE_TARGET_RATIO as DEFAULT_ROOFLINE_TARGET_RATIO,
    collect_roofline_progress as collect_roofline_progress,
    _normalize_roofline_snapshot as _normalize_roofline_snapshot,
    collect_kernel_roofline as collect_kernel_roofline,
    _normalize_kernel_roofline_entry as _normalize_kernel_roofline_entry,
)
from .explore import (
    _shape_ledger as _shape_ledger,
    _shape_winners_history as _shape_winners_history,
    collect_explore_search as collect_explore_search,
    _VARIANT_NAME_RE as _VARIANT_NAME_RE,
    _scan_sweep_variants as _scan_sweep_variants,
    _shape_sweep_point as _shape_sweep_point,
    collect_sweep as collect_sweep,
)
from .attribution import (
    _normalize_specialist_key as _normalize_specialist_key,
    _action_family as _action_family,
    _promote_legacy_gain_entries as _promote_legacy_gain_entries,
    collect_attribution as collect_attribution,
    _collect_phase_breakdown as _collect_phase_breakdown,
    _reconstruct_gain_ledger as _reconstruct_gain_ledger,
)
from .optimizations import (
    OPTIMIZATIONS_SCHEMA_VERSION as OPTIMIZATIONS_SCHEMA_VERSION,
    collect_recorded_optimizations as collect_recorded_optimizations,
)
from .decision import (
    _TOKEN_IN_KEY as _TOKEN_IN_KEY,
    _TOKEN_OUT_KEY as _TOKEN_OUT_KEY,
    _TOKEN_CACHE_CREATE_KEY as _TOKEN_CACHE_CREATE_KEY,
    _TOKEN_CACHE_READ_KEY as _TOKEN_CACHE_READ_KEY,
    _TOKEN_KEYS_ALL as _TOKEN_KEYS_ALL,
    _coerce_token as _coerce_token,
    _empty_token_bucket as _empty_token_bucket,
    _fold_call_into_bucket as _fold_call_into_bucket,
    _load_llm_calls as _load_llm_calls,
    _load_proposal_task_map as _load_proposal_task_map,
    _attribute_critic_calls as _attribute_critic_calls,
    _load_dispatch_history_all as _load_dispatch_history_all,
    _build_phase_windows as _build_phase_windows,
    _phase_at as _phase_at,
    _OVERHEAD_COMPONENTS as _OVERHEAD_COMPONENTS,
    _decision_key as _decision_key,
    _token_convenience as _token_convenience,
    collect_token_usage as collect_token_usage,
    collect_langfuse as collect_langfuse,
    _proposal_scores_by_variant as _proposal_scores_by_variant,
    collect_decision_trace as collect_decision_trace,
    _write_decision_trace_jsonl as _write_decision_trace_jsonl,
)
from .telemetry import (
    collect_critic_robustness as collect_critic_robustness,
    _critic_kb_writes_summary as _critic_kb_writes_summary,
    _scan_all_benchmark_reports as _scan_all_benchmark_reports,
    _scan_run_dirs as _scan_run_dirs,
    _scan_server_logs as _scan_server_logs,
    _aggregate_gpu_monitor as _aggregate_gpu_monitor,
    _collect_lane_timeline as _collect_lane_timeline,
    collect_telemetry as collect_telemetry,
    collect_kb_provenance as collect_kb_provenance,
    _collect_flusher_status as _collect_flusher_status,
    _coerce_round_id as _coerce_round_id,
    collect_specialist_runs as collect_specialist_runs,
    _normalize_specialist_domain_breakdown as _normalize_specialist_domain_breakdown,
    _domain_for_task as _domain_for_task,
)
from .geak import (
    _geak_accepted_kernels_from_journey as _geak_accepted_kernels_from_journey,
    _geak_reconstruct_from_disk as _geak_reconstruct_from_disk,
    collect_geak as collect_geak,
)
from .v6 import (
    collect_v6_metadata as collect_v6_metadata,
    collect_v6_outcome as collect_v6_outcome,
    collect_v6_timeline as collect_v6_timeline,
)
from .v6_close import collect_v6_close as collect_v6_close
from .v6_stages import (
    project_baseline_event as project_baseline_event,
    project_conc_sweep_event as project_conc_sweep_event,
    project_sweep_event as project_sweep_event,
)

__all__ = [
    "collect_attribution",
    "collect_baseline",
    "collect_capability_summary",
    "collect_critic_robustness",
    "collect_decision_trace",
    "collect_final",
    "collect_explore_search",
    "collect_kb_provenance",
    "collect_kernel_invocations",
    "collect_kernel_lifecycle",
    "collect_collective",
    "collect_geak",
    "collect_phase_segments",
    "TimelineDedup",
    "collect_phase_timeline",
    "collect_session",
    "collect_source_files",
    "collect_specialist_runs",
    "collect_sweep",
    "collect_telemetry",
    "collect_token_usage",
    "collect_v6_close",
    "collect_v6_metadata",
    "collect_v6_outcome",
    "collect_v6_timeline",
    "project_baseline_event",
    "project_conc_sweep_event",
    "project_sweep_event",
    "collect_workload",
    "collect_model_info",
    "collect_recorded_optimizations",
    "geak_route_evidence",
    "session_elapsed_minutes",
]
