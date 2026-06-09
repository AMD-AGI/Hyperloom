# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SharedState — single-writer (Coordinator) persisted session state, backed by atomic JSON at ``$SESSION_DIR/state.json``; enforces CORE_STATE_FIELDS guards.

fields:

    session_id          str   — set by Coordinator at session creation
    model_name          str   — e.g. "meta-llama/Llama-3.1-8B-Instruct"
    model_path          str   — local NFS path to weights
    model_class         str   — categorical key supplied via --model-class
    model_arch          dict  — advisory architecture profile (hybrid
                                structured + free-text notes) loaded from
                                the launcher's ``$USER_DATA_PATH/model_arch.json``;
                                prompt-context only, no deterministic gating
    model_architectures list  — config.json ``architectures``; stamped into
                                the recipe-snapshot ``extras`` as a KB tag
    model_type          str   — config.json ``model_type``; stamped into
                                the recipe-snapshot ``extras`` as a KB tag
    target_summary      str   — set by `target_analysis` action
    baseline_tput       float — tok/s/GPU after `baseline` action
    baseline_accuracy   float — GSM8K score after `baseline`
    current_best        dict  — {action: str, tput: float, accuracy: float}
    cumulative_gain     float — % over baseline
    stop_reason         str   — set when graceful stop fires (§9)
    current_action      str   — what's running right now (set by Orchestration)
    crash_count         int   — incremented by Robustness on real failures
    pruned_families     list[str]  — set by Robustness via PRUNE_BRANCH
    start_ts            str   — ISO timestamp
    max_minutes         int   — wall-clock budget (0 = unlimited)
    last_profile_trace  str   — set by Coordinator when `profile` returns a
                                trace path; consumed by Orch to populate
                                `trace_analyze` REQUEST `trace_input` param
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# Ordered (key, label) projection for advisory ``model_arch``; empty/None keys dropped.
_MODEL_ARCH_STRUCTURED_FIELDS: tuple[tuple[str, str], ...] = (
    ("decoder_type", "decoder"),
    ("attention", "attention"),
    ("layer_mix", "layers"),
    ("kv_cache_per_token", "kv/token"),
    ("active_params", "params"),
    ("num_experts", "experts"),
    ("experts_per_tok", "experts/tok"),
    ("mtp", "mtp"),
    ("swa_window", "swa_window"),
    ("norm", "norm"),
)


def render_model_arch_compact(arch: dict | None) -> str:
    """Render the advisory ``model_arch`` profile as a single compact line (``""`` when empty/not a dict)."""
    if not isinstance(arch, dict) or not arch:
        return ""
    parts: list[str] = []
    for key, label in _MODEL_ARCH_STRUCTURED_FIELDS:
        val = arch.get(key)
        if val is None or val == "":
            continue
        parts.append(f"{label}={val}")
    notes = str(arch.get("notes") or "").strip()
    if notes:
        parts.append(f"notes={notes}")
    return "; ".join(parts)


# Default partial-attempt cap for run_optimization; override via env in ``record_kernel_opt`` (1 disables second chance).
_DEFAULT_KERNEL_OPT_MAX_PARTIAL = 2
# Backend ladder without a KEEP retires the kernel; override via ``INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES`` (>=1).
_DEFAULT_KERNEL_OPT_MAX_FAILURES = 1
# Hot-kernel report gate: reusable hot kernels >= this GPU share need a kernel_opt attempt/rejection before ``report``.
_DEFAULT_HOT_KERNEL_MIN_GPU_PCT = 3.0
# Only the top-N reusable hot kernels are enforced.
_DEFAULT_HOT_KERNEL_GATE_TOP_N = 5

# Per-action audit history cap (``<action>_attempts`` lists keep most recent N).
_DEFAULT_ATTEMPTS_HISTORY = 20

# Global ``last_action_failures`` rolling-log cap.
_DEFAULT_LAST_FAILURES = 10

# phase_history cap (record_phase_transition).
_PHASE_HISTORY_CAP = 100

# roofline_snapshots history cap (record_trace_analyze).
_ROOFLINE_SNAPSHOTS_CAP = 50

# gap ledger caps; both enforced in upsert_gap.
_GAPS_MAX_ENTRIES = 50
_GAPS_ATTEMPTS_HISTORY = 20

# Per-action audit trail kinds; kernel-owned actions excluded (dedicated structures).
_AUDIT_ACTIONS: frozenset[str] = frozenset({
    "baseline", "profile", "sweep", "explore",
    # ``roofline`` runs profile + trace_analyze atomically; audited for snapshot id/path visibility.
    "roofline",
})

# audit-action name -> (result-dict key, prompt-display label / key_metric_kind).
_KEY_METRIC_MAP: dict[str, tuple[str, str]] = {
    "baseline": ("output_throughput", "output_throughput"),
    "profile":  ("output_throughput", "output_throughput"),
    "sweep":    ("output_throughput", "output_throughput"),
    "explore":  ("best_gain_pct",     "gain_pct"),
    # ``roofline`` key metric is the monotonic snapshot id.
    "roofline": ("snapshot_id",       "snapshot_id"),
}


#: top-level state.json schema version; absent key treated as v1 and migrated to LATEST_STATE_SCHEMA_VERSION on first save.
LATEST_STATE_SCHEMA_VERSION: int = 2


# Phase4 compat (read-only): rename ``extra_sglang_args`` -> ``extra_server_args`` one-shot on load.
_PHASE4_LEGACY_KEY_RENAMES: dict[str, str] = {
    "extra_sglang_args":           "extra_server_args",
    "candidate_extra_sglang_args": "candidate_extra_server_args",
}


def _migrate_legacy_extra_sglang_args_keys(obj: Any) -> int:
    """Recursively rewrite legacy ``extra_sglang_args`` field names in-place; returns count rewritten (canonical kept when both present)."""
    migrated = 0
    if isinstance(obj, dict):
        for legacy_key, canonical_key in _PHASE4_LEGACY_KEY_RENAMES.items():
            if legacy_key in obj:
                if canonical_key not in obj:
                    obj[canonical_key] = obj.pop(legacy_key)
                else:
                    del obj[legacy_key]
                migrated += 1
        for v in obj.values():
            migrated += _migrate_legacy_extra_sglang_args_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            migrated += _migrate_legacy_extra_sglang_args_keys(item)
    return migrated


@dataclass
class TraceAnalyzeSnapshot:
    """Reference shape for ``SharedState.last_trace_analyze`` (``F0_pre_merge.MD`` §9); on-disk shape stays a plain dict for Inv-10.1."""

    trace_input: str = ""
    candidates_path: str = ""
    kernel_roofline_path: str = ""
    hot_kernels_top15: list[dict[str, Any]] = field(default_factory=list)
    kernel_roofline_top15: list[dict[str, Any]] = field(default_factory=list)
    task_groups: list[dict[str, Any]] = field(default_factory=list)
    reusable_native_kernel_ids: list[str] = field(default_factory=list)
    trace_health_warnings: list[dict[str, Any]] = field(default_factory=list)
    analysis_md_path: str = ""
    analysis_md_text: str = ""
    roofline_snapshot_id: int = 0
    roofline_baseline_gain_at_snapshot: float = 0.0
    ts: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "TraceAnalyzeSnapshot":
        d = d or {}
        return cls(
            trace_input=str(d.get("trace_input") or ""),
            candidates_path=str(d.get("candidates_path") or ""),
            kernel_roofline_path=str(d.get("kernel_roofline_path") or ""),
            hot_kernels_top15=list(d.get("hot_kernels_top15") or []),
            kernel_roofline_top15=list(d.get("kernel_roofline_top15") or []),
            task_groups=list(d.get("task_groups") or []),
            reusable_native_kernel_ids=list(
                d.get("reusable_native_kernel_ids") or []
            ),
            trace_health_warnings=list(d.get("trace_health_warnings") or []),
            analysis_md_path=str(d.get("analysis_md_path") or ""),
            analysis_md_text=str(d.get("analysis_md_text") or ""),
            roofline_snapshot_id=int(d.get("roofline_snapshot_id") or 0),
            roofline_baseline_gain_at_snapshot=float(
                d.get("roofline_baseline_gain_at_snapshot") or 0.0
            ),
            ts=str(d.get("ts") or ""),
        )


@dataclass
class SharedState:
    # versioned state.json schema; bumped by from_dict migration. Fresh sessions born at latest.
    schema_version: int = LATEST_STATE_SCHEMA_VERSION
    session_id: str = ""
    # Primus-Claw session UUID (empty standalone); joins Hyperloom to claw sessions in manifest/breakdown.
    claw_session_id: str = ""
    # Primus-Claw sandbox user id (empty standalone).
    sandbox_user_id: str = ""
    model_name: str = ""
    model_path: str = ""
    model_class: str = ""
    # Advisory architecture profile; prompt-context only, no deterministic gating (those stay on ``model_class``).
    model_arch: dict = field(default_factory=dict)
    # KB tags from config.json (``architectures`` + ``model_type``); stamped into recipe-snapshot ``extras`` so fine-tuned models carry base arch identity.
    model_architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    framework: str = ""
    gpu_type: str = ""
    # Workload metadata mirrored from manifest.json at session start; resume re-exports env vars. Avoids TP=1 default self-veto in _warm_specialist_params.
    tp: int = 0
    # Expert-parallel size for MoE; mirror of ``EP`` env var. Resume-safe so KB warm-start/recipe ``ep`` tag survive a fresh shell.
    ep: int = 0
    precision: str = ""
    # ``framework_version`` — only recipe-snapshot v2 canonical-id member not derivable from other fields; empty => ``unknown_version``.
    framework_version: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    kernel_enabled: bool = True
    # When False (``--no-explore``) EXPLORE is skipped: PRELUDE/FRAMEWORK_PR route to KERNEL (or SWEEP).
    explore_enabled: bool = True
    # After FP8 GEMM tuning succeeds, continue into source-level kernel_opt by default.
    continue_kernel_after_gemm: bool = True
    # SWEEP-phase post-sweep concurrency sweep; opt out via ``--no-enable-conc-sweep``.
    conc_sweep_enabled: bool = True
    # CONC ladder for conc_sweep (mirrors conc_sweep.DEFAULT_CONCS). Empty => skip_reason=empty_conc_list.
    conc_sweep_concs: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128],
    )
    # Total wall-clock budget (s) for conc_sweep. 0 disables the gate.
    conc_sweep_total_budget_sec: int = 9000
    # Per-variant Magpie subprocess timeout (s), clamped to remaining total budget.
    conc_sweep_variant_timeout_sec: int = 1800
    target_summary: str = ""
    baseline_tput: float = 0.0
    # Hot-server measure-round tput from the baseline cold-start double-run
    # (kept for reporting only). ``baseline_tput`` is the single-fresh-server
    # warmup-round number used as the fair comparison ANCHOR for explore /
    # sweep variants (which each restart the server + run one round). When
    # the double-run is disabled / ineligible, this stays 0.0 and
    # ``baseline_tput`` carries the only measured number.
    baseline_hot_tput: float = 0.0
    baseline_accuracy: float = 0.0
    baseline_failure_streak: int = 0
    # Baseline-materialized YAML path; injected downstream as ``config_path`` so variants inherit the contract.
    baseline_config_path: str = ""
    # Runtime component versions for recipe writes (framework/runtime/ROCm/aiter/image digest); empty values stripped.
    stack_fingerprint_meta: dict = field(default_factory=dict)
    # Extra workload-shape fields from baseline YAML; warm-start/lesson filters, not part of recipe canonical id.
    baseline_workload_extra: dict = field(default_factory=dict)
    # One-shot guard for PRELUDE warm-recipe replay (resume can't re-enqueue).
    warm_replay_attempted: bool = False
    # One-shot guard for injecting warm-recipe history into explore ledger.
    warm_history_injected: bool = False
    # Structured warm-replay outcome for reports/prompts (status reproduced|drift|failed|skipped, etc.).
    warm_replay_outcome: dict = field(default_factory=dict)
    # Baseline Magpie runtime (s, success path); ExploreExecutor derives overtime-kill deadline. Zero => no-op.
    baseline_runtime_sec: float = 0.0
    current_best: dict[str, Any] = field(default_factory=dict)
    # Full accepted configuration stack across action families; current_best keeps the materialized full args/env.
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    # Index-aligned with ``optimization_stack``: per-entry incremental gain pct; missing => None.
    gain_per_stack_entry: list[float | None] = field(default_factory=list)
    cumulative_gain: float = 0.0
    # Validated cumulative gain: re-baselined fresh server with every KEEP (per-round gains don't compose linearly); standalone validate_stack denied by PolicyGate.
    cumulative_gain_validated: float = 0.0
    cumulative_gain_validated_ts: str = ""
    # ``optimization_stack`` length at last successful inline rebench; longer => new KEEPs need validation.
    cumulative_gain_validated_stack_len: int = 0
    # Tput watermark for gain-driven roofline refresh; Coordinator re-enqueues at a compound 10% step.
    last_roofline_tput: float = 0.0
    stop_reason: str = ""
    # Closing phase — set when wall-clock deadline fires; Coordinator only drains a ``report`` task. Cleared on resume.
    closing_phase: bool = False
    closing_started_unix: float = 0.0
    closing_report_task_id: str = ""
    # True at END of CLOSE 5-step sequencer; cli.finally short-circuits emergency breakdown write. Resume clears it (idempotent).
    close_sequence_done: bool = False
    # Auto-roofline gate (EXPLORE-entry): pending roofline task_id; blocks first-round specialist dispatch until snapshot lands.
    auto_roofline_pending_task_id: str = ""
    current_action: str = ""
    crash_count: int = 0
    # Last Coordinator-side exception caught by the tick-loop guard (gives postmortems a traceback).
    last_tick_exception: dict[str, Any] = field(default_factory=dict)
    pruned_families: list[str] = field(default_factory=list)
    start_ts: str = field(default_factory=_now_iso)
    max_minutes: int = 0
    last_profile_trace: str = ""
    # ``succeeded``/``failed`` for most recent profile; failed allows re-run even when last_profile_trace is non-empty.
    last_profile_status: str = ""
    # Rolling log of PolicyGate denials (newest last, cap 50).
    policy_denial_history: list[dict[str, Any]] = field(default_factory=list)
    # Per-(action_name, rule) consecutive denial counter.
    policy_denial_streak: dict[str, int] = field(default_factory=dict)
    # Set when AST flag discovery cannot locate framework source files.
    discovered_flags_error: str = ""
    # Server EXTRA_SGLANG_ARGS in effect when last_profile_trace was captured; identical args means the same trace.
    last_profile_args: str = ""

    # Roofline-v2 trace-analyze cache; ``last_trace_analyze`` canonical 11-field dict from record_trace_analyze; ``roofline_snapshot_id`` mirrors nested value for hot-path access.
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    roofline_snapshot_id: int = 0
    # Append-only compact roofline snapshots for report.py (first baseline kept for before/after); capped at MAX_ROOFLINE_SNAPSHOTS.
    roofline_snapshots: list[dict[str, Any]] = field(default_factory=list)
    # Outer roofline failure counter; bumped on fail, reset on success.
    roofline_failure_streak: int = 0

    # Feature toggles (mirrored from ``cli.py`` flags at session start).
    # FRAMEWORK_PR phase toggle (PRELUDE → FRAMEWORK_PR → EXPLORE); ``--no-framework`` opts out. Absent from PHASE_LLM_PROPOSABLE_ACTIONS so PolicyGate R1 blocks LLM proposal.
    framework_phase_enabled: bool = True
    # FRAMEWORK_PR progress: one entry per candidate benchmark; used by breakdown + plateau exit judgment.
    framework_pr_phase_progress: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # One row per phase-discover batch; read by exit_normal_framework_pr plateau gate (3 batches <1% => exit).
    framework_pr_batches: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # True when FRAMEWORK_PR loop has no more candidates; compute_next_phase uses it for framework_pr_phase_done exit.
    framework_pr_phase_done: bool = False
    # Consecutive ``fa phase-discover`` failures; phase marked done only after DISCOVER_FAILURE_RETRY_LIMIT (default 3).
    framework_pr_discover_failures: int = 0
    # Per-repo candidate cap for ``fa phase-discover``; 0 => DEFAULT_FRAMEWORK_PR_MAX_CANDIDATES.
    framework_pr_max_candidates: int = 0
    # FRAMEWORK_PR Critic-gate decisions; cache lets resume avoid re-calling the Critic.
    framework_pr_critic_decisions: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # Default True: FRAMEWORK_PR pump dispatches a write-capable serving_specialist per candidate alongside diff-only track. False restores diff-only.
    framework_pr_authoring_enabled: bool = True
    # Default True: Coordinator auto-analysis is ``roofline`` (profile+trace_analyze+analysis.md); False enqueues plain ``profile``. Absent from PHASE_LLM_PROPOSABLE_ACTIONS (PolicyGate R1 denies LLM proposal).
    enable_roofline: bool = True
    # ExploreExecutor per-variant overtime kill multiplier; >0 and baseline_runtime_sec>0 kills past baseline_runtime_sec*ratio (outcome='KILLED_OVERTIME'). Stack-rebench exempt. Default +10%.
    explore_overtime_kill_ratio: float = 1.10
    # ExploreExecutor per-variant hard timeout override; 0 => auto-derive from baseline_runtime_sec*(kill_ratio+safety_margin).
    explore_variant_timeout_sec_override: int = 0
    # Headroom added to kill_ratio for auto-derived hard cap (default 0.5); no effect when override > 0.
    explore_variant_timeout_safety_margin: float = 0.5
    # Most recent workload sweep (CONC/ISL/OSL frontier).
    last_sweep: dict[str, Any] = field(default_factory=dict)
    # Mirrors last_sweep for the conc_sweep post-hook so SWEEP→CLOSE exits on conc_sweep completion.
    last_conc_sweep: dict[str, Any] = field(default_factory=dict)
    # Most recent run_optimization_done so Orch doesn't re-dispatch the same kernel_id every tick.
    last_kernel_opt: dict[str, Any] = field(default_factory=dict)
    # Per-action audit (kernel parity): each ``last_<action>`` is the most recent attempt snapshot; ``<action>_attempts`` is a capped list.
    last_baseline: dict[str, Any] = field(default_factory=dict)
    last_profile: dict[str, Any] = field(default_factory=dict)
    # GEAK FP8 GEMM tuning snapshot (kernel-owned): aiter A8W8 tuned CSV + SGLang dispatch patch before kernel_opt.
    last_gemm_tuning: dict[str, Any] = field(default_factory=dict)
    # merged explore action snapshot (same schema as other ``last_<action>`` mirrors).
    last_explore: dict[str, Any] = field(default_factory=dict)
    # Composite roofline action audit snapshot plus capped history.
    last_roofline: dict[str, Any] = field(default_factory=dict)
    baseline_attempts: list[dict[str, Any]] = field(default_factory=list)
    profile_attempts: list[dict[str, Any]] = field(default_factory=list)
    gemm_tuning_attempts: list[dict[str, Any]] = field(default_factory=list)
    sweep_attempts: list[dict[str, Any]] = field(default_factory=list)
    # explore audit log (capped per _DEFAULT_ATTEMPTS_HISTORY).
    explore_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Capped roofline audit log with snapshot ids and analysis paths.
    roofline_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Global rolling log of unpromotable task results (cap _DEFAULT_LAST_FAILURES); rich failure context for self-correction. Covers every task kind.
    last_action_failures: list[dict[str, Any]] = field(default_factory=list)
    # Per-kernel run_optimization history by kernel_id; record_kernel_opt retires kernels stuck in PARTIAL (default 2; override via INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL).
    kernel_opt_attempts: dict[str, Any] = field(default_factory=dict)
    # Cross-round params/backends/sweep aggregation (cap 10); _promote_to_shared_state detects a consistent sub-1%-per-shot winner across rounds.
    params_winner_history: list[dict[str, Any]] = field(default_factory=list)
    # Consecutive grid-runner tasks with no new current_best; Robustness nudges Orch off the plateau. Reset on advance.
    params_no_promote_streak: int = 0
    # Unified explore ledger — persistent DFS state for merged ``explore``; ``tested`` keyed by canonical_fingerprint. ``accepted`` includes inlined stack-rebench survivors; rebench-evicted entries live in rejected with reason='stack_unstable'.
    explore_search: dict[str, Any] = field(default_factory=dict)
    # specialist sub-agent rolling state; one entry per EXPLORE round (round_id, tasks, proposals_total/kept/rejected/skipped, etc.).
    specialist_rounds: list[dict[str, Any]] = field(default_factory=list)
    # Per-domain "empty proposal_set" streak; reset on non-empty specialist_done. Robustness escalates on persistent emptiness.
    specialist_domain_empty_streak: dict[str, int] = field(default_factory=dict)
    # Legacy session_steward slots (steward removed in P3_17); kept only for resume + report.py back-compat, never written.
    last_remaining_gaps_assessment: dict[str, Any] = field(default_factory=dict)
    remaining_gaps_assessments: list[dict[str, Any]] = field(default_factory=list)
    steward_continuation_used: bool = False
    steward_infra_failures_by_round: dict[str, int] = field(
        default_factory=dict,
    )
    # last specialist task snapshot (parity with other ``last_<action>`` mirrors).
    last_specialist: dict[str, Any] = field(default_factory=dict)
    # Per-specialist patch verdict ledger by task_id; Critic must approve/advise before PolicyGate allows the integrate_patch delegate.
    specialist_patch_verdicts: dict[str, str] = field(default_factory=dict)
    # Intervention-mix ledger ({change_type∈{config,code_patch}, action, task_id, ts, delta_pct}); Robustness detects config-only loops.
    intervention_mix: list[dict[str, Any]] = field(default_factory=list)
    # Current run of contiguous config KEEPs; resets when a code_patch KEEP lands.
    consecutive_config_only_rounds: int = 0
    # Research scout bookkeeping; master switch ``--no-research-scout``; seen_pr_ids shared with FRAMEWORK_PR to avoid re-mining.
    research_scout_enabled: bool = True
    research_scout_interval: int = 3
    # Master switch for advisory "External target gap" prompt block (``--no-target-advisory``); never gates Objective.
    target_advisory_enabled: bool = True
    # Master switch for sedimenting KEEP/REVERT provenance into the persistent recipe; off => recipe stays ephemeral.
    recipe_sediment_enabled: bool = True
    research_scout_runs: int = 0
    research_scout_seen_pr_ids: list[str] = field(default_factory=list)
    # Round id of last scout dispatch so K-round re-dispatch fires once per qualifying round.
    research_scout_last_round: int = -1
    # Total specialist dispatches in current EXPLORE entry; reset on fresh entry. Robustness detects specialist storms.
    explore_specialist_dispatched_count: int = 0
    # Research-lane capacity locked at session start (core field; PolicyGate denies mid-session mutation).
    research_lane_capacity: int = 1
    # GPU pool capacity for needs_gpu specialists (0 disables); locked at session start.
    gpu_specialist_capacity: int = 0
    # escalate_strategy_change carry-over: Coordinator writes validated next_action_hint here for compute_next_phase, then clears it once acted on.
    pending_escalate_hint: str = ""
    # last cleared escalate hint (audit only) for the breakdown.
    last_consumed_escalate_hint: str = ""
    last_consumed_escalate_hint_ts: str = ""
    # per-phase plateau threshold overrides locked at session start (CLI flags); empty => library defaults.
    plateau_overrides: dict[str, Any] = field(default_factory=dict)
    # E2E integrate bookkeeping keyed by kernel_id+patch_path+args; prevents re-validating the same patch after NEEDS_REVIEW/REVERT.
    kernel_integrate_attempts: dict[str, Any] = field(default_factory=dict)
    rejected_kernel_patches: list[dict[str, Any]] = field(default_factory=list)
    # Kernel ids with no remaining automated path (from REVERTs + exhausted integrate attempts).
    rejected_kernel_ids: list[str] = field(default_factory=list)

    # Search-space expansion ledger surfaced in the Orchestration prompt.
    discovered_flags: dict[str, Any] = field(default_factory=dict)
    # Rolling per-action winners log (cap 20) for dynamic idea generation.
    backend_winners_history: list[dict[str, Any]] = field(default_factory=list)
    # Synergy combo keys already tested this session; prevents re-running combinations.
    synergy_attempted: list[str] = field(default_factory=list)

    # Monotonic Coordinator tick counter; stable anchor for plateau/phase budget math.
    tick: int = 0
    # Remaining gain-pct target gap (0.0 => no target); fact for the "Mission progress" line, not a priority.
    target_gap_pct: float = 0.0

    # Phase state machine fields
    # ``phase`` — run-level pipeline phase (PRELUDE/FRAMEWORK_PR/EXPLORE/KERNEL/SWEEP/CLOSE); Coordinator-only (CORE_STATE_FIELDS). Empty => not yet initialised.
    phase: str = ""
    # ISO UTC timestamp the current phase was entered (breakdown.phase_segments + budget judge).
    phase_started_ts: str = ""
    # Unix epoch matching ``phase_started_ts`` so the budget judge skips ISO re-parsing.
    phase_started_unix: float = 0.0
    # Append-only log of phase transitions (rows from phase_state.make_history_row; reason in PHASE_EXIT_REASONS). Capped at _PHASE_HISTORY_CAP.
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    # Wall-clock budget percentages per phase (from CLI flags/defaults); persisted for resume. Empty => library defaults.
    phase_budget_pct: dict[str, float] = field(default_factory=dict)

    # Cortex KB integration fields — Coordinator-only writers (CORE_STATE_FIELDS; LLM update_state denied).
    # ``cortex_session_id`` — hyperloom-local id carried into KB fact-write attrs (source_session_id); defaults to session_dir.name.
    cortex_session_id: str = ""
    # Kept (always ``{}``) for resume back-compat; breakdown.kb_provenance.commit derived from drain_pending instead.
    cortex_session_summary: dict[str, Any] = field(default_factory=dict)
    # Snapshot of ``find-recipe`` output (parsed dict); empty on first session for a (workload, hw) pair.
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    # Snapshot of ``pitfalls`` output (negative priors), list of KB point dicts; consumed by specialist prompt § 5c. Resume tolerates older snapshots.
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 snapshot of ``lessons`` output (positive priors), symmetric with warm_start_pitfalls; consumed by specialist prompt § 5b. Empty under --degraded-kb or T0 failure.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    # ISO UTC timestamp of the T0 snapshot; empty under --degraded-kb or T0 failure.
    warm_start_ts: str = ""

    # structured gaps ledger: dedup'd unresolved bottlenecks (Coordinator-only _refresh_gaps; CORE_STATE_FIELDS); dedup keyed by canonical_id, attempts capped 20/gap, list capped _GAPS_MAX_ENTRIES.
    gaps: list[dict[str, Any]] = field(default_factory=list)

    # Orchestration working memory — durable compacted reasoning snapshot for compaction + crash-recovery rebuild; Coordinator-only writer, not in session_breakdown.
    orchestration_memory: dict[str, Any] = field(default_factory=dict)

    # Persistence
    @classmethod
    def state_path(cls, session_dir: Path) -> Path:
        return Path(session_dir) / "state.json"

    @classmethod
    def load_or_init(cls, session_dir: Path) -> "SharedState":
        """Load existing ``state.json`` or return a fresh blank instance."""
        path = cls.state_path(session_dir)
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SharedState":
        # Unified migration entry point; absent schema_version treated as 1. Idempotent (latest version short-circuits).
        incoming_version = int(raw.get("schema_version") or 1)
        needs_migration = incoming_version < LATEST_STATE_SCHEMA_VERSION
        migration_events: list[str] = []

        # Migrate ``extra_sglang_args`` -> ``extra_server_args`` (+ candidate_) across all nested ledgers; next save emits canonical only.
        legacy_migrations = _migrate_legacy_extra_sglang_args_keys(raw)
        if legacy_migrations:
            migration_events.append(
                f"extra_server_args rename: migrated {legacy_migrations} legacy "
                f"extra_sglang_args / candidate_extra_sglang_args key(s) "
                f"to extra_server_args / candidate_extra_server_args"
            )

        # Filter to known fields; unknown keys dropped, missing keys default.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        # Legacy scoreboard fields (read-only compat); already dropped by the filter, listed only to count/log in ``warn`` mode.
        _legacy_drop_fields = (
            "action_scores",
            "score_violation",
            "cooldown_until_tick",
            "locked_reason",
            "ucb_bonus",
            "aging_bonus",
            "score_mult",
            "effective_score",
            "last_action_score_snapshot",
            "last_select_kernels",
        )
        legacy_seen: list[tuple[str, int]] = []
        for legacy in _legacy_drop_fields:
            if legacy in raw:
                payload = raw.get(legacy)
                size = (
                    len(payload) if isinstance(payload, (dict, list, str))
                    else 1
                )
                legacy_seen.append((legacy, int(size)))
            filtered.pop(legacy, None)
        if legacy_seen:
            mode = os.environ.get(
                "INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", "drop",
            ).strip().lower()
            import logging as _logging
            log = _logging.getLogger(__name__)
            summary = ", ".join(f"{k}={n}" for k, n in legacy_seen)
            migration_events.append(
                f"§3.9 dropped scoreboard fields ({summary})"
            )
            if mode == "warn":
                log.warning(
                    "v0.8 §3.9: dropped legacy scoreboard fields from "
                    "state.json (%s). set "
                    "--legacy-action-scores=drop to silence this.",
                    summary,
                )
            else:
                log.info(
                    "v0.8 §3.9: dropped legacy scoreboard fields from "
                    "state.json (%s).", summary,
                )
        # Normalize the unified ``explore_search`` ledger at load; winners/synergy history folded in.
        filtered["explore_search"] = cls._build_explore_search(
            existing=filtered.get("explore_search"),
            backend_winners_history=filtered.get("backend_winners_history"),
            params_winner_history=filtered.get("params_winner_history"),
            synergy_attempted=filtered.get("synergy_attempted"),
        )

        # fact-layer integrity check (Inv-10.1): strict (default) aborts when a fact-layer key was present but didn't load; lenient warns.
        if needs_migration and raw:
            mode = os.environ.get(
                "INFERENCE_OPTIMIZER_MIGRATION_MODE", "strict",
            ).strip().lower()
            fact_layer_keys = (
                "baseline_tput", "baseline_accuracy", "current_best",
                "cumulative_gain", "cumulative_gain_validated",
                "optimization_stack",
                # Steward fields safe to default (missing => no priors).
                "last_remaining_gaps_assessment",
                "remaining_gaps_assessments",
            )
            missing: list[str] = []
            for key in fact_layer_keys:
                if key in raw and key not in filtered:
                    missing.append(key)
            if missing:
                import logging as _logging
                log = _logging.getLogger(__name__)
                fmt = (
                    "v0.8 §3.10: fact-layer field(s) %s present in "
                    "state.json but not loaded into SharedState "
                    "(Inv-10.1 violation)."
                )
                if mode == "lenient":
                    log.warning(
                        fmt + " --migration-mode=lenient → continuing.",
                        ", ".join(missing),
                    )
                else:
                    log.error(fmt, ", ".join(missing))
                    raise ValueError(
                        f"v0.8 §3.10 strict migration failed: fact-layer "
                        f"field(s) {missing!r} lost. Re-run with "
                        f"--migration-mode=lenient to continue."
                    )

        filtered["schema_version"] = LATEST_STATE_SCHEMA_VERSION

        # operator-visible migration summary so resume traces are self-describing.
        if needs_migration:
            import logging as _logging
            log = _logging.getLogger(__name__)
            event_str = "; ".join(migration_events) or "(no field changes)"
            log.info(
                "v0.8 §3.10: state.json migrated v%d → v%d. Events: %s",
                incoming_version, LATEST_STATE_SCHEMA_VERSION, event_str,
            )

        return cls(**filtered)

    @staticmethod
    def _build_explore_search(
        *,
        existing: Any,
        backend_winners_history: Any,
        params_winner_history: Any,
        synergy_attempted: Any,
    ) -> dict[str, Any]:
        """Shape the unified ``explore_search`` ledger at load time; folds live history so resume preserves cross-round aggregation."""
        from .action_executors._grid_runner import variant_fingerprint as _fp

        existing = existing if isinstance(existing, dict) else {}
        out: dict[str, Any] = dict(existing)
        out.setdefault("schema_version", 1)
        out.setdefault("tested", {})
        out.setdefault("accepted", [])
        out.setdefault("rejected", [])
        out.setdefault("discovered_flags", [])
        out.setdefault("domains_round_summary", [])
        out.setdefault("name_index", {})
        out.setdefault("cursor", len(out.get("tested") or {}))
        out.setdefault("last_round", {})

        # winners_history: fold live history + prior rows, sorted by (round_id, ts).
        wh: list[dict[str, Any]] = []
        for source_list in (
            backend_winners_history,
            params_winner_history,
            existing.get("winners_history") or [],
        ):
            if not isinstance(source_list, list):
                continue
            for entry in source_list:
                if not isinstance(entry, dict):
                    continue
                fp_val = entry.get("fingerprint") or _fp(
                    str(entry.get("extra_server_args") or ""),
                    dict(entry.get("extra_envs") or {}),
                )
                wh.append({
                    "round_id": str(entry.get("round_id") or ""),
                    "variant_name": str(entry.get("variant_name")
                                          or entry.get("name") or ""),
                    "fingerprint": str(fp_val),
                    "gain_pct": entry.get("gain_pct"),
                    "extra_args": str(entry.get("extra_args")
                                       or entry.get("extra_server_args") or ""),
                    "extra_envs": dict(entry.get("extra_envs") or {}),
                    "provenance": str(entry.get("provenance") or ""),
                    "ts": str(entry.get("ts") or ""),
                })
        wh.sort(key=lambda r: (str(r.get("round_id") or ""), str(r.get("ts") or "")))
        out["winners_history"] = wh

        # synergy_attempted: fold live field + executor-side additions, deduped.
        sa_set: set[tuple[str, ...]] = set()

        def _normalize_combo(c: Any) -> tuple[str, ...] | None:
            if isinstance(c, list):
                items = tuple(sorted(str(x) for x in c if isinstance(x, str)))
                return items if items else None
            if isinstance(c, str) and c.strip():
                parts = tuple(sorted(p for p in c.split("+") if p))
                return parts if parts else None
            return None

        for source in (synergy_attempted, existing.get("synergy_attempted") or []):
            if not isinstance(source, list):
                continue
            for c in source:
                norm = _normalize_combo(c)
                if norm:
                    sa_set.add(norm)
        out["synergy_attempted"] = [list(c) for c in sorted(sa_set)]
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, session_dir: Path) -> None:
        """Atomically write state.json (tmp + os.replace)."""
        path = self.state_path(session_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # Mutators (Coordinator only — LLM agents go via intents)
    def add_pruned_family(self, family: str) -> bool:
        """Idempotent add. Returns True iff the family was newly added."""
        if family in self.pruned_families:
            return False
        self.pruned_families.append(family)
        return True

    def is_pruned(self, family: str) -> bool:
        return family in self.pruned_families

    def prune_family(self, family: str) -> bool:
        """Alias for :meth:`add_pruned_family` (policy-loop stop-loss)."""
        return self.add_pruned_family(family)

    _POLICY_DENIAL_HISTORY_CAP = 50

    def record_policy_denial(
        self,
        *,
        action_name: str,
        rule: str,
        hint: str,
        intent_type: str,
        tick: int,
        intent_payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a denial row and bump per-(action,rule) streak."""
        key = f"{action_name or '*'}:{rule}"
        streak = int(self.policy_denial_streak.get(key, 0)) + 1
        self.policy_denial_streak[key] = streak
        entry = {
            "tick": int(tick),
            "action_name": action_name or "",
            "rule": rule,
            "hint": hint or "",
            "intent_type": intent_type,
            "streak": streak,
            "ts": _now_iso(),
        }
        if intent_payload:
            entry["intent_payload_keys"] = sorted(intent_payload.keys())
        history = list(self.policy_denial_history or [])
        history.append(entry)
        if len(history) > self._POLICY_DENIAL_HISTORY_CAP:
            history = history[-self._POLICY_DENIAL_HISTORY_CAP :]
        self.policy_denial_history = history
        return streak

    def reset_policy_denial_streak(self, action_name: str) -> None:
        if not action_name:
            return
        prefix = f"{action_name}:"
        self.policy_denial_streak = {
            k: v
            for k, v in (self.policy_denial_streak or {}).items()
            if not k.startswith(prefix)
        }

    # stop_reason ENUM validator
    def set_stop_reason(
        self,
        value: str,
        *,
        strict: bool | None = None,
    ) -> str:
        """Validated writer for :attr:`stop_reason` (Inv-8.3 closed vocab): values outside ``STOP_REASON_VOCAB`` map to ``"unknown"`` (lenient) or raise (``strict=True``, default env ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON``). Returns value written."""
        from .phase_state import STOP_REASON_VOCAB, is_valid_stop_reason
        text = str(value or "").strip()
        if not text:
            self.stop_reason = ""
            return ""
        if is_valid_stop_reason(text):
            self.stop_reason = text
            return text
        if strict is None:
            strict_env = os.environ.get(
                "INFERENCE_OPTIMIZER_STRICT_STOP_REASON", "",
            ).strip().lower()
            strict = strict_env in ("1", "true", "yes")
        if strict:
            raise ValueError(
                f"stop_reason={text!r} not in STOP_REASON_VOCAB "
                f"({sorted(STOP_REASON_VOCAB)!r})"
            )
        # Lenient: map to "unknown" and warn (original observable via warnings log).
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "stop_reason=%r not in STOP_REASON_VOCAB; mapped to 'unknown' "
            ". Set "
            "INFERENCE_OPTIMIZER_STRICT_STOP_REASON=1 to fail-fast.",
            text,
        )
        self.stop_reason = "unknown"
        return "unknown"

    # escalate hint plumbing
    def set_pending_escalate_hint(self, hint: str) -> str:
        """Stash the LLM-supplied hint for the next phase compute pass; unknown hints dropped (Inv-8.2: closed vocab). Returns value written."""
        from .phase_state import is_valid_escalate_hint
        text = str(hint or "").strip()
        if text and not is_valid_escalate_hint(text):
            return ""
        self.pending_escalate_hint = text
        return text

    def consume_pending_escalate_hint(self) -> str:
        """Pop the pending hint (recording consumption in audit fields) so the next tick doesn't re-trigger; returns cleared hint."""
        hint = (self.pending_escalate_hint or "").strip()
        if not hint:
            return ""
        self.pending_escalate_hint = ""
        self.last_consumed_escalate_hint = hint
        self.last_consumed_escalate_hint_ts = _now_iso()
        return hint

    # phase machine writer (Coordinator-only, single writer)
    def record_phase_transition(
        self,
        *,
        to_phase: str,
        reason: str,
        evidence: dict[str, Any] | None = None,
        ts: str | None = None,
        ts_unix: float | None = None,
    ) -> dict[str, Any]:
        """Append a phase_history row and atomically update ``phase`` fields; ``phase``/``phase_history`` are CORE_STATE_FIELDS so LLM update_state is rejected. Returns the inserted row."""
        from datetime import datetime as _dt, timezone as _tz
        import time as _time
        # Lazy import to avoid an import-time cycle.
        from .phase_state import make_history_row

        now_ts = ts or _dt.now(_tz.utc).isoformat(timespec="seconds")
        now_unix = float(ts_unix if ts_unix is not None else _time.time())
        row = make_history_row(
            from_phase=self.phase or "",
            to_phase=to_phase,
            reason=reason,
            evidence=evidence,
            ts=now_ts,
            ts_unix=now_unix,
        )
        history = list(self.phase_history or [])
        history.append(row)
        if len(history) > _PHASE_HISTORY_CAP:
            history = history[-_PHASE_HISTORY_CAP:]
        self.phase_history = history
        self.phase = row["to_phase"]
        self.phase_started_ts = now_ts
        self.phase_started_unix = now_unix
        return row

    def to_policy_denial_summary(self, *, top_k: int = 6) -> str:
        if not self.policy_denial_history:
            return ""
        rows = list(self.policy_denial_history)[-top_k:]
        lines = [
            "=== Recent policy denials "
            f"(newest last, total={len(self.policy_denial_history)}) ==="
        ]
        for r in rows:
            lines.append(
                f"  tick={r.get('tick')} action={r.get('action_name')!r} "
                f"rule={r.get('rule')!r} streak={r.get('streak')} "
                f"hint={str(r.get('hint') or '')[:140]!r}"
            )
        return "\n".join(lines)

    def increment_crash_count(self, by: int = 1) -> int:
        self.crash_count += by
        return self.crash_count

    def record_tick_exception(
        self,
        *,
        tick: int,
        stage: str,
        exc_type: str,
        message: str,
        traceback_text: str,
        agent: str = "",
    ) -> dict[str, Any]:
        """Persist a compact Coordinator exception summary for postmortems."""
        entry = {
            "tick": int(tick or 0),
            "ts": _now_iso(),
            "stage": str(stage or ""),
            "agent": str(agent or ""),
            "type": str(exc_type or ""),
            "message": str(message or "")[:1000],
            "traceback": str(traceback_text or "")[:12000],
        }
        self.last_tick_exception = entry
        return entry

    def apply_changes(self, changes: dict[str, Any], *, allow_core: bool) -> dict[str, Any]:
        """Merge a non-empty changes dict into this state; does NOT re-validate the role/source allowlist (PolicyGate filters upstream). Returns fields actually written."""
        if not changes:
            return {}
        applied: dict[str, Any] = {}
        for key, value in changes.items():
            if key not in self.__dataclass_fields__:
                continue
            setattr(self, key, value)
            applied[key] = value
        return applied

    def _format_last_kernel_opt(self) -> str:
        """Single-line repr of last kernel-opt outcome for prompt injection."""
        if not self.last_kernel_opt:
            return "(none)"
        ko = self.last_kernel_opt
        kid = str(ko.get("kernel_id") or "")
        attempts_entry = self.kernel_opt_attempts.get(kid) or {}
        history_tag = ""
        if attempts_entry:
            history_tag = (
                f" history=attempts={attempts_entry.get('attempts', 0)}"
                f"/partial={attempts_entry.get('partial_count', 0)}"
            )
            rej_reason = attempts_entry.get("rejected_reason")
            if rej_reason:
                history_tag += f"/retired={rej_reason}"
        return (
            f"kernel_id={kid or '?'} "
            f"decision={ko.get('decision','?')} "
            f"speedup={ko.get('micro_speedup','?')}"
            f"{history_tag}"
        )

    def _resolve_kernel_patch_identity(
        self, payload: dict[str, Any] | None,
    ) -> tuple[str, str, str, str]:
        payload = payload or {}
        kernel_id = str(payload.get("kernel_id") or "")
        patch_path = str(
            payload.get("patch_path")
            or payload.get("best_artifact_path")
            or ""
        )
        if (
            not patch_path
            and kernel_id
            and str((self.last_kernel_opt or {}).get("kernel_id") or "") == kernel_id
        ):
            patch_path = str(
                (self.last_kernel_opt or {}).get("best_artifact_path")
                or (self.last_kernel_opt or {}).get("patch_path")
                or ""
            )
        target_file = str(
            payload.get("target_file")
            or payload.get("source_file")
            or ""
        )
        # External envelope; route through compat helper so legacy ``extra_sglang_args`` still resolves.
        from ..compat.payload_aliases import read_extra_server_args
        extra_args = read_extra_server_args(payload).strip()
        return kernel_id, patch_path, target_file, extra_args

    def kernel_patch_key(self, payload: dict[str, Any] | None) -> str:
        kernel_id, patch_path, _target_file, extra_args = (
            self._resolve_kernel_patch_identity(payload)
        )
        if not kernel_id or not patch_path:
            return ""
        return "|".join([kernel_id, patch_path, extra_args])

    def find_rejected_kernel_patch(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        key = self.kernel_patch_key(payload)
        if not key:
            return None
        for entry in self.rejected_kernel_patches:
            if isinstance(entry, dict) and entry.get("key") == key:
                return entry
        return None

    def record_kernel_integrate_result(
        self,
        result: dict[str, Any],
        *,
        max_attempts: int = 3,
        keep_threshold_pct: float = 1.0,
    ) -> dict[str, Any] | None:
        """Persist one integrate E2E result and reject exhausted patch attempts."""
        if not isinstance(result, dict):
            return None
        key = self.kernel_patch_key(result)
        if not key:
            return None
        kernel_id, patch_path, target_file, extra_args = (
            self._resolve_kernel_patch_identity(result)
        )
        entry = dict(self.kernel_integrate_attempts.get(key) or {})
        attempts = list(entry.get("attempts") or [])
        attempt = {
            "decision": result.get("decision"),
            "status": result.get("status"),
            "new_tput": result.get("new_tput"),
            "gain_pct": result.get("gain_pct"),
            "workspace": result.get("workspace"),
            "report_path": result.get("report_path"),
            "ts": _now_iso(),
        }
        attempts.append(attempt)
        best_gain = max(
            (
                float(a.get("gain_pct"))
                for a in attempts
                if isinstance(a, dict) and isinstance(a.get("gain_pct"), (int, float))
            ),
            default=0.0,
        )
        entry.update({
            "key": key,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "target_file": target_file,
            "extra_server_args": extra_args,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "best_gain_pct": best_gain,
            "last_decision": result.get("decision"),
            "last_status": result.get("status"),
            "updated_at": _now_iso(),
        })
        self.kernel_integrate_attempts[key] = entry

        if result.get("decision") == "KEEP":
            return entry

        should_reject = (
            result.get("decision") == "REVERT"
            or len(attempts) >= max_attempts
        )
        if not should_reject:
            return entry

        reason = (
            "revert_decision"
            if result.get("decision") == "REVERT"
            else f"max_e2e_attempts_{max_attempts}_without_keep"
        )
        rejected = {
            "key": key,
            "kernel_id": kernel_id,
            "patch_path": patch_path,
            "target_file": target_file,
            "extra_server_args": extra_args,
            "attempt_count": len(attempts),
            "best_gain_pct": best_gain,
            "keep_threshold_pct": keep_threshold_pct,
            "last_decision": result.get("decision"),
            "reason": reason,
            "ts": _now_iso(),
        }
        self.rejected_kernel_patches = [
            r for r in self.rejected_kernel_patches
            if not (isinstance(r, dict) and r.get("key") == key)
        ]
        self.rejected_kernel_patches.append(rejected)
        if kernel_id and kernel_id not in self.rejected_kernel_ids:
            self.rejected_kernel_ids.append(kernel_id)
        entry["rejected"] = rejected
        self.kernel_integrate_attempts[key] = entry
        return entry

    def record_kernel_opt(self, result: dict[str, Any]) -> None:
        """Capture kernel_optimization_handler result for the next Orch turn; empty kernel_id no-op, non-KEEP can't overwrite a pending KEEP, retires kernel_id (r24 guard) after >= max_partial PARTIALs (INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL)."""
        if not isinstance(result, dict):
            return
        kernel_id = str(result.get("kernel_id") or "")
        if not kernel_id:
            # Metadata-less failure: preserve prior streaming-record KEEP.
            return

        verification = result.get("verification") or {}
        proposal = result.get("proposal") or {}
        decision = str(proposal.get("decision", "")).upper()
        micro_speedup = verification.get("micro_speedup", 0.0)
        try:
            micro_float = float(micro_speedup)
        except (TypeError, ValueError):
            micro_float = 0.0
        best_artifact_path = str(verification.get("best_artifact_path", "") or "")
        source_file = str(
            result.get("source_file")
            or (result.get("candidate") or {}).get("source_file")
            or ""
        )
        status = str(result.get("status") or "").lower()
        err_class = str(result.get("error_class") or "")
        # Pure infra failure = backend ladder with no verdict; kept distinct from REVERT/PARTIAL so retirement counters don't double-count.
        is_infra_failure = (
            decision == ""
            and (
                status in {"failed", "error", "timeout"}
                or err_class in {
                    "subtask_exception",
                    "handler_exception",
                    "subprocess_timeout",
                    "kernel_agent_root_missing",
                    "missing_integration_inputs",
                }
            )
        )
        ts = _now_iso()

        entry = dict(self.kernel_opt_attempts.get(kernel_id) or {})
        history = list(entry.get("history") or [])
        history.append({
            "decision": decision, "micro": micro_float,
            "status": status, "ts": ts,
        })
        history = history[-10:]
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        # Per-source attempts so a Python wrapper and its device file don't share a retry quota.
        per_source = dict(entry.get("attempts_per_source") or {})
        src_key = source_file or ""
        per_source[src_key] = int(per_source.get(src_key, 0)) + 1
        entry["attempts_per_source"] = per_source
        if decision == "PARTIAL":
            entry["partial_count"] = int(entry.get("partial_count", 0)) + 1
        elif decision == "KEEP":
            # Success resets streaks so a future regression isn't auto-retired on stale history.
            entry["partial_count"] = 0
            entry["failure_count"] = 0
        if is_infra_failure:
            entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
        entry["last_decision"] = decision
        entry["last_status"] = status
        entry["last_micro_speedup"] = micro_float
        entry["last_artifact_path"] = best_artifact_path
        entry["last_source_file"] = source_file
        entry["last_ts"] = ts
        entry["history"] = history

        # last_kernel_opt overwrite policy: KEEP always wins; non-KEEP writes only when no pending KEEP to protect.
        prev = self.last_kernel_opt or {}
        prev_decision = str(prev.get("decision", "")).upper()
        prev_kid = str(prev.get("kernel_id", ""))
        integrated_ids = self._kernel_ids_in_optimization_stack()
        prev_pending = (
            prev_decision == "KEEP"
            and bool(prev_kid)
            and prev_kid not in (self.rejected_kernel_ids or [])
            and prev_kid not in integrated_ids
        )
        if decision == "KEEP" or not prev_pending:
            self.last_kernel_opt = {
                "kernel_id": kernel_id,
                "decision": decision,
                "reasons": proposal.get("reasons", []),
                "micro_speedup": micro_float,
                "compile_passed": verification.get("compile_passed"),
                "correctness_passed": verification.get("correctness_passed"),
                "best_artifact_path": best_artifact_path,
                "source_file": source_file,
                "ts": ts,
            }

        max_partial = _DEFAULT_KERNEL_OPT_MAX_PARTIAL
        env_v = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL")
        if env_v:
            try:
                max_partial = max(1, int(env_v))
            except (TypeError, ValueError):
                pass

        # One backend ladder without a KEEP retires the kernel by default; raise threshold for flaky backends.
        max_failures = _DEFAULT_KERNEL_OPT_MAX_FAILURES
        env_f = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES")
        if env_f:
            try:
                max_failures = max(1, int(env_f))
            except (TypeError, ValueError):
                pass

        should_reject = (
            decision == "REVERT"
            or int(entry.get("partial_count", 0)) >= max_partial
            or int(entry.get("failure_count", 0)) >= max_failures
        )
        if should_reject:
            if kernel_id not in self.rejected_kernel_ids:
                self.rejected_kernel_ids.append(kernel_id)
            entry["rejected_reason"] = (
                "revert_decision"
                if decision == "REVERT"
                else (
                    f"max_partial_attempts_{max_partial}_without_keep"
                    if int(entry.get("partial_count", 0)) >= max_partial
                    else f"max_failures_{max_failures}_without_keep"
                )
            )

        self.kernel_opt_attempts[kernel_id] = entry

    def record_gemm_tuning(self, result: dict[str, Any]) -> None:
        """Capture the GEAK GEMM tuning result for sequencing and prompts."""
        if not isinstance(result, dict):
            result = {"status": "failed", "error": "non-dict gemm tuning result"}
        entry = dict(result)
        entry.setdefault("ts", _now_iso())
        self.last_gemm_tuning = entry
        attempts = list(self.gemm_tuning_attempts or [])
        attempts.append(entry)
        self.gemm_tuning_attempts = attempts[-_DEFAULT_ATTEMPTS_HISTORY:]

    # Multi-KEEP integrate queue helpers.
    def _kernel_ids_in_optimization_stack(self) -> set[str]:
        """kernel_ids already absorbed into optimization_stack as integrate entries."""
        return {
            str(e.get("kernel_id"))
            for e in (self.optimization_stack or [])
            if isinstance(e, dict)
            and e.get("action") == "integrate"
            and e.get("kernel_id")
        }

    def _source_files_in_optimization_stack(self) -> set[str]:
        """source_file paths already touched by an integrate entry; enforces "same source_file, only strongest KEEP integrated" (apply_kernel_patch is a whole-file overwrite)."""
        sources: set[str] = set()
        for e in (self.optimization_stack or []):
            if not isinstance(e, dict) or e.get("action") != "integrate":
                continue
            src = str(e.get("target_file") or e.get("source_file") or "")
            if src:
                sources.add(src)
        return sources

    def next_pending_keep_kernel_id(self) -> str:
        """Return next KEEP kernel_id awaiting integrate ("" if drained); excludes integrated/rejected/same-file-conflict KEEPs, picks highest ``last_micro_speedup`` first."""
        integrated_ids = self._kernel_ids_in_optimization_stack()
        integrated_sources = self._source_files_in_optimization_stack()
        rejected = set(self.rejected_kernel_ids or [])

        best_kid = ""
        best_micro = float("-inf")
        for kid, entry in (self.kernel_opt_attempts or {}).items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("last_decision", "")).upper() != "KEEP":
                continue
            if kid in integrated_ids or kid in rejected:
                continue
            src = str(entry.get("last_source_file") or "")
            if src and src in integrated_sources:
                # Same-file conflict: a stronger KEEP on this file was already integrated.
                continue
            try:
                micro = float(entry.get("last_micro_speedup") or 0.0)
            except (TypeError, ValueError):
                micro = 0.0
            if micro > best_micro:
                best_micro = micro
                best_kid = kid
        return best_kid

    def pending_keep_kernel_ids(self) -> list[str]:
        """All KEEP kernel_ids awaiting integrate, sorted strongest-first; surfaced in the prompt so the LLM doesn't propose ``report`` before draining them."""
        integrated_ids = self._kernel_ids_in_optimization_stack()
        integrated_sources = self._source_files_in_optimization_stack()
        rejected = set(self.rejected_kernel_ids or [])
        # Mirror next_pending_keep_kernel_id same-file guard: only strongest KEEP per source_file is queueable.
        claimed_sources: set[str] = set()
        ranked: list[tuple[float, str, str]] = []
        for kid, entry in (self.kernel_opt_attempts or {}).items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("last_decision", "")).upper() != "KEEP":
                continue
            if kid in integrated_ids or kid in rejected:
                continue
            src = str(entry.get("last_source_file") or "")
            if src and src in integrated_sources:
                continue
            try:
                micro = float(entry.get("last_micro_speedup") or 0.0)
            except (TypeError, ValueError):
                micro = 0.0
            ranked.append((micro, kid, src))
        ranked.sort(key=lambda x: x[0], reverse=True)
        result: list[str] = []
        for _micro, kid, src in ranked:
            if src and src in claimed_sources:
                continue
            if src:
                claimed_sources.add(src)
            result.append(kid)
        return result

    @property
    def has_keep_pending_integrate(self) -> bool:
        return bool(self.next_pending_keep_kernel_id())

    @property
    def kernel_opt_attempts_count(self) -> int:
        return len(self.kernel_opt_attempts or {})

    # Hot-kernel report gate: report blocked until meaningful reusable hot kernels are attempted/rejected.
    def untried_hot_reusable_kernels(
        self,
        *,
        min_gpu_pct: float | None = None,
        top_n: int | None = None,
    ) -> list[str]:
        """Hot kernels still owing a ``kernel_opt`` attempt (reusable, gpu_pct >= min_gpu_pct, untouched); capped to top_n by gpu_pct, one kernel_id per task_group."""
        info = self.last_trace_analyze or {}
        hot = info.get("hot_kernels_top15") or info.get("hot_kernels") or []
        task_groups = info.get("task_groups") or []
        if not isinstance(hot, list):
            return []

        if min_gpu_pct is None:
            try:
                min_gpu_pct = float(os.environ.get(
                    "HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT",
                    _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
                ))
            except (TypeError, ValueError):
                min_gpu_pct = _DEFAULT_HOT_KERNEL_MIN_GPU_PCT
        if top_n is None:
            try:
                top_n = int(os.environ.get(
                    "HYPERLOOM_KERNEL_OPT_GATE_TOP_N",
                    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
                ))
            except (TypeError, ValueError):
                top_n = _DEFAULT_HOT_KERNEL_GATE_TOP_N
        top_n = max(1, int(top_n))

        kid_to_group: dict[str, list[str]] = {}
        for g in task_groups:
            if not isinstance(g, dict):
                continue
            members = [str(m) for m in (g.get("kernel_ids") or []) if m]
            for m in members:
                kid_to_group[m] = members

        integrated_ids = self._kernel_ids_in_optimization_stack()
        integrated_sources = self._source_files_in_optimization_stack()
        rejected = set(self.rejected_kernel_ids or [])
        attempts = self.kernel_opt_attempts or {}

        # Sort by gpu_pct desc so dedup picks the strongest member of each task_group.
        rows: list[tuple[float, str, str, list[str]]] = []
        for k in hot:
            if not isinstance(k, dict):
                continue
            if k.get("reusable_native_kernel") is not True:
                continue
            try:
                gpu_pct = float(k.get("gpu_pct") or 0.0)
            except (TypeError, ValueError):
                gpu_pct = 0.0
            if gpu_pct < min_gpu_pct:
                continue
            kid = str(k.get("kernel_id") or "")
            if not kid:
                continue
            src = str(k.get("source_file") or "")
            members = sorted(kid_to_group.get(kid, [kid]))
            rows.append((gpu_pct, kid, src, members))
        rows.sort(key=lambda x: x[0], reverse=True)

        ranked: list[tuple[float, str, str, list[str]]] = []
        seen_groups: set[tuple[str, ...]] = set()
        for row in rows:
            group_key = tuple(row[3])
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            ranked.append(row)
        ranked = ranked[:top_n]

        untried: list[str] = []
        for _pct, kid, src, members in ranked:
            if members and all(m in rejected for m in members):
                continue
            if any(m in integrated_ids for m in members):
                continue
            if src and src in integrated_sources:
                continue
            if any(int((attempts.get(m) or {}).get("attempts", 0)) > 0
                   for m in members):
                continue
            untried.append(kid)
        return untried

    # Per-action audit (kernel parity for non-kernel actions)
    @staticmethod
    def _truncate_excerpt(value: Any, *, limit: int = 800) -> str | None:
        """Coerce ``value`` to str and trim to ``limit`` chars; None for falsy inputs (renderer shows ``err=(none)``)."""
        if value is None:
            return None
        text = str(value)
        if not text:
            return None
        if len(text) <= limit:
            return text
        return text[:limit]

    @staticmethod
    def _stderr_tail(value: Any, *, limit: int = 1000) -> str | None:
        """Pull the last ``limit`` chars from a subprocess error blob (stderr's actionable signal is at the end)."""
        if value is None:
            return None
        text = str(value)
        if not text:
            return None
        return text[-limit:] if len(text) > limit else text

    def record_action_attempt(
        self,
        action: str,
        *,
        task_id: str,
        status: str,
        decision: str,
        result: dict[str, Any] | None,
        extras: dict[str, Any] | None = None,
        max_history: int = _DEFAULT_ATTEMPTS_HISTORY,
    ) -> dict[str, Any] | None:
        """Append one attempt to ``<action>_attempts`` and refresh ``last_<action>``. Returns the entry, or None when ``action`` not in the audit set. Does NOT call :meth:`save`."""
        if action not in _AUDIT_ACTIONS:
            return None
        attempts_attr = f"{action}_attempts"
        last_attr = f"last_{action}"
        if not hasattr(self, attempts_attr) or not hasattr(self, last_attr):
            return None
        result = result or {}
        metric_key, metric_kind = _KEY_METRIC_MAP.get(
            action, ("output_throughput", "output_throughput"),
        )
        raw_metric = result.get(metric_key)
        try:
            key_metric: float | None = (
                float(raw_metric) if isinstance(raw_metric, (int, float))
                else None
            )
        except (TypeError, ValueError):
            key_metric = None
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "task_id": str(task_id or ""),
            "status": str(status or ""),
            "decision": str(decision or ""),
            "key_metric": key_metric,
            "key_metric_kind": metric_kind,
            "workspace": (
                str(result.get("workspace"))
                if result.get("workspace") else None
            ),
            "error_class": (
                str(result.get("error_class"))
                if result.get("error_class") else None
            ),
            "error_excerpt": self._truncate_excerpt(result.get("error")),
            "raw_result_path": (
                str(result.get("raw_result_path"))
                if result.get("raw_result_path") else None
            ),
            "reported_success": result.get("reported_success"),
            "extras": dict(extras or {}),
        }
        history: list[dict[str, Any]] = list(getattr(self, attempts_attr) or [])
        history.append(entry)
        if len(history) > max_history:
            history = history[-max_history:]
        setattr(self, attempts_attr, history)
        setattr(self, last_attr, dict(entry))
        return entry

    def record_action_failure(
        self,
        *,
        action: str,
        task_id: str,
        result: dict[str, Any] | None,
        max_history: int = _DEFAULT_LAST_FAILURES,
    ) -> dict[str, Any]:
        """Append one rich failure record to :attr:`last_action_failures` for self-correction; invoked for EVERY unpromotable task kind, unlike :meth:`record_action_attempt`."""
        result = result or {}
        error_class = result.get("error_class")
        error_class_str = str(error_class) if error_class else None
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "action": str(action or ""),
            "task_id": str(task_id or ""),
            "error_class": error_class_str,
            "error_excerpt": self._truncate_excerpt(result.get("error")),
            "stderr_tail": (
                self._stderr_tail(result.get("error"))
                if error_class_str in {"subprocess_nonzero", "timeout"}
                else None
            ),
            "workspace": (
                str(result.get("workspace"))
                if result.get("workspace") else None
            ),
            "raw_result_path": (
                str(result.get("raw_result_path"))
                if result.get("raw_result_path") else None
            ),
            "reported_success": result.get("reported_success"),
        }
        history = list(self.last_action_failures or [])
        history.append(entry)
        if len(history) > max_history:
            history = history[-max_history:]
        self.last_action_failures = history
        return entry

    def record_trace_analyze(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Write the canonical 11-field ``last_trace_analyze`` dict (single writer). ``roofline_snapshot_id`` increments monotonically; PR #321 retired ``last_trace_analyze_baseline`` (roofline_snapshots feeds report.py Roofline Comparison)."""
        if not isinstance(result, dict):
            return
        trace_input = (
            (payload or {}).get("trace_input")
            or (payload or {}).get("trace_dir")
            or ""
        )
        candidates_path = result.get("candidates_path") or ""
        if not candidates_path:
            artifacts = result.get("artifact_paths") or {}
            if isinstance(artifacts, dict):
                candidates_path = artifacts.get("kernel_candidates", "") or ""
        kernel_roofline_path = result.get("kernel_roofline_path") or ""
        if not kernel_roofline_path:
            artifacts = result.get("artifact_paths") or {}
            if isinstance(artifacts, dict):
                kernel_roofline_path = artifacts.get("kernel_roofline", "") or ""
        hot = result.get("hot_kernels") or []
        summary: list[dict[str, Any]] = []
        kernel_roofline: list[dict[str, Any]] = []
        reusable_ids: list[str] = []
        for entry in hot[:15] if isinstance(hot, list) else []:
            if not isinstance(entry, dict):
                continue
            kid = entry.get("kernel_id")
            reusable = bool(entry.get("reusable_native_kernel"))
            arithmetic_intensity = entry.get("arithmetic_intensity")
            if arithmetic_intensity is None:
                arithmetic_intensity = entry.get("flops_per_byte")
            efficiency_percent = entry.get("efficiency_percent")
            if efficiency_percent is None:
                efficiency_percent = entry.get("efficiency_pct")
            summary_entry = {
                "kernel_id": kid,
                "name": entry.get("name"),
                # TraceLens kernel_category bucket; passthrough or downstream by_kernel rows get empty string.
                "kernel_category": entry.get("kernel_category") or "",
                "gpu_pct": entry.get("gpu_pct"),
                "bottleneck": entry.get("bottleneck"),
                "bound_type": entry.get("bound_type"),
                "arithmetic_intensity": arithmetic_intensity,
                "flops_per_byte": entry.get("flops_per_byte"),
                "efficiency_percent": efficiency_percent,
                "compute_utilization_pct": entry.get("compute_utilization_pct"),
                "bandwidth_utilization_pct": entry.get("bandwidth_utilization_pct"),
                "suggestion": entry.get("suggestion") or "",
                "roofline_name": entry.get("roofline_name"),
                "source_file": entry.get("source_file"),
                "reusable_native_kernel": reusable,
                "recommended_backends": entry.get("recommended_backends") or [],
                "recommended_actions": entry.get("recommended_actions") or [],
            }
            summary.append(summary_entry)
            if any(
                summary_entry.get(key) not in (None, "", [])
                for key in (
                    "bound_type",
                    "arithmetic_intensity",
                    "flops_per_byte",
                    "efficiency_percent",
                    "compute_utilization_pct",
                    "bandwidth_utilization_pct",
                    "suggestion",
                    "roofline_name",
                )
            ):
                kernel_roofline.append(dict(summary_entry))
            if reusable and kid:
                reusable_ids.append(str(kid))

        # Project skipped (non-routable) candidates so the LLM sees unoptimizable operators and doesn't hallucinate kernel_ids.
        skipped = result.get("skipped_kernels") or []
        skipped_summary: list[dict[str, Any]] = []
        if isinstance(skipped, list):
            skipped_sorted = sorted(
                (e for e in skipped if isinstance(e, dict)),
                key=lambda e: float(e.get("gpu_pct") or 0.0),
                reverse=True,
            )
            for entry in skipped_sorted[:15]:
                skipped_summary.append({
                    "kernel_id": entry.get("kernel_id"),
                    "name": entry.get("name"),
                    "skip_reason": entry.get("skip_reason") or "",
                    "gpu_pct": entry.get("gpu_pct"),
                })

        raw_warnings = result.get("trace_health_warnings") or []
        warnings_cleaned: list[dict[str, Any]] = []
        if isinstance(raw_warnings, list):
            for entry in raw_warnings:
                if isinstance(entry, dict) and entry.get("code"):
                    warnings_cleaned.append(dict(entry))

        # Monotonic snapshot counter: read previous value + 1.
        prev_snapshot_id = 0
        if isinstance(self.last_trace_analyze, dict):
            prev_raw = self.last_trace_analyze.get("roofline_snapshot_id")
            if isinstance(prev_raw, int):
                prev_snapshot_id = prev_raw
        snapshot_id = prev_snapshot_id + 1

        analysis_md_path = result.get("trace_report_path") or ""
        analysis_md_text = ""
        if analysis_md_path:
            try:
                # Stored verbatim; prompt path strips base64 data-URLs via strip_base64_data_urls.
                analysis_md_text = Path(analysis_md_path).read_text(
                    encoding="utf-8", errors="replace",
                )
            except (OSError, ValueError):
                analysis_md_text = ""

        task_groups = result.get("task_groups") or []
        if not isinstance(task_groups, list):
            task_groups = []

        ts_iso = _now_iso()
        self.last_trace_analyze = {
            "trace_input": str(trace_input),
            "candidates_path": str(candidates_path),
            "kernel_roofline_path": str(kernel_roofline_path),
            "hot_kernels_top15": summary,
            "kernel_roofline_top15": kernel_roofline,
            "skipped_kernels_top": skipped_summary,
            "task_groups": task_groups,
            "reusable_native_kernel_ids": reusable_ids,
            "trace_health_warnings": warnings_cleaned,
            "analysis_md_path": str(analysis_md_path),
            "analysis_md_text": analysis_md_text,
            "roofline_snapshot_id": snapshot_id,
            "roofline_baseline_gain_at_snapshot": float(
                self.cumulative_gain_validated,
            ),
            "ts": ts_iso,
        }
        # Top-level mirror so PolicyGate/Coordinator skip the nested-dict lookup.
        self.roofline_snapshot_id = snapshot_id

        # Append compact history for report-side Roofline Comparison; best-effort, parse errors degrade to None.
        try:
            from .roofline_snapshot import build_roofline_snapshot
            # Stamp decode-roofline ceiling (session constant) + measured tput (current_best.tput else baseline_tput).
            from .roofline_ceiling import (
                RooflineBreakdown,
                compute_roofline_breakdown_from_state,
            )
            # Two-sided roofline; peak_tput stays min(mem, cmp) for the dashboard field.
            breakdown = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")
            try:
                breakdown = compute_roofline_breakdown_from_state(self)
            except Exception:  # noqa: BLE001 — ceiling is best-effort
                pass
            peak_tput = float(breakdown.peak_tok_per_sec or 0.0)
            achieved_tput = 0.0
            cb = self.current_best if isinstance(self.current_best, dict) else {}
            cb_tput = cb.get("tput")
            if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                achieved_tput = float(cb_tput)
            elif isinstance(self.baseline_tput, (int, float)) and self.baseline_tput > 0:
                achieved_tput = float(self.baseline_tput)
            history_entry = build_roofline_snapshot(
                snapshot_id=snapshot_id,
                ts=ts_iso,
                analysis_md_path=str(analysis_md_path),
                theoretical_peak_tok_per_sec=peak_tput,
                achieved_tok_per_sec=achieved_tput,
                mem_ceiling_tok_per_sec=float(breakdown.mem_tok_per_sec or 0.0),
                cmp_ceiling_tok_per_sec=float(breakdown.cmp_tok_per_sec or 0.0),
                bound_kind=breakdown.bound_kind,
            )
            history_entry["trace_input"] = str(trace_input)
            history_entry["analysis_md_path"] = str(analysis_md_path)
            # 9fe4609 sidecar artifact pointer for per-kernel roofline data.
            history_entry["kernel_roofline_path"] = str(kernel_roofline_path)
            if not isinstance(self.roofline_snapshots, list):
                self.roofline_snapshots = []
            self.roofline_snapshots.append(history_entry)
            if len(self.roofline_snapshots) > _ROOFLINE_SNAPSHOTS_CAP:
                # Always keep snapshot #1 so the report's baseline anchor never rotates away.
                base = self.roofline_snapshots[0]
                tail = self.roofline_snapshots[-(_ROOFLINE_SNAPSHOTS_CAP - 1):]
                self.roofline_snapshots = [base, *tail]
        except Exception:  # noqa: BLE001 — never block record on render concerns
            pass

    def record_sweep(self, result: dict[str, Any]) -> None:
        if not isinstance(result, dict):
            return
        grid = result.get("sweep_grid") or []
        best = None
        if isinstance(grid, list):
            best = max(
                (
                    e for e in grid
                    if isinstance(e, dict)
                    and e.get("status") == "succeeded"
                    and isinstance(e.get("output_throughput"), (int, float))
                ),
                default=None,
                key=lambda e: e.get("output_throughput") or 0.0,
            )
        self.last_sweep = {
            "ts": _now_iso(),
            "grid_size": result.get("grid_size", len(grid) if isinstance(grid, list) else 0),
            "best_overall": best or {},
            "best_for_each_conc": result.get("best_for_each_conc") or {},
            "pareto_front": result.get("pareto_front") or [],
            "workspace": result.get("workspace", ""),
        }

    def record_conc_sweep(self, result: dict[str, Any]) -> None:
        """Record conc_sweep task completion (mirrors record_sweep). Bug #12: status lets exit_normal_sweep return conc_sweep_done so SWEEP→CLOSE fires on conc_sweep alone."""
        if not isinstance(result, dict):
            return
        self.last_conc_sweep = {
            "ts":               _now_iso(),
            "status":           str(result.get("status") or "succeeded"),
            "skip_reason":      str(result.get("skip_reason") or ""),
            "was_skipped":      bool(result.get("was_skipped", False)),
            "budget_exhausted": bool(result.get("budget_exhausted", False)),
            "summary":          dict(result.get("summary") or {}),
            "workspace":        str(result.get("workspace") or ""),
        }

    # specialist round bookkeeping
    def record_specialist_round(self, entry: dict[str, Any]) -> None:
        """Append one round summary to ``specialist_rounds``; idempotent on ``round_id`` (re-record overwrites)."""
        if not isinstance(entry, dict) or not entry:
            return
        round_id = str(entry.get("round_id") or "").strip()
        if not round_id:
            self.specialist_rounds.append(dict(entry))
            return
        existing = self.specialist_rounds
        for i, prev in enumerate(existing):
            if isinstance(prev, dict) and str(prev.get("round_id") or "") == round_id:
                existing[i] = dict(entry)
                return
        existing.append(dict(entry))

    def bump_specialist_domain_empty_streak(
        self, domain: str, *, empty: bool,
    ) -> int:
        """Increment/reset the per-domain empty-proposal streak; returns new value (escalation threshold lives in ``KB_design §3.9``, not here)."""
        d = str(domain or "").strip() or "unknown"
        if empty:
            self.specialist_domain_empty_streak[d] = int(
                self.specialist_domain_empty_streak.get(d, 0) or 0
            ) + 1
        else:
            self.specialist_domain_empty_streak[d] = 0
        return self.specialist_domain_empty_streak[d]

    # gaps ledger helpers
    def find_gap(self, canonical_id: str) -> dict[str, Any] | None:
        """Return the gap entry matching ``canonical_id`` (or ``None``)."""
        if not canonical_id:
            return None
        cid = str(canonical_id)
        for gap in self.gaps:
            if isinstance(gap, dict) and str(gap.get("canonical_id") or "") == cid:
                return gap
        return None

    def upsert_gap(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Insert or update one gap row, keyed by ``canonical_id``. Coordinator-only writer (Inv-1 single-writer + CORE_STATE_FIELDS lock). Returns the merged entry."""
        if not isinstance(entry, dict):
            return {}
        cid = str(entry.get("canonical_id") or "").strip()
        if not cid:
            return {}
        now = _now_iso()
        existing = self.find_gap(cid)
        if existing is None:
            merged: dict[str, Any] = {
                "canonical_id":    cid,
                "symptom":         str(entry.get("symptom") or ""),
                "layer":           str(entry.get("layer") or ""),
                "severity":        str(entry.get("severity") or "medium"),
                "domain_hint":     str(entry.get("domain_hint") or ""),
                "source":          str(entry.get("source") or ""),
                # Optional origin reference (PR/blog URL) sedimented into the recipe with KEEP/REVERT provenance.
                "provenance":      str(entry.get("provenance") or ""),
                "first_seen_ts":   str(entry.get("first_seen_ts") or now),
                "last_updated_ts": now,
                "attempts":        list(entry.get("attempts") or []),
            }
            if len(merged["attempts"]) > _GAPS_ATTEMPTS_HISTORY:
                merged["attempts"] = merged["attempts"][-_GAPS_ATTEMPTS_HISTORY:]
            self.gaps.append(merged)
        else:
            # Field-wise merge: incoming non-empty values win except ``first_seen_ts`` (preserve oldest).
            for key in ("symptom", "layer", "severity", "domain_hint",
                        "source", "provenance"):
                incoming = entry.get(key)
                if incoming:
                    existing[key] = str(incoming)
            existing.setdefault("first_seen_ts", str(entry.get("first_seen_ts") or now))
            existing["last_updated_ts"] = now
            incoming_attempts = list(entry.get("attempts") or [])
            if incoming_attempts:
                merged_attempts = list(existing.get("attempts") or []) + incoming_attempts
                # Capped tail; callers supply newest-last lists (convention).
                if len(merged_attempts) > _GAPS_ATTEMPTS_HISTORY:
                    merged_attempts = merged_attempts[-_GAPS_ATTEMPTS_HISTORY:]
                existing["attempts"] = merged_attempts
            merged = existing
        # Enforce global cap, trimming oldest after the upsert so the just-touched gap is retained.
        if len(self.gaps) > _GAPS_MAX_ENTRIES:
            keep_cid = cid
            others = [g for g in self.gaps if g is not merged]

            def _sort_key(g: dict[str, Any]) -> str:
                return str(g.get("last_updated_ts") or g.get("first_seen_ts") or "")

            others.sort(key=_sort_key)
            keep_count = _GAPS_MAX_ENTRIES - 1
            others = others[-keep_count:] if keep_count > 0 else []
            self.gaps = others + [merged]
            del keep_cid  # silence linters when the local isn't used elsewhere
        return merged

    def append_gap_attempt(
        self,
        canonical_id: str,
        attempt: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Append one attempt row to an existing gap; returns the gap or ``None`` when unknown (caller may ``upsert_gap`` instead)."""
        gap = self.find_gap(canonical_id)
        if gap is None:
            return None
        attempts = list(gap.get("attempts") or [])
        attempts.append(dict(attempt) | {"ts": str(attempt.get("ts") or _now_iso())})
        if len(attempts) > _GAPS_ATTEMPTS_HISTORY:
            attempts = attempts[-_GAPS_ATTEMPTS_HISTORY:]
        gap["attempts"] = attempts
        gap["last_updated_ts"] = _now_iso()
        return gap

    def replace_gaps(self, entries: list[dict[str, Any]]) -> None:
        """Bulk-replace ``gaps`` with a fresh dedup'd list (discards stale rows wholesale); idempotent."""
        if not isinstance(entries, list):
            return
        dedup: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cid = str(entry.get("canonical_id") or "").strip()
            if not cid:
                continue
            if cid not in dedup:
                order.append(cid)
            dedup[cid] = dict(entry)
        # Apply the per-entry cap on attempts.
        new_list: list[dict[str, Any]] = []
        for cid in order:
            row = dedup[cid]
            attempts = list(row.get("attempts") or [])
            if len(attempts) > _GAPS_ATTEMPTS_HISTORY:
                attempts = attempts[-_GAPS_ATTEMPTS_HISTORY:]
            row["attempts"] = attempts
            new_list.append(row)
        if len(new_list) > _GAPS_MAX_ENTRIES:
            new_list = new_list[-_GAPS_MAX_ENTRIES:]
        self.gaps = new_list

    def record_intervention(
        self,
        *,
        change_type: str,
        action: str,
        task_id: str = "",
        delta_pct: float | None = None,
    ) -> None:
        """Append one intervention entry and update config-only counters. consecutive-config counter advances on ``"config"``, resets on ``"code_patch"``; ``"code_patch_attempt"`` is telemetry-only."""
        ct = str(change_type or "").strip().lower()
        entry = {
            "change_type": ct,
            "action": str(action or ""),
            "task_id": str(task_id or ""),
            "delta_pct": delta_pct,
            "ts": _now_iso(),
        }
        self.intervention_mix.append(entry)
        if ct == "config":
            self.consecutive_config_only_rounds = (
                int(self.consecutive_config_only_rounds or 0) + 1
            )
        elif ct == "code_patch":
            self.consecutive_config_only_rounds = 0

    def get_intervention_mix(self, *, recent_window: int = 5) -> dict[str, Any]:
        """Summarise the intervention-mix ledger as derived counts (config vs code_patch totals, recent window, consecutive_config_only, config_heavy). Read-only; unknown change_types ignored in tallies but break the trailing config-only run."""
        ledger = [e for e in (self.intervention_mix or []) if isinstance(e, dict)]

        def _ct(entry: dict[str, Any]) -> str:
            return str(entry.get("change_type") or "").strip().lower()

        total_config = sum(1 for e in ledger if _ct(e) == "config")
        total_code_patch = sum(1 for e in ledger if _ct(e) == "code_patch")
        total_code_patch_attempt = sum(
            1 for e in ledger
            if _ct(e) in ("code_patch", "code_patch_attempt")
        )
        window = ledger[-recent_window:] if recent_window > 0 else ledger
        recent_config = sum(1 for e in window if _ct(e) == "config")
        recent_code_patch = sum(1 for e in window if _ct(e) == "code_patch")
        recent_code_patch_attempt = sum(
            1 for e in window
            if _ct(e) in ("code_patch", "code_patch_attempt")
        )

        consecutive_config_only = 0
        for e in reversed(ledger):
            ct = _ct(e)
            if ct == "config":
                consecutive_config_only += 1
            else:
                break

        return {
            "total_config": total_config,
            "total_code_patch": total_code_patch,
            "total_code_patch_attempt": total_code_patch_attempt,
            "recent_config": recent_config,
            "recent_code_patch": recent_code_patch,
            "recent_code_patch_attempt": recent_code_patch_attempt,
            "consecutive_config_only": consecutive_config_only,
            "config_heavy": total_config >= recent_window and total_code_patch == 0,
        }

    def bump_specialist_dispatched(self, n: int = 1) -> int:
        """Increment the per-EXPLORE specialist dispatch counter; returns post-increment value."""
        self.explore_specialist_dispatched_count = (
            int(self.explore_specialist_dispatched_count or 0) + int(n)
        )
        return self.explore_specialist_dispatched_count

    def reset_specialist_dispatched(self) -> None:
        """Zero the per-EXPLORE specialist dispatch counter (on fresh EXPLORE entry)."""
        self.explore_specialist_dispatched_count = 0

    def bump_research_scout_runs(self, n: int = 1) -> int:
        """Increment the research-scout dispatch counter; return new total."""
        self.research_scout_runs = int(self.research_scout_runs or 0) + int(n)
        return self.research_scout_runs

    def register_seen_pr_ids(self, pr_ids: Any) -> int:
        """Add PR ids to the shared seen-set (scout + FRAMEWORK_PR dedup); returns count newly added."""
        seen = set(self.research_scout_seen_pr_ids or [])
        added = 0
        for raw in pr_ids or []:
            pid = str(raw or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            self.research_scout_seen_pr_ids.append(pid)
            added += 1
        return added

    def has_seen_pr_id(self, pr_id: Any) -> bool:
        """True iff ``pr_id`` was already surfaced by scout / FRAMEWORK_PR."""
        pid = str(pr_id or "").strip()
        return bool(pid) and pid in set(self.research_scout_seen_pr_ids or [])

    # Explore plateau proxy
    def reset_explore_plateau_proxy(self) -> None:
        """Reset the legacy explore plateau proxy counter."""
        self.params_no_promote_streak = 0

    def note_explore_outcome(self, *, promoted: bool) -> None:
        """Update the legacy plateau proxy after one explore task (KEEP resets, no-promote increments)."""
        if promoted:
            self.reset_explore_plateau_proxy()
        else:
            self.params_no_promote_streak += 1

    def to_intervention_mix_summary(self) -> str:
        """Render the intervention ledger as neutral telemetry (one-line counts summary; ``""`` when empty). No directive emitted — config-vs-patch is the LLM's choice."""
        mix = self.intervention_mix or []
        if not mix:
            return ""
        n_config = sum(
            1 for m in mix if (m or {}).get("change_type") == "config"
        )
        n_patch = sum(
            1 for m in mix if (m or {}).get("change_type") == "code_patch"
        )
        n_patch_attempt = sum(
            1 for m in mix
            if (m or {}).get("change_type") in (
                "code_patch", "code_patch_attempt",
            )
        )
        n_config_attempt = sum(
            1 for m in mix if (m or {}).get("change_type") == "config_attempt"
        )
        consec = int(self.consecutive_config_only_rounds or 0)
        return (
            f"config_keeps={n_config} config_attempts={n_config_attempt} "
            f"code_patch_keeps={n_patch} code_patch_attempts={n_patch_attempt} "
            f"consecutive_config_only_rounds={consec}"
        )

    def record_specialist_patch_verdict(
        self, specialist_task_id: str, verdict: str,
    ) -> None:
        """Record the Critic verdict for a specialist worktree patch; idempotent (later verdict overwrites), empty ``verdict`` clears the entry to force re-review."""
        sid = str(specialist_task_id or "").strip()
        if not sid:
            return
        v = str(verdict or "").strip().lower()
        if not v:
            self.specialist_patch_verdicts.pop(sid, None)
            return
        self.specialist_patch_verdicts[sid] = v

    def get_specialist_patch_verdict(
        self, specialist_task_id: str,
    ) -> str:
        """Return the patch verdict, or empty when no Critic decision exists."""
        sid = str(specialist_task_id or "").strip()
        if not sid:
            return ""
        return self.specialist_patch_verdicts.get(sid, "") or ""

    def update_last_specialist(self, snapshot: dict[str, Any]) -> None:
        """Snapshot the most recent specialist task (parity with last_*)."""
        if isinstance(snapshot, dict):
            self.last_specialist = dict(snapshot)

    def apply_explore_search_update(self, update: dict[str, Any]) -> None:
        """Merge an ExploreExecutor search update into persistent state (v0.8 M3); executor never writes ``accepted`` directly — :meth:`record_explore_accepted` is the single writer for that bucket."""
        if not isinstance(update, dict):
            return
        prior = self.explore_search if isinstance(self.explore_search, dict) else {}
        merged = dict(prior)
        merged["schema_version"] = int(update.get("schema_version") or 1)
        merged["tested"] = dict(update.get("tested") or prior.get("tested") or {})
        merged["rejected"] = list(update.get("rejected") or prior.get("rejected") or [])
        merged["name_index"] = dict(
            update.get("name_index") or prior.get("name_index") or {}
        )
        merged["cursor"] = int(update.get("cursor") or len(merged["tested"]))
        merged["last_round"] = dict(update.get("last_round") or {})
        # Append-only history fields — merge instead of overwrite.
        wh = list(prior.get("winners_history") or [])
        for entry in update.get("winners_history") or []:
            if isinstance(entry, dict):
                wh.append(dict(entry))
        merged["winners_history"] = wh
        sa: set[tuple[str, ...]] = set()
        for src in (prior.get("synergy_attempted"), update.get("synergy_attempted")):
            for c in src or []:
                if isinstance(c, list):
                    items = tuple(sorted(str(x) for x in c if isinstance(x, str)))
                    if items:
                        sa.add(items)
                elif isinstance(c, str) and c:
                    items = tuple(sorted(c.split("+")))
                    if items:
                        sa.add(items)
        merged["synergy_attempted"] = [list(c) for c in sorted(sa)]
        merged["discovered_flags"] = list(
            update.get("discovered_flags") or prior.get("discovered_flags") or []
        )
        merged["domains_round_summary"] = list(
            update.get("domains_round_summary")
            or prior.get("domains_round_summary") or []
        )
        # Preserve accepted bucket from prior runs (record_explore_accepted is its writer).
        merged["accepted"] = list(prior.get("accepted") or [])
        # Drop merged_from_legacy_sig so a later load re-runs the legacy union (defensive vs interleaved fallback session).
        merged.pop("merged_from_legacy_sig", None)
        self.explore_search = merged

    def record_explore_accepted(self, variant: dict[str, Any]) -> None:
        """Append one promoted variant to ``explore_search.accepted``; dedupes by ``fingerprint`` and removes any matching ``rejected`` entry so a variant isn't in both buckets."""
        if not isinstance(variant, dict) or not variant:
            return
        from .action_executors._canonical_fingerprint import canonical_fingerprint
        args = str(
            variant.get("candidate_extra_server_args")
            or variant.get("extra_server_args") or ""
        )
        envs = dict(variant.get("extra_envs") or {})
        fp = str(variant.get("fingerprint") or canonical_fingerprint(args, envs))
        entry = {
            "fingerprint": fp,
            "name": str(variant.get("name") or ""),
            "extra_server_args": args,
            "extra_envs": envs,
            "note": str(variant.get("note") or ""),
            "tput": variant.get("output_throughput") or variant.get("tput"),
            "gain_pct": variant.get("gain_pct"),
            "stack_index": variant.get("stack_index"),
            "accepted_at_round": str(variant.get("accepted_at_round") or ""),
            "ts": str(variant.get("ts") or _now_iso()),
            "provenance": str(variant.get("provenance") or "llm_direct"),
        }
        search = dict(self.explore_search or {})
        search.setdefault("schema_version", 1)
        accepted = [
            v for v in (search.get("accepted") or [])
            if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        accepted.append(entry)
        search["accepted"] = accepted
        search["rejected"] = [
            v for v in (search.get("rejected") or [])
            if not (isinstance(v, dict) and v.get("fingerprint") == fp)
        ]
        name_index = dict(search.get("name_index") or {})
        if entry["name"]:
            name_index[entry["name"]] = fp
        search["name_index"] = name_index
        # Append a winners_history row so plateau judges needn't crawl optimization_stack.
        wh = list(search.get("winners_history") or [])
        wh.append({
            "round_id": entry["accepted_at_round"],
            "variant_name": entry["name"],
            "fingerprint": fp,
            "gain_pct": entry["gain_pct"],
            "extra_args": args,
            "extra_envs": envs,
            "provenance": entry["provenance"],
            "ts": entry["ts"],
        })
        search["winners_history"] = wh
        self.explore_search = search

    # search-space expansion bookkeeping
    def record_discovered_flags(
        self,
        *,
        framework: str,
        backend_flags: list[str] | None = None,
        param_flags: list[str] | None = None,
        source_path: str = "",
    ) -> None:
        """Persist the AST-discovered flag list for a framework; the prompt surfaces the union so the LLM synthesizes new GridVariants beyond DEFAULT_*_GRID. Idempotent per-framework."""
        fw = (framework or "").strip().lower() or "unknown"
        entry = dict(self.discovered_flags.get(fw) or {})
        if backend_flags is not None:
            entry["backend_flags"] = sorted(set(str(f) for f in backend_flags))
        if param_flags is not None:
            entry["param_flags"] = sorted(set(str(f) for f in param_flags))
        if source_path:
            entry["source_path"] = str(source_path)
        entry["ts"] = _now_iso()
        self.discovered_flags[fw] = entry

    # No action-score API; ``increment_tick`` is a pure monotonic counter for plateau/phase budget math.
    def increment_tick(self) -> int:
        """Bump the Coordinator tick counter and return the new value."""
        self.tick = int(self.tick or 0) + 1
        return self.tick

    def append_stack_gain_entry(
        self,
        *,
        action: str,
        variant_name: str | None,
        new_tput: float,
        extra_server_args: str = "",
        ts: str | None = None,
    ) -> float | None:
        """Mirror an optimization_stack append into gain_per_stack_entry; computes ``(new_tput-baseline_tput)/baseline_tput*100`` and appends. Returns gain_pct (None when baseline_tput is 0 or new_tput non-positive)."""
        try:
            base = float(self.baseline_tput or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        try:
            tput = float(new_tput or 0.0)
        except (TypeError, ValueError):
            tput = 0.0
        gain_pct: float | None
        if base > 0 and tput > 0:
            gain_pct = (tput - base) / base * 100.0
        else:
            gain_pct = None
        self.gain_per_stack_entry.append(gain_pct)
        return gain_pct

    def seed_stack_from_current_best(self) -> None:
        """Backfill stack for old sessions that only had current_best."""
        if self.optimization_stack or not isinstance(self.current_best, dict):
            return
        variant = self.current_best.get("variant_name")
        extra_args = self.current_best.get("extra_server_args")
        if not variant and not extra_args:
            return
        self.optimization_stack = [{
            "action": self.current_best.get("action", "unknown"),
            "variant_name": variant or "legacy_current_best",
            "extra_server_args": extra_args or "",
            "extra_envs": dict(self.current_best.get("extra_envs") or {}),
            "tput": self.current_best.get("tput"),
            "workspace": self.current_best.get("workspace"),
            "source": "seeded_from_current_best",
        }]
        # Keep gain_per_stack_entry aligned with optimization_stack (None == unknown gain for seeded entries).
        if len(self.gain_per_stack_entry) < len(self.optimization_stack):
            self.gain_per_stack_entry.extend(
                [None] * (len(self.optimization_stack) - len(self.gain_per_stack_entry))
            )

    # Time-budget helpers (consumed by Coordinator._compose_prompt)
    def elapsed_minutes(self, *, now: datetime | None = None) -> float:
        """Wall-clock minutes since ``start_ts`` (0.0 when empty/unparseable)."""
        if not self.start_ts:
            return 0.0
        try:
            start = datetime.fromisoformat(self.start_ts)
        except ValueError:
            return 0.0
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        delta = (now_dt - start).total_seconds() / 60.0
        return max(0.0, delta)

    def remaining_minutes(self, *, now: datetime | None = None) -> float | None:
        """Minutes left in the wall-clock budget; ``None`` when ``max_minutes`` unset (unbounded), else clamped at 0."""
        if not self.max_minutes:
            return None
        return max(0.0, float(self.max_minutes) - self.elapsed_minutes(now=now))

    def optimization_stack_has_unvalidated_keeps(self) -> bool:
        """True iff a new KEEP landed since the last inline stack rebench (purely a stack-length check vs ``cumulative_gain_validated_stack_len``)."""
        return len(self.optimization_stack) > int(self.cumulative_gain_validated_stack_len)

    def to_mission_summary(self, *, now: datetime | None = None) -> str:
        """Mission-progress block printed at the top of every tick (outcome-shaped state: raw/validated gain, time vs budget, stack staleness); distinct from :meth:`to_prompt_summary`."""
        elapsed = self.elapsed_minutes(now=now)
        remaining = self.remaining_minutes(now=now)
        budget_line = (
            f"time      : elapsed={elapsed:.1f}min "
            f"remaining={remaining:.1f}min "
            f"budget={self.max_minutes}min"
        ) if remaining is not None else (
            f"time      : elapsed={elapsed:.1f}min budget=unlimited"
        )
        validated_age = ""
        if self.cumulative_gain_validated_ts:
            validated_age = f" (ts={self.cumulative_gain_validated_ts})"
        unvalidated = self.optimization_stack_has_unvalidated_keeps()
        unvalidated_tag = (
            " ⚠ stack changed since last rebench — RUN `explore` "
            "(per-KEEP stack rebench is inlined)"
            if unvalidated else ""
        )
        lines = [
            f"baseline  : {self.baseline_tput} tok/s/GPU",
            f"current   : {self._format_current_best_for_mission()}",
            f"gain      : per-round-sum={self.cumulative_gain:.2f}% "
            f"validated={self.cumulative_gain_validated:.2f}%{validated_age}",
            f"stack     : {len(self.optimization_stack)} entries "
            f"(validated_at_len={self.cumulative_gain_validated_stack_len})"
            f"{unvalidated_tag}",
        ]
        # Surface reusable hot kernels still owing a kernel_opt attempt (visible without a checklist).
        untried_hot = self.untried_hot_reusable_kernels()
        if untried_hot:
            lines.append(
                f"untried_hot_kernels: {', '.join(untried_hot)}"
            )
        lines.append(budget_line)
        return "\n".join(lines)

    def _format_current_best_for_mission(self) -> str:
        if not isinstance(self.current_best, dict) or not self.current_best:
            return "(none)"
        return (
            f"action={self.current_best.get('action','?')} "
            f"tput={self.current_best.get('tput','?')} "
            f"variant={self.current_best.get('variant_name','?')}"
        )

    def to_phase_status_summary(
        self,
        *,
        budget_pct: dict[str, float] | None = None,
        now_unix: float | None = None,
    ) -> str:
        """Render the per-tick ``=== Phase ===`` block (v0.8 §3.3); compact (≤5 lines). EXPLORE adds a ``force_exit`` line showing runway before the hard force-exit gate."""
        from .phase_state import (
            DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
            DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
            PHASE_EXPLORE,
            llm_proposable_actions_for_with_interleave,
            normalize_budget_pct,
            phase_budget_remaining_seconds,
            phase_elapsed_seconds,
            session_remaining_seconds,
        )

        phase = (self.phase or "").strip().upper() or "UNSET"
        elapsed = int(phase_elapsed_seconds(self, now_unix=now_unix))
        budget = normalize_budget_pct(budget_pct or self.phase_budget_pct)
        budget_pct_for_phase = budget.get(phase, 0.0)
        remaining = phase_budget_remaining_seconds(
            self, budget_pct=budget, now_unix=now_unix,
        )
        budget_line: str
        if remaining is None:
            budget_line = (
                f"budget    : pct={budget_pct_for_phase:.2f} (unlimited run; "
                f"no per-phase cap)"
            )
        else:
            budget_line = (
                f"budget    : pct={budget_pct_for_phase:.2f} "
                f"elapsed_sec={elapsed} remaining_sec={int(remaining)}"
            )
        proposable = tuple(sorted(
            llm_proposable_actions_for_with_interleave(phase)
        ))
        allowed_line = (
            f"allowed   : {', '.join(proposable) if proposable else '(none)'}"
        )
        lines = [
            f"phase     : {phase}",
            f"entered   : {self.phase_started_ts or '(unset)'}",
            budget_line,
            allowed_line,
        ]
        # EXPLORE-only: distance to hard force-exit alongside the soft budget.
        if phase == PHASE_EXPLORE:
            overrides = self.plateau_overrides or {}
            hours_thresh = float(overrides.get(
                "force_exit_hours_remaining",
                DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING,
            ))
            pct_thresh = float(overrides.get(
                "force_exit_budget_pct",
                DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT,
            ))
            session_remaining = session_remaining_seconds(
                self, now_unix=now_unix,
            )
            session_buffer = (
                int(session_remaining - hours_thresh * 3600.0)
                if session_remaining is not None else None
            )
            if remaining is not None and budget_pct_for_phase > 0:
                mm = float(self.max_minutes or 0)
                phase_total_sec = mm * 60.0 * budget_pct_for_phase
                phase_remaining_pct = (
                    remaining / phase_total_sec if phase_total_sec > 0 else 0.0
                )
            else:
                phase_remaining_pct = None
            force_line = (
                f"force_exit: hours_thresh={hours_thresh:.1f}h "
                f"pct_thresh={pct_thresh:.2f}"
            )
            if session_buffer is not None:
                force_line += f" session_buffer_sec={session_buffer}"
            if phase_remaining_pct is not None:
                force_line += (
                    f" phase_remaining_pct={phase_remaining_pct:.3f}"
                )
            lines.append(force_line)
        return "\n".join(lines)

    def to_phase_budget_telemetry(
        self,
        *,
        budget_pct: dict[str, float] | None = None,
        now_unix: float | None = None,
    ) -> str:
        """Render the per-phase budget telemetry block for Robustness (one ``phase: elapsed=Xs cap=Ys (Z%)`` line per phase) so it can spot budget overruns."""
        from .phase_state import (
            DEFAULT_PHASE_BUDGET_PCT,
            PHASE_NAMES,
            normalize_budget_pct,
            phase_elapsed_seconds,
        )

        budget = normalize_budget_pct(budget_pct or self.phase_budget_pct)
        # Aggregate elapsed per phase using phase_history.
        elapsed_per_phase: dict[str, float] = {}
        history = self.phase_history or []
        for idx, row in enumerate(history):
            if not isinstance(row, dict):
                continue
            phase = str(row.get("to_phase") or "").upper()
            entered = float(row.get("ts_unix") or 0.0)
            if not phase or entered <= 0:
                continue
            if idx + 1 < len(history) and isinstance(history[idx + 1], dict):
                exited = float(history[idx + 1].get("ts_unix") or entered)
            else:
                # Currently-active segment — measure to now.
                elapsed_now = phase_elapsed_seconds(self, now_unix=now_unix)
                exited = entered + elapsed_now
            elapsed_per_phase[phase] = (
                elapsed_per_phase.get(phase, 0.0) + max(0.0, exited - entered)
            )
        if not elapsed_per_phase:
            return "(no phase history yet)"
        mm = float(self.max_minutes or 0.0)
        total_budget_sec = mm * 60.0
        lines: list[str] = []
        # Stable order — iterate PHASE_NAMES so new phases render automatically.
        for phase in PHASE_NAMES:
            if phase not in elapsed_per_phase:
                continue
            elapsed = elapsed_per_phase[phase]
            pct = budget.get(phase, DEFAULT_PHASE_BUDGET_PCT.get(phase, 0.0))
            cap_sec = total_budget_sec * pct if total_budget_sec > 0 else 0.0
            used_pct = (elapsed / cap_sec * 100.0) if cap_sec > 0 else 0.0
            cap_line = f"cap={int(cap_sec)}s" if cap_sec > 0 else "cap=unlimited"
            lines.append(
                f"  {phase}: elapsed={int(elapsed)}s {cap_line} used={used_pct:.0f}%"
            )
        return "\n".join(lines) or "(no phase history yet)"

    def to_warm_start_summary(self, *, max_lines: int = 12) -> str:
        """Render T0 warm-start snapshot for the ``=== Warm start ===`` prompt section (v0.8 §3.3 §4.1); empty when no recipe/pitfalls. Capped; full JSON at runtime/cortex/.kb_warm.json / .kb_pitfalls.json."""
        recipe = self.warm_start_recipe or {}
        pitfalls = self.warm_start_pitfalls or []
        if not recipe and not pitfalls:
            return ""
        out: list[str] = []
        workload = str(recipe.get("workload") or "") if isinstance(recipe, dict) else ""
        hw = str(recipe.get("hw") or "") if isinstance(recipe, dict) else ""
        if workload or hw:
            out.append(f"recipe: workload={workload or '?'} hw={hw or '?'}")
        raw = str(recipe.get("raw") or "") if isinstance(recipe, dict) else ""
        # Trim recipe raw text — at most 5 lines, 240 chars each.
        if raw.strip():
            kept = 0
            for line in raw.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                out.append(f"  · {stripped[:240]}")
                kept += 1
                if kept >= 5:
                    break
            if kept == 0:
                out.append("  · (recipe present but text was empty)")
        else:
            out.append("  · (no recipe text — first session for this workload/hw)")
        if pitfalls:
            out.append(f"pitfalls ({len(pitfalls)}):")
            for entry in pitfalls[:5]:
                if not isinstance(entry, dict):
                    continue
                snippet = str(entry.get("raw") or entry.get("symptom") or "")
                if not snippet.strip():
                    continue
                first_line = snippet.splitlines()[0].strip()
                out.append(f"  · {first_line[:240]}")
        if max_lines and len(out) > max_lines:
            out = out[:max_lines]
            out.append(f"  · (truncated to {max_lines} lines; "
                       f"see runtime/cortex/.kb_warm.json for full snapshot)")
        return "\n".join(out)

    def to_gaps_summary(self, *, max_entries: int = 10) -> str:
        """Render :attr:`gaps` for prompt injection (KB_design §3.3/§3.5); empty when no gaps. Capped at ``max_entries`` newest rows."""
        if not self.gaps:
            return ""
        # Newest first by last_updated_ts (deterministic fallback to first_seen_ts/insertion).
        ordered = list(self.gaps)
        ordered.sort(
            key=lambda g: str(
                g.get("last_updated_ts") or g.get("first_seen_ts") or "",
            ),
            reverse=True,
        )
        rows: list[str] = []
        for gap in ordered[:max_entries]:
            if not isinstance(gap, dict):
                continue
            cid = str(gap.get("canonical_id") or "?")
            layer = str(gap.get("layer") or "?")
            severity = str(gap.get("severity") or "?")
            symptom = str(gap.get("symptom") or "").replace("\n", " ").strip()
            if len(symptom) > 200:
                symptom = symptom[:197] + "..."
            attempts = gap.get("attempts") or []
            attempt_n = len(attempts) if isinstance(attempts, list) else 0
            last_tag = ""
            if isinstance(attempts, list) and attempts:
                last = attempts[-1]
                if isinstance(last, dict):
                    last_tag = (
                        f" last={last.get('action','?')}:"
                        f"{last.get('outcome','?')}"
                    )
            rows.append(
                f"  - {cid} [{layer}/{severity}] {symptom}\n"
                f"      attempts={attempt_n}{last_tag}"
            )
        if len(ordered) > max_entries:
            rows.append(
                f"  · (+{len(ordered) - max_entries} older gaps elided; "
                f"see state.json `gaps[]`)"
            )
        return "\n".join(rows)

    def to_proposal_scores_summary(self, *, max_rounds: int = 2) -> str:
        """Render advisory multi-model proposal scores for Orchestration. NO mean/sorting (Inv-9.1: no system-side scoreboard); rater identities anonymized to avoid brand bias. Empty when no recent round carries scores."""
        rounds = [
            r for r in (self.specialist_rounds or [])
            if isinstance(r, dict)
            and isinstance(r.get("ensemble_scores"), dict)
            and (r["ensemble_scores"].get("models") or {})
        ]
        if not rounds:
            return ""
        shown = rounds[-max_rounds:]
        # Stable, anonymized rater labels: map each real slug to ``rater_N`` (slug never reaches the prompt).
        all_slugs: set[str] = set()
        for r in shown:
            models = r["ensemble_scores"].get("models") or {}
            all_slugs.update(str(s) for s in models.keys())
            errs = r["ensemble_scores"].get("errors") or {}
            all_slugs.update(str(s) for s in errs.keys())
        rater_label = {
            slug: f"rater_{i}"
            for i, slug in enumerate(sorted(all_slugs), start=1)
        }
        rows: list[str] = [
            "(Advisory only — one reference among many, NOT a ranking "
            "directive. Scores are 0-10 likelihood-of-throughput-gain "
            "priors from independent anonymized raters; weigh on merit "
            "alongside gaps / KB / analysis.md.)",
        ]
        for r in shown:
            ens = r["ensemble_scores"]
            models = ens.get("models") or {}
            scale = str(ens.get("scale") or "0-10")
            round_id = str(r.get("round_id") or "?")
            domain = str(r.get("domain") or "?")
            rows.append(f"round={round_id} domain={domain} scale={scale}")
            # Collect variant names across models, preserving proposal_set order when available.
            ordered_names: list[str] = []
            seen: set[str] = set()
            for variant in (r.get("proposal_set") or []):
                if isinstance(variant, dict):
                    nm = str(variant.get("name") or "")
                    if nm and nm not in seen:
                        ordered_names.append(nm)
                        seen.add(nm)
            for per_model in models.values():
                if isinstance(per_model, dict):
                    for nm in per_model:
                        if nm not in seen:
                            ordered_names.append(nm)
                            seen.add(nm)
            # Render raters in stable label order so a column means the same model across rounds.
            ordered_slugs = sorted(
                (s for s in models if s in rater_label),
                key=lambda s: rater_label[s],
            )
            for nm in ordered_names:
                parts: list[str] = []
                for model_slug in ordered_slugs:
                    per_model = models.get(model_slug)
                    if not isinstance(per_model, dict):
                        continue
                    label = rater_label[model_slug]
                    cell = per_model.get(nm)
                    if isinstance(cell, dict) and cell.get("score") is not None:
                        reason = str(cell.get("reason") or "").replace("\n", " ")
                        if len(reason) > 80:
                            reason = reason[:77] + "..."
                        parts.append(
                            f"{label}={float(cell['score']):.1f} "
                            f"(\"{reason}\")"
                        )
                    else:
                        parts.append(f"{label}=n/a")
                rows.append(f"  - {nm}: " + ", ".join(parts))
            errors = ens.get("errors") or {}
            if errors:
                err_labels = ", ".join(
                    sorted(
                        rater_label.get(str(s), "rater_?")
                        for s in errors
                    )
                )
                rows.append(f"  · raters unavailable this round: {err_labels}")
        return "\n".join(rows)

    def to_prompt_summary(self) -> str:
        """Compact, human-readable snapshot for prompt injection (DESIGN §8.3)."""
        lines = [
            f"session_id={self.session_id or '(unset)'}",
            f"model={self.model_name or '(unset)'}  class={self.model_class or '(unset)'}",
        ]
        # Advisory architecture profile; prompt-context only (TraceLens analysis_md is ground truth). Omitted when no profile.
        _arch_line = render_model_arch_compact(self.model_arch)
        if _arch_line:
            lines.append(
                f"model_arch(advisory; subordinate to TraceLens analysis_md)={_arch_line}"
            )
        lines += [
            f"baseline_tput={self.baseline_tput}  baseline_acc={self.baseline_accuracy}",
            f"baseline_failure_streak={self.baseline_failure_streak}",
            f"current_best={self.current_best or '(none)'}",
            f"optimization_stack={self._format_optimization_stack()}",
            f"cumulative_gain={self.cumulative_gain}%",
            (
                f"cumulative_gain_validated={self.cumulative_gain_validated}% "
                f"(stack_len_at_validation={self.cumulative_gain_validated_stack_len}, "
                f"ts={self.cumulative_gain_validated_ts or '(never)'})"
            ),
            f"last_sweep={self._format_last_sweep()}",
            f"current_action={self.current_action or '(idle)'}",
            f"crash_count={self.crash_count}",
            f"pruned_families={self.pruned_families or '(none)'}",
            f"last_profile_trace={self.last_profile_trace or '(none)'}",
            f"last_profile_status={self.last_profile_status or '(none)'}",
            f"last_profile_args='{self.last_profile_args}'",
            f"discovered_flags_error={self.discovered_flags_error or '(none)'}",
            f"last_trace_analyze={self._format_last_trace_analyze()}",
            # Full TraceLens analysis.md so the LLM grounds propose_action in the actual report.
            f"analysis_md={self._format_analysis_md_full()}",
            # Streak counter is a readable fact (KEEP/REVERT counts allowed); plateau judges also consume it on legacy resume.
            f"params_no_promote_streak={self.params_no_promote_streak}",
            f"explore_search={self._format_explore_search()}",
            f"discovered_flags={self._format_discovered_flags()}",
            f"backend_winners_history={self._format_backend_winners_history()}",
            f"synergy_attempted={len(self.synergy_attempted)} combos",
            f"last_kernel_opt={self._format_last_kernel_opt()}",
            # Pending KEEPs the integrate gate will drain, plus per-kernel attempt count.
            (
                "pending_keep_kernels="
                f"{self.pending_keep_kernel_ids() or '(none)'}"
            ),
            (
                "has_keep_pending_integrate="
                f"{'true' if self.has_keep_pending_integrate else 'false'}"
            ),
            f"kernel_opt_attempts_count={self.kernel_opt_attempts_count}",
            f"rejected_kernel_patches={self._format_rejected_kernel_patches()}",
            f"rejected_kernel_ids={self.rejected_kernel_ids or '(none)'}",
            f"last_baseline={self._format_attempt(self.last_baseline)}",
            f"last_profile={self._format_attempt(self.last_profile)}",
            f"last_gemm_tuning={self._format_attempt(self.last_gemm_tuning)}",
            f"last_explore={self._format_attempt(self.last_explore)}",
            f"last_sweep={self._format_attempt(self.last_sweep)}",
            f"attempts_history={self._format_attempts_history()}",
            f"last_action_failures={self._format_last_action_failures()}",
            f"tick={int(self.tick or 0)}  "
            f"target_gap_pct={float(self.target_gap_pct or 0.0):.2f}",
            f"stop_reason={self.stop_reason or '(none)'}",
            f"closing_phase={self.closing_phase}  "
            f"closing_started_unix={self.closing_started_unix or 0.0}  "
            f"closing_report_task_id={self.closing_report_task_id or '(none)'}",
        ]
        return "\n".join(lines)

    # Audit-trail renderers (per-action attempts + global failure log); compact one-liners.
    @staticmethod
    def _format_attempt(entry: dict[str, Any] | None) -> str:
        """Render one ``last_<action>`` snapshot or attempts[-1] entry."""
        if not isinstance(entry, dict) or not entry:
            return "(none)"
        metric = entry.get("key_metric")
        metric_kind = entry.get("key_metric_kind") or "metric"
        metric_str = (
            f"{metric_kind}={metric:.2f}"
            if isinstance(metric, (int, float)) else f"{metric_kind}=N/A"
        )
        err = entry.get("error_class") or "-"
        ws = entry.get("workspace") or "-"
        return (
            f"status={entry.get('status','?')} "
            f"decision={entry.get('decision','?')} "
            f"{metric_str} err={err} ws={ws} "
            f"task_id={entry.get('task_id','?')} ts={entry.get('ts','?')}"
        )

    def _format_attempts_history(self) -> str:
        """One-line summary across the audit actions (``baseline:total(s<succ>,f<fail>) ...``) so the LLM gauges reliability without 6x20 rows."""
        parts: list[str] = []
        for action in sorted(_AUDIT_ACTIONS):
            attempts_attr = f"{action}_attempts"
            history = getattr(self, attempts_attr, None) or []
            if not history:
                continue
            total = len(history)
            succ = sum(
                1 for e in history
                if isinstance(e, dict) and e.get("status") == "succeeded"
            )
            fail = sum(
                1 for e in history
                if isinstance(e, dict) and e.get("status") == "failed"
            )
            parts.append(f"{action}:{total}(s{succ},f{fail})")
        return " ".join(parts) if parts else "(no attempts recorded)"

    def _format_last_action_failures(self) -> str:
        """Render up to the 3 most-recent global failures (rich-context companion to crash_count/baseline_failure_streak); full list on disk."""
        if not self.last_action_failures:
            return "(none)"
        rows: list[str] = []
        for entry in self.last_action_failures[-3:]:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action") or "?"
            error_class = entry.get("error_class") or "?"
            ts = entry.get("ts") or "?"
            excerpt = entry.get("error_excerpt") or ""
            ws = entry.get("workspace") or "-"
            excerpt_short = excerpt.splitlines()[0][:200] if excerpt else ""
            rows.append(
                f"[{action}/{error_class}@{ts}] err=\"{excerpt_short}\" ws={ws}"
            )
        suffix = (
            f" [+{len(self.last_action_failures) - 3} earlier]"
            if len(self.last_action_failures) > 3 else ""
        )
        return " | ".join(rows) + suffix if rows else "(none)"

    def _format_rejected_kernel_patches(self) -> str:
        if not self.rejected_kernel_patches:
            return "(none)"
        return [
            (
                f"{r.get('kernel_id','?')}: attempts={r.get('attempt_count','?')} "
                f"best_gain={r.get('best_gain_pct','?')} reason={r.get('reason','?')}"
            )
            for r in self.rejected_kernel_patches[-5:]
            if isinstance(r, dict)
        ] or "(none)"

    def _format_discovered_flags(self) -> str:
        if not self.discovered_flags:
            return "(none — first backends/params round will populate)"
        parts: list[str] = []
        for fw, entry in sorted(self.discovered_flags.items()):
            if not isinstance(entry, dict):
                continue
            n_b = len(entry.get("backend_flags") or [])
            n_p = len(entry.get("param_flags") or [])
            parts.append(f"{fw}:backend={n_b}/param={n_p}")
        return ", ".join(parts) or "(none)"

    @staticmethod
    def _format_variant_line(entry: dict[str, Any]) -> str:
        """One-line render of a search variant for prompt blocks."""
        name = str(entry.get("name") or "?")
        gain = entry.get("gain_pct")
        tput = entry.get("tput") or entry.get("output_throughput")
        gain_s = (
            f"{gain:+.2f}%" if isinstance(gain, (int, float)) else " no_meas"
        )
        tput_s = (
            f" (tput={tput:.1f})"
            if isinstance(tput, (int, float)) and tput > 0
            else ""
        )
        args = (
            str(entry.get("extra_server_args") or "").strip()
            or "(no-flag)"
        )
        envs = entry.get("extra_envs") or {}
        envs_s = (
            " " + " ".join(f"{k}={v}" for k, v in sorted(envs.items()))
            if envs else ""
        )
        return f"{name:28s} {gain_s:>9}{tput_s}  {args}{envs_s}"

    @staticmethod
    def _enrich_with_tested_gain(
        entry: dict[str, Any], tested: dict[str, Any],
    ) -> dict[str, Any]:
        """Backfill ``gain_pct``/``tput`` from the matching ``tested[fp]`` at render time (some accepted entries don't persist gain_pct; avoids a second writer)."""
        if (
            entry.get("gain_pct") is not None
            and entry.get("tput") is not None
        ):
            return entry
        fp = str(entry.get("fingerprint") or "")
        snap = tested.get(fp) if fp else None
        if not isinstance(snap, dict):
            return entry
        out = dict(entry)
        if out.get("gain_pct") is None:
            out["gain_pct"] = snap.get("gain_pct")
        if out.get("tput") is None:
            result = (
                snap.get("result")
                if isinstance(snap.get("result"), dict) else {}
            )
            out["tput"] = (
                snap.get("tput") or (result or {}).get("output_throughput")
            )
        return out

    def _format_backend_winners_history(self) -> str:
        """Multi-line render of the explore-round winners history (last 5 rounds: per-winner gain_pct/tput/flags); older rounds collapse to an elision line."""
        if not self.backend_winners_history:
            return "(no explore rounds completed)"
        last = self.backend_winners_history[-5:]
        out: list[str] = [""]
        for r in last:
            if not isinstance(r, dict):
                continue
            best = r.get("best") if isinstance(r.get("best"), dict) else None
            best_gain = best.get("gain_pct") if best else None
            gain_tag = (
                f" {best_gain:+.2f}%"
                if isinstance(best_gain, (int, float)) else ""
            )
            base = float(r.get("base_tput", 0.0) or 0.0)
            out.append(
                f"    {r.get('round_id','?')} ({r.get('action','?')}): "
                f"base_tput={base:.1f}  "
                f"best={(best.get('name') if best else '(none)')}{gain_tag}"
            )
            winners = [
                w for w in (r.get("winners") or []) if isinstance(w, dict)
            ]
            if not winners:
                out.append("      (no winners this round)")
                continue
            for w in winners:
                out.append("      • " + SharedState._format_variant_line(w))
        if len(self.backend_winners_history) > 5:
            out.append(
                f"    [+{len(self.backend_winners_history) - 5} "
                f"earlier rounds elided]"
            )
        return "\n".join(out)

    def _format_explore_search(self) -> str:
        return self._format_search_state(self.explore_search)

    @staticmethod
    def _format_search_state(search: dict[str, Any] | None) -> str:
        """Multi-line render of a ``*_search`` dedup ledger; each entry surfaces real ``gain_pct``. Counts on the head line; bodies show last 5 per bucket (only the prompt body is truncated)."""
        if not search:
            return "(none)"
        accepted = list(search.get("accepted") or [])
        rejected = list(search.get("rejected") or [])
        tested = search.get("tested") or {}
        cursor = search.get("cursor", 0)
        out: list[str] = [
            "",
            f"    cursor={cursor}  accepted={len(accepted)}  "
            f"rejected={len(rejected)}  tested={len(tested)}",
        ]
        if accepted:
            out.append("    accepted:")
            for entry in accepted[-5:]:
                if not isinstance(entry, dict):
                    continue
                out.append("      • " + SharedState._format_variant_line(
                    SharedState._enrich_with_tested_gain(entry, tested)
                ))
        if rejected:
            out.append("    rejected (last 5):")
            for entry in rejected[-5:]:
                if not isinstance(entry, dict):
                    continue
                out.append("      • " + SharedState._format_variant_line(
                    entry
                ))
        return "\n".join(out)

    def _format_optimization_stack(self) -> str:
        if not self.optimization_stack:
            return "(none)"
        parts = []
        for entry in self.optimization_stack:
            if not isinstance(entry, dict):
                continue
            parts.append(
                f"{entry.get('action','?')}:{entry.get('variant_name','?')}"
            )
        return parts or "(none)"

    @staticmethod
    def _strip_base64_data_urls(text: str) -> str:
        """Drop base64 image payloads before prompt injection (in-memory only; on-disk file intact). Delegates to ``inference_optimizer.tracelens_md``."""
        if not text:
            return text or ""
        from inference_optimizer.tracelens_md import strip_base64_data_urls
        return strip_base64_data_urls(text)

    def _format_analysis_md_full(self) -> str:
        """Inject TraceLens analysis.md verbatim (Roofline composite design §6.1: no truncation/interpretation) between ``=== TraceLens Analysis ... ===`` bookends; header carries snapshot id + gain. Empty cache → one-line hint to propose ``roofline``."""
        cached = self.last_trace_analyze or {}
        md_text = cached.get("analysis_md_text") or ""
        if not md_text:
            return (
                "(no TraceLens snapshot yet — analysis is auto-enqueued "
                "by the Coordinator at the end of PRELUDE and on every "
                "+10% validated-gain crossing; wait for the pending "
                "task to land, or continue with specialist / explore "
                "work that does not need analysis.md. `roofline` and "
                "`profile` are Coordinator-managed and absent from "
                "`PHASE_LLM_PROPOSABLE_ACTIONS`, so PolicyGate R1 "
                "denies any LLM-emitted propose_action/delegate "
                "against either name with rule `phase_incompatible`.)"
            )
        md_text = self._strip_base64_data_urls(md_text)
        snap = cached.get("roofline_snapshot_id", "?")
        gain = cached.get("roofline_baseline_gain_at_snapshot", 0.0)
        try:
            gain_str = f"{float(gain):.2f}"
        except (TypeError, ValueError):
            gain_str = "?"
        return (
            f"\n=== TraceLens Analysis (snapshot #{snap}, "
            f"gain at snapshot = {gain_str}%) ===\n"
            f"{md_text}\n"
            f"=== End TraceLens Analysis ===\n"
        )

    def _format_last_trace_analyze(self) -> str:
        return self._format_trace_analyze_blob(self.last_trace_analyze)

    def _format_trace_analyze_blob(self, blob: dict[str, Any] | None) -> str:
        if not blob:
            return "(none)"
        ids = [
            str(e.get("kernel_id"))
            for e in blob.get("hot_kernels_top15", [])
            if isinstance(e, dict) and e.get("kernel_id")
        ]
        reusable = list(blob.get("reusable_native_kernel_ids", []))
        base = (
            f"trace={blob.get('trace_input','?')} "
            f"candidates_path={blob.get('candidates_path','?')} "
            f"top={ids or []} reusable_native={reusable or []}"
        )
        # With no routable candidates, surface skipped operators so the LLM doesn't echo invalid kernel_ids.
        skipped_suffix = ""
        if not ids:
            sk = blob.get("skipped_kernels_top") or []
            rendered_sk = [
                f"{s.get('kernel_id')}:{s.get('name')}:{s.get('skip_reason') or '?'}"
                for s in sk
                if isinstance(s, dict) and s.get("kernel_id")
            ]
            if rendered_sk:
                skipped_suffix = (
                    f" skipped_kernels_top=[{'; '.join(rendered_sk)}]"
                )
        # Surface TraceLens routing signals inline so the LLM grounds the next action; omitted in steady-state.
        warnings = blob.get("trace_health_warnings") or []
        if not warnings:
            return base + skipped_suffix
        rendered: list[str] = []
        for w in warnings:
            if not isinstance(w, dict):
                continue
            code = str(w.get("code") or "unknown")
            extras: list[str] = []
            if "idle_pct" in w and "threshold_pct" in w:
                extras.append(f"idle={w['idle_pct']}%")
                extras.append(f"threshold={w['threshold_pct']}%")
            if "returncode" in w:
                extras.append(f"rc={w['returncode']}")
            if extras:
                rendered.append(f"{code}({','.join(extras)})")
            else:
                rendered.append(code)
        return f"{base}{skipped_suffix} warnings=[{'; '.join(rendered)}]"

    def _format_last_sweep(self) -> str:
        if not self.last_sweep:
            return "(none)"
        best = self.last_sweep.get("best_overall") or {}
        if not best:
            return f"grid_size={self.last_sweep.get('grid_size', 0)} best=(none)"
        return (
            f"grid_size={self.last_sweep.get('grid_size', 0)} "
            f"best={best.get('name','?')} "
            f"tput={best.get('output_throughput','?')} "
            f"conc={best.get('conc','?')} isl={best.get('isl','?')} osl={best.get('osl','?')}"
        )


__all__ = ["SharedState", "render_model_arch_compact"]
