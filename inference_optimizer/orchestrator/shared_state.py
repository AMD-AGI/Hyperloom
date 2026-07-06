# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SharedState — single-writer (Coordinator) persisted session state, backed by atomic JSON at ``$SESSION_DIR/state.json``; enforces CORE_STATE_FIELDS guards.

Fields::

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
    baseline_tput       float — primary throughput after `baseline` action;
                                tok/s/GPU for serving frameworks, img/s for
                                scriptable xDiT (displayed as e2el_mean_ms)
    baseline_accuracy   float — GSM8K score after `baseline`
    current_best        dict  — {action: str, tput: float, accuracy: float}
    cumulative_gain     float — % over baseline
    stop_reason         str   — set when graceful stop fires (§9)
    current_action      str   — what's running right now (set by Orchestration)
    crash_count         int   — incremented by the Coordinator when a tick/agent
                                exception is recorded; also appends to
                                crash_timestamps (Robustness only reads it)
    pruned_families     list[str]  — set by Robustness via PRUNE_BRANCH
    start_ts            str   — ISO timestamp
    max_minutes         int   — wall-clock budget (0 = unlimited)
    last_profile_trace  str   — set by Coordinator when `profile` returns a
                                trace path; consumed by Orch to populate
                                `trace_analyze` REQUEST `trace_input` param
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._time import now_iso

log = logging.getLogger(__name__)

# Upper bound on retained crash timestamps (trailing-window rate needs only the
# recent tail; older entries age out of any sane emergency window anyway).
_CRASH_TIMESTAMP_CAP: int = 200

# microseconds + ``+00:00`` (canonical helper; kept importable for callers).
_now_iso = now_iso


def _first_positive_tput(d: Any) -> float:
    """Return the first positive ``tput``/``output_throughput`` from a dict.

    Args:
        d: A metrics dict (non-dicts are treated as empty).

    Returns:
        The first of ``tput`` then ``output_throughput`` that is a positive
        number, as a float; ``0.0`` when neither is present/positive.
    """
    src = d if isinstance(d, dict) else {}
    for key in ("tput", "output_throughput"):
        val = src.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return 0.0


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
    """Render the advisory ``model_arch`` profile as a single compact line (``""`` when empty/not a dict).

    Args:
        arch (dict | None): The advisory architecture profile mapping
            (structured keys plus an optional free-text ``notes`` field).

    Returns:
        str: A ``"; "``-joined ``label=value`` line over the recognized
            structured fields (plus ``notes`` when present), or ``""`` when
            ``arch`` is empty or not a dict.
    """
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
# Backend ladder infra failures can be transient; require two failed ladders
# before retiring the kernel. Override via
# ``INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES`` (>=1).
_DEFAULT_KERNEL_OPT_MAX_FAILURES = 2


def resolve_kernel_opt_max_failures() -> int:
    """Resolve the infra-failure retry budget (>=1).

    Shared by ``record_kernel_opt``, ``kernel_work_pending``, and batch
    dispatch so ``INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES`` stays
    consistent across layers.
    """
    env_f = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES")
    if env_f:
        try:
            return max(1, int(env_f))
        except (TypeError, ValueError):
            pass
    return _DEFAULT_KERNEL_OPT_MAX_FAILURES


# Integration faults (environment / apply / bench crashes) are distinct from a
# genuine gate REVERT (measured gain below threshold / accuracy regression). A
# fault means the patch was never fairly measured, so it must not burn the
# ``max_attempts`` REVERT quota; it gets its own small retry budget instead.
#
# The reliable discriminator is ``status``: a genuine gate verdict (KEEP /
# REVERT / NEEDS_REVIEW on a real measurement) is always emitted by
# ``integrate_handler`` with ``status:"ok"``; every unmeasured path returns
# ``status:"failed"`` (several also stamp ``decision:"REVERT"`` for backward
# compat, which is exactly why ``decision`` alone is *not* trustworthy). See
# :meth:`_is_integrate_fault`. The error-class set below is a secondary,
# belt-and-suspenders signal — it stays meaningful even if a future caller
# emits a fault with a drifted ``status`` — but it is no longer the only thing
# standing between an environment crash and a discarded patch.
_INTEGRATE_FAULT_ERROR_CLASSES = frozenset(
    {
        "missing_integration_inputs",
        "patch_not_applied",
        "apply_failed",
        "mn_server_restart_failed_post_patch",
        "rebaseline_exception",
        "cpp_itfs_rebuild_not_verified",
        "framework_script_mismatch",
        "bench_exception",
        "subtask_exception",
        "handler_exception",
        "subprocess_timeout",
    }
)
# Independent bounded budget for integration-fault *attempts* (separate from the
# REVERT ``max_attempts`` quota). NOTE: this counts total fault attempts, not
# retries-after-the-first — a value of 2 means "one initial fault plus one
# retry, then reject". Total attempts are still capped globally by the
# crash-rate emergency stop in the coordinator.
_MAX_INTEGRATE_FAULT_ATTEMPTS = 2

# Hot-kernel report gate: reusable hot kernels >= this GPU share need a kernel_opt attempt/rejection before ``report``.
_DEFAULT_HOT_KERNEL_MIN_GPU_PCT = 3.0
# Only the top-N reusable hot kernels are enforced.
_DEFAULT_HOT_KERNEL_GATE_TOP_N = 5

# How many hot / skipped kernels ``record_trace_analyze`` keeps in the trace
# summary (the ``hot_kernels_top15`` / ``kernel_roofline_top15`` field names
# reflect this value).
_TRACE_HOT_KERNEL_TOP_N = 15

# Per-action audit history cap (``<action>_attempts`` lists keep most recent N).
_DEFAULT_ATTEMPTS_HISTORY = 20

# Global ``last_action_failures`` rolling-log cap.
_DEFAULT_LAST_FAILURES = 10

# phase_history cap (record_phase_transition).
_PHASE_HISTORY_CAP = 100

# Lifecycle-event log cap. Unlike phase_history (≤6 transitions),
# lifecycle events fire at every step boundary (trace_analyze /
# run_optimization / integrate / report, ×N kernels ×M rounds), so the cap
# is generous but still bounds state.json growth on a long run.
_LIFECYCLE_CAP = 500

# roofline_snapshots history cap (record_trace_analyze).
_ROOFLINE_SNAPSHOTS_CAP = 50

# gap ledger caps; both enforced in upsert_gap.
_GAPS_MAX_ENTRIES = 50
_GAPS_ATTEMPTS_HISTORY = 20

# Long-run (multi-day) bounded-growth caps for append-only telemetry ledgers.
# Tail-trim idiom (``list[-CAP:]``) matching phase_history / roofline_snapshots.
# Any derived totals computed over a trimmed ledger saturate at the cap; that is
# advisory only — authoritative control counters (e.g.
# ``consecutive_config_only_rounds``) are tracked separately and never trimmed.
_INTERVENTION_MIX_CAP = 500
_SPECIALIST_ROUNDS_CAP = 200
_SEEN_PR_IDS_CAP = 2000
_WINNERS_HISTORY_CAP = 200
# Negative ledger (explore_search["tested"]); dict keyed by fingerprint, oldest
# insertion-order keys evicted first. Bounds state.json across multi-day runs.
_EXPLORE_TESTED_CAP = 5000

# Per-action audit trail kinds; kernel_agent-owned actions excluded (dedicated structures).
_AUDIT_ACTIONS: frozenset[str] = frozenset(
    {
        "baseline",
        "profile",
        "sweep",
        "explore",
        # ``roofline`` runs profile + trace_analyze atomically; audited for snapshot id/path visibility.
        "roofline",
    }
)

# audit-action name -> (result-dict key, prompt-display label / key_metric_kind).
_KEY_METRIC_MAP: dict[str, tuple[str, str]] = {
    "baseline": ("output_throughput", "output_throughput"),
    "profile": ("output_throughput", "output_throughput"),
    "sweep": ("output_throughput", "output_throughput"),
    "explore": ("best_gain_pct", "gain_pct"),
    # ``roofline`` key metric is the monotonic snapshot id.
    "roofline": ("snapshot_id", "snapshot_id"),
}


#: top-level state.json schema version; absent key treated as v1 and migrated to LATEST_STATE_SCHEMA_VERSION on first save.
LATEST_STATE_SCHEMA_VERSION: int = 2


# Legacy key renames applied once on load: ``extra_sglang_args`` -> ``extra_server_args``.
_PHASE4_LEGACY_KEY_RENAMES: dict[str, str] = {
    "extra_sglang_args": "extra_server_args",
    "candidate_extra_sglang_args": "candidate_extra_server_args",
}


def _cap_tested_ledger(tested: dict[str, Any]) -> dict[str, Any]:
    """Bound the explore_search negative ledger for multi-day runs.

    ``tested`` is keyed by canonical fingerprint; Python dicts preserve insertion
    order, so retaining the last ``_EXPLORE_TESTED_CAP`` keys evicts the oldest
    rejections first. Dropping a stale rejection only risks one re-exploration,
    which the round-level dedup still catches in-session.

    Args:
        tested (dict[str, Any]): The explore_search negative ledger keyed by
            canonical fingerprint.

    Returns:
        dict[str, Any]: The ledger trimmed to the most recent
            ``_EXPLORE_TESTED_CAP`` keys, or an empty dict when ``tested`` is
            not a dict.
    """
    if not isinstance(tested, dict) or len(tested) <= _EXPLORE_TESTED_CAP:
        return tested if isinstance(tested, dict) else {}
    keys = list(tested.keys())[-_EXPLORE_TESTED_CAP:]
    return {k: tested[k] for k in keys}


def _stamp_cycle_on_tested(
    tested: dict[str, Any],
    cycle: int,
    bottleneck: str = "",
) -> dict[str, Any]:
    """Bucket negative-ledger entries by macro-cycle + bottleneck (R3).

    The executor builds ``tested`` without cycle awareness; SharedState is the
    single point that knows ``macro_cycle`` and the live ``bottleneck``. Existing
    ``cycle`` / ``bottleneck`` values are preserved so an entry stays attributed
    to the cycle + bottleneck that first rejected it, enabling per-cycle /
    per-bottleneck bucketing of veto fingerprints across bottleneck shifts.

    Args:
        tested (dict[str, Any]): The negative ledger keyed by fingerprint;
            entry dict values are stamped in place.
        cycle (int): The macro-cycle to attribute newly-stamped entries to.
        bottleneck (str): The live bottleneck label to stamp; blank values
            skip the bottleneck stamp.

    Returns:
        dict[str, Any]: The same ``tested`` mapping with ``cycle`` /
            ``bottleneck`` stamped on entries lacking them, or an empty dict
            when ``tested`` is not a dict.
    """
    if not isinstance(tested, dict):
        return {}
    bn = (bottleneck or "").strip()
    for v in tested.values():
        if isinstance(v, dict):
            if "cycle" not in v:
                v["cycle"] = int(cycle)
            if bn and "bottleneck" not in v:
                v["bottleneck"] = bn
    return tested


def _stamp_cycle_on_rejected(
    rejected: list[Any],
    cycle: int,
    bottleneck: str = "",
) -> list[Any]:
    """Bucket rejected entries by macro-cycle + bottleneck (R3).

    Args:
        rejected (list[Any]): The rejected-entries list; dict items are
            stamped in place.
        cycle (int): The macro-cycle to attribute newly-stamped entries to.
        bottleneck (str): The live bottleneck label to stamp; blank values
            skip the bottleneck stamp.

    Returns:
        list[Any]: The same ``rejected`` list with ``cycle`` /
            ``bottleneck`` stamped on entries lacking them, or an empty list
            when ``rejected`` is not a list.
    """
    if not isinstance(rejected, list):
        return []
    bn = (bottleneck or "").strip()
    for v in rejected:
        if isinstance(v, dict):
            if "cycle" not in v:
                v["cycle"] = int(cycle)
            if bn and "bottleneck" not in v:
                v["bottleneck"] = bn
    return rejected


def _migrate_legacy_extra_sglang_args_keys(obj: Any) -> int:
    """Recursively rewrite legacy ``extra_sglang_args`` field names in-place; returns count rewritten (canonical kept when both present).

    Args:
        obj (Any): An arbitrarily nested structure (dict / list / scalar);
            dicts and lists are walked recursively and mutated in place.

    Returns:
        int: The number of legacy keys rewritten or dropped across the whole
            structure.
    """
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


from ._shared_state.render import _RenderMixin


from ._shared_state.explore_state import _ExploreStateMixin


@dataclass
class SharedState(_RenderMixin, _ExploreStateMixin):
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
    # Degraded-mode advisory: True when the run is knowingly running a
    # multimodal checkpoint on the text-only path (--allow-mm-text-fallback).
    # Advisory only — never a stop_reason, never gates Objective/scoring.
    degraded_mode: bool = False
    # Structured degraded-mode / model-compat warnings (e.g. multimodal text
    # fallback). Surfaced verbatim in reports/final.{json,md}.
    model_warnings: list[dict[str, Any]] = field(default_factory=list)
    # KB tags from config.json (``architectures`` + ``model_type``); stamped into recipe-snapshot ``extras`` so fine-tuned models carry base arch identity.
    model_architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    # config.json-derived structural summary (attention_type / heads / MoE / quant); persisted in state.json for downstream collectors.
    model_info: dict = field(default_factory=dict)
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
    # Profile-phase output length (from --profile-osl). 0 = unset (profile
    # defaults to min(osl, 1024)). Persisted so a fresh-shell resume keeps the
    # operator's explicit profile OSL instead of reverting to the default.
    profile_osl: int = 0
    max_model_len: int = 0
    kernel_enabled: bool = True
    # KERNEL-phase optimizer: "native" (GEAK/per-kernel loop, default) or
    # "perfskills" (one-shot whole-pipeline e2e optimizer cloned from upstream;
    # see kernel-agent/tools/backends/perfskills_runner.py).
    kernel_optimizer: str = "native"
    # Snapshot of the last PerfSkills e2e run (result.json + final_launch.sh /
    # bench_e2e.sh handles the SWEEP phase reuses).
    perfskills_result: dict[str, Any] = field(default_factory=dict)
    # When False (``--no-explore``) EXPLORE is skipped: PRELUDE/FRAMEWORK_AGENT route to KERNEL (or SWEEP).
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
    # Discarded first-round tput from the baseline cold-start double-run.
    # Kept only for audit/debugging; conclusion fields and gain math use the
    # hot measure-round value in ``baseline_tput``.
    baseline_cold_tput: float = 0.0
    # Deprecated mirror of the hot measure-round tput. Kept for readers that
    # already inspect this field; it should match ``baseline_tput`` whenever
    # the double-run path is eligible.
    baseline_hot_tput: float = 0.0
    baseline_accuracy: float = 0.0
    # Standalone baseline-arm roofline ceiling computed right after baseline
    # lands; backs up snapshot ceiling so the frontend has data even when the
    # roofline (profile + trace_analyze) step fails. Empty until baseline runs.
    baseline_roofline_ceiling: dict[str, Any] = field(default_factory=dict)
    baseline_failure_streak: int = 0
    baseline_arg_error_streak: int = 0
    # Combined backstop: ANY baseline failure (regardless of error_class) counts
    # here. Mixed classes split the per-class streaks above so neither hits its
    # threshold; this total catches that and fast-fails (P5 anti time-exhaustion).
    baseline_total_failures: int = 0
    # One-shot: cuda-graph capture failure asks the next baseline to retry
    # by disabling cuda-graph capture (framework-correct flag injected by the
    # BaselineExecutor). Set on failure, consumed by BaselineExecutor.
    baseline_eager_fallback: bool = False
    # Enablement path (framework-agent) state.
    # ``enablement_launch_log``: captured launch/traceback text when baseline
    #   cannot launch.
    # ``enablement_dispatched``: in-flight guard that an authoring attempt is
    #   queued/running; cleared when the attempt REVERTs.
    # ``enablement_attempts``: number of dispatches (drives candidate rotation
    #   and idempotency).
    # ``enablement_succeeded``: terminal KEEP guard.
    enablement_launch_log: str = ""
    enablement_dispatched: bool = False
    enablement_attempts: int = 0
    enablement_succeeded: bool = False
    # ``enablement_kept_patches``: ordered, deduped list of patch file paths
    #   from prior enablement rounds that made *forward progress* (cleared the
    #   previous crash but revealed a new, deeper gap). These are re-applied as
    #   a base before the next round's patch so serial gaps stack instead of
    #   being reverted and re-derived. See ``_maybe_rearm_enablement``.
    enablement_kept_patches: list = field(default_factory=list)
    # ``enablement_setup_commands``: ordered, deduped list of allowlisted
    #   environment-setup shell commands (e.g. ``pip install transformers>=5.12``,
    #   ``apt-get install -y gh``) that prior enablement rounds ran to make the
    #   combo buildable/runnable. Re-run (idempotently) by integrate_patch before
    #   applying patches + booting, so an install a specialist relied on is
    #   durable and reproducible across re-bench / pod restarts rather than an
    #   ephemeral side effect of one specialist session. Stacked like
    #   ``enablement_kept_patches``.
    enablement_setup_commands: list = field(default_factory=list)
    # ``enablement_stall_streak``: consecutive enablement rounds that neither
    #   became runnable nor advanced to a new failure signature. When it reaches
    #   ``_ENABLEMENT_MAX_STALL`` the loop stops with stop_reason
    #   ``enablement_stalled`` instead of spinning on the same gap forever.
    enablement_stall_streak: int = 0
    # Launch-log hashes already recorded as needs_human_review (UNKNOWN /
    # non-actionable failures); at most one review record per distinct log.
    enablement_human_review_logged: list = field(default_factory=list)
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
    # This is the COLD (warmup-round) full boot+bench wall-clock — the hard-cap anchor.
    baseline_runtime_sec: float = 0.0
    # Baseline WARM measure-round wall-clock (client-only, no boot). Anchors the explore
    # decision-round overtime kill so it is post-startup apples-to-apples with the warm variant
    # measurement. Zero => fall back to baseline_runtime_sec (cold anchor).
    baseline_warm_runtime_sec: float = 0.0
    current_best: dict[str, Any] = field(default_factory=dict)
    # Reference launch recipe (from --reference-script or auto-discovery): lowest-priority
    # base server args/envs seeding every baseline. Fact-layer => persisted + restored on resume.
    reference_server_args: str = ""
    reference_envs: dict[str, str] = field(default_factory=dict)
    reference_model: str = ""
    reference_source: str = ""
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
    # Resume sentinels. ``pending_integrate`` is written before a
    # non-transactional integrate_patch window and normally cleared after
    # stack/current best are persisted. After a crash, resume recovery replays
    # or rolls back that window before clearing the sentinel.
    # ``resume_pending_revalidation`` tells reports/ops that accepted stack
    # entries need a fresh post-resume stack rebench.
    pending_integrate: dict[str, Any] = field(default_factory=dict)
    resume_pending_revalidation: bool = False
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
    # Unix timestamps of recent crashes (bounded), used for the trailing-window
    # emergency-stop rate so old crashes age out instead of accumulating forever.
    crash_timestamps: list[float] = field(default_factory=list)
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
    # FRAMEWORK_AGENT phase toggle (PRELUDE → FRAMEWORK_AGENT → EXPLORE); ``--no-framework-agent`` opts out. Absent from PHASE_LLM_PROPOSABLE_ACTIONS so PolicyGate R1 blocks LLM proposal.
    framework_agent_phase_enabled: bool = True
    # FRAMEWORK progress: one entry per candidate benchmark; used by breakdown + plateau exit judgment.
    framework_agent_phase_progress: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # One row per phase-discover batch; read by exit_normal_framework_agent plateau gate (3 batches <1% => exit).
    framework_agent_batches: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # True when FRAMEWORK loop has no more candidates; compute_next_phase uses it for framework_agent_phase_done exit.
    framework_agent_phase_done: bool = False
    # Consecutive ``fa phase-discover`` failures; phase marked done only after DISCOVER_FAILURE_RETRY_LIMIT (default 3).
    framework_agent_discover_failures: int = 0
    # Consecutive empty-but-valid ``fa phase-discover`` batches within the current
    # FRAMEWORK phase; tolerate up to DISCOVER_FAILURE_RETRY_LIMIT before exiting so a
    # transient upstream blip (cortex/PR-Monitor/GitHub) doesn't skip the phase.
    # Reset to 0 on any non-empty batch.
    framework_agent_empty_discoveries: int = 0
    # Consecutive FRAMEWORK_AGENT phase completions that discovered zero candidates
    # (empty_discovery). Drives the Step-1 advisory ("framework phase ineffective");
    # reset whenever a phase completes having tested >=1 candidate.
    framework_consecutive_empty_discoveries: int = 0
    # Per-repo candidate cap for ``fa phase-discover``; 0 => DEFAULT_FRAMEWORK_MAX_CANDIDATES.
    framework_max_candidates: int = 0
    # FRAMEWORK Critic-gate decisions; cache lets resume avoid re-calling the Critic.
    framework_agent_critic_decisions: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # Default True: FRAMEWORK pump dispatches a write-capable serving_specialist per candidate alongside diff-only track. False restores diff-only.
    framework_agent_authoring_enabled: bool = True
    # Maps an authoring specialist task_id -> originating FRAMEWORK candidate id
    # (PR URL). The downstream integrate_patch task only carries
    # ``specialist_task_id``, so the authored-outcome bridge resolves the real
    # candidate id through this map; without it the progress row is keyed on the
    # specialist/integrate task_id and never matches the PR-URL key that
    # ``_select_next_framework_agent_candidate`` checks -> pump livelock.
    framework_agent_specialist_candidate_map: dict[str, str] = field(
        default_factory=dict,
    )
    # Re-author rounds per candidate id (capped); needs_review verdicts increment.
    specialist_reauthor_attempts: dict[str, int] = field(
        default_factory=dict,
    )
    # P0-4 backstop: per-candidate-key count of Critic-review submissions. If a
    # candidate is submitted for review more than the abort threshold (a
    # terminal-row leak somewhere let it be re-selected), the pump force-stamps
    # ``repeated_review_abort`` and stops re-selecting it, so no single candidate
    # can burn the whole phase budget even if a new path forgets its terminal row.
    framework_agent_review_counts: dict[str, int] = field(
        default_factory=dict,
    )
    # Default True: Coordinator auto-analysis is ``roofline`` (profile+trace_analyze+analysis.md); False enqueues plain ``profile``. Absent from PHASE_LLM_PROPOSABLE_ACTIONS (PolicyGate R1 denies LLM proposal).
    enable_roofline: bool = True
    # ExploreExecutor per-variant overtime kill multiplier; >0 kills the decision run past anchor*ratio (outcome='KILLED_OVERTIME'). Anchor is the WARM measure time when warm-decision is active, else the cold baseline. The kill clock starts at the server-ready marker so it measures only the post-ready (pure hot client) phase, matching the anchor. Warmup + stack-rebench exempt. Default 2.0x.
    explore_overtime_kill_ratio: float = 2.0
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
    # Most recent run_optimization dispatch skipped with no eligible kernels
    # (empty batch, no named kernel). Recorded honestly as a non-failure so the
    # breakdown can surface it; it is otherwise invisible in sbd (the result has
    # no backend and no kernel_id).
    last_kernel_opt_dispatch_skip: dict[str, Any] = field(default_factory=dict)
    # Per-action audit (kernel parity): each ``last_<action>`` is the most recent attempt snapshot; ``<action>_attempts`` is a capped list.
    last_baseline: dict[str, Any] = field(default_factory=dict)
    last_profile: dict[str, Any] = field(default_factory=dict)
    # GEAK FP8 GEMM tuning snapshot (kernel_agent-owned): aiter A8W8 tuned CSV + SGLang dispatch patch before kernel_opt.
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
    # Unified explore ledger — persistent explore-search state for merged ``explore``; ``tested`` keyed by canonical_fingerprint. ``accepted`` includes inlined stack-rebench survivors; rebench-evicted entries live in rejected with reason='stack_unstable'.
    explore_search: dict[str, Any] = field(default_factory=dict)
    # specialist sub-agent rolling state; one entry per EXPLORE round (round_id, tasks, proposals_total/kept/rejected/skipped, etc.).
    specialist_rounds: list[dict[str, Any]] = field(default_factory=list)
    # Per-domain "empty proposal_set" streak; reset on non-empty specialist_done. Robustness escalates on persistent emptiness.
    specialist_domain_empty_streak: dict[str, int] = field(default_factory=dict)
    # Per-kb_anchor coverage counters (per-domain lower-bound).
    # ``rounds_since_last_specialist`` — EXPLORE rounds since a specialist for
    # that anchor was dispatched; ``rounds_since_last_keep`` — rounds since a
    # KEEP landed for it. Both ++ once per EXPLORE round (bump_domain_round_counters),
    # reset on dispatch / KEEP. Threshold consumed by the Coordinator escalation.
    rounds_since_last_specialist: dict[str, int] = field(default_factory=dict)
    rounds_since_last_keep: dict[str, int] = field(default_factory=dict)
    # Legacy session_steward slots (steward removed); kept only for resume + report.py back-compat, never written.
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
    # Research scout bookkeeping; master switch ``--no-research-scout``; seen_pr_ids shared with FRAMEWORK to avoid re-mining.
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
    # Static-recon specialist bookkeeping (explore-opt-5 capability A); master
    # switch ``--no-static-recon``. PRELUDE-only one-shot source reconnaissance.
    static_recon_enabled: bool = True
    static_recon_runs: int = 0
    # Total specialist dispatches in current EXPLORE entry; reset on fresh entry. Robustness detects specialist storms.
    explore_specialist_dispatched_count: int = 0
    # Research-lane capacity locked at session start (core field; PolicyGate denies mid-session mutation).
    research_lane_capacity: int = 1
    # GPU pool capacity for needs_gpu specialists (0 disables); locked at
    # session start. The dataclass default is a placeholder for tests/direct
    # construction; the CLI/manifest default is whole-machine GPU detection.
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
    # Crash-safe stack-validation checkpoints (SWEEP-entry combo E2E).
    pending_stack_validation_result: dict[str, Any] = field(default_factory=dict)
    pending_stack_validation_apply_results: list[dict[str, Any]] = field(
        default_factory=list,
    )
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
    # ``phase`` — run-level pipeline phase (PRELUDE/FRAMEWORK_AGENT/EXPLORE/KERNEL/SWEEP/CLOSE); Coordinator-only (CORE_STATE_FIELDS). Empty => not yet initialised.
    phase: str = ""
    # ISO UTC timestamp the current phase was entered (breakdown.phase_segments + budget judge).
    phase_started_ts: str = ""
    # Unix epoch matching ``phase_started_ts`` so the budget judge skips ISO re-parsing.
    phase_started_unix: float = 0.0
    # Append-only log of phase transitions (rows from phase_state.make_history_row; reason in PHASE_EXIT_REASONS). Capped at _PHASE_HISTORY_CAP.
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    # Append-only operator-facing lifecycle log (#266). Each row is built
    # by :func:`phase_state.make_lifecycle_event` and records a phase/step
    # boundary (START / END / ERROR) plus the artifact paths it produced,
    # so a launcher can surface "phase X ran, outputs at <path>" in chat.
    # Coordinator-only writer (PolicyGate adds it to ``CORE_STATE_FIELDS``);
    # capped at ``_LIFECYCLE_CAP``.
    lifecycle: list[dict[str, Any]] = field(default_factory=list)
    # Wall-clock budget percentages per phase (from CLI flags/defaults); persisted for resume. Empty => library defaults.
    phase_budget_pct: dict[str, float] = field(default_factory=dict)
    # Cyclic phase machine: macro-cycle counter. cycle 0 is the first pass
    # (PRELUDE→…→SWEEP); each SWEEP→EXPLORE loopback increments it. Stamped onto
    # every phase_history row. 0 for legacy/non-cyclic runs.
    macro_cycle: int = 0
    # Per-cycle budget: wall-clock minutes allotted to ONE macro-cycle.
    # When > 0 the per-phase budget math (phase_budget_remaining_seconds /
    # EXPLORE force-exit) is computed against this window instead of the total
    # ``max_minutes``. 0 disables (legacy: phase budgets are % of total).
    cycle_minutes: float = 0.0
    # Global-convergence tracking: validated cumulative gain at the current
    # macro-cycle's start, and the consecutive no-gain cycle streak. Updated by
    # the Coordinator on each SWEEP→EXPLORE loopback / terminal close.
    gain_at_cycle_start: float = 0.0
    no_gain_cycle_streak: int = 0
    # Cyclic bottleneck re-direction: set when a cyclic EXPLORE plateau
    # winds the cycle down with ``switch_bottleneck`` — the next macro-cycle's
    # orchestration prompt surfaces a redirect advisory steering specialists off
    # ``last_cycle_bottleneck`` toward the current dominant roofline bottleneck.
    # Cleared once the live top bottleneck actually drifts off the plateaued one.
    pending_bottleneck_switch: bool = False
    last_cycle_bottleneck: str = ""
    # Latest roofline saturation per specialist-domain family and
    # prev-cycle -> current-cycle bottleneck movement. Coordinator/record_trace
    # are the only writers; orchestration only sees advisory prompt text.
    saturated_directions: dict[str, dict[str, Any]] = field(default_factory=dict)
    bottleneck_shift: dict[str, Any] = field(default_factory=dict)
    # Deterministic per-cycle advisory focus log. Each entry is a
    # compact, persisted fact row so cycle strategy survives compaction/resume.
    cycle_strategy_log: list[dict[str, Any]] = field(default_factory=list)

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
    # Model-facing WarmStartContext built by ``cortex_t0`` from the KB
    # recipe row. PARALLEL to ``warm_start_recipe`` (which stays the raw
    # KB envelope that breakdown / kb_explorer read). Carries an explicit
    # ``status`` (``hit`` / ``seed_only`` / ``miss`` / ``error``) plus a
    # ready-to-replay ``recommended_replay`` champion and the experiential
    # lists, so warm-replay / specialist / ledger consumers branch on
    # status without re-deriving it from tier+confidence. Empty dict when
    # T0 was bypassed or failed.
    warm_start_context: dict[str, Any] = field(default_factory=dict)

    # structured gaps ledger: dedup'd unresolved bottlenecks (Coordinator-only _refresh_gaps; CORE_STATE_FIELDS); dedup keyed by canonical_id, attempts capped 20/gap, list capped _GAPS_MAX_ENTRIES.
    gaps: list[dict[str, Any]] = field(default_factory=list)

    # Orchestration working memory — durable compacted reasoning snapshot for compaction + crash-recovery rebuild; Coordinator-only writer, not in session_breakdown.
    orchestration_memory: dict[str, Any] = field(default_factory=dict)

    # Bounded rollback ring of prior good ``orchestration_memory`` records;
    # capped at 10 by the Coordinator. Lets a later degenerate compaction be
    # recovered from a prior snapshot via INFERENCE_OPTIMIZER_ORCH_MEMORY_ROLLBACK.
    orchestration_memory_history: list[dict[str, Any]] = field(default_factory=list)

    # Non-field instance attr (set in load_or_init / save): session dir used by
    # breakdown instrumentation. Plain class attr => not a dataclass field, so
    # asdict()/state.json never serialize it.
    _session_dir = None

    # Persistence
    @classmethod
    def state_path(cls, session_dir: Path) -> Path:
        """Return the canonical ``state.json`` path for a session directory.

        Args:
            session_dir (Path): The session root directory.

        Returns:
            Path: ``session_dir / "state.json"``.
        """
        return Path(session_dir) / "state.json"

    @classmethod
    def load_or_init(cls, session_dir: Path) -> "SharedState":
        """Load existing ``state.json`` or return a fresh blank instance.

        Reads and migrates the persisted state via :meth:`from_dict` when
        the file exists; otherwise constructs a default instance for a
        brand-new session.

        Args:
            session_dir (Path): The session root directory containing (or
                that will contain) ``state.json``.

        Returns:
            SharedState: The loaded-and-migrated state, or a fresh default
                instance when no ``state.json`` exists yet.
        """
        path = cls.state_path(session_dir)
        if not path.exists():
            inst = cls()
        else:
            with path.open(encoding="utf-8") as f:
                raw = json.load(f)
            inst = cls.from_dict(raw)
        # Remember the session dir so breakdown instrumentation can record
        # fragments at author time (non-field attr; not serialized by asdict).
        inst._session_dir = Path(session_dir)
        return inst

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SharedState":
        """Construct a :class:`SharedState` from a raw mapping, migrating it.

        Acts as the unified migration entry point: an absent
        ``schema_version`` is treated as 1, legacy keys are renamed to
        their canonical form, and unknown keys are dropped. The operation
        is idempotent and short-circuits when already at the latest schema.

        Args:
            raw: Decoded state mapping (e.g. from JSON on disk).

        Returns:
            A fully-populated, migrated :class:`SharedState` instance.
        """
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
        # Legacy scoreboard fields; already dropped by the filter, listed only to count/log in ``warn`` mode.
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
                size = len(payload) if isinstance(payload, (dict, list, str)) else 1
                legacy_seen.append((legacy, int(size)))
            filtered.pop(legacy, None)
        if legacy_seen:
            mode = (
                os.environ.get(
                    "INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES",
                    "drop",
                )
                .strip()
                .lower()
            )
            import logging as _logging

            log = _logging.getLogger(__name__)
            summary = ", ".join(f"{k}={n}" for k, n in legacy_seen)
            migration_events.append(f"§3.9 dropped scoreboard fields ({summary})")
            if mode == "warn":
                log.warning(
                    "v0.8 §3.9: dropped legacy scoreboard fields from "
                    "state.json (%s). set "
                    "--legacy-action-scores=drop to silence this.",
                    summary,
                )
            else:
                log.info(
                    "v0.8 §3.9: dropped legacy scoreboard fields from state.json (%s).",
                    summary,
                )
        # Normalize the unified ``explore_search`` ledger at load; winners/synergy history folded in.
        filtered["explore_search"] = cls._build_explore_search(
            existing=filtered.get("explore_search"),
            backend_winners_history=filtered.get("backend_winners_history"),
            params_winner_history=filtered.get("params_winner_history"),
            synergy_attempted=filtered.get("synergy_attempted"),
        )

        # fact-layer integrity check: strict (default) aborts when a fact-layer key was present but didn't load; lenient warns.
        if needs_migration and raw:
            mode = (
                os.environ.get(
                    "INFERENCE_OPTIMIZER_MIGRATION_MODE",
                    "strict",
                )
                .strip()
                .lower()
            )
            fact_layer_keys = (
                "baseline_tput",
                "baseline_cold_tput",
                "baseline_hot_tput",
                "baseline_accuracy",
                "current_best",
                "cumulative_gain",
                "cumulative_gain_validated",
                "optimization_stack",
                # Reference recipe must survive resume or restarts re-fail baseline.
                "reference_server_args", "reference_envs",
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
                incoming_version,
                LATEST_STATE_SCHEMA_VERSION,
                event_str,
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
        """Shape the unified ``explore_search`` ledger at load time; folds live history so resume preserves cross-round aggregation.

        Args:
            existing (Any): The persisted ``explore_search`` dict (or any
                value; non-dicts are treated as empty).
            backend_winners_history (Any): Legacy backend-winners rows folded
                into the unified ``winners_history``.
            params_winner_history (Any): Legacy params-winner rows folded into
                the unified ``winners_history``.
            synergy_attempted (Any): Legacy synergy combos folded (deduped)
                into ``synergy_attempted``.

        Returns:
            dict[str, Any]: The normalized ``explore_search`` ledger with all
                required keys defaulted and live history folded in.
        """
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
                wh.append(
                    {
                        "round_id": str(entry.get("round_id") or ""),
                        "variant_name": str(entry.get("variant_name") or entry.get("name") or ""),
                        "fingerprint": str(fp_val),
                        "gain_pct": entry.get("gain_pct"),
                        "extra_args": str(entry.get("extra_args") or entry.get("extra_server_args") or ""),
                        "extra_envs": dict(entry.get("extra_envs") or {}),
                        "provenance": str(entry.get("provenance") or ""),
                        "ts": str(entry.get("ts") or ""),
                    }
                )
        wh.sort(key=lambda r: (str(r.get("round_id") or ""), str(r.get("ts") or "")))
        out["winners_history"] = wh

        # synergy_attempted: fold live field + executor-side additions, deduped.
        sa_set: set[tuple[str, ...]] = set()

        def _normalize_combo(c: Any) -> tuple[str, ...] | None:
            """Normalize a synergy combo to a sorted tuple of flag names.

            Args:
                c (Any): A list of flag-name strings or a ``"+"``-joined
                    combo string.

            Returns:
                tuple[str, ...] | None: The sorted flag-name tuple, or
                    ``None`` when the input yields no usable names.
            """
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
        """Serialize this state to a plain JSON-compatible dict.

        Returns:
            dict[str, Any]: A deep ``dataclasses.asdict`` copy suitable for
                JSON serialization.
        """
        return asdict(self)

    def save(self, session_dir: Path) -> None:
        """Atomically write ``state.json`` (tmp file + ``os.replace``).

        Serializes via :meth:`to_dict` and writes to a temp file in the
        same directory before an atomic rename, so concurrent readers never
        observe a partial blob. The temp file is cleaned up on failure.

        Args:
            session_dir (Path): The session root directory; created if it
                does not already exist.
        """
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
        # Author-time breakdown capture: snapshot state-owned sections into the
        # recorder spool right after persisting. Best-effort; never blocks save.
        self._session_dir = Path(session_dir)
        try:
            from ..breakdown.recorder import instrument

            instrument.snapshot_state_sections(session_dir, self)
        except Exception:  # noqa: BLE001 — author-time capture must never block save
            log.debug("snapshot_state_sections failed", exc_info=True)
        # Read-only derived artifact: re-render current_setting.sh from the
        # current best route so the operator can audit / re-feed it via
        # --reference-script. Never parsed back (resume restores from state.json).
        try:
            cb = self.current_best or {}
            if cb:
                from ..reference_script import render_reference_script
                text = render_reference_script(
                    framework=os.environ.get("FRAMEWORK", "sglang"),
                    server_args=str(cb.get("extra_server_args") or ""),
                    envs=dict(cb.get("extra_envs") or {}),
                    model=self.reference_model or os.environ.get("MODEL_PATH"),
                )
                (Path(session_dir) / "current_setting.sh").write_text(
                    text, encoding="utf-8",
                )
        except Exception:  # noqa: BLE001 — derived artifact, never fatal
            log.debug("current_setting.sh render failed", exc_info=True)

    # Mutators (Coordinator only — LLM agents go via intents)
    def add_pruned_family(self, family: str) -> bool:
        """Idempotently mark an action family as pruned.

        Args:
            family (str): The action family identifier to prune.

        Returns:
            bool: ``True`` iff the family was newly added; ``False`` when it
                was already present.
        """
        if family in self.pruned_families:
            return False
        self.pruned_families.append(family)
        return True

    def is_pruned(self, family: str) -> bool:
        """Report whether an action family has been pruned.

        Args:
            family (str): The action family identifier to check.

        Returns:
            bool: ``True`` when ``family`` is in :attr:`pruned_families`.
        """
        return family in self.pruned_families

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
        """Forwarding shim — implementation in :mod:`.policy`."""
        from . import policy as _m
        return _m.record_policy_denial(self, action_name=action_name, rule=rule, hint=hint, intent_type=intent_type, tick=tick, intent_payload=intent_payload)

    def reset_policy_denial_streak(self, action_name: str) -> None:
        """Forwarding shim — implementation in :mod:`.policy`."""
        from . import policy as _m
        return _m.reset_policy_denial_streak(self, action_name)

    # stop_reason ENUM validator
    def set_stop_reason(
        self,
        value: str,
        *,
        strict: bool | None = None,
    ) -> str:
        """Validated writer for :attr:`stop_reason` (Inv-8.3 closed vocab): values outside ``STOP_REASON_VOCAB`` map to ``"unknown"`` (lenient) or raise (``strict=True``, default env ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON``). Returns value written.

        Args:
            value (str): The proposed stop reason; blank clears
                :attr:`stop_reason`.
            strict (bool | None): When ``True`` an out-of-vocab value raises;
                when ``None`` the mode is read from
                ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON``.

        Returns:
            str: The value actually written (``""``, the validated reason, or
                ``"unknown"`` in lenient mode).

        Raises:
            ValueError: When ``strict`` is enabled and ``value`` is not in
                ``STOP_REASON_VOCAB``.
        """
        from .phase_state import STOP_REASON_VOCAB, is_valid_stop_reason

        text = str(value or "").strip()
        if not text:
            self.stop_reason = ""
            return ""
        if is_valid_stop_reason(text):
            self.stop_reason = text
            return text
        if strict is None:
            strict_env = (
                os.environ.get(
                    "INFERENCE_OPTIMIZER_STRICT_STOP_REASON",
                    "",
                )
                .strip()
                .lower()
            )
            strict = strict_env in ("1", "true", "yes")
        if strict:
            raise ValueError(f"stop_reason={text!r} not in STOP_REASON_VOCAB ({sorted(STOP_REASON_VOCAB)!r})")
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
        """Stash the LLM-supplied hint for the next phase compute pass; unknown hints dropped (Inv-8.2: closed vocab). Returns value written.

        Args:
            hint (str): The proposed escalate hint; values outside the closed
                vocab (and blanks) are dropped.

        Returns:
            str: The hint actually stored (``""`` when dropped or blank).
        """
        from .phase_state import is_valid_escalate_hint

        text = str(hint or "").strip()
        if text and not is_valid_escalate_hint(text):
            return ""
        self.pending_escalate_hint = text
        return text

    def consume_pending_escalate_hint(self) -> str:
        """Pop the pending hint (recording consumption in audit fields) so the next tick doesn't re-trigger; returns cleared hint.

        Returns:
            str: The consumed hint (``""`` when none was pending).
        """
        hint = (self.pending_escalate_hint or "").strip()
        if not hint:
            return ""
        self.pending_escalate_hint = ""
        self.last_consumed_escalate_hint = hint
        self.last_consumed_escalate_hint_ts = _now_iso()
        return hint

    def enablement_close_guard_active(self) -> bool:
        """True while a not-yet-enabled run must be protected from premature close.

        Until a baseline can be *established*, the entire optimization loop is
        blocked on it — you cannot run EXPLORE/KERNEL/SWEEP without a baseline.
        So a ``skip_to_close`` (whether the Orchestration LLM or Robustness emits
        it) is *premature* here: giving up before the model even runs is not a
        legitimate finish. While this guard is active the ``skip_to_close`` hint
        is dropped; the only ways a not-yet-enabled run may terminate are the
        *honest* ones that do not route through ``skip_to_close``:

        * ``enablement_stalled`` — the enablement loop's own stall cap
          (:data:`_ENABLEMENT_MAX_STALL` consecutive no-progress rounds),
        * ``prelude_baseline_failed`` — repeated baseline failures with no
          enablement engaged,
        * the wall-clock deadline / time-exhausted exits,
        * hard aborts (crash threshold, signal, emergency, user stop).

        Returns:
            bool: ``True`` in PRELUDE / FRAMEWORK_AGENT while ``baseline_tput``
            has never gone positive and enablement has not yet succeeded.
        """
        phase = (self.phase or "").strip().upper()
        return (
            phase in ("PRELUDE", "FRAMEWORK_AGENT")
            and float(getattr(self, "baseline_tput", 0.0) or 0.0) <= 0.0
            and not bool(getattr(self, "enablement_succeeded", False))
        )

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
        """Forwarding shim — implementation in :mod:`.phase_state`."""
        from . import phase_state as _m
        return _m.record_phase_transition(self, to_phase=to_phase, reason=reason, evidence=evidence, ts=ts, ts_unix=ts_unix)

    def current_top_bottleneck(self) -> str:
        """Return the latest roofline snapshot's ``top_bottleneck`` ("" when none).

        Single accessor so the R3 redirect logic and prompt advisory read the
        same value (latest = ``roofline_snapshots[-1]``).

        Returns:
            str: The latest snapshot's ``top_bottleneck``, or ``""`` when no
                snapshot exists.
        """
        snaps = self.roofline_snapshots if isinstance(self.roofline_snapshots, list) else []
        if not snaps:
            return ""
        latest = snaps[-1]
        if isinstance(latest, dict):
            return str(latest.get("top_bottleneck") or "")
        return ""

    def mark_bottleneck_switch(self, prev_bottleneck: str = "") -> None:
        """Flag that the next macro-cycle should redirect off ``prev_bottleneck`` (R3).

        Called when a cyclic EXPLORE plateau winds the cycle down. Records the
        bottleneck we plateaued on so the redirect advisory can steer specialists
        away from it; falls back to the live top bottleneck when not supplied.

        Args:
            prev_bottleneck (str): The bottleneck plateaued on; when blank the
                live top bottleneck is used instead.
        """
        self.pending_bottleneck_switch = True
        pb = (prev_bottleneck or "").strip() or self.current_top_bottleneck()
        if pb:
            self.last_cycle_bottleneck = pb

    def clear_bottleneck_switch(self) -> None:
        """Clear the pending bottleneck-switch handoff (R3)."""
        self.pending_bottleneck_switch = False
        self.last_cycle_bottleneck = ""

    def maybe_clear_bottleneck_switch_on_drift(self, new_top_bottleneck: str) -> bool:
        """Retire a pending switch once the live top bottleneck has drifted (R3).

        Returns True when the flag was cleared. A fresh roofline whose top
        bottleneck differs from the plateaued one means the redirect succeeded,
        so the orchestration prompt should stop nagging.

        Args:
            new_top_bottleneck (str): The current live top bottleneck.

        Returns:
            bool: ``True`` when a pending switch was cleared, ``False``
                otherwise.
        """
        if not bool(getattr(self, "pending_bottleneck_switch", False)):
            return False
        nt = (new_top_bottleneck or "").strip()
        if nt and nt != (self.last_cycle_bottleneck or ""):
            self.clear_bottleneck_switch()
            return True
        return False

    def record_lifecycle_event(
        self,
        *,
        step: str,
        status: str,
        phase: str | None = None,
        label: str | None = None,
        artifacts: dict[str, str] | None = None,
        detail: str = "",
        duration_s: float | None = None,
        ts: str | None = None,
    ) -> dict[str, Any]:
        """Forwarding shim — implementation in :mod:`.phase_state`."""
        from . import phase_state as _m
        return _m.record_lifecycle_event(self, step=step, status=status, phase=phase, label=label, artifacts=artifacts, detail=detail, duration_s=duration_s, ts=ts)


    def increment_crash_count(self, by: int = 1) -> int:
        """Increment the cumulative crash counter and record crash times.

        ``crash_count`` stays a monotonic telemetry total; each crash also
        appends the current time to :attr:`crash_timestamps` (bounded to the
        most recent entries) so the emergency stop can use a trailing-window
        rate instead of the never-decaying total.

        Args:
            by (int): Amount to add to :attr:`crash_count` (default 1).

        Returns:
            int: The post-increment crash count.
        """
        self.crash_count += by
        now = time.time()
        for _ in range(max(1, int(by))):
            self.crash_timestamps.append(now)
        if len(self.crash_timestamps) > _CRASH_TIMESTAMP_CAP:
            del self.crash_timestamps[:-_CRASH_TIMESTAMP_CAP]
        return self.crash_count

    def recent_crash_count(self, *, window_sec: float, now: float | None = None) -> int:
        """Count crashes recorded within the trailing ``window_sec`` seconds.

        Args:
            window_sec (float): Trailing window width in seconds.
            now (float | None): Reference time; defaults to ``time.time()``.

        Returns:
            int: Number of crash timestamps newer than ``now - window_sec``.
        """
        ref = time.time() if now is None else now
        cutoff = ref - float(window_sec)
        return sum(1 for t in self.crash_timestamps if t >= cutoff)

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
        """Persist a compact Coordinator exception summary for postmortems.

        Args:
            tick (int): The Coordinator tick at which the exception fired.
            stage (str): The tick-loop stage that raised.
            exc_type (str): The exception class name.
            message (str): The exception message (truncated to 1000 chars).
            traceback_text (str): The formatted traceback (truncated to
                12000 chars).
            agent (str): Optional agent identifier associated with the stage.

        Returns:
            dict[str, Any]: The recorded exception summary now stored in
                :attr:`last_tick_exception`.
        """
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
        """Merge a non-empty changes dict into this state; does NOT re-validate the role/source allowlist (PolicyGate filters upstream). Returns fields actually written.

        Args:
            changes (dict[str, Any]): Field-name -> value mapping to apply;
                keys not matching a dataclass field are ignored.
            allow_core (bool): When False, keys in
                :data:`policy.CORE_STATE_FIELDS` are dropped (defense in depth:
                PolicyGate already rejects them upstream, but this ensures a
                caller reaching here off the intent path still cannot write
                Coordinator-only fields). When True, all known fields are
                written (Coordinator/trusted callers).

        Returns:
            dict[str, Any]: The subset of ``changes`` actually written to
                dataclass fields.
        """
        if not changes:
            return {}
        core_fields: frozenset[str] = frozenset()
        if not allow_core:
            # Lazy import to avoid a shared_state <-> policy import cycle; reuse
            # the single CORE_STATE_FIELDS source of truth rather than a copy.
            from .policy import CORE_STATE_FIELDS

            core_fields = CORE_STATE_FIELDS
        applied: dict[str, Any] = {}
        for key, value in changes.items():
            if key not in self.__dataclass_fields__:
                continue
            if key in core_fields:
                log.warning(
                    "apply_changes: dropping core state field %r (allow_core=False)",
                    key,
                )
                continue
            setattr(self, key, value)
            applied[key] = value
        return applied


    def _resolve_kernel_patch_identity(
        self,
        payload: dict[str, Any] | None,
    ) -> tuple[str, str, str, str]:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m._resolve_kernel_patch_identity(self, payload)

    def kernel_patch_key(self, payload: dict[str, Any] | None) -> str:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.kernel_patch_key(self, payload)

    def find_rejected_kernel_patch(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.find_rejected_kernel_patch(self, payload)

    @staticmethod
    def _is_integrate_fault(result: dict[str, Any]) -> bool:
        """True when an integrate result is an integration *fault*, not a verdict.

        A fault is an environment/apply/bench crash that prevented the patch
        from being fairly measured. It must not burn the REVERT quota.

        The discriminator is ``status``, not ``decision`` or ``error_class``:
        ``integrate_handler`` only ever reaches a genuine gate verdict (KEEP /
        REVERT / NEEDS_REVIEW) on a real measurement, and always stamps those
        with ``status:"ok"``. Every unmeasured path returns ``status:"failed"``
        — and several of those *also* carry ``decision:"REVERT"`` for backward
        compat (e.g. apply-failed, "re-baseline did not succeed"), so keying on
        ``decision`` alone misclassifies them as terminal reverts and discards
        the patch on the first environment hiccup. Keying on ``status="failed"``
        catches them all, including fault paths that forget a top-level
        ``error_class``.

        The :data:`_INTEGRATE_FAULT_ERROR_CLASSES` membership test is a
        secondary signal kept for robustness if a future caller emits a fault
        with a non-``failed`` status.
        """
        status = str(result.get("status") or "").strip().lower()
        if status == "failed":
            return True
        err_class = str(result.get("error_class") or "").strip()
        return err_class in _INTEGRATE_FAULT_ERROR_CLASSES

    def record_kernel_integrate_result(
        self,
        result: dict[str, Any],
        *,
        max_attempts: int = 3,
        keep_threshold_pct: float = 1.0,
        max_fault_attempts: int = _MAX_INTEGRATE_FAULT_ATTEMPTS,
    ) -> dict[str, Any] | None:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.record_kernel_integrate_result(
            self,
            result,
            max_attempts=max_attempts,
            keep_threshold_pct=keep_threshold_pct,
            max_fault_attempts=max_fault_attempts,
        )

    def record_kernel_opt(self, result: dict[str, Any]) -> None:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.record_kernel_opt(self, result)

    def record_gemm_tuning(self, result: dict[str, Any]) -> None:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.record_gemm_tuning(self, result)

    # Multi-KEEP integrate queue helpers.
    def _kernel_ids_in_optimization_stack(self) -> set[str]:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m._kernel_ids_in_optimization_stack(self)

    def _source_files_in_optimization_stack(self) -> set[str]:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m._source_files_in_optimization_stack(self)

    def _kernel_ids_with_integrate_attempts(self) -> set[str]:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m._kernel_ids_with_integrate_attempts(self)

    def integrate_attempt_count_for_kernel(self, kernel_id: str) -> int:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.integrate_attempt_count_for_kernel(self, kernel_id)

    def _kernel_trace_impact_pct(self, kernel_id: str) -> float:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m._kernel_trace_impact_pct(self, kernel_id)

    def next_pending_keep_kernel_id(self) -> str:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.next_pending_keep_kernel_id(self)

    def pending_keep_kernel_ids(self) -> list[str]:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.pending_keep_kernel_ids(self)

    @property
    def has_keep_pending_integrate(self) -> bool:
        """True when kernel KEEP results still await kernel ``integrate``.

        This is separate from ``pending_integrate``, the integrate_patch
        crash-recovery sentinel.
        """
        from . import kernel_request_handlers as _m
        return _m.has_keep_pending_integrate(self)

    @property
    def kernel_opt_attempts_count(self) -> int:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.kernel_opt_attempts_count(self)

    # Hot-kernel report gate: report blocked until meaningful reusable hot kernels are attempted/rejected.
    def untried_hot_reusable_kernels(
        self,
        *,
        min_gpu_pct: float | None = None,
        top_n: int | None = None,
    ) -> list[str]:
        """Forwarding shim — implementation in :mod:`.kernel_request_handlers`."""
        from . import kernel_request_handlers as _m
        return _m.untried_hot_reusable_kernels(self, min_gpu_pct=min_gpu_pct, top_n=top_n)

    # Per-action audit (kernel parity for non-kernel actions)
    @staticmethod
    def _truncate_excerpt(value: Any, *, limit: int = 800) -> str | None:
        """Coerce ``value`` to str and trim to ``limit`` chars; None for falsy inputs (renderer shows ``err=(none)``).

        Args:
            value (Any): The value to coerce to a string excerpt.
            limit (int): Maximum retained length in characters (default 800).

        Returns:
            str | None: The trimmed string, or ``None`` when ``value`` is
                falsy.
        """
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
        """Pull the last ``limit`` chars from a subprocess error blob (stderr's actionable signal is at the end).

        Args:
            value (Any): The error blob to coerce and tail.
            limit (int): Maximum retained trailing length in characters
                (default 1000).

        Returns:
            str | None: The trailing slice of the string, or ``None`` when
                ``value`` is falsy.
        """
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
        """Append one attempt to ``<action>_attempts`` and refresh ``last_<action>``. Entry schema {ts, task_id, status, decision, key_metric, key_metric_kind, workspace, error_class, error_excerpt, stderr_tail, raw_result_path, reported_success, extras}. Returns the entry, or None when ``action`` not in the audit set (kernel_agent-owned actions use bespoke recorders). Does NOT call :meth:`save`.

        Args:
            action (str): The audited action name (must be in
                ``_AUDIT_ACTIONS``).
            task_id (str): The task id this attempt belongs to.
            status (str): The task status string.
            decision (str): The promotion decision string.
            result (dict[str, Any] | None): The action result envelope;
                ``None`` treated as empty.
            extras (dict[str, Any] | None): Optional extra fields recorded on
                the entry.
            max_history (int): Cap on retained ``<action>_attempts`` entries.

        Returns:
            dict[str, Any] | None: The recorded attempt entry, or ``None``
                when ``action`` is not audited.
        """
        if action not in _AUDIT_ACTIONS:
            return None
        attempts_attr = f"{action}_attempts"
        last_attr = f"last_{action}"
        if not hasattr(self, attempts_attr) or not hasattr(self, last_attr):
            return None
        result = result or {}
        metric_key, metric_kind = _KEY_METRIC_MAP.get(
            action,
            ("output_throughput", "output_throughput"),
        )
        raw_metric = result.get(metric_key)
        try:
            key_metric: float | None = float(raw_metric) if isinstance(raw_metric, (int, float)) else None
        except (TypeError, ValueError):
            key_metric = None
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "task_id": str(task_id or ""),
            "status": str(status or ""),
            "decision": str(decision or ""),
            "key_metric": key_metric,
            "key_metric_kind": metric_kind,
            "workspace": (str(result.get("workspace")) if result.get("workspace") else None),
            "error_class": (str(result.get("error_class")) if result.get("error_class") else None),
            "error_excerpt": self._truncate_excerpt(result.get("error")),
            # stderr_tail mirrors record_action_failure: only for subprocess
            # failures, so the breakdown exporter can surface the raw crash.
            "stderr_tail": (
                self._stderr_tail(result.get("error"))
                if str(result.get("error_class") or "") in {"subprocess_nonzero", "timeout"}
                else None
            ),
            "stderr_log_path": (str(result.get("stderr_log_path")) if result.get("stderr_log_path") else None),
            "raw_result_path": (str(result.get("raw_result_path")) if result.get("raw_result_path") else None),
            "reported_success": result.get("reported_success"),
            "extras": dict(extras or {}),
        }
        history: list[dict[str, Any]] = list(getattr(self, attempts_attr) or [])
        history.append(entry)
        if len(history) > max_history:
            history = history[-max_history:]
        setattr(self, attempts_attr, history)
        setattr(self, last_attr, dict(entry))
        # Author-time breakdown capture: one phase_timeline event per attempt.
        try:
            from ..breakdown.recorder import instrument

            instrument.record_phase_event(
                getattr(self, "_session_dir", None),
                action=action,
                entry=entry,
            )
        except Exception:  # noqa: BLE001 — author-time capture must never block record
            log.debug("record_phase_event capture failed", exc_info=True)
        return entry

    def record_action_failure(
        self,
        *,
        action: str,
        task_id: str,
        result: dict[str, Any] | None,
        max_history: int = _DEFAULT_LAST_FAILURES,
    ) -> dict[str, Any]:
        """Append one rich failure record to :attr:`last_action_failures` for self-correction; invoked for EVERY unpromotable task kind, unlike :meth:`record_action_attempt`.

        Args:
            action (str): The failed action name.
            task_id (str): The task id that failed.
            result (dict[str, Any] | None): The failure result envelope;
                ``None`` treated as empty.
            max_history (int): Cap on retained ``last_action_failures``
                entries.

        Returns:
            dict[str, Any]: The recorded failure entry.
        """
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
                self._stderr_tail(result.get("error")) if error_class_str in {"subprocess_nonzero", "timeout"} else None
            ),
            "stderr_log_path": (str(result.get("stderr_log_path")) if result.get("stderr_log_path") else None),
            "workspace": (str(result.get("workspace")) if result.get("workspace") else None),
            "raw_result_path": (str(result.get("raw_result_path")) if result.get("raw_result_path") else None),
            "reported_success": result.get("reported_success"),
        }
        history = list(self.last_action_failures or [])
        history.append(entry)
        if len(history) > max_history:
            history = history[-max_history:]
        self.last_action_failures = history
        return entry

    def _resolve_baseline_achieved_tput(self) -> float:
        """Baseline throughput for a baseline-arm roofline snapshot.

        Prefers ``baseline_tput``; falls back to ``last_baseline``'s
        ``tput``/``output_throughput`` so a state that lost ``baseline_tput``
        still stamps an achieved value (avoids an empty within/gap pct).

        Returns:
            float: The resolved baseline throughput, or ``0.0`` when none is
                available.
        """
        if isinstance(self.baseline_tput, (int, float)) and self.baseline_tput > 0:
            return float(self.baseline_tput)
        return _first_positive_tput(self.last_baseline)

    def _resolve_current_best_achieved_tput(self) -> float:
        """Optimized-arm throughput for a current_best roofline snapshot.

        Reads ``current_best``'s ``tput``/``output_throughput`` so a
        current_best-tagged snapshot keeps its arm even when ``tput`` is
        momentarily absent (avoids silently downgrading to the baseline arm).

        Returns:
            float: The resolved current_best throughput, or ``0.0`` when none
                is available.
        """
        return _first_positive_tput(self.current_best)

    def _scriptable_latency_roofline(
        self, framework: str, achieved_tput: float, kernel_roofline_path: Any
    ) -> tuple[float, float]:
        """Resolve the (measured e2e latency, ideal latency floor) ms pair.

        Scriptable/diffusion workloads have a compute-latency roofline rather
        than a decode-throughput one. Returns ``(0.0, 0.0)`` for serving
        frameworks and on any failure so the caller's serving path and legacy
        behaviour are unchanged.

        Args:
            framework: Session framework name.
            achieved_tput: Snapshot-time ``output_throughput`` (img/s for
                scriptable xDiT).
            kernel_roofline_path: Path to ``reports/kernel_roofline.json``; the
                ``diffusion_roofline.json`` sidecar sits at its run-dir root.

        Returns:
            ``(e2e_mean_ms, roofline_ideal_ms)``; either element is ``0.0``
            when unavailable.
        """
        try:
            from .. import framework_registry

            if not framework_registry.is_scriptable(framework):
                return 0.0, 0.0
            # img/s -> per-image e2e latency (ms); the achieved metric.
            e2e_mean_ms = float(
                framework_registry.primary_metric_value(framework, achieved_tput) or 0.0
            )
            # Ideal per-image latency floor from the diffusion roofline sidecar
            # (run_dir/diffusion_roofline.json; kernel_roofline is run_dir/reports/).
            roofline_ideal_ms = 0.0
            try:
                sidecar = Path(str(kernel_roofline_path)).parent.parent / "diffusion_roofline.json"
                if sidecar.is_file():
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                    analytic = data.get("analytic_dit_ceiling") if isinstance(data, dict) else None
                    totals = data.get("totals") if isinstance(data, dict) else None
                    if isinstance(analytic, dict) and float(analytic.get("ideal_compute_us") or 0.0) > 0:
                        roofline_ideal_ms = float(analytic["ideal_compute_us"]) / 1000.0
                    elif isinstance(totals, dict) and float(totals.get("sigma_ideal_roofline_us") or 0.0) > 0:
                        roofline_ideal_ms = float(totals["sigma_ideal_roofline_us"]) / 1000.0
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                roofline_ideal_ms = 0.0
            return e2e_mean_ms, roofline_ideal_ms
        except Exception:  # noqa: BLE001 — best-effort enrichment, never blocks
            return 0.0, 0.0

    def record_baseline_roofline_ceiling(self) -> dict[str, Any]:
        """Compute a standalone baseline-arm roofline ceiling and cache it.

        Runs purely off the baseline materialized yaml + model config (no
        profile trace), so it succeeds whenever baseline ran. Stamps the same
        ceiling/perfmodel fields a snapshot carries (trace-only fields stay
        absent) into ``baseline_roofline_ceiling`` as a frontend backup for
        when the roofline (profile + trace_analyze) step fails. Best-effort;
        returns ``{}`` and leaves the field empty on any failure.
        """
        try:
            from .roofline_ceiling import (
                RooflineBreakdown,
                compute_roofline_breakdown_from_state,
            )
            from .roofline_snapshot import (
                attach_perfmodel_breakdown,
                build_roofline_snapshot,
            )
        except Exception:  # noqa: BLE001 — import guard, best-effort
            return {}

        achieved = self._resolve_baseline_achieved_tput()
        breakdown = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")
        try:
            breakdown = compute_roofline_breakdown_from_state(
                self, arm="baseline",
            )
        except Exception:  # noqa: BLE001 — ceiling is best-effort
            pass
        peak_tput = float(breakdown.peak_tok_per_sec or 0.0)
        if peak_tput <= 0:
            return {}

        ts_iso = _now_iso()
        ceiling = build_roofline_snapshot(
            snapshot_id=None,
            ts=ts_iso,
            analysis_md_path="",
            theoretical_peak_tok_per_sec=peak_tput,
            achieved_tok_per_sec=achieved,
            mem_ceiling_tok_per_sec=float(breakdown.mem_tok_per_sec or 0.0),
            cmp_ceiling_tok_per_sec=float(breakdown.cmp_tok_per_sec or 0.0),
            bound_kind=breakdown.bound_kind,
            framework=str(getattr(self, "framework", "") or ""),
        )
        # Mark provenance: this is the baseline-arm ceiling backup, not a
        # trace-derived snapshot.
        ceiling["ceiling_arm"] = "baseline"

        # Per-op PerfModel breakdown + provenance (mirrors record_trace_analyze).
        attach_perfmodel_breakdown(ceiling, self, arm="baseline")

        self.baseline_roofline_ceiling = ceiling
        return ceiling

    def record_trace_analyze(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Write the canonical 11-field ``last_trace_analyze`` dict (single writer). ``roofline_snapshot_id`` increments monotonically; PR #321 retired ``last_trace_analyze_baseline`` (roofline_snapshots feeds report.py Roofline Comparison).

        Args:
            payload (dict[str, Any]): The trace_analyze task payload (supplies
                ``trace_input`` / ``trace_dir`` and optional ``roofline_arm``).
            result (dict[str, Any]): The trace_analyze result envelope; a
                non-dict result is a no-op.
        """
        if not isinstance(result, dict):
            return
        trace_input = (payload or {}).get("trace_input") or (payload or {}).get("trace_dir") or ""
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
        rocprof_by_kernel_id: dict[str, Any] = {}
        if kernel_roofline_path:
            try:
                roofline_payload = json.loads(Path(kernel_roofline_path).read_text(encoding="utf-8"))
                for row in roofline_payload.get("kernels") or []:
                    if not isinstance(row, dict) or not row.get("kernel_id"):
                        continue
                    rocprof_by_kernel_id[str(row["kernel_id"])] = row.get("rocprof_roofline")
            except Exception:  # noqa: BLE001 — sidecar merge is best-effort
                rocprof_by_kernel_id = {}
        for entry in hot[:_TRACE_HOT_KERNEL_TOP_N] if isinstance(hot, list) else []:
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
            rocprof_roofline = entry.get("rocprof_roofline")
            if rocprof_roofline is None and kid is not None:
                rocprof_roofline = rocprof_by_kernel_id.get(str(kid))
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
                "rocprof_roofline": rocprof_roofline,
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
                    "rocprof_roofline",
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
            for entry in skipped_sorted[:_TRACE_HOT_KERNEL_TOP_N]:
                skipped_summary.append(
                    {
                        "kernel_id": entry.get("kernel_id"),
                        "name": entry.get("name"),
                        "skip_reason": entry.get("skip_reason") or "",
                        "gpu_pct": entry.get("gpu_pct"),
                    }
                )

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
                    encoding="utf-8",
                    errors="replace",
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
            from .roofline_snapshot import (
                attach_perfmodel_breakdown,
                build_roofline_snapshot,
            )

            # Stamp decode-roofline ceiling (session constant) + measured tput (current_best.tput else baseline_tput).
            from .roofline_ceiling import (
                RooflineBreakdown,
                compute_roofline_breakdown_from_state,
            )

            # Resolve which arm this snapshot measures FIRST so the ceiling's
            # precision is anchored to the SAME arm as achieved: a baseline
            # snapshot keeps its dtype on baseline even after current_best is
            # promoted (else a later fp8 best would retro-inflate the ceiling).
            # An explicit ``roofline_arm`` on the payload wins — a delayed
            # PRELUDE roofline must record as baseline even though current_best
            # has already been promoted by warm-replay. Absent it, fall back to
            # inferring the arm from current_best.tput.
            forced_arm = str((payload or {}).get("roofline_arm") or "").strip()
            # Unknown arm values fall through to current_best inference; warn so
            # a typo'd payload tag is visible instead of silently mis-anchored.
            if forced_arm and forced_arm not in ("baseline", "current_best"):
                log.warning(
                    "record_trace_analyze: ignoring unknown roofline_arm=%r; falling back to current_best inference",
                    forced_arm,
                )
                forced_arm = ""
            cb = self.current_best if isinstance(self.current_best, dict) else {}
            cb_tput = cb.get("tput")
            if forced_arm == "baseline":
                snapshot_arm = "baseline"
                achieved_tput = self._resolve_baseline_achieved_tput()
            elif forced_arm == "current_best":
                # Explicit tag wins: keep the optimized arm even if tput is
                # momentarily absent, never downgrade to baseline (the profile
                # actually ran current_best params, so the ceiling must match).
                snapshot_arm = "current_best"
                achieved_tput = self._resolve_current_best_achieved_tput()
            elif isinstance(cb_tput, (int, float)) and cb_tput > 0:
                snapshot_arm = "current_best"
                achieved_tput = float(cb_tput)
            else:
                snapshot_arm = "baseline"
                achieved_tput = self._resolve_baseline_achieved_tput()
            # Primary decode ceiling plus memory/compute sides; PerfModel bottom-up formula, legacy is fallback only.
            breakdown = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")
            try:
                breakdown = compute_roofline_breakdown_from_state(
                    self,
                    arm=snapshot_arm,
                )
            except Exception:  # noqa: BLE001 — ceiling is best-effort
                pass
            peak_tput = float(breakdown.peak_tok_per_sec or 0.0)
            # Scriptable/diffusion has no tok/s decode ceiling. Surface the
            # compute-latency roofline instead: the measured per-image e2e
            # latency (achieved) vs the compute-roofline ideal floor read from
            # the diffusion_roofline.json sidecar. Both best-effort → 0 keeps
            # the serving path and legacy callers unchanged.
            fw = str(getattr(self, "framework", "") or "")
            e2e_mean_ms, roofline_ideal_ms = self._scriptable_latency_roofline(
                fw, achieved_tput, kernel_roofline_path
            )
            history_entry = build_roofline_snapshot(
                snapshot_id=snapshot_id,
                ts=ts_iso,
                analysis_md_path=str(analysis_md_path),
                theoretical_peak_tok_per_sec=peak_tput,
                achieved_tok_per_sec=achieved_tput,
                mem_ceiling_tok_per_sec=float(breakdown.mem_tok_per_sec or 0.0),
                cmp_ceiling_tok_per_sec=float(breakdown.cmp_tok_per_sec or 0.0),
                bound_kind=breakdown.bound_kind,
                framework=fw,
                e2e_mean_ms=e2e_mean_ms,
                roofline_ideal_ms=roofline_ideal_ms,
            )
            # Per-op PerfModel breakdown for dashboard visualization.
            attach_perfmodel_breakdown(history_entry, self, arm=snapshot_arm)
            history_entry["trace_input"] = str(trace_input)
            history_entry["macro_cycle"] = int(getattr(self, "macro_cycle", 0) or 0)
            history_entry["analysis_md_path"] = str(analysis_md_path)
            # 9fe4609 sidecar artifact pointer for per-kernel roofline data.
            history_entry["kernel_roofline_path"] = str(kernel_roofline_path)
            if not isinstance(self.roofline_snapshots, list):
                self.roofline_snapshots = []
            self.roofline_snapshots.append(history_entry)
            try:
                from .roofline_snapshot import direction_saturation

                sat = direction_saturation(history_entry)
                direction = str(sat.get("direction") or "")
                hint = sat.get("domain_hint") if isinstance(sat.get("domain_hint"), dict) else {}
                domain_key = str(hint.get("domain") or direction or "unknown")
                latest_cycle = int(history_entry.get("macro_cycle") or 0)
                prev_cycle_snapshot: dict[str, Any] = {}
                for prev in reversed(self.roofline_snapshots[:-1]):
                    if not isinstance(prev, dict):
                        continue
                    try:
                        prev_cycle = int(prev.get("macro_cycle", 0) or 0)
                    except (TypeError, ValueError):
                        prev_cycle = 0
                    if prev_cycle < latest_cycle:
                        prev_cycle_snapshot = prev
                        break
                if not prev_cycle_snapshot and len(self.roofline_snapshots) >= 2:
                    prev = self.roofline_snapshots[-2]
                    prev_cycle_snapshot = prev if isinstance(prev, dict) else {}
                prev_sat = direction_saturation(prev_cycle_snapshot) if prev_cycle_snapshot else {}
                if not isinstance(self.saturated_directions, dict):
                    self.saturated_directions = {}
                self.saturated_directions[domain_key] = {
                    **sat,
                    "domain": domain_key,
                    "tag": str(hint.get("tag") or ""),
                    "macro_cycle": latest_cycle,
                    "snapshot_id": snapshot_id,
                    "top_bottleneck": history_entry.get("top_bottleneck"),
                }
                self.bottleneck_shift = {
                    "from": prev_sat.get("direction", "") if prev_sat else "",
                    "to": sat.get("direction", ""),
                    "from_domain": (prev_sat.get("domain_hint") or {}).get("domain", "") if prev_sat else "",
                    "to_domain": domain_key,
                    "prev_cycle": prev_cycle_snapshot.get("macro_cycle") if prev_cycle_snapshot else None,
                    "cycle": latest_cycle,
                    "within_delta": (
                        round(float(sat["within_pct"]) - float(prev_sat["within_pct"]), 2)
                        if prev_sat
                        and isinstance(sat.get("within_pct"), (int, float))
                        and isinstance(prev_sat.get("within_pct"), (int, float))
                        else None
                    ),
                    "gap_delta": (
                        round(float(sat["gap_pct"]) - float(prev_sat["gap_pct"]), 2)
                        if prev_sat
                        and isinstance(sat.get("gap_pct"), (int, float))
                        and isinstance(prev_sat.get("gap_pct"), (int, float))
                        else None
                    ),
                    "bound_kind_changed": (
                        bool(prev_sat)
                        and str(prev_sat.get("bound_kind") or "") != str(sat.get("bound_kind") or "")
                    ),
                    "current": sat,
                    "previous": prev_sat,
                }
            except Exception:  # noqa: BLE001 — saturation telemetry is advisory only
                pass
            # R3: a fresh roofline whose top bottleneck has drifted off the
            # plateaued one means the cycle redirect succeeded — retire the
            # pending switch so the orchestration prompt stops nagging.
            self.maybe_clear_bottleneck_switch_on_drift(
                str(history_entry.get("top_bottleneck") or ""),
            )
            if len(self.roofline_snapshots) > _ROOFLINE_SNAPSHOTS_CAP:
                # Always keep snapshot #1 so the report's baseline anchor never rotates away.
                base = self.roofline_snapshots[0]
                tail = self.roofline_snapshots[-(_ROOFLINE_SNAPSHOTS_CAP - 1) :]
                self.roofline_snapshots = [base, *tail]
        except Exception:  # noqa: BLE001 — never block record on render concerns
            pass

    def record_sweep(self, result: dict[str, Any]) -> None:
        """Snapshot the most recent workload sweep into ``last_sweep``.

        Selects the best succeeded grid entry by ``output_throughput`` and
        records grid size, per-concurrency bests, the Pareto front, and the
        workspace path.

        Args:
            result (dict[str, Any]): The sweep executor result envelope. A
                non-dict result is a no-op.
        """
        if not isinstance(result, dict):
            return
        grid = result.get("sweep_grid") or []
        best = None
        if isinstance(grid, list):
            best = max(
                (
                    e
                    for e in grid
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
            # Watermark of validated gain at the moment this sweep ran, so a later
            # SWEEP entry (cyclic reloop) can skip a redundant full sweep when no
            # validated improvement landed since (see Coordinator._on_enter_sweep).
            "cumulative_gain_validated_at_record": float(
                getattr(self, "cumulative_gain_validated", 0.0) or 0.0
            ),
        }

    def record_conc_sweep(self, result: dict[str, Any]) -> None:
        """Record conc_sweep task completion (mirrors record_sweep). The status field lets exit_normal_sweep return conc_sweep_done so SWEEP→CLOSE can fire on conc_sweep alone.

        Args:
            result (dict[str, Any]): The conc_sweep result envelope; a
                non-dict result is a no-op.
        """
        if not isinstance(result, dict):
            return
        self.last_conc_sweep = {
            "ts": _now_iso(),
            "status": str(result.get("status") or "succeeded"),
            "skip_reason": str(result.get("skip_reason") or ""),
            "was_skipped": bool(result.get("was_skipped", False)),
            "budget_exhausted": bool(result.get("budget_exhausted", False)),
            "summary": dict(result.get("summary") or {}),
            "workspace": str(result.get("workspace") or ""),
        }






























    # No action-score API; ``increment_tick`` is a pure monotonic counter for plateau/phase budget math.
    def increment_tick(self) -> int:
        """Bump the Coordinator tick counter.

        Returns:
            int: The post-increment monotonic tick value.
        """
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
        """Mirror an optimization_stack append into gain_per_stack_entry; computes ``(new_tput-baseline_tput)/baseline_tput*100`` and appends. Returns gain_pct (None when baseline_tput is 0 or new_tput non-positive).

        Args:
            action (str): The action that produced the stack entry.
            variant_name (str | None): The variant name, when applicable.
            new_tput (float): The measured throughput for the entry.
            extra_server_args (str): The extra server args for the entry.
            ts (str | None): Optional ISO timestamp for the entry.

        Returns:
            float | None: The computed incremental gain pct, or ``None`` when
                ``baseline_tput`` is 0 or ``new_tput`` is non-positive.
        """
        try:
            base = float(self.baseline_tput or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        try:
            tput = float(new_tput or 0.0)
        except (TypeError, ValueError):
            tput = 0.0
        from .gain_math import gain_pct

        entry_gain_pct = gain_pct(tput, base)
        self.gain_per_stack_entry.append(entry_gain_pct)
        return entry_gain_pct

    def seed_stack_from_current_best(self) -> None:
        """Backfill stack for old sessions that only had current_best."""
        if self.optimization_stack or not isinstance(self.current_best, dict):
            return
        variant = self.current_best.get("variant_name")
        extra_args = self.current_best.get("extra_server_args")
        if not variant and not extra_args:
            return
        self.optimization_stack = [
            {
                "action": self.current_best.get("action", "unknown"),
                "variant_name": variant or "legacy_current_best",
                "extra_server_args": extra_args or "",
                "extra_envs": dict(self.current_best.get("extra_envs") or {}),
                "tput": self.current_best.get("tput"),
                "workspace": self.current_best.get("workspace"),
                "source": "seeded_from_current_best",
            }
        ]
        # Keep gain_per_stack_entry aligned with optimization_stack (None == unknown gain for seeded entries).
        if len(self.gain_per_stack_entry) < len(self.optimization_stack):
            self.gain_per_stack_entry.extend([None] * (len(self.optimization_stack) - len(self.gain_per_stack_entry)))

    # Time-budget helpers (consumed by Coordinator._compose_prompt)
    def elapsed_minutes(self, *, now: datetime | None = None) -> float:
        """Wall-clock minutes since ``start_ts`` (0.0 when empty/unparseable).

        Args:
            now (datetime | None): Reference time; defaults to the current UTC
                time.

        Returns:
            float: Minutes elapsed since ``start_ts`` (clamped at 0.0; 0.0
                when ``start_ts`` is empty or unparseable).
        """
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
        """Minutes left in the wall-clock budget; ``None`` when ``max_minutes`` unset (unbounded), else clamped at 0.

        Args:
            now (datetime | None): Reference time; defaults to the current UTC
                time.

        Returns:
            float | None: Minutes remaining in the budget (clamped at 0.0), or
                ``None`` when ``max_minutes`` is unset.
        """
        if not self.max_minutes:
            return None
        return max(0.0, float(self.max_minutes) - self.elapsed_minutes(now=now))

    def optimization_stack_has_unvalidated_keeps(self) -> bool:
        """True iff a new KEEP landed since the last inline stack rebench (purely a stack-length check vs ``cumulative_gain_validated_stack_len``).

        Returns:
            bool: ``True`` when ``optimization_stack`` is longer than
                ``cumulative_gain_validated_stack_len``.
        """
        return len(self.optimization_stack) > int(self.cumulative_gain_validated_stack_len)



























__all__ = ["SharedState", "render_model_arch_compact"]
