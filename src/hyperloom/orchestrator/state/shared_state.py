# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SharedState — single-writer (Coordinator) persisted session state, backed by atomic JSON at ``$SESSION_DIR/state.json``; enforces CORE_STATE_FIELDS guards.

Fields::

    session_id          str   — set by Coordinator at session creation
    model_name          str   — e.g. "meta-llama/Llama-3.1-8B-Instruct"
    model_path          str   — local NFS path to weights
    model_class         str   — categorical key supplied via --model-class
    model_arch          dict  — advisory architecture profile (hybrid
                                structured + free-text notes) loaded from
                                the launcher's ``<session_dir>/model_arch.json``;
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
    current_best        dict  — champion snapshot: ``action`` + ``tput`` plus
                                per-writer detail (variant_name, extra_server_args,
                                extra_envs, workspace, latency means)
    cumulative_gain_validated float — % over baseline at the last measurement
                                that promoted (an explore KEEP's decision round,
                                or a full-stack revalidation), on whichever axis
                                the session grades (see resolve_graded_comparison)
    stop_reason         str   — set when graceful stop fires
    stop_ts             str   — ISO timestamp of the first stop_reason write
    resumed_ts          str   — ISO timestamp of the most recent --resume
    current_action      str   — what's running right now (set by Orchestration)
    crash_count         int   — incremented by the Coordinator when a tick/agent
                                exception is recorded; also appends to
                                crash_timestamps (Robustness only reads it)
    pruned_families     list[str]  — set by Robustness via PRUNE_BRANCH
    start_ts            str   — ISO timestamp
    max_minutes         int   — wall-clock budget (0 = unlimited)
    deadline_unix       float — absolute session deadline (0 = unset/unbounded)
    last_profile_trace  str   — set by Coordinator when `profile` returns a
                                trace path; consumed by Orch to populate
                                `trace_analyze` REQUEST `trace_input` param
    last_collective     dict  — latest collective campaign and integration state
    collective_attempts list  — capped collective campaign audit
    collective_only_mode bool — disable non-collective KERNEL lanes
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from hyperloom.common.coerce import to_str_list, to_unix
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.io import atomic_write_json
from hyperloom.common.jsonio import read_json
from hyperloom.common.profile_args import sanitize_profile_server_args

if TYPE_CHECKING:  # import cycle: perf_metric is imported lazily at call time
    from hyperloom.common.perf_metric import GradedComparison

from . import kernel_decision_settings as _kernel_decision_settings
from ._shared_state.enablement_round import EnablementRound

log = logging.getLogger(__name__)

# Upper bound on retained crash timestamps (trailing-window rate needs only the
# recent tail).
_CRASH_TIMESTAMP_CAP: int = 200

# Compatibility aliases kept on shared_state for existing callers/tests.
_DEFAULT_ATTEMPTS_HISTORY = _kernel_decision_settings._DEFAULT_ATTEMPTS_HISTORY
_DEFAULT_HOT_KERNEL_MIN_GPU_PCT = _kernel_decision_settings._DEFAULT_HOT_KERNEL_MIN_GPU_PCT
_MAX_INTEGRATE_FAULT_ATTEMPTS = _kernel_decision_settings._MAX_INTEGRATE_FAULT_ATTEMPTS
_now_iso = _kernel_decision_settings._now_iso
resolve_hot_kernel_min_gpu_pct = _kernel_decision_settings.resolve_hot_kernel_min_gpu_pct
resolve_kernel_opt_max_failures = _kernel_decision_settings.resolve_kernel_opt_max_failures


def first_positive_tput(d: Any) -> float:
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


def resolve_anchor_with_drift(snapshot_tput: float, state: Any) -> tuple[float, bool]:
    """Grade against the live anchor when a KEEP landed after this task snapshotted its params.

    A task carries ``base_tput`` from the moment it was created; a KEEP landing
    while it queued makes that snapshot stale and would grade the candidate
    against a recipe it no longer sits on top of.

    Args:
        snapshot_tput: The anchor recorded in the task's params.
        state: Any object exposing ``current_best`` / ``baseline_tput``.

    Returns:
        ``(anchor, drifted)`` — the anchor to grade against, and whether the
        live value displaced a positive snapshot (i.e. worth logging).
    """
    live = resolve_grading_anchor_tput(state)
    if live > snapshot_tput:
        return live, snapshot_tput > 0
    return snapshot_tput, False


def framework_is_scriptable(framework: str | None) -> bool:
    """Whether *framework* reports an image-quality gate instead of token throughput.

    An unset name reads as a serving framework: a session that has not resolved
    one yet is not scriptable until it says so.
    """
    name = str(framework or "").strip()
    if not name:
        return False
    from hyperloom.inference_optimizer import framework_registry

    return bool(framework_registry.is_scriptable(name))


def resolve_grading_anchor_tput(state: Any) -> float:
    """Output throughput a new candidate is composed on top of.

    Candidates are launched with ``current_best``'s args/envs, so grading them
    against ``baseline_tput`` compares a measurement to a configuration it was
    never taken on: anything that beats the bare baseline but regresses against
    the established recipe (e.g. a warm-replay bundle) reads as a win and drags
    ``current_best`` down. ``baseline_tput`` is the fallback only before any
    validated layer exists.

    Always the output axis. This is not the KEEP grader -- that is
    :func:`resolve_graded_comparison` -- and its consumers are not grading a
    candidate: the ``base_tput`` seeded into task params, the drift check, and
    the two objective resolvers, whose targets are operator-supplied output
    figures.

    Args:
        state: Any object exposing ``current_best`` / ``baseline_tput``
            (``None`` and partial test doubles are tolerated).

    Returns:
        ``current_best``'s output throughput when positive, else
        ``baseline_tput``; ``0.0`` when neither is established.
    """
    if state is None:
        return 0.0
    best = first_positive_tput(getattr(state, "current_best", None))
    if best > 0:
        return best
    baseline = getattr(state, "baseline_tput", 0.0)
    return float(baseline) if isinstance(baseline, (int, float)) and baseline > 0 else 0.0


def resolve_graded_comparison(
    state: Any,
    measurement: Any,
    *,
    against_baseline: bool = False,
) -> "GradedComparison":
    """Resolve what a KEEP decision grades: candidate and reference, one axis.

    Under total-token-throughput grading both sides come from perf snapshots,
    which exist only when the axis pair (total, intvty p90) is present, so a
    lane cannot half-apply the objective. When either side cannot supply the
    pair both read the output axis and ``degrade_reason`` names why. The
    interactivity constraint is resolved here as ``vetoed`` because it belongs
    to the total-throughput objective rather than sitting beside it.

    Args:
        state: The session state (``current_best`` / ``baseline_tput`` /
            ``baseline_perf`` / ``framework`` / ``benchmark_mode``).
        measurement: The candidate's measurement mapping.
        against_baseline: Grade against the session baseline (cumulative
            realized gain) rather than the recipe the candidate was composed on.

    Returns:
        A :class:`GradedComparison` whose ``candidate`` and ``reference`` are
        on the axis it names.
    """
    from hyperloom.common.perf_metric import (
        GRADED_OUTPUT,
        GRADED_TOTAL,
        GradedComparison,
        output_tput_of,
        passes_intvty_gate,
        perf_snapshot_from_mapping,
        resolve_grading_anchor_perf,
        total_tput_of,
        total_tput_serving_grading_enabled,
    )

    degrade_reason = ""
    if total_tput_serving_grading_enabled(
        scriptable=framework_is_scriptable(getattr(state, "framework", None)),
        benchmark_mode=str(getattr(state, "benchmark_mode", "") or ""),
    ):
        if against_baseline:
            ref_perf = perf_snapshot_from_mapping(getattr(state, "baseline_perf", None))
            reason = "" if ref_perf else "baseline_axes_missing"
        else:
            ref_perf, reason = resolve_grading_anchor_perf(state)
        cand_perf = perf_snapshot_from_mapping(measurement)
        if ref_perf and cand_perf:
            return GradedComparison(
                objective=GRADED_TOTAL,
                candidate=total_tput_of(cand_perf),
                reference=total_tput_of(ref_perf),
                vetoed=not passes_intvty_gate(cand_perf, ref_perf),
            )
        degrade_reason = reason or "candidate_axes_missing"

    reference = (
        float(getattr(state, "baseline_tput", 0.0) or 0.0) if against_baseline else resolve_grading_anchor_tput(state)
    )
    return GradedComparison(
        objective=GRADED_OUTPUT,
        candidate=output_tput_of(measurement),
        reference=reference,
        degrade_reason=degrade_reason,
    )


def _normalize_envs(value: Any) -> dict[str, str]:
    """Coerce an env mapping to ``str -> str``; a resumed non-dict reads as empty."""
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


# ``base_*`` task param -> the ``current_best`` field it mirrors and its normalizer.
_STACK_BASE_FIELDS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("base_extra_args", "extra_server_args", lambda v: str(v or "").strip()),
    ("base_extra_envs", "extra_envs", _normalize_envs),
    ("base_remove_args", "remove_args", to_str_list),
    ("base_unset_envs", "unset_envs", to_str_list),
    ("base_args_mode", "args_mode", lambda v: "replace" if str(v or "").strip().lower() == "replace" else ""),
)


def stack_base_params(current_best: Any) -> dict[str, Any]:
    """``base_*`` params projected from the fields ``current_best`` carries.

    Args:
        current_best: A ``current_best`` snapshot (non-dicts read as empty).

    Returns:
        The normalized ``base_*`` params; keys absent from ``current_best`` are
        omitted rather than defaulted, so a caller can tell "no config" from
        "empty config".
    """
    cb = current_best if isinstance(current_best, dict) else {}
    return {key: normalize(cb[source]) for key, source, normalize in _STACK_BASE_FIELDS if source in cb}


def inject_stack_base_params(
    params: dict[str, Any],
    state: Any,
    *,
    anchor: bool = False,
    overwrite: bool = False,
) -> None:
    """Seed a task's base config from ``current_best``, in place.

    A candidate is graded against an anchor while being launched on top of that
    anchor's args/envs, so the two are one unit; ``anchor=True`` takes both from
    the same snapshot rather than leaving a caller to seed one and omit the other.

    Args:
        params: Task params, mutated in place.
        state: Any object exposing ``current_best`` / ``baseline_tput``
            (``None`` and partial test doubles are tolerated).
        anchor: Also seed ``base_tput``.
        overwrite: Replace keys already present in ``params``, for an
            execution-time rebind; dispatch-time seeding must not clobber an
            operator- or LLM-supplied value.
    """

    def _put(key: str, value: Any) -> None:
        if overwrite:
            params[key] = value
        else:
            params.setdefault(key, value)

    if anchor:
        anchor_tput = resolve_grading_anchor_tput(state)
        if anchor_tput > 0:
            _put("base_tput", anchor_tput)
    for key, value in stack_base_params(getattr(state, "current_best", None)).items():
        # Empty means "no config"; on a rebind it is what clears a superseded layer.
        if value or overwrite:
            _put(key, value)


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


# Integration faults (environment / apply / bench crashes) are distinct from a
# genuine gate REVERT; a fault means the patch was never fairly measured, so it
# gets its own small retry budget instead of burning the REVERT quota. The
# reliable discriminator is ``status`` (see :meth:`_is_integrate_fault`); this
# error-class set is a secondary signal.
_INTEGRATE_FAULT_ERROR_CLASSES = frozenset(
    {
        "missing_integration_inputs",
        "patch_not_applied",
        "apply_failed",
        # The served context cannot host an eval request, so the accuracy gate
        # can never return a verdict under this configuration. The patch was
        # never fairly measured, so it must not spend a KEEP attempt.
        "eval_context_too_small",
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
# How many hot / skipped kernels ``record_trace_analyze`` keeps in the trace
# summary (matches the ``*_top15`` field names).
_TRACE_HOT_KERNEL_TOP_N = 15

# Session-level kernel-roofline report the analyzer writes for a non-close run;
# read back when the trace_analyze envelope arrives without its payload keys.
_DEFAULT_ROOFLINE_REPORT_NAME = "kernel_roofline_current.json"

# Global ``last_action_failures`` rolling-log cap.
_DEFAULT_LAST_FAILURES = 30

# phase_history cap (record_phase_transition).
_PHASE_HISTORY_CAP = 100

# Lifecycle-event log cap (fires at every step boundary, so generous but bounded).
_LIFECYCLE_CAP = 500

# roofline_snapshots history cap (record_trace_analyze).
_ROOFLINE_SNAPSHOTS_CAP = 50

# gap ledger caps; both enforced in upsert_gap.
_GAPS_MAX_ENTRIES = 50
_GAPS_ATTEMPTS_HISTORY = 20

# Long-run bounded-growth caps for append-only telemetry ledgers (tail-trim).
_INTERVENTION_MIX_CAP = 500
_SPECIALIST_ROUNDS_CAP = 200
_SEEN_PR_IDS_CAP = 2000
_WINNERS_HISTORY_CAP = 200
# Negative ledger (explore_search["tested"]); oldest insertion-order keys
# evicted first.
_EXPLORE_TESTED_CAP = 5000

# Per-action audit trail kinds; kernel_agent-owned actions excluded (dedicated structures).
_AUDIT_ACTIONS: frozenset[str] = frozenset(
    {
        "baseline",
        "profile",
        "explore",
        # ``roofline`` runs profile + trace_analyze atomically.
        "roofline",
    }
)

# audit-action name -> (result-dict key, key_metric_kind).
_KEY_METRIC_MAP: dict[str, tuple[str, str]] = {
    "baseline": ("output_throughput", "output_throughput"),
    "profile": ("output_throughput", "output_throughput"),
    "explore": ("best_gain_pct", "gain_pct"),
    "roofline": ("snapshot_id", "snapshot_id"),
}


#: top-level state.json schema version; absent key treated as v1 and migrated to LATEST_STATE_SCHEMA_VERSION on first save.
LATEST_STATE_SCHEMA_VERSION: int = 6

#: FRAMEWORK fields renamed by the framework_agent rename, old name -> current
#: name. A state written before that rename spells them the old way, and the
#: unknown-key filter in ``from_dict`` drops anything not in this table, which
#: is why an un-migrated resume silently restarted the phase from scratch.
#: ``framework_pr_max_candidates`` and ``framework_pr_critic_decisions`` are
#: deliberately absent: both fields have since been removed, so there is
#: nothing left to migrate them into.
_FRAMEWORK_FIELD_RENAMES_V5: dict[str, str] = {
    "framework_phase_enabled": "framework_agent_phase_enabled",
    "framework_pr_phase_progress": "framework_agent_phase_progress",
    "framework_pr_batches": "framework_agent_batches",
    "framework_pr_phase_done": "framework_agent_phase_done",
    "framework_pr_discover_failures": "framework_agent_discover_failures",
    "framework_pr_consecutive_empty_discoveries": "framework_consecutive_empty_discoveries",
    "framework_pr_authoring_enabled": "framework_agent_authoring_enabled",
    "framework_pr_specialist_candidate_map": "framework_agent_specialist_candidate_map",
}

#: KERNEL-entry dispatch switch renamed by the auto-dispatch rename, old name ->
#: current name. The old spelling tied the switch to GEMM tuning, which stopped
#: being true once the dispatch moved into the shared entry tail. Without this
#: table the unknown-key filter in ``from_dict`` would drop the old spelling and
#: a resumed opt-out session would silently start dispatching again.
_KERNEL_OPT_FIELD_RENAMES_V6: dict[str, str] = {
    "continue_kernel_after_gemm": "auto_kernel_opt_enabled",
}

#: Stack action label for FRAMEWORK entries, and the prefix promote used to glue
#: onto their ``variant_name``. Resume reconciliation keys on the bare candidate
#: key, so an entry still carrying the prefix reads as an orphaned KEEP and
#: misses the ``(action, variant_name)`` dedup that stops a second append.
_FRAMEWORK_STACK_ACTION_V5: str = "framework"
_FRAMEWORK_VARIANT_PREFIX_V5: str = "framework:"


def effective_closing_grace_sec(
    max_minutes: float | None,
    closing_grace_sec: float | None,
) -> float:
    """Resolve the closing-phase grace window after the wall-clock deadline.

    Explicit ``closing_grace_sec`` (including ``0`` to disable the closing
    phase) wins; otherwise default to ``min(120, max_minutes * 60 * 0.02)``.

    Lives beside the budget accessors rather than next to its Coordinator
    caller because the window is also the reserve those accessors hold back,
    and this module is the leaf both sides can import.

    Args:
        max_minutes (float | None): The wall-clock budget in minutes, used for
            the default.
        closing_grace_sec (float | None): Explicit grace window in seconds;
            when not ``None`` it is used verbatim.

    Returns:
        float: The closing-phase grace window in seconds.
    """
    if closing_grace_sec is not None:
        return float(closing_grace_sec)
    return min(120.0, (max_minutes or 0.0) * 60.0 * 0.02)


@contextmanager
def timed_teardown_step(state: Any, name: str) -> Iterator[None]:
    """Record how long one post-deadline teardown step took on ``state``.

    The session's own record never said what the unbudgeted tail cost. A step
    that cannot be timed is the same gap as a step that never ran.

    Args:
        state: Object exposing :meth:`SharedState.record_teardown_timing`;
            missing the method is a no-op so tests can pass a stub.
        name: Step key stored on ``teardown_timings_sec``.
    """
    started = time.monotonic()
    try:
        yield
    finally:
        recorder = getattr(state, "record_teardown_timing", None)
        if callable(recorder):
            recorder(name, time.monotonic() - started)


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
    # Advisory: True when knowingly running a multimodal checkpoint on the
    # text-only path (--allow-mm-text-fallback). Never gates Objective/scoring.
    degraded_mode: bool = False
    # Structured degraded-mode / model-compat warnings (e.g. multimodal text
    # fallback). Surfaced verbatim in reports/final.{json,md}.
    model_warnings: list[dict[str, Any]] = field(default_factory=list)
    # KB tags from config.json (``architectures`` + ``model_type``); stamped into recipe-snapshot ``extras`` so fine-tuned models carry base arch identity.
    model_architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    # config.json-derived structural summary (attention_type / heads / MoE / quant).
    model_info: dict = field(default_factory=dict)
    framework: str = ""
    gpu_type: str = ""
    # Workload metadata mirrored from manifest.json at session start; resume re-exports env vars.
    tp: int = 0
    # Expert-parallel size for MoE; mirror of ``EP`` env var. Resume-safe.
    ep: int = 0
    precision: str = ""
    # ``framework_version`` — only recipe-snapshot v2 canonical-id member not derivable from other fields; empty => ``unknown_version``.
    framework_version: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    # Profile-phase output length (from --profile-osl). 0 = unset (profile
    # defaults to min(osl, 1024)). Persisted across resume.
    profile_osl: int = 0
    max_model_len: int = 0
    kernel_enabled: bool = True
    # KERNEL-phase optimizer: "geak" (default, one-shot whole-pipeline e2e) or
    # "native" (per-kernel loop when explicitly requested).
    kernel_optimizer: str = "geak"
    # Snapshot of the last GEAK e2e run (result.json + final_launch.sh /
    # bench_e2e.sh handles the SWEEP phase reuses).
    geak_result: dict[str, Any] = field(default_factory=dict)
    # Whether KERNEL entry dispatches the source-level kernel_opt batch itself
    # (``--no-auto-kernel-opt`` opts out). Independent of GEMM tuning, and it
    # only governs the entry's own dispatch: orchestration can still request
    # kernel_opt explicitly, and the fusion/collective lanes have their own gates.
    auto_kernel_opt_enabled: bool = True
    # SWEEP-phase post-sweep concurrency sweep; opt out via ``--no-enable-conc-sweep``.
    conc_sweep_enabled: bool = True
    # Which benchmark workload this session measures: "agentx" (agentic trace
    # replay) or "synthetic" (ISL/OSL). Recorded at seed time and asserted on
    # resume: the KEEP ledger is keyed on server args alone, so measurements
    # from the two modes would silently overwrite each other in the same rows.
    # Empty on sessions predating the field (treated as "not asserted").
    benchmark_mode: str = ""
    # Generation counter for AgentX measurements. Bumped whenever a change makes
    # previously recorded AgentX numbers incomparable (aiperf/scenario upgrade,
    # corpus generation, a fixed measurement defect). A resume whose stored
    # epoch differs must not reuse the old KEEPs or baseline anchor.
    agentx_epoch: int = 0
    # CONC ladder for conc_sweep, seeded from the workload's own ladder by
    # ``_parse_conc_sweep_concs``. Empty => skip_reason=empty_conc_list.
    conc_sweep_concs: list[int] = field(default_factory=list)
    # Total wall-clock budget (s) for conc_sweep. 0 disables the gate.
    conc_sweep_total_budget_sec: int = 9000
    # Per-variant Magpie subprocess timeout (s), clamped to remaining total budget.
    conc_sweep_variant_timeout_sec: int = 1800
    target_summary: str = ""
    baseline_tput: float = 0.0
    # Baseline AgentX perf snapshot: total tok/s objective plus the intvty p90
    # the veto is measured against, and the reported axes the summary renders.
    baseline_perf: dict[str, Any] = field(default_factory=dict)
    # Internal-only baseline cold+hot double-run switch; default-on keeps the optimisation phase
    # warm-decision apples-to-apples with the baseline measurement basis.
    baseline_double_run: bool = True
    baseline_accuracy: float = 0.0
    # ``--no-eval``: no accuracy eval anywhere. ``baseline_accuracy`` stays 0,
    # which the candidate gates already read as "grade on throughput only".
    eval_disabled: bool = False
    # Standalone baseline-arm roofline ceiling computed right after baseline
    # lands; backs up snapshot ceiling so the frontend has data even when the
    # roofline (profile + trace_analyze) step fails. Empty until baseline runs.
    baseline_roofline_ceiling: dict[str, Any] = field(default_factory=dict)
    baseline_failure_streak: int = 0
    baseline_arg_error_streak: int = 0
    # Combined backstop counting ANY baseline failure regardless of error_class,
    # catching mixed classes that never trip a per-class streak (anti time-exhaustion).
    baseline_total_failures: int = 0
    # One-shot: a cuda-graph capture failure asks the next baseline to retry with
    # cuda-graph capture disabled. Set on failure, consumed by BaselineExecutor.
    baseline_eager_fallback: bool = False
    # Admission for both enablement lanes, from ``--enablement``: ``launch``,
    # ``eval``, ``all`` (default) or ``off`` — with ``off`` a broken baseline
    # fast-fails instead of opening an authoring loop.
    enablement_mode: str = "all"
    # All per-round enablement fields are nested here.
    enablement: EnablementRound = field(default_factory=EnablementRound)
    # Off-loop targeted-build sentinel (task_id/pid/pgid/attempt_root/
    # aiter_jit_dir/deadline/action); own sentinel, resume-cleared.
    pending_targeted_build: dict = field(default_factory=dict)
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
    # Crash-safe rollback/bookkeeping state for the combined Recipe + Kernel
    # PRELUDE validation. Cleared only after KEEP/REVERT settles.
    warm_replay_pending: dict = field(default_factory=dict)
    # Session-authoritative InferenceX checkout. A successful isolated warm
    # replay promotes its already-patched checkout here without re-applying.
    active_inferencex_path: str = ""
    # Durable post-save handoffs for section KB staging. Rows are removed only
    # after the idempotent draft write succeeds.
    kb_stage_outbox: list = field(default_factory=list)
    # Owner sections dropped because their persisted artifacts disappeared.
    kb_stage_dead_letter: list = field(default_factory=list)
    # Idempotent terminal Recipe publication state. Independent from CLOSE
    # report completion so a failed remote write remains retryable at teardown.
    recipe_finalize_status: str = ""
    recipe_finalize_attempts: int = 0
    recipe_finalize_outcome: dict = field(default_factory=dict)
    # One-shot guard for PRELUDE warm-kernel KB read/apply (resume can't re-fire).
    warm_kernel_kb_attempted: bool = False
    # Resolved prior-champion kernel columns (gemm/fusion/rewrite) loaded at
    # PRELUDE from the Recipe ``value.kernel``; read back by the combined promote.
    warm_kernel_kb_plan: list = field(default_factory=list)
    # Baseline COLD (warmup-round) full boot+bench wall-clock; the hard-cap
    # anchor from which ExploreExecutor derives the overtime-kill deadline.
    baseline_runtime_sec: float = 0.0
    # Baseline WARM measure-round wall-clock (client-only, no boot); anchors the
    # explore overtime kill apples-to-apples. Zero => fall back to the cold anchor.
    baseline_warm_runtime_sec: float = 0.0
    # Whether the baseline's hot pass was dropped because the budget could not
    # cover it plus one variant to read against it, leaving the cold pass's
    # depressed figure as the anchor. Routes PRELUDE to CLOSE rather than letting
    # later phases compare against a denominator that was never the baseline, and
    # keeps a resumed session from treating preparation as finished.
    baseline_measure_round_dropped: bool = False
    # The benchmark's own share of the COLD round above, from the server-ready
    # marker onward. Kept beside that total rather than replacing it because the
    # difference between them is what booting this workload costs, and every
    # later variant boots again: the pair prices one variant, while this figure
    # alone prices a pass that re-attaches. Zero => never measured.
    baseline_post_ready_runtime_sec: float = 0.0
    current_best: dict[str, Any] = field(default_factory=dict)
    # Immutable measurement evidence captured with the current-best promotion.
    # GEAK may only use its tput after the stored launch identity matches the
    # configuration it is about to reproduce.
    current_best_measurement: dict[str, Any] = field(default_factory=dict)
    # Reference launch recipe from the operator's --reference-script: lowest-priority
    # base server args/envs seeding every baseline. Persisted.
    reference_server_args: str = ""
    reference_envs: dict[str, str] = field(default_factory=dict)
    reference_model: str = ""
    reference_source: str = ""
    # Operator launch shape, persisted so a bare --resume serves the same contract.
    # ``--server-args``: merged with per-task extra_server_args, above the reference base.
    operator_server_args: str = ""
    # ``--extra-env NAME=VALUE`` pins.
    operator_extra_env: dict[str, str] = field(default_factory=dict)
    # Operator-supplied custom-workload paths. Fresh launch publishes them as
    # env from ``--framework-path`` / ``--benchmark-scripts-dir``; a resume
    # that does not re-pass those flags must re-export them from here or the
    # scriptable runner cannot find the entrypoint (rc=2, no measurement).
    bypass_scripts_dir: str = ""
    framework_repo_path: str = ""
    # ``HYPERLOOM_BENCHMARK_BACKEND`` at seed time (``bypass`` for custom).
    benchmark_backend: str = ""
    # The card's compute-partition shape this session was measured in, as
    # observed at launch: mode, partition count, CU and memory per partition,
    # streams per partition. Empty when the card reported nothing. Part of the
    # measurement contract, not a tuning knob -- the same configuration in SPX
    # and in CPX is two different experiments.
    compute_partition: dict[str, Any] = field(default_factory=dict)
    # ``--nodes``, feeding the robustness defaults and the IR-8 check. NOT the
    # cluster hand-off, which is resolved from argv before this state loads.
    nodes: int = 1
    # Resolved robustness-agent ``request.options``; a resume layers its own flags
    # on top, per-key. Stored resolved because the resolution folds in
    # multi-node / scriptable policy that the individual flags do not carry.
    robustness_options: dict[str, Any] = field(default_factory=dict)
    # Warm-recipe replay gates (``--no-warm-replay`` / ``--warm-replay-min-*``).
    warm_replay_enabled: bool = True
    warm_replay_min_confidence: float = 0.7
    warm_replay_min_reproduce_pct: float = 0.8
    # Full accepted configuration stack across action families; current_best keeps the materialized full args/env.
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    # Index-aligned with ``optimization_stack``: per-entry incremental gain pct; missing => None.
    gain_per_stack_entry: list[float | None] = field(default_factory=list)
    # Total gain over ``baseline_tput``, stamped only from a measurement taken
    # with the whole stack applied; standalone validate_stack denied by PolicyGate.
    cumulative_gain_validated: float = 0.0
    cumulative_gain_validated_ts: str = ""
    # ``optimization_stack`` length at the last validated measurement; longer => new KEEPs need validation.
    cumulative_gain_validated_stack_len: int = 0
    # Resume sentinels. ``pending_integrate`` is written before a
    # non-transactional integrate_patch window and cleared after stack/current
    # best persist; after a crash resume replays or rolls back the window.
    # ``resume_pending_revalidation`` flags that accepted stack entries need a
    # fresh post-resume stack rebench.
    pending_integrate: dict[str, Any] = field(default_factory=dict)
    resume_pending_revalidation: bool = False
    # A GEAK e2e candidate with a self-reported win not yet confirmed by a
    # main-flow rebench; kept OUT of current_best / optimization_stack / the
    # headline gain until validated. Cleared once promoted from a measured rebench.
    geak_pending: dict[str, Any] = field(default_factory=dict)
    # Tput watermark for gain-driven roofline refresh; Coordinator re-enqueues at a compound 10% step.
    last_roofline_tput: float = 0.0
    stop_reason: str = ""
    # When the session first stopped, and therefore its end time for
    # consumers. Stamped by the first ``set_stop_reason`` write and left alone
    # by later ones, so the CLOSE sequence's own artifacts and any re-export
    # quote the same end; cleared with the reason on resume.
    stop_ts: str = ""
    # When the current run leg began, i.e. the most recent ``--resume``; empty
    # for a session that has only ever run once. ``start_ts`` cannot answer
    # this: a resume after a clean stop deliberately keeps it so the wall-clock
    # budget still counts from the original start, which leaves this the only
    # record of where the previous leg ended.
    resumed_ts: str = ""
    # Closing phase — set when wall-clock deadline fires; Coordinator only drains a ``report`` task. Cleared on resume.
    closing_phase: bool = False
    closing_started_unix: float = 0.0
    closing_report_task_id: str = ""
    # True at END of CLOSE 7-step sequencer; cli.finally short-circuits emergency breakdown write. Resume clears it (idempotent).
    close_sequence_done: bool = False
    # Auto-roofline gate (optimisation-phase entry): pending roofline task_id; blocks first-round specialist dispatch until snapshot lands.
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
    # Absolute unix deadline for a bounded session. Stamped once from
    # ``start_ts + max_minutes`` so a resume cannot reissue a full budget.
    # ``0.0`` means unset or unbounded.
    deadline_unix: float = 0.0
    # Wall-clock seconds spent in post-deadline teardown, keyed by step.
    teardown_timings_sec: dict[str, float] = field(default_factory=dict)
    # Operator's ``--closing-grace-sec``; ``None`` derives it from max_minutes.
    closing_grace_sec: float | None = None
    last_profile_trace: str = ""
    # ``succeeded``/``failed`` for most recent profile; failed allows re-run even when last_profile_trace is non-empty.
    last_profile_status: str = ""
    # Workload context captured with ``last_profile_trace``; strict matching
    # prevents consumers from reusing runtime shapes after workload changes.
    last_profile_workload: dict[str, Any] = field(default_factory=dict)
    # ``current_best.action`` in effect when ``last_profile_workload`` was
    # recorded, so the backfill in ``current_profile_workload_context`` can tell
    # a same-arm reuse from a stale one. Empty on legacy sessions.
    last_profile_workload_action: str = ""
    # Rolling log of PolicyGate denials (newest last, cap 50).
    policy_denial_history: list[dict[str, Any]] = field(default_factory=list)
    # Per-(action_name, rule) consecutive denial counter.
    policy_denial_streak: dict[str, int] = field(default_factory=dict)
    # Set when AST flag discovery cannot locate framework source files.
    discovered_flags_error: str = ""
    # Server EXTRA_SGLANG_ARGS in effect when last_profile_trace was captured; identical args means the same trace.
    last_profile_args: str = ""
    # Per-kernel GPU time breakdown JSON from the most recent profile. Read by
    # ``_framework_gap_composer`` to add a bottleneck keyword to the framework
    # arm's gap description.
    last_profile_kernel_breakdown: str = ""
    # Merged host-side rewrite evidence document from the most recent profile
    # (see ``_framework_rewrite_evidence``). Distinct from the kernel breakdown:
    # it reports redundant *host* work — collective round-trips, device-to-host
    # syncs, repeated host-to-device copies, recomputed loop invariants — none of
    # which appears in a kernel timeline, and which is where framework-level
    # source rewrites (as opposed to kernel rewrites) find their wins.
    last_framework_rewrite_evidence: str = ""
    # Why the field above is empty, when it is. "No candidates" and "the probe
    # never ran" both render as no evidence, and only one of them means the
    # workload has nothing left to rewrite; without this the framework phase
    # cannot tell a genuine negative from a broken instrument.
    last_framework_rewrite_evidence_status: str = ""
    # Environment switches behind accepted framework-level source rewrites,
    # registered as search levers. Each row carries its rewrite category, its
    # dependency edges, whether it is currently on (``default_on``), and its
    # individually measured contribution once the explore phase has attributed
    # it. This is what turns an authored bundle of rewrites into per-rewrite
    # numbers and a searchable combination space, instead of a patch that is
    # kept or reverted whole.
    authored_framework_levers: list[dict[str, Any]] = field(default_factory=list)

    # Roofline-v2 trace-analyze cache written by record_trace_analyze.
    # roofline_snapshot_id is a property derived from this dict.
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    # Append-only compact roofline snapshots for report.py; capped at ``_ROOFLINE_SNAPSHOTS_CAP`` (snapshot #1 always retained as the report's baseline anchor).
    roofline_snapshots: list[dict[str, Any]] = field(default_factory=list)
    # Outer roofline failure counter; bumped on fail, reset on success.
    roofline_failure_streak: int = 0

    # Feature toggles (mirrored from ``cli.py`` flags at session start).
    # FRAMEWORK_AGENT phase toggle (PRELUDE → FRAMEWORK_AGENT → KERNEL_AGENT); ``--no-framework-agent`` opts out.
    framework_agent_phase_enabled: bool = True
    # FRAMEWORK progress: one entry per candidate benchmark; used by breakdown + plateau exit judgment.
    framework_agent_phase_progress: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # One row per discovery batch; read by the source arm's plateau gate (3 batches <1% => exit).
    framework_agent_batches: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # True when the source arm has no more candidates; read by its plateau gate.
    framework_agent_phase_done: bool = False
    # Consecutive discovery failures; the arm declines only after DISCOVER_FAILURE_RETRY_LIMIT (default 3).
    framework_agent_discover_failures: int = 0
    # Consecutive empty-but-valid discovery batches; tolerate up to
    # DISCOVER_FAILURE_RETRY_LIMIT before exiting. Reset on any non-empty batch.
    framework_agent_empty_discoveries: int = 0
    # Consecutive FRAMEWORK_AGENT phase completions that discovered zero candidates
    # (empty_discovery). Drives the Step-1 advisory ("framework phase ineffective");
    # reset whenever a phase completes having tested >=1 candidate.
    framework_consecutive_empty_discoveries: int = 0
    # Default True: FRAMEWORK pump dispatches a write-capable serving_specialist per candidate alongside diff-only track. False restores diff-only.
    framework_agent_authoring_enabled: bool = True
    # Default True: when PR discovery is empty/exhausted (or the ranker prefers
    # it), the FRAMEWORK pump dispatches a candidate-free authoring specialist
    # that authors a throughput patch from the live source + profile evidence
    # instead of skipping the phase. Requires framework_agent_authoring_enabled;
    # --no-framework-local-explore opts out (restores discover-exhaustion exit).
    framework_local_explore_enabled: bool = True
    # Maps an authoring specialist task_id -> originating FRAMEWORK candidate id
    # (PR URL), so the authored-outcome bridge can key the progress row on the
    # PR-URL that ``_select_next_framework_agent_candidate`` checks.
    framework_agent_specialist_candidate_map: dict[str, str] = field(
        default_factory=dict,
    )
    # Re-author rounds per candidate id (capped); needs_review verdicts increment.
    specialist_reauthor_attempts: dict[str, int] = field(
        default_factory=dict,
    )
    # Backstop: per-candidate-key count of Critic-review submissions; past the
    # abort threshold the pump force-stamps ``repeated_review_abort`` and stops
    # re-selecting it, bounding one candidate's share of the phase budget.
    framework_agent_review_counts: dict[str, int] = field(
        default_factory=dict,
    )
    # Apply-failure re-author attempts per candidate id, capped by
    # ``_AUTHORED_LANE_MAX_ATTEMPTS``. Declared (not set ad hoc) because
    # ``to_dict`` is ``asdict``, which walks declared fields only: an undeclared
    # attribute is dropped at every save, so the cap would be re-spent from zero
    # on every resume.
    apply_fail_reauthor_attempts: dict[str, int] = field(
        default_factory=dict,
    )
    # Retry contexts queued by the authored lane and drained by
    # ``_drain_apply_fail_retry_pending``. Declared for the same reason: an
    # undeclared attribute means a resume silently discards queued retries.
    apply_fail_retry_pending: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # Default True: Coordinator auto-analysis is ``roofline`` (profile+trace_analyze+analysis.md); False enqueues plain ``profile``.
    enable_roofline: bool = True
    # ExploreExecutor per-variant overtime kill multiplier; >0 kills the decision
    # run past anchor*ratio (outcome='KILLED_OVERTIME'). Anchor is the WARM
    # measure time when active else the cold baseline; the warmup round is exempt.
    explore_overtime_kill_ratio: float = 2.0
    # ExploreExecutor per-variant hard timeout override; 0 => auto-derive from baseline_runtime_sec*(kill_ratio+safety_margin).
    explore_variant_timeout_sec_override: int = 0
    # Headroom added to kill_ratio for auto-derived hard cap (default 0.5); no effect when override > 0.
    explore_variant_timeout_safety_margin: float = 0.5
    # The concurrency ladder's terminal state; SWEEP→CLOSE exits on it.
    last_conc_sweep: dict[str, Any] = field(default_factory=dict)
    # Durable watermark from the last real conc_sweep measurement; survives the
    # macro-cycle reloop clearing ``last_conc_sweep`` so redundant closeout is
    # skipped when no validated gain landed since the prior conc_sweep.
    last_conc_sweep_watermark: dict[str, Any] = field(default_factory=dict)
    # Most recent run_optimization_done so Orch doesn't re-dispatch the same kernel_id every tick.
    last_kernel_opt: dict[str, Any] = field(default_factory=dict)
    # Most recent forge-fusion run result and its e2e integrate result; persisted
    # so resume does not rerun a completed fusion loop or lose the adoption audit.
    last_fusion: dict[str, Any] = field(default_factory=dict)
    last_fusion_integrate: dict[str, Any] = field(default_factory=dict)
    # How many times forge-fusion aborted on infrastructure this session, so KERNEL
    # entry can stop re-arming a cause that does not heal. Counted here rather than
    # inside ``last_fusion`` because that record is replaced by every run: an
    # unrelated failure landing between two aborts would carry no count forward and
    # silently reset the cap. Monotonic -- every outcome that would justify a reset
    # already stops the gate on its own.
    fusion_infra_aborts: int = 0
    # Most recent collective campaign and capped integration audit.
    last_collective: dict[str, Any] = field(default_factory=dict)
    collective_attempts: list[dict[str, Any]] = field(default_factory=list)
    collective_only_mode: bool = False
    # Most recent run_optimization dispatch skipped with no eligible kernels;
    # recorded as a non-failure so the breakdown can surface it.
    last_kernel_opt_dispatch_skip: dict[str, Any] = field(default_factory=dict)
    # Per-action audit (kernel parity): each ``last_<action>`` is the most recent attempt snapshot; ``<action>_attempts`` is a capped list.
    last_baseline: dict[str, Any] = field(default_factory=dict)
    last_profile: dict[str, Any] = field(default_factory=dict)
    # GEAK FP8 GEMM tuning snapshot (kernel_agent-owned): aiter A8W8 tuned CSV + SGLang dispatch patch before kernel_opt.
    last_gemm_tuning: dict[str, Any] = field(default_factory=dict)
    # merged explore action snapshot (same schema as other ``last_<action>`` mirrors).
    last_explore: dict[str, Any] = field(default_factory=dict)
    last_roofline: dict[str, Any] = field(default_factory=dict)
    baseline_attempts: list[dict[str, Any]] = field(default_factory=list)
    profile_attempts: list[dict[str, Any]] = field(default_factory=list)
    gemm_tuning_attempts: list[dict[str, Any]] = field(default_factory=list)
    # explore audit log (capped per _DEFAULT_ATTEMPTS_HISTORY).
    explore_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Capped roofline audit log with snapshot ids and analysis paths.
    roofline_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Global rolling log of unpromotable task results (cap _DEFAULT_LAST_FAILURES); rich failure context for self-correction. Covers every task kind.
    last_action_failures: list[dict[str, Any]] = field(default_factory=list)
    # Structured per-variant failure evidence, capped and keyed by failure_id (last-wins).
    failures: list[dict[str, Any]] = field(default_factory=list)
    # Authoritative per-kernel optimization history keyed by stable task identity.
    kernel_opt_task_attempts: dict[str, Any] = field(default_factory=dict)
    # Immutable KEEP snapshots awaiting E2E integration, keyed by integration_id.
    # Their lifecycle is independent of trace-local kernel ordinals.
    pending_kernel_integrations: dict[str, Any] = field(default_factory=dict)
    # Consecutive grid-runner tasks with no new current_best; Robustness nudges Orch off the plateau. Reset on advance.
    params_no_promote_streak: int = 0
    # Unified persistent explore-search ledger; ``tested`` keyed by canonical_fingerprint, ``accepted`` holds the round's KEEPs, everything graded down moves to rejected.
    explore_search: dict[str, Any] = field(default_factory=dict)
    # specialist sub-agent rolling state; one entry per config-arm round (round_id, tasks, proposals_total/kept/rejected/skipped, etc.).
    specialist_rounds: list[dict[str, Any]] = field(default_factory=list)
    # Per-kb_anchor coverage counters: config-arm rounds since a specialist was
    # dispatched / since a KEEP landed. Both ++ once per round, reset on
    # dispatch / KEEP.
    rounds_since_last_specialist: dict[str, int] = field(default_factory=dict)
    rounds_since_last_keep: dict[str, int] = field(default_factory=dict)
    # last specialist task snapshot (parity with other ``last_<action>`` mirrors).
    last_specialist: dict[str, Any] = field(default_factory=dict)
    # Patch verdict ledger keyed by review subject (a specialist task_id, or a candidate id for a PR pre-screen); Critic must approve/advise before PolicyGate allows the integrate_patch delegate.
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
    # Research-lane capacity locked at session start (core field; PolicyGate denies mid-session mutation).
    research_lane_capacity: int = 1
    # GPU pool capacity for needs_gpu specialists (0 disables); locked at
    # session start. The dataclass default is a placeholder for tests/direct
    # construction; the CLI/manifest default is whole-machine GPU detection.
    gpu_specialist_capacity: int = 0
    # escalate_strategy_change carry-over: Coordinator writes validated next_action_hint here for compute_next_phase, then clears it either by consuming it (drove a transition) or discarding it (an unrelated transition fired while it was pending).
    pending_escalate_hint: str = ""
    # last hint that actually drove a phase transition (audit only) for the breakdown.
    last_consumed_escalate_hint: str = ""
    last_consumed_escalate_hint_ts: str = ""
    # last hint thrown away by an unrelated transition, never acted on (audit only) for the breakdown. Distinct from last_consumed_escalate_hint: that field means "this drove a transition", which a discarded hint never did.
    last_discarded_escalate_hint: str = ""
    last_discarded_escalate_hint_ts: str = ""
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
    # Consecutive KERNEL_AGENT ticks that observed no forward motion AND no
    # kernel-lane task in flight. Lets the phase machine wind down to SWEEP
    # instead of spinning on hallucinated kernel-id requests / no-intent turns
    # until the wall-clock cap. Reset to 0 outside KERNEL_AGENT and whenever
    # ``kernel_progress_fingerprint`` changes.
    kernel_idle_ticks: int = 0
    # Digest of the KERNEL progress signals (attempt ledger, rejected ids, last
    # kernel_opt, stack depth, in-flight kernel task ids) seen on the tick that
    # opened the current idle streak. The streak used to be driven by
    # ``kernel_work_pending``, which reports whether the ledger still holds
    # anything unresolved — attempts that can never be advanced answer "yes"
    # forever, so the counter was pinned at 0 through 1130 idle ticks. A digest
    # answers the question that actually matters: did anything change?
    kernel_progress_fingerprint: str = ""
    # Unix time the current idle streak opened. Ticks alone are a poor stall
    # measure (a few seconds each, and the phase machine advances more than once
    # per coordinator tick), so the wind-down also requires the streak to have
    # lasted ``KERNEL_IDLE_MIN_SECONDS``. Rebased to now while a kernel-lane task
    # is in flight so a 30-minute build never accrues idle time.
    kernel_idle_since_unix: float = 0.0
    # Unix time an inline kernel request (``integrate``, ``run_optimization``,
    # ...) last reported itself running. Those are awaited straight in the intent
    # router and never become a row in the task registry, so the KERNEL idle
    # guard's in-flight probe cannot see them and reads a long one as a dead
    # phase. A timestamp rather than a flag, so one left behind by a process that
    # died mid-step goes stale instead of muting the guard for the next run.
    kernel_inline_step_seen_unix: float = 0.0

    # Which macro cycle's kernel nomination pass has run to completion. A kernel
    # a nominator looked at and passed over leaves no ledger row, so it stays
    # "untried" forever and the phase-pending predicate never goes quiet. Stored
    # as the cycle rather than a bare flag so the next cycle retires it without
    # anyone having to clear it.
    kernel_auto_pass_cycle: int | None = None

    # Search-space expansion ledger surfaced in the Orchestration prompt.
    discovered_flags: dict[str, Any] = field(default_factory=dict)

    # Monotonic Coordinator tick counter; stable anchor for plateau/phase budget math.
    tick: int = 0
    # Percent improvement still needed to reach the objective (0.0 => none/reached); fact for the "Mission progress" line, not a priority.
    target_gap_pct: float = 0.0

    # Phase state machine fields
    # ``phase`` — run-level pipeline phase (PRELUDE/FRAMEWORK_AGENT/KERNEL_AGENT/SWEEP/CLOSE); Coordinator-only (CORE_STATE_FIELDS). Empty => not yet initialised.
    phase: str = ""
    # ISO UTC timestamp the current phase was entered (breakdown.phase_segments + budget judge).
    phase_started_ts: str = ""
    # Unix epoch matching ``phase_started_ts`` so the budget judge skips ISO re-parsing.
    phase_started_unix: float = 0.0
    # Append-only log of phase transitions (rows from machine_state.make_history_row; reason in PHASE_EXIT_REASONS). Capped at _PHASE_HISTORY_CAP.
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    # Durable sum of completed optimisation-phase segments. None means a legacy
    # resumed state whose pre-upgrade total is unknowable. The current active
    # segment is added at status-render time; accumulating completed segments
    # avoids undercounting long macro-cycle runs after phase_history is capped.
    explore_elapsed_accum_s: float | None = 0.0
    # Durable per-phase sum of COMPLETED segments, keyed by phase name.
    # ``phase_started_unix`` is reset on EVERY phase entry, so a budget guard
    # reading it alone measures the CURRENT entry only — and each phase is
    # re-entered once per macro-cycle. That turned "KERNEL gets 15% of the run"
    # into "KERNEL gets 15% of the run every time it is entered": three entries
    # burned 288% of the cap while the other phases starved. The guards read
    # this total plus the live segment instead. ``bank_phase_segment`` is the
    # only writer; unlike ``explore_elapsed_accum_s`` there is no "unknown"
    # sentinel, because a budget guard must never read "unknown" as "no cap"
    # (see ``from_raw`` for how a pre-upgrade state is reconstructed).
    phase_elapsed_totals: dict[str, float] = field(default_factory=dict)
    # Append-only operator-facing lifecycle log. Each row (built by
    # :func:`machine_state.make_lifecycle_event`) records a phase/step boundary
    # plus artifact paths. Coordinator-only writer; capped at ``_LIFECYCLE_CAP``.
    lifecycle: list[dict[str, Any]] = field(default_factory=list)
    # Wall-clock budget percentages per phase (from CLI flags/defaults); persisted for resume. Empty => library defaults.
    phase_budget_pct: dict[str, float] = field(default_factory=dict)
    # Cyclic phase machine macro-cycle counter (cycle 0 is the first pass; each
    # SWEEP→FRAMEWORK_AGENT loopback increments it). Stamped onto every phase_history row.
    macro_cycle: int = 0
    # Per-cycle budget: wall-clock minutes for ONE macro-cycle. When > 0 the
    # per-phase budget math is computed against this window instead of
    # ``max_minutes``. 0 disables (phase budgets are % of total).
    cycle_minutes: float = 0.0
    # Global-convergence tracking: validated cumulative gain at the current
    # macro-cycle's start, and the consecutive no-gain cycle streak.
    gain_at_cycle_start: float = 0.0
    no_gain_cycle_streak: int = 0
    # Cyclic bottleneck re-direction: set when a cyclic config plateau winds the
    # cycle down; the next macro-cycle's prompt surfaces a redirect advisory off
    # ``last_cycle_bottleneck``. Cleared once the live top bottleneck drifts off it.
    pending_bottleneck_switch: bool = False
    last_cycle_bottleneck: str = ""
    # Latest roofline saturation per specialist-domain family and prev->current
    # cycle bottleneck movement. Coordinator/record_trace are the only writers.
    saturated_directions: dict[str, dict[str, Any]] = field(default_factory=dict)
    bottleneck_shift: dict[str, Any] = field(default_factory=dict)
    # Per-cycle advisory focus log; persisted so cycle strategy survives resume.
    cycle_strategy_log: list[dict[str, Any]] = field(default_factory=list)

    # Recipe KB integration fields — Coordinator-only writers.
    # ``recipe_kb_session_id`` — hyperloom-local id carried into KB fact-write attrs; defaults to session_dir.name.
    recipe_kb_session_id: str = ""
    # Snapshot of ``recipe_kb_t0._cascade_warm_start_search`` output (parsed dict); empty on first session for a (workload, hw) pair.
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    # Snapshot of ``pitfalls`` output (negative priors), list of KB point dicts; consumed by the specialist prompt. Resume tolerates older snapshots.
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 snapshot of ``lessons`` output (positive priors), symmetric with warm_start_pitfalls; consumed by the specialist prompt. Empty under --degraded-kb or T0 failure.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    # ISO UTC timestamp of the T0 snapshot; empty under --degraded-kb or T0 failure.
    warm_start_ts: str = ""
    # Model-facing advisory context built by ``recipe_kb_t0``. Current remote
    # records carry match/history/KG data only; local legacy records may also
    # carry their compatibility replay projection.
    warm_start_context: dict[str, Any] = field(default_factory=dict)

    # structured gaps ledger: dedup'd unresolved bottlenecks (Coordinator-only _refresh_gaps; CORE_STATE_FIELDS); dedup keyed by canonical_id, attempts capped 20/gap, list capped _GAPS_MAX_ENTRIES.
    gaps: list[dict[str, Any]] = field(default_factory=list)

    # Orchestration working memory — durable compacted reasoning snapshot for compaction + crash-recovery rebuild; Coordinator-only writer.
    orchestration_memory: dict[str, Any] = field(default_factory=dict)

    # Bounded rollback ring (cap 10) of prior good ``orchestration_memory``
    # records; recovers a later degenerate compaction from a prior snapshot.
    orchestration_memory_history: list[dict[str, Any]] = field(default_factory=list)

    # Census of orchestration prompt pushes: {"seed": n, "delta": n}; a ratio
    # near 1:0 means compaction is re-seeding the conversation every tick.
    orchestration_prompt_modes: dict[str, int] = field(default_factory=dict)

    # Bounded ring (cap 10) of per-macro-cycle directives injected into the
    # orchestration system prompt; entries: {cycle, directive, source, ts}.
    cycle_directive_history: list[dict[str, Any]] = field(default_factory=list)

    # Non-field instance attr (set in load_or_init / save): session dir for
    # breakdown instrumentation. Plain class attr => not serialized.
    _session_dir = None

    #: Fields of :meth:`profile_workload_context` that say *what was profiled*.
    #: The rest -- ``server_args``, ``extra_envs``, ``remove_args``,
    #: ``unset_envs``, ``args_mode`` -- say how the profile task was
    #: parameterized, and are only populated when the recorder had those params
    #: to hand. Two call sites record the same trace differently for that reason
    #: alone, so comparing them makes a perfectly fresh trace read as stale.
    #: ``serving_config`` is excluded here too: it has its own comparison, which
    #: comes from ``current_best`` on both sides and is therefore symmetric.
    #:
    #: ``ClassVar`` because a bare annotation would make this constant a
    #: dataclass field: it would be written into every ``state.json``, accepted
    #: back from disk, and writable through ``apply_changes`` since a constant
    #: is not something ``CORE_STATE_FIELDS`` thinks to lock. None of that
    #: changes behaviour while the sole reader goes through ``cls``, which is
    #: exactly what makes it worth closing -- it decides trace staleness, so an
    #: instance-scoped read added later would let a stored value govern whether
    #: a profile is reused or re-run.
    PROFILE_WORKLOAD_IDENTITY_KEYS: ClassVar[tuple[str, ...]] = (
        "framework",
        "precision",
        "model_path",
        "tp",
        "conc",
        "isl",
        "osl",
        "max_model_len",
    )

    @classmethod
    def profile_workload_identity(cls, context: Any) -> dict[str, Any]:
        """Project a workload context down to what identifies the profiled run.

        Args:
            context (Any): A :meth:`profile_workload_context` result, or
                anything else (treated as carrying no identity).

        Returns:
            dict[str, Any]: The identity fields, missing ones included as
            ``None`` so a recorded context and a freshly built one compare
            equal when they describe the same workload.
        """
        if not isinstance(context, Mapping):
            return {}
        return {key: context.get(key) for key in cls.PROFILE_WORKLOAD_IDENTITY_KEYS}

    def profile_workload_context(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the normalized workload and runtime identity for a profile trace."""
        params = overrides if isinstance(overrides, dict) else {}
        context: dict[str, Any] = {}
        for name in ("framework", "precision", "model_path"):
            value = params.get(name)
            if value in (None, ""):
                value = getattr(self, name, "")
            normalized = str(value or "").strip()
            if name in ("framework", "precision"):
                normalized = normalized.lower()
            elif normalized:
                path = Path(normalized).expanduser()
                if path.exists():
                    try:
                        normalized = str(path.resolve())
                    except OSError:
                        log.debug(
                            "profile workload path resolution failed for %s; keeping the unresolved path",
                            path,
                            exc_info=True,
                        )
            context[name] = normalized
        for name in ("tp", "conc", "isl", "osl", "max_model_len"):
            value = params.get(name)
            if value in (None, ""):
                value = getattr(self, name, 0)
            try:
                context[name] = int(value or 0)
            except (TypeError, ValueError):
                context[name] = 0
        raw_server_args = (
            params.get("extra_server_args") if "extra_server_args" in params else params.get("base_extra_args")
        )
        server_args = str(raw_server_args or "").strip()
        server_args = sanitize_profile_server_args(server_args)
        try:
            context["server_args"] = shlex.join(shlex.split(server_args)) if server_args else ""
        except ValueError:
            context["server_args"] = " ".join(server_args.split())

        raw_envs = params.get("extra_envs") if "extra_envs" in params else params.get("base_extra_envs")
        if isinstance(raw_envs, dict):
            context["extra_envs"] = {
                str(key): str(value) for key, value in sorted(raw_envs.items(), key=lambda item: str(item[0]))
            }
        else:
            context["extra_envs"] = {}

        def _normalized_list(base_name: str, direct_name: str) -> list[str]:
            raw = params.get(direct_name) if direct_name in params else params.get(base_name)
            values = [raw] if isinstance(raw, str) else list(raw or [])
            return sorted({str(value).strip() for value in values if str(value).strip()})

        context["remove_args"] = _normalized_list("base_remove_args", "remove_args")
        context["unset_envs"] = _normalized_list("base_unset_envs", "unset_envs")
        args_mode = params.get("args_mode") if "args_mode" in params else params.get("base_args_mode")
        context["args_mode"] = str(args_mode or "append").strip().lower()

        current_best = self.current_best if isinstance(self.current_best, dict) else {}
        raw_envs = current_best.get("extra_envs") or {}
        extra_envs = (
            {
                str(key): str(value)
                for key, value in sorted(raw_envs.items(), key=lambda item: str(item[0]))
                if str(key).strip()
            }
            if isinstance(raw_envs, dict)
            else {}
        )
        serving_config = {
            "extra_server_args": str(current_best.get("extra_server_args") or "").strip(),
            "extra_envs": extra_envs,
        }
        if any((serving_config["extra_server_args"], extra_envs)):
            context["serving_config"] = serving_config
        return context

    def profile_trace_matches_workload(
        self,
        expected: dict[str, Any] | None = None,
    ) -> bool:
        """Whether the recorded profile trace is fresh for a target workload.

        Single source of truth for the "latest profile succeeded AND its recorded
        workload matches the active (or given) workload" freshness rule shared by
        the forge shape resolvers and the kernel-entry reprofile gate. Returns
        ``False`` when the last profile did not succeed or recorded no workload.
        """
        if str(getattr(self, "last_profile_status", "") or "").strip().lower() != "succeeded":
            return False
        recorded = getattr(self, "last_profile_workload", None)
        if not isinstance(recorded, dict) or not recorded:
            return False
        # Default to the *current-best* runtime identity, not the bare context:
        # last_profile_workload is recorded with the real profile params (actual
        # server_args / extra_envs), while profile_workload_context() with no
        # overrides reports server_args="" and skips the current_best runtime
        # backfill, so any workload with server args/extra envs would compare
        # unequal and every fresh profile would be discarded as stale. This
        # matches the vLLM block-FP8 path, which passes
        # current_profile_workload_context() as ``expected``.
        target = expected if isinstance(expected, dict) and expected else self.current_profile_workload_context()
        return recorded == target

    def record_profile_workload(
        self,
        params: dict[str, Any] | None = None,
        *,
        arm: str = "",
    ) -> dict[str, Any]:
        """Record the workload identity and originating arm for a profile trace.

        Args:
            params (dict[str, Any] | None): The profile task params the trace
                was produced with.
            arm (str): The profiled roofline arm; ``"baseline"`` pins the
                recorded action, anything else resolves it from
                ``current_best``.

        Returns:
            dict[str, Any]: The recorded workload context.
        """
        self.last_profile_workload = self.profile_workload_context(params or {})
        if arm == "baseline":
            self.last_profile_workload_action = "baseline"
        else:
            current_best = self.current_best if isinstance(self.current_best, dict) else {}
            self.last_profile_workload_action = str(current_best.get("action") or "")
        self.last_profile_args = str(self.last_profile_workload.get("server_args") or "")
        return self.last_profile_workload

    def current_profile_workload_context(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the profile identity for the active current-best runtime."""
        current_best = self.current_best if isinstance(self.current_best, dict) else {}
        incoming = overrides if isinstance(overrides, dict) else {}
        params: dict[str, Any] = {}
        runtime_keys = {
            "extra_server_args",
            "extra_envs",
            "remove_args",
            "unset_envs",
            "args_mode",
            "base_extra_args",
            "base_extra_envs",
            "base_remove_args",
            "base_unset_envs",
            "base_args_mode",
        }
        has_runtime_override = any(key in current_best for key in runtime_keys) or any(
            key in incoming for key in runtime_keys
        )
        recorded = self.last_profile_workload if isinstance(self.last_profile_workload, dict) else {}
        # A bare baseline/profile ``current_best`` carries no runtime fields, so
        # the only record of what the server actually ran with is the last
        # profile. Backfilling it keeps "no override" from reading as an empty
        # runtime, but only when that profile measured the same arm: a runtime
        # recorded off a tuned arm would otherwise be reported as still active.
        recorded_action = str(self.last_profile_workload_action or "").strip()
        if (
            not has_runtime_override
            and current_best.get("action") in {"baseline", "profile"}
            and recorded
            # Legacy sessions predate the recorded action; keep reusing them
            # rather than forcing a reprofile on every run.
            and recorded_action in {"", "baseline", "profile"}
        ):
            params.update(
                {
                    "base_extra_args": recorded.get("server_args", ""),
                    "base_extra_envs": dict(recorded.get("extra_envs") or {}),
                    "base_remove_args": list(recorded.get("remove_args") or []),
                    "base_unset_envs": list(recorded.get("unset_envs") or []),
                    "base_args_mode": recorded.get("args_mode", "append"),
                }
            )
        for source, target in (
            ("extra_server_args", "base_extra_args"),
            ("remove_args", "base_remove_args"),
            ("unset_envs", "base_unset_envs"),
            ("args_mode", "base_args_mode"),
        ):
            if source in current_best:
                params[target] = current_best[source]
            if source in incoming:
                params[target] = incoming[source]
            if target in incoming:
                params[target] = incoming[target]

        raw_current_envs = current_best.get("extra_envs")
        merged_envs = dict(raw_current_envs) if isinstance(raw_current_envs, dict) else {}
        if isinstance(incoming.get("base_extra_envs"), dict):
            merged_envs.update(incoming["base_extra_envs"])
        if isinstance(incoming.get("extra_envs"), dict):
            merged_envs.update(incoming["extra_envs"])
        if merged_envs:
            params["base_extra_envs"] = merged_envs

        for name in (
            "framework",
            "precision",
            "model_path",
            "tp",
            "conc",
            "isl",
            "osl",
            "max_model_len",
        ):
            if name in incoming:
                params[name] = incoming[name]
        return self.profile_workload_context(params)

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

        Raises:
            ValueError: If ``state.json`` exists but is not a JSON object.
        """
        path = cls.state_path(session_dir)
        if not path.exists():
            inst = cls()
        else:
            inst = cls.from_dict(read_json(path, require_dict=True, strict=True))
        # Remember the session dir for breakdown instrumentation (not serialized).
        inst._session_dir = Path(session_dir)
        return inst

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SharedState":
        """Construct a :class:`SharedState` from a raw mapping, migrating it.

        Acts as the unified migration entry point: an absent
        ``schema_version`` is treated as 1 and unknown keys are dropped. The
        operation is idempotent and short-circuits when already at the latest
        schema.

        Args:
            raw: Decoded state mapping (e.g. from JSON on disk).

        Returns:
            A fully-populated, migrated :class:`SharedState` instance.
        """
        # Unified migration entry point; absent schema_version treated as 1. Idempotent (latest version short-circuits).
        incoming_version = int(raw.get("schema_version") or 1)

        # Filter to known fields; unknown keys dropped, missing keys default.
        # ``fields()`` rather than ``__dataclass_fields__``: the latter also
        # holds ClassVar pseudo-fields, which ``__init__`` does not accept, so a
        # state.json naming one would raise here instead of being ignored.
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        # A pre-telemetry state may already have completed optimisation segments,
        # but their exact sum cannot be reconstructed once phase_history has
        # been capped. Preserve "unknown" instead of reporting a misleading
        # partial zero after resume. Fresh states still start from 0.0.
        if "explore_elapsed_accum_s" not in raw:
            filtered["explore_elapsed_accum_s"] = None
        # A state written before per-phase totals existed still records every
        # transition in phase_history, so the completed segments are
        # reconstructible. Rebuilding beats defaulting to an empty dict: an empty
        # dict silently re-arms the per-entry bug for the rest of a resumed run.
        # The rebuild errs in both directions — the cap drops old segments, and a
        # segment straddling a resume carries the idle gap (see the helper).
        if not isinstance(filtered.get("phase_elapsed_totals"), dict):
            from ..phases.machine_state import phase_elapsed_totals_from_history

            filtered["phase_elapsed_totals"] = phase_elapsed_totals_from_history(
                filtered.get("phase_history"),
            )
        if not isinstance(filtered.get("specialist_patch_verdicts"), dict):
            filtered["specialist_patch_verdicts"] = {}
        if not isinstance(filtered.get("kernel_opt_task_attempts"), dict):
            filtered["kernel_opt_task_attempts"] = {}
        if not isinstance(filtered.get("pending_kernel_integrations"), dict):
            filtered["pending_kernel_integrations"] = {}
        # Normalize the unified ``explore_search`` ledger at load.
        filtered["explore_search"] = cls._build_explore_search(
            existing=filtered.get("explore_search"),
        )

        if incoming_version < 4:
            # Lift flat enablement_* keys from old state.json into EnablementRound.
            _ENABLEMENT_ROUND_FIELDS = {f.name for f in fields(EnablementRound)}
            flat = {
                k[len("enablement_") :]: v
                for k, v in raw.items()
                if k.startswith("enablement_") and k[len("enablement_") :] in _ENABLEMENT_ROUND_FIELDS
            }
            # Prefer any nested blob already present (from a partial migration).
            if not isinstance(filtered.get("enablement"), dict):
                filtered["enablement"] = flat
            else:
                for k, v in flat.items():
                    filtered["enablement"].setdefault(k, v)

        if incoming_version < 5:
            # Carry the pre-rename FRAMEWORK fields over. Read from ``raw``:
            # the old spellings are not dataclass fields, so the filter above
            # has already discarded them. A state holding both spellings is
            # mid-migration, and the current one wins.
            for legacy, current in _FRAMEWORK_FIELD_RENAMES_V5.items():
                if legacy in raw and current not in raw:
                    filtered[current] = raw[legacy]

            # Renaming the fields is not enough: a session that already promoted
            # a FRAMEWORK KEEP has stack entries whose variant_name still carries
            # the promote-side prefix, and reconciliation keys on the bare
            # candidate key. Left alone they read as orphaned KEEPs for the rest
            # of the session and no longer collide with the dedup key.
            stack = filtered.get("optimization_stack")
            if isinstance(stack, list):
                # Rebuilt rather than edited in place: ``filtered`` is a shallow
                # copy, so mutating an entry would also rewrite the caller's
                # ``raw`` — and ``from_dict`` takes a mapping it does not own.
                filtered["optimization_stack"] = [
                    {**entry, "variant_name": str(entry["variant_name"])[len(_FRAMEWORK_VARIANT_PREFIX_V5) :]}
                    if (
                        isinstance(entry, dict)
                        and str(entry.get("action") or "") == _FRAMEWORK_STACK_ACTION_V5
                        and str(entry.get("variant_name") or "").startswith(_FRAMEWORK_VARIANT_PREFIX_V5)
                    )
                    else entry
                    for entry in stack
                ]

        if incoming_version < 6:
            # Carry the pre-rename KERNEL-entry dispatch switch over. Same
            # reasoning as the v5 block: the old spelling is not a dataclass
            # field, so the filter above has already dropped it, and a state
            # holding both spellings is mid-migration with the current one
            # winning.
            for legacy, current in _KERNEL_OPT_FIELD_RENAMES_V6.items():
                if legacy in raw and current not in raw:
                    filtered[current] = bool(raw[legacy])

        if isinstance(filtered.get("enablement"), dict):
            filtered["enablement"] = EnablementRound.from_dict(filtered["enablement"])
        elif not isinstance(filtered.get("enablement"), EnablementRound):
            filtered["enablement"] = EnablementRound()

        filtered["schema_version"] = LATEST_STATE_SCHEMA_VERSION

        return cls(**filtered)

    @staticmethod
    def _build_explore_search(
        *,
        existing: Any,
    ) -> dict[str, Any]:
        """Shape the unified ``explore_search`` ledger at load time.

        Args:
            existing (Any): The persisted ``explore_search`` dict (or any
                value; non-dicts are treated as empty).

        Returns:
            dict[str, Any]: The normalized ``explore_search`` ledger with all
                required keys defaulted and history normalized.
        """
        from ..actions.executors._canonical_fingerprint import canonical_fingerprint as _fp

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

        # winners_history: normalize persisted rows, sorted by (round_id, ts).
        wh: list[dict[str, Any]] = []
        for source_list in (existing.get("winners_history") or [],):
            if not isinstance(source_list, list):
                continue
            for entry in source_list:
                if not isinstance(entry, dict):
                    continue
                controls: dict[str, Any] = {}
                for key in ("remove_args", "unset_envs"):
                    raw = entry.get(key)
                    if isinstance(raw, str):
                        vals = [raw.strip()] if raw.strip() else []
                    elif isinstance(raw, (list, tuple, set)):
                        vals = [str(v).strip() for v in raw if str(v).strip()]
                    else:
                        vals = []
                    if vals:
                        controls[key] = vals
                mode = str(entry.get("args_mode") or "append").strip().lower()
                if mode == "replace":
                    controls["args_mode"] = "replace"
                fp_val = entry.get("fingerprint") or _fp(
                    str(entry.get("extra_server_args") or ""),
                    dict(entry.get("extra_envs") or {}),
                    **controls,
                )
                wh.append(
                    {
                        "round_id": str(entry.get("round_id") or ""),
                        "variant_name": str(entry.get("variant_name") or entry.get("name") or ""),
                        "fingerprint": str(fp_val),
                        "gain_pct": entry.get("gain_pct"),
                        "extra_args": str(entry.get("extra_args") or entry.get("extra_server_args") or ""),
                        "extra_envs": dict(entry.get("extra_envs") or {}),
                        **controls,
                        "provenance": str(entry.get("provenance") or ""),
                        "ts": str(entry.get("ts") or ""),
                    }
                )
        wh.sort(key=lambda r: (str(r.get("round_id") or ""), str(r.get("ts") or "")))
        out["winners_history"] = wh

        # synergy_attempted: normalize executor-side combos, deduped.
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

        for source in (existing.get("synergy_attempted") or [],):
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
        # Backfill scriptable/diffusion (xDiT) ``e2el_mean_ms`` from ``tput``
        # so current_best carries the primary latency metric. Best-effort.
        self._backfill_scriptable_latency()
        path = self.state_path(session_dir)
        atomic_write_json(path, self.to_dict(), indent=2, sort_keys=True)
        # Author-time breakdown capture: snapshot state-owned sections into the
        # recorder spool right after persisting. Best-effort; never blocks save.
        self._session_dir = Path(session_dir)
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.snapshot_state_sections(session_dir, self)
        except Exception:  # noqa: BLE001 — author-time capture must never block save
            log.debug("snapshot_state_sections failed", exc_info=True)
        # Derived artifact: re-render current_setting.sh from the current best
        # route so the operator can audit / re-feed it via --reference-script.
        try:
            cb = self.current_best or {}
            if cb:
                from hyperloom.inference_optimizer.reference_script import render_reference_script

                text = render_reference_script(
                    framework=str(self.framework or os.environ.get("FRAMEWORK", "sglang")),
                    server_args=str(cb.get("extra_server_args") or ""),
                    envs=dict(cb.get("extra_envs") or {}),
                    model=self.reference_model or os.environ.get("MODEL_PATH"),
                    tp=int(self.tp or 0) or None,
                    max_model_len=int(self.max_model_len or 0) or None,
                    gpu_type=str(self.gpu_type or os.environ.get("GPU_TYPE", "")) or None,
                )
                (Path(session_dir) / "current_setting.sh").write_text(
                    text,
                    encoding="utf-8",
                )
        except Exception:  # noqa: BLE001 — derived artifact, never fatal
            log.debug("current_setting.sh render failed", exc_info=True)
        # Live status mirror: reflect the persisted snapshot into Langfuse for
        # real-time status. Throttled and best-effort; never blocks the save path.
        try:
            from ..trace.langfuse_emitter import record_status as _lf_record_status

            _lf_record_status(session_dir, self._langfuse_status_summary())
        except Exception:  # noqa: BLE001 — status mirror must never block save
            log.debug("langfuse status mirror failed", exc_info=True)

    def _backfill_scriptable_latency(self) -> None:
        """Derive ``current_best.e2el_mean_ms`` from ``tput`` for scriptable runs.

        No-op for serving frameworks, when there is no current best, or when a
        measured ``e2el_mean_ms`` is already present. For scriptable xDiT the
        per-image e2e latency equals ``1000 / img_per_s`` (single-stream), so the
        stored ``tput`` fully determines it. Never raises.
        """
        try:
            from hyperloom.inference_optimizer import framework_registry

            fw = str(getattr(self, "framework", "") or "")
            if not framework_registry.is_scriptable(fw):
                return
            cb = self.current_best
            if not isinstance(cb, dict):
                return
            if cb.get("e2el_mean_ms") is not None:
                return
            tput = cb.get("tput")
            if not isinstance(tput, (int, float)) or tput <= 0:
                return
            e2el = framework_registry.primary_metric_value(fw, float(tput))
            if e2el is not None and e2el > 0:
                cb["e2el_mean_ms"] = round(float(e2el), 4)
        except Exception:  # noqa: BLE001 — derived backfill, never blocks save
            pass

    def _langfuse_status_summary(self) -> dict[str, Any]:
        """Flatten the state into an OTEL-friendly scalar status snapshot.

        Only top-level scalars (str/bool/int/float) are emitted so each key
        lands as a directly-filterable Langfuse trace-metadata attribute
        (nested values would be JSON-stringified). Float gains/throughput are
        rounded so tiny per-tick deltas don't defeat the emitter's on-change
        throttle. Best-effort: any field access is defensive.

        Returns:
            dict[str, Any]: The flat scalar status summary to mirror.
        """
        cb = self.current_best if isinstance(self.current_best, dict) else {}
        last = self.lifecycle[-1] if self.lifecycle else {}
        summary: dict[str, Any] = {
            "phase": self.phase or "",
            "stop_reason": self.stop_reason or "",
            "closing_phase": bool(self.closing_phase),
            "degraded_mode": bool(self.degraded_mode),
            "cumulative_gain_validated": round(
                float(self.cumulative_gain_validated or 0.0),
                2,
            ),
            "baseline_failure_streak": int(self.baseline_failure_streak or 0),
            "macro_cycle": int(self.macro_cycle or 0),
            "session_id": self.session_id or "",
            "model_name": self.model_name or "",
            "framework": self.framework or os.environ.get("FRAMEWORK", "") or "",
            "kb_hit": str((self.warm_start_context or {}).get("status") or ""),
        }
        try:
            from ..phases.machine_state import explore_elapsed_seconds

            session_elapsed_s = max(0.0, self.elapsed_minutes() * 60.0)
            summary["session_elapsed_s"] = int(round(session_elapsed_s))
            explore_elapsed_s = explore_elapsed_seconds(self)
            if explore_elapsed_s is not None:
                summary["explore_elapsed_s"] = int(round(explore_elapsed_s))
                summary["explore_ratio"] = (
                    round(explore_elapsed_s / session_elapsed_s, 4) if session_elapsed_s > 0.0 else 0.0
                )
        except Exception:  # noqa: BLE001 - status telemetry must stay best-effort
            log.debug("explore runtime telemetry derivation failed", exc_info=True)
        tput = cb.get("tput") if isinstance(cb, dict) else None
        if isinstance(tput, (int, float)) and not isinstance(tput, bool):
            summary["current_best_tput"] = round(float(tput), 1)
        if isinstance(last, dict):
            if last.get("seq") is not None:
                summary["last_seq"] = last.get("seq")
            if last.get("status"):
                summary["last_lifecycle_status"] = str(last.get("status"))
            if last.get("phase"):
                summary["last_lifecycle_phase"] = str(last.get("phase"))
        return {k: v for k, v in summary.items() if isinstance(v, (str, bool, int, float))}

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
        from ..policy import gate as _m

        return _m.record_policy_denial(
            self,
            action_name=action_name,
            rule=rule,
            hint=hint,
            intent_type=intent_type,
            tick=tick,
            intent_payload=intent_payload,
        )

    def reset_policy_denial_streak(self, action_name: str) -> None:
        """Forwarding shim — implementation in :mod:`.policy`."""
        from ..policy import gate as _m

        return _m.reset_policy_denial_streak(self, action_name)

    # stop_reason ENUM validator
    def set_stop_reason(
        self,
        value: str,
        *,
        strict: bool | None = None,
    ) -> str:
        """Validated writer for :attr:`stop_reason` (Inv-8.3 closed vocab): values outside ``STOP_REASON_VOCAB`` map to ``"unknown"`` (lenient) or raise (``strict=True``, default env ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON``). Returns value written.

        The first write also stamps :attr:`stop_ts`, so the session's end is
        recorded once by its producer instead of being guessed by whoever reads
        the state later.

        Args:
            value (str): The proposed stop reason; blank clears
                :attr:`stop_reason` and :attr:`stop_ts`.
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
        from ..phases.machine_state import STOP_REASON_VOCAB, is_valid_stop_reason

        text = str(value or "").strip()
        if not text:
            self.stop_reason = ""
            self.stop_ts = ""
            return ""
        if is_valid_stop_reason(text):
            return self._commit_stop_reason(text)
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
        # Lenient: map to "unknown" and warn.
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "stop_reason=%r not in STOP_REASON_VOCAB; mapped to 'unknown' "
            ". Set "
            "INFERENCE_OPTIMIZER_STRICT_STOP_REASON=1 to fail-fast.",
            text,
        )
        return self._commit_stop_reason("unknown")

    def _commit_stop_reason(self, reason: str) -> str:
        """Write a validated stop reason, stamping the end time on the first one.

        The session ends when its first terminal reason is recorded. Later
        calls may refine the reason -- CLOSE stops the session on entry and the
        Coordinator's ``finally`` re-asserts it -- but the CLOSE sequence writes
        ``session_breakdown.json`` in between, so moving the timestamp would
        leave the shipped artifact and the state disagreeing about when the run
        ended.

        Args:
            reason (str): The validated, non-blank reason to record.

        Returns:
            str: ``reason``, so callers can return it unchanged.
        """
        self.stop_reason = reason
        if not self.stop_ts:
            self.stop_ts = _now_iso()
        return reason

    # escalate hint plumbing
    def set_pending_escalate_hint(self, hint: str) -> str:
        """Stash the LLM-supplied hint for the next phase compute pass; unknown hints dropped (Inv-8.2: closed vocab). Returns value written.

        Args:
            hint (str): The proposed escalate hint; values outside the closed
                vocab (and blanks) are dropped.

        Returns:
            str: The hint actually stored (``""`` when dropped or blank).
        """
        from ..phases.machine_state import is_valid_escalate_hint

        text = str(hint or "").strip()
        if text and not is_valid_escalate_hint(text):
            return ""
        self.pending_escalate_hint = text
        return text

    def consume_pending_escalate_hint(self) -> str:
        """Pop the pending hint because it drove a phase transition; returns cleared hint.

        Records the hint into ``last_consumed_escalate_hint`` — an audit field
        that specifically means "this hint drove a transition". A hint that
        was thrown away without acting on it is a different event and must go
        through :meth:`discard_pending_escalate_hint` instead, or a discard
        would misreport itself as a consumption in the breakdown.

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

    def discard_pending_escalate_hint(self) -> str:
        """Pop the pending hint because an unrelated transition fired without acting on it; returns cleared hint.

        Records the hint into ``last_discarded_escalate_hint`` rather than
        ``last_consumed_escalate_hint`` — the hint never drove anything, so
        recording it as consumed would tell the breakdown the opposite of
        what happened.

        Returns:
            str: The discarded hint (``""`` when none was pending).
        """
        hint = (self.pending_escalate_hint or "").strip()
        if not hint:
            return ""
        self.pending_escalate_hint = ""
        self.last_discarded_escalate_hint = hint
        self.last_discarded_escalate_hint_ts = _now_iso()
        return hint

    def enablement_close_guard_active(self) -> bool:
        """True while a not-yet-enabled run must be protected from premature close.

        While this guard is active a ``skip_to_close`` hint is dropped; a
        not-yet-enabled run may only terminate via honest paths that do not route
        through ``skip_to_close`` (``enablement_stalled``,
        ``prelude_baseline_failed``, the wall-clock/time-exhausted exits, or hard
        aborts).

        Returns:
            bool: ``True`` in PRELUDE / FRAMEWORK_AGENT while ``baseline_tput``
            has never gone positive and enablement has not yet succeeded.
        """
        phase = (self.phase or "").strip().upper()
        return (
            phase in ("PRELUDE", "FRAMEWORK_AGENT")
            and float(getattr(self, "baseline_tput", 0.0) or 0.0) <= 0.0
            and not self.enablement.succeeded
        ) or self.enablement.validation_pending

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
        """Forwarding shim — implementation in :mod:`hyperloom.orchestrator.phases.machine_state`."""
        from ..phases import machine_state as _m

        return _m.record_phase_transition(
            self, to_phase=to_phase, reason=reason, evidence=evidence, ts=ts, ts_unix=ts_unix
        )

    def append_phase_history_event(
        self,
        *,
        reason: str,
        evidence: dict[str, Any] | None = None,
        ts: str | None = None,
        ts_unix: float | None = None,
    ) -> dict[str, Any]:
        """Forwarding shim — implementation in :mod:`hyperloom.orchestrator.phases.machine_state`."""
        from ..phases import machine_state as _m

        return _m.append_phase_history_event(
            self,
            reason=reason,
            evidence=evidence,
            ts=ts,
            ts_unix=ts_unix,
        )

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

    def current_comm_pct(self) -> float | None:
        """Return the latest exposed-communication percentage."""
        snaps = self.roofline_snapshots if isinstance(self.roofline_snapshots, list) else []
        if not snaps:
            return None
        latest = snaps[-1]
        if not isinstance(latest, dict):
            raise ValueError("Latest roofline snapshot must be a mapping")
        value = latest.get("comm_pct")
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Latest comm_pct must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("Latest comm_pct must be finite and non-negative")
        return parsed

    @staticmethod
    def _collective_attempt_snapshot(result: dict[str, Any]) -> dict[str, Any]:
        """Build one compact collective campaign record."""
        return {
            "collective_attempt_id": str(result["collective_attempt_id"]),
            "integration_id": str(result.get("integration_id") or ""),
            "experiment_id": str(result.get("experiment_id") or ""),
            "analysis_key": str(result.get("analysis_key") or ""),
            "status": str(result.get("status") or ""),
            "decision": str(result.get("decision") or ""),
            "kept": result["kept"],
            "requires_e2e_validation": result["requires_e2e_validation"],
            "patch_cleanup_status": str(result.get("patch_cleanup_status") or result.get("integration_status") or ""),
            "integration_decision": str(result.get("integration_decision") or ""),
            "kernel_id": str(result.get("kernel_id") or ""),
            "kernel_name": str(result.get("kernel_name") or ""),
            "source_file": str(result.get("source_file") or result.get("target_file") or ""),
            "kernel_repo": str(result.get("kernel_repo") or ""),
            "backend": "forge_collective",
            "engine": str(result.get("engine") or "forge_collective"),
            "kernel_speedup": result.get("kernel_speedup"),
            "gpu_pct": result.get("gpu_pct"),
            "collective_op": str(result.get("collective_op") or ""),
            "world_size": result.get("world_size"),
            "workspace": str(result.get("workspace") or ""),
            "patch_path": str(result.get("patch") or result.get("patch_path") or ""),
            "iterations": result.get("iterations"),
            "salvaged": bool(result.get("salvaged")),
            "duration_sec": (result.get("duration_sec") or result.get("elapsed_sec") or result.get("runtime_sec")),
            "error_class": str(result.get("error_class") or ""),
            "error": str(result.get("error") or "")[-1200:],
            "ts": str(result.get("ts") or _now_iso()),
        }

    @staticmethod
    def _collective_integration_snapshot(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the integration fields stored on a collective campaign."""
        revert = result.get("revert_result")
        finalize = result.get("finalize_result")
        # Fall back to legacy field names for --resume compat with older sessions.
        patch_cleanup_status = str(result.get("patch_cleanup_status") or result.get("integration_status") or "")
        patch_cleanup_action = str(
            result.get("patch_cleanup_action") or result.get("integration_recovery_action") or ""
        )
        return {
            "patch_cleanup_status": patch_cleanup_status,
            "patch_cleanup_action": patch_cleanup_action,
            "integration_decision": str(result["decision"]).strip().upper(),
            "integration_result_status": str(result.get("status") or ""),
            "integration_gain_pct": result.get("gain_pct"),
            "integration_base_tput": result.get("base_tput"),
            "integration_new_tput": result.get("new_tput"),
            "integration_workspace": str(result.get("workspace") or ""),
            "integration_report_path": str(result.get("report_path") or ""),
            "integration_error_class": str(result.get("error_class") or ""),
            "integration_error": str(result.get("error") or "")[-1200:],
            "integration_revert_status": (str(revert.get("status") or "") if isinstance(revert, dict) else ""),
            "integration_finalize_status": (str(finalize.get("status") or "") if isinstance(finalize, dict) else ""),
            "integration_ts": _now_iso(),
        }

    def record_collective(self, result: dict[str, Any], session_dir: Path) -> None:
        """Upsert and persist one collective campaign."""
        if not isinstance(result, dict):
            raise TypeError("Collective result must be a mapping")
        incoming = dict(result)
        incoming_attempt_id = str(incoming.get("collective_attempt_id") or "").strip()
        previous = (
            dict(self.last_collective)
            if isinstance(self.last_collective, dict)
            and incoming_attempt_id
            and str(self.last_collective.get("collective_attempt_id") or "").strip() == incoming_attempt_id
            else {}
        )
        recorded = {**previous, **incoming}
        recorded.setdefault("ts", _now_iso())
        status = recorded.get("status")
        decision = recorded.get("decision")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("Collective result is missing status")
        if decision not in {"KEEP", "REVERT"}:
            raise ValueError("Collective result has invalid decision")
        if recorded.get("engine") != "forge_collective":
            raise ValueError("Collective result has invalid engine")
        kept = recorded.setdefault("kept", False)
        requires_e2e = recorded.setdefault(
            "requires_e2e_validation",
            False,
        )
        if not isinstance(kept, bool) or not isinstance(requires_e2e, bool):
            raise ValueError("Collective result E2E flags must be boolean")
        if kept != (decision == "KEEP") or requires_e2e != kept:
            raise ValueError("Collective result contract is inconsistent")
        attempt_id = str(recorded.get("collective_attempt_id") or "").strip()
        if not attempt_id:
            raise ValueError("Collective result is missing a stable attempt identity")
        recorded["collective_attempt_id"] = attempt_id
        if kept and not str(recorded.get("integration_id") or "").strip():
            raise ValueError("Collective KEEP is missing integration_id")
        for field_name in ("kernel_speedup", "gpu_pct"):
            value = recorded.get(field_name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (field_name == "kernel_speedup" and value <= 0)
                or (field_name == "gpu_pct" and value < 0)
            ):
                raise ValueError(f"Collective result has invalid {field_name}")
        recorded.setdefault(
            "patch_cleanup_status",
            "pending" if requires_e2e else "complete",
        )

        snapshot = self._collective_attempt_snapshot(recorded)
        if not isinstance(self.collective_attempts, list) or any(
            not isinstance(item, dict) for item in self.collective_attempts
        ):
            raise ValueError("collective_attempts must contain mappings")
        history = [dict(item) for item in self.collective_attempts]
        for index, item in enumerate(history):
            if str(item.get("collective_attempt_id") or "") == attempt_id:
                history[index] = snapshot
                break
        else:
            history.append(snapshot)
        previous_last = self.last_collective
        previous_history = self.collective_attempts
        self.last_collective = recorded
        self.collective_attempts = history[-_DEFAULT_ATTEMPTS_HISTORY:]
        try:
            self.save(session_dir)
        except Exception:
            self.last_collective = previous_last
            self.collective_attempts = previous_history
            raise

    def record_collective_integration(
        self,
        result: dict[str, Any],
        session_dir: Path,
        *,
        integration_id: str = "",
    ) -> None:
        """Attach and persist an integration verdict to its campaign."""
        if not isinstance(result, dict):
            raise TypeError("Collective integration result must be a mapping")
        integration = dict(result)
        integration_id = str(integration_id or integration.get("integration_id") or "").strip()
        if not integration_id:
            raise ValueError("Collective integration is missing integration_id")
        decision = str(integration.get("decision") or "").strip().upper()
        if decision not in {"KEEP", "REVERT", "NEEDS_REVIEW"}:
            raise ValueError(f"Invalid collective integration decision: {decision!r}")
        # Fall back to legacy field names for --resume compat with older sessions.
        patch_cleanup_status = str(
            integration.get("patch_cleanup_status") or integration.get("integration_status") or ""
        ).strip()
        if patch_cleanup_status not in {"complete", "recovery_required"}:
            raise ValueError("Collective patch_cleanup_status must be complete or recovery_required")
        recovery_action = str(
            integration.get("patch_cleanup_action") or integration.get("integration_recovery_action") or ""
        ).strip()
        if patch_cleanup_status == "complete" and recovery_action:
            raise ValueError("Completed collective integration cannot require recovery")
        if patch_cleanup_status == "recovery_required" and recovery_action not in {"finalize", "revert"}:
            raise ValueError("Collective recovery action must be finalize or revert")
        for field_name in ("gain_pct", "base_tput", "new_tput"):
            value = integration.get(field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Collective integration has invalid {field_name}")
        integration["decision"] = decision
        integration["patch_cleanup_status"] = patch_cleanup_status
        integration["patch_cleanup_action"] = recovery_action
        if not isinstance(self.last_collective, dict):
            raise ValueError("last_collective must be a mapping")
        last = dict(self.last_collective)
        if str(last.get("integration_id") or "") != integration_id:
            raise ValueError("Collective integration_id does not match last_collective")
        attempt_id = str(last.get("collective_attempt_id") or "").strip()
        if not attempt_id:
            raise ValueError("last_collective is missing collective_attempt_id")
        if not isinstance(self.collective_attempts, list) or any(
            not isinstance(item, dict) for item in self.collective_attempts
        ):
            raise ValueError("collective_attempts must contain mappings")
        history = [dict(item) for item in self.collective_attempts]
        matches = [
            index
            for index, item in enumerate(history)
            if str(item.get("collective_attempt_id") or "") == attempt_id
            and str(item.get("integration_id") or "") == integration_id
        ]
        if len(matches) != 1:
            raise ValueError("Collective integration must match exactly one campaign")
        integration_fields = self._collective_integration_snapshot(integration)
        last.update(integration_fields)
        history[matches[0]].update(integration_fields)

        previous_last = self.last_collective
        previous_history = self.collective_attempts
        self.last_collective = last
        self.collective_attempts = history
        try:
            self.save(session_dir)
        except Exception:
            self.last_collective = previous_last
            self.collective_attempts = previous_history
            raise

    def mark_bottleneck_switch(self, prev_bottleneck: str = "") -> None:
        """Flag that the next macro-cycle should redirect off ``prev_bottleneck`` (R3).

        Called when a cyclic config plateau winds the cycle down. Records the
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
        """Forwarding shim — implementation in :mod:`hyperloom.orchestrator.phases.machine_state`."""
        from ..phases import machine_state as _m

        return _m.record_lifecycle_event(
            self,
            step=step,
            status=status,
            phase=phase,
            label=label,
            artifacts=artifacts,
            detail=detail,
            duration_s=duration_s,
            ts=ts,
        )

    def merge_lifecycle_events(self, incoming: Any) -> None:
        """Union ``incoming`` lifecycle rows into this state, ordered by timestamp.

        Used when a nested run records lifecycle events on its own
        :class:`SharedState` instance: overwriting would drop whatever the live
        state recorded in the meantime, so the two logs are unioned instead.
        Rows are deduplicated on ``(step, status, ts)`` and ``seq`` is renumbered
        so it stays monotonic, which also makes repeated merges idempotent.

        Args:
            incoming (Any): Lifecycle rows to merge; ignored unless a non-empty
                list.
        """
        if not isinstance(incoming, list) or not incoming:
            return
        from copy import deepcopy

        existing = self.lifecycle if isinstance(self.lifecycle, list) else []

        def _key(row: dict[str, Any]) -> tuple[str, str, str]:
            return (str(row.get("step")), str(row.get("status")), str(row.get("ts")))

        merged = [row for row in existing if isinstance(row, dict)]
        seen = {_key(row) for row in merged}
        for row in incoming:
            if not isinstance(row, dict) or _key(row) in seen:
                continue
            seen.add(_key(row))
            merged.append(deepcopy(row))
        merged.sort(key=lambda row: str(row.get("ts") or ""))
        if len(merged) > _LIFECYCLE_CAP:
            merged = merged[-_LIFECYCLE_CAP:]
        for index, row in enumerate(merged):
            row["seq"] = index
        self.lifecycle = merged

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
            # Lazy import to avoid a shared_state <-> policy import cycle.
            from ..policy.gate import CORE_STATE_FIELDS

            core_fields = CORE_STATE_FIELDS
        applied: dict[str, Any] = {}
        # ``fields()`` excludes ClassVar pseudo-fields, so a class constant is
        # not writable here. CORE_STATE_FIELDS locks the session fields someone
        # thought to lock, and nobody thinks to lock a constant.
        writable = {f.name for f in fields(self)}
        for key, value in changes.items():
            if key not in writable:
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
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m._resolve_kernel_patch_identity(self, payload)

    def find_rejected_kernel_patch(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.find_rejected_kernel_patch(self, payload)

    @staticmethod
    def _is_integrate_fault(result: dict[str, Any]) -> bool:
        """True when an integrate result is an integration *fault*, not a verdict.

        A fault is an environment/apply/bench crash that prevented the patch from
        being fairly measured; it must not burn the REVERT quota. The
        discriminator is ``status`` (a genuine gate verdict stamps ``status:"ok"``
        while every unmeasured path returns ``status:"failed"``);
        :data:`_INTEGRATE_FAULT_ERROR_CLASSES` is a secondary signal.
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
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.record_kernel_integrate_result(
            self,
            result,
            max_attempts=max_attempts,
            keep_threshold_pct=keep_threshold_pct,
            max_fault_attempts=max_fault_attempts,
        )

    def record_kernel_opt(self, result: dict[str, Any]) -> None:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.record_kernel_opt(self, result)

    def record_gemm_tuning(self, result: dict[str, Any]) -> None:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        _m.record_gemm_tuning(self, result)

    # Multi-KEEP integrate queue helpers.
    def _kernel_ids_with_integrate_attempts(self) -> set[str]:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m._kernel_ids_with_integrate_attempts(self)

    def integrate_attempt_count_for_kernel(self, kernel_id: str) -> int:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.integrate_attempt_count_for_kernel(self, kernel_id)

    def integrate_attempt_count_for_integration(
        self,
        integration_id: str,
    ) -> int:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.integrate_attempt_count_for_integration(
            self,
            integration_id,
        )

    def next_pending_keep_kernel_id(self) -> str:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.next_pending_keep_kernel_id(self)

    def pending_keep_kernel_ids(self) -> list[str]:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.pending_keep_kernel_ids(self)

    def pending_kernel_integration_records(self) -> list[dict[str, Any]]:
        """Return immutable pending KEEP snapshots in integration priority order."""
        from ..kernel import _kernel_decisions as _m

        return _m.pending_kernel_integration_records(self)

    @property
    def kernel_opt_attempts(self) -> dict[str, Any]:
        """``kernel_opt_task_attempts`` re-indexed by the trace-local kernel id.

        Returns:
            A fresh dict; mutating it does not touch the ledger, but the entry
            values are the ledger's own dicts.
        """
        from ..kernel import _kernel_decisions as _m

        return _m.index_attempts_by_kernel_id(self.kernel_opt_task_attempts)

    @kernel_opt_attempts.setter
    def kernel_opt_attempts(self, value: dict[str, Any]) -> None:
        """Seed ``kernel_opt_task_attempts`` from an ordinal-keyed dict.

        Args:
            value: ``{kernel_id: attempt}``. Each attempt is stamped with its
                ``current_kernel_id`` / ``stable_task_key`` and filed under the
                stable key; an entry already filed under that key wins.
        """
        from ..kernel._kernel_decisions import _stable_kernel_task_key

        if not isinstance(self.kernel_opt_task_attempts, dict):
            object.__setattr__(self, "kernel_opt_task_attempts", {})
        for kernel_id, entry in value.items():
            if not isinstance(entry, dict):
                continue
            stamped = dict(entry)
            stamped.setdefault("current_kernel_id", str(kernel_id))
            task_key = stamped.get("stable_task_key") or _stable_kernel_task_key(
                task_group_key=str(stamped.get("task_group_key") or ""),
                kernel_id=str(kernel_id),
                source_file=str(stamped.get("last_source_file") or ""),
            )
            stamped.setdefault("stable_task_key", task_key)
            self.kernel_opt_task_attempts.setdefault(task_key, stamped)

    @property
    def roofline_snapshot_id(self) -> int:
        """Counter of the newest roofline snapshot, or 0 before the first one.

        Lives inside ``last_trace_analyze`` so clearing that cache resets the
        counter with it — ``record_trace_analyze`` then restarts from 1.
        """
        raw = (self.last_trace_analyze or {}).get("roofline_snapshot_id")
        return int(raw) if isinstance(raw, int) else 0

    @property
    def has_keep_pending_integrate(self) -> bool:
        """True when kernel KEEP results still await kernel ``integrate``.

        This is separate from ``pending_integrate``, the integrate_patch
        crash-recovery sentinel.
        """
        from ..kernel import _kernel_decisions as _m

        return _m.has_keep_pending_integrate(self)

    @property
    def kernel_opt_attempts_count(self) -> int:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.kernel_opt_attempts_count(self)

    # Reusable hot kernels still owing a kernel_opt attempt. Advisory only — no PolicyGate rule denies
    # ``report`` on this basis; feeds kernel_work_pending, the report annotations and the prompt guidance.
    def untried_hot_reusable_kernels(
        self,
        *,
        min_gpu_pct: float | None = None,
        top_n: int | None = None,
    ) -> list[str]:
        """Forwarding shim — implementation in :mod:`._kernel_decisions`."""
        from ..kernel import _kernel_decisions as _m

        return _m.untried_hot_reusable_kernels(self, min_gpu_pct=min_gpu_pct, top_n=top_n)

    # Per-action audit (kernel parity for non-kernel actions)
    @staticmethod
    def _truncate_excerpt(value: Any, *, limit: int = 1200) -> str | None:
        """Coerce ``value`` to str and trim to ``limit`` chars; None for falsy inputs (renderer shows ``err=(none)``).

        Args:
            value (Any): The value to coerce to a string excerpt.
            limit (int): Maximum retained length in characters.

        Returns:
            str | None: The trimmed string, or ``None`` when ``value`` is
                falsy.
        """
        if value is None:
            return None
        text = redact_secret_values(str(value))
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
        text = redact_secret_values(str(value))
        if not text:
            return None
        return text[-limit:] if len(text) > limit else text

    def _common_result_fields(self, result: dict[str, Any]) -> dict[str, Any]:
        """Build the failure/diagnostic fields shared by the attempt + failure logs.

        Single source of truth for the overlapping fields recorded by both
        :meth:`record_action_attempt` and :meth:`record_action_failure`, so the
        two writers can never drift.

        ``stderr_tail`` is captured for EVERY failure carrying an ``error`` blob
        (no ``error_class`` whitelist). The tail is the actionable end of a
        server/subprocess crash — e.g. a vLLM ``server_init_dead`` whose root
        cause (``ValueError: No common block size``) lives in the server.log
        excerpt the executor already folded into ``error``. Both the breakdown
        RCA exporter and the orchestration prompt consume it, so gating it by
        error_class silently dropped the one field that explains the failure.

        Args:
            result (dict[str, Any]): The action result envelope.

        Returns:
            dict[str, Any]: The shared diagnostic fields (``error_class`` /
                ``error_excerpt`` / ``stderr_tail`` / ``stderr_log_path`` /
                ``workspace`` / ``raw_result_path`` / ``reported_success`` /
                ``variant_name``).
        """
        return {
            "error_class": (str(result.get("error_class")) if result.get("error_class") else None),
            "error_excerpt": self._truncate_excerpt(result.get("error")),
            "stderr_tail": self._stderr_tail(result.get("error")),
            "stderr_log_path": (str(result.get("stderr_log_path")) if result.get("stderr_log_path") else None),
            "workspace": (str(result.get("workspace")) if result.get("workspace") else None),
            "raw_result_path": (str(result.get("raw_result_path")) if result.get("raw_result_path") else None),
            "reported_success": result.get("reported_success"),
            "variant_name": (str(result.get("variant_name")) if result.get("variant_name") else None),
            "failure_id": result.get("failure_id"),
        }

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
        """Append one attempt to ``<action>_attempts`` and refresh ``last_<action>``. Entry schema {ts, task_id, status, decision, key_metric, key_metric_kind, workspace, error_class, error_excerpt, stderr_tail, stderr_log_path, raw_result_path, reported_success, variant_name, extras}. Returns the entry, or None when ``action`` not in the audit set (kernel_agent-owned actions use bespoke recorders). Does NOT call :meth:`save`.

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
            **self._common_result_fields(result),
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
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            capture_result = dict(result)
            capture_result.setdefault(
                "workload",
                {
                    "framework": str(getattr(self, "framework", "") or ""),
                    "model_name": str(getattr(self, "model_name", "") or ""),
                    "gpu_type": str(getattr(self, "gpu_type", "") or ""),
                    "precision": str(getattr(self, "precision", "") or ""),
                    "tp": int(getattr(self, "tp", 0) or 0),
                    "ep": int(getattr(self, "ep", 0) or 0),
                    "conc": int(getattr(self, "conc", 0) or 0),
                    "isl": int(getattr(self, "isl", 0) or 0),
                    "osl": int(getattr(self, "osl", 0) or 0),
                },
            )
            instrument.record_phase_event(
                getattr(self, "_session_dir", None),
                action=action,
                entry=entry,
                result=capture_result,
                phase=str(getattr(self, "phase", "") or ""),
                macro_cycle=int(getattr(self, "macro_cycle", 0) or 0),
                tick=int(getattr(self, "tick", 0) or 0),
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
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "action": str(action or ""),
            "task_id": str(task_id or ""),
            **self._common_result_fields(result),
        }
        history = list(self.last_action_failures or [])
        history.append(entry)
        if len(history) > max_history:
            history = history[-max_history:]
        self.last_action_failures = history
        return entry

    def record_failure_evidence(self, fe: "dict[str, Any]") -> None:
        """Append one structured failure packet to :attr:`failures` (last-wins on ``failure_id``).

        Also mirrors the packet to ``<session_dir>/reports/failures/`` so it
        survives state.json compaction.

        Args:
            fe: A failure evidence dict as produced by
                :func:`~hyperloom.orchestrator.state.failure_evidence.failure_from_variant_outcome`.
        """
        fid = str(fe.get("failure_id") or "")
        history = [e for e in (self.failures or []) if e.get("failure_id") != fid]
        history.append(fe)
        self.failures = history[-_DEFAULT_LAST_FAILURES:]

        session_dir = getattr(self, "_session_dir", None)
        if session_dir:
            from hyperloom.common.io import atomic_write_json
            from hyperloom.inference_optimizer.session.session_paths import failure_evidence_path

            try:
                atomic_write_json(failure_evidence_path(session_dir, fid), fe, make_parents=True)
            except OSError:
                log.debug("failure evidence mirror failed for %s", fid, exc_info=True)

    def find_failure(self, failure_id: str) -> "dict[str, Any] | None":
        """Return the :attr:`failures` entry for ``failure_id``, else ``None``.

        Args:
            failure_id: The stable failure id to look up.

        Returns:
            The matching entry, or ``None`` when not found.
        """
        fid = str(failure_id or "").strip()
        for entry in reversed(self.failures or []):
            if entry.get("failure_id") == fid:
                return entry
        return None

    def failures_for_task(self, task_id: str) -> "list[dict[str, Any]]":
        """Return the ``task_id`` entries from :attr:`failures`, newest first.

        Args:
            task_id: The task id to filter on.

        Returns:
            A list of matching failure evidence dicts.
        """
        tid = str(task_id or "").strip()
        return [e for e in reversed(self.failures or []) if e.get("task_id") == tid]

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
        return first_positive_tput(self.last_baseline)

    def _resolve_current_best_achieved_tput(self) -> float:
        """Optimized-arm throughput for a current_best roofline snapshot.

        Reads ``current_best``'s ``tput``/``output_throughput`` so a
        current_best-tagged snapshot keeps its arm even when ``tput`` is
        momentarily absent (avoids silently downgrading to the baseline arm).

        Returns:
            float: The resolved current_best throughput, or ``0.0`` when none
                is available.
        """
        return first_positive_tput(self.current_best)

    def _locate_diffusion_roofline_sidecar(self, kernel_roofline_path: Any) -> Path | None:
        """Locate the ``diffusion_roofline.json`` sidecar for the latest trace run.

        The sidecar sits at the TraceLens run-dir root
        (``<session>/kernel-agent/runs/<ts>/<ts>_tl-*/diffusion_roofline.json``).
        Diffusion/xDiT trace_analyze emits ONLY this sidecar (no
        ``kernel_roofline.json``), so ``kernel_roofline_path`` is empty and the
        run-dir cannot be derived from it. Resolve, in order:

          1. ``kernel_roofline_path`` run-dir root (serving/kernel path, when set).
          2. Newest sidecar under ``<session>/kernel-agent/runs`` (diffusion path).

        Args:
            kernel_roofline_path: Path to ``reports/kernel_roofline.json`` when
                present; empty for diffusion sessions.

        Returns:
            The resolved sidecar path, or ``None`` when none is found.
        """
        krp = str(kernel_roofline_path or "").strip()
        if krp:
            cand = Path(krp).parent.parent / "diffusion_roofline.json"
            if cand.is_file():
                return cand
        session_dir = getattr(self, "_session_dir", None)
        if session_dir:
            try:
                sidecars = [
                    p for p in Path(session_dir).glob("kernel-agent/runs/**/diffusion_roofline.json") if p.is_file()
                ]
                if sidecars:
                    return max(sidecars, key=lambda p: p.stat().st_mtime)
            except OSError:
                return None
        return None

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
            from hyperloom.inference_optimizer import framework_registry

            if not framework_registry.is_scriptable(framework):
                return 0.0, 0.0
            # img/s -> per-image e2e latency (ms); the achieved metric.
            e2e_mean_ms = float(framework_registry.primary_metric_value(framework, achieved_tput) or 0.0)
            # Ideal per-image latency floor from the diffusion roofline sidecar.
            roofline_ideal_ms = 0.0
            sidecar = self._locate_diffusion_roofline_sidecar(kernel_roofline_path)
            if sidecar is not None:
                try:
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                    data = data if isinstance(data, dict) else {}
                    # Priority: full-pipeline analytic ceiling (ms) > DiT-only
                    # analytic ceiling (us) > trace-summed per-kernel ideal (us).
                    approach_a = data.get("analytic_ceiling")
                    analytic = data.get("analytic_dit_ceiling")
                    totals = data.get("totals")
                    if isinstance(approach_a, dict) and float(approach_a.get("ideal_ms") or 0.0) > 0:
                        roofline_ideal_ms = float(approach_a["ideal_ms"])
                    elif isinstance(analytic, dict) and float(analytic.get("ideal_compute_us") or 0.0) > 0:
                        roofline_ideal_ms = float(analytic["ideal_compute_us"]) / 1000.0
                    elif isinstance(totals, dict) and float(totals.get("sigma_ideal_roofline_us") or 0.0) > 0:
                        roofline_ideal_ms = float(totals["sigma_ideal_roofline_us"]) / 1000.0
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    roofline_ideal_ms = 0.0
            return e2e_mean_ms, roofline_ideal_ms
        except Exception:  # noqa: BLE001 — best-effort enrichment, never blocks
            return 0.0, 0.0

    def _roofline_throughput_unit(self) -> str:
        """Return the throughput unit for roofline snapshots of this workload.

        Delegates to the framework registry (xDiT = img/s,
        text-gen = tok/s, …). The numeric ``*_tok_per_sec`` fields keep their
        names for wire stability; this unit tells consumers how to render them.
        """
        from hyperloom.inference_optimizer import framework_registry

        framework = str(getattr(self, "framework", "") or "").strip().lower()
        return framework_registry.throughput_unit(framework)

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
            from ..kernel.roofline_ceiling import (
                RooflineBreakdown,
                compute_roofline_breakdown_from_state,
            )
            from ..kernel.roofline_snapshot import (
                attach_perfmodel_breakdown,
                build_roofline_snapshot,
            )
        except Exception:  # noqa: BLE001 — import guard, best-effort
            return {}

        achieved = self._resolve_baseline_achieved_tput()
        breakdown = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")
        try:
            breakdown = compute_roofline_breakdown_from_state(
                self,
                arm="baseline",
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
            throughput_unit=self._roofline_throughput_unit(),
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
        """Write ``last_trace_analyze`` (single writer); ``roofline_snapshot_id`` increments from the previous nested value, restarting from 1 when the cache was cleared.

        Args:
            payload (dict[str, Any]): The trace_analyze task payload (supplies
                ``trace_input`` / ``trace_dir`` and optional ``roofline_arm``).
            result (dict[str, Any]): The trace_analyze result envelope; a
                non-dict result is a no-op.
        """
        if not isinstance(result, dict):
            return
        trace_input = (payload or {}).get("trace_input") or (payload or {}).get("trace_dir") or ""
        artifacts = result.get("artifact_paths") or {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        candidates_path = result.get("candidates_path") or ""
        if not candidates_path:
            candidates_path = artifacts.get("kernel_candidates", "") or ""
        kernel_roofline_path = result.get("kernel_roofline_path") or ""
        if not kernel_roofline_path:
            kernel_roofline_path = artifacts.get("kernel_roofline", "") or ""
        disk = self._read_session_roofline_report(payload, kernel_roofline_path)
        if disk:
            kernel_roofline_path = kernel_roofline_path or str(disk["path"])
            candidates_path = candidates_path or str(disk["payload"].get("kernel_candidates_path") or "")
            if not result.get("trace_report_path"):
                result = dict(result)
                result["trace_report_path"] = str(disk["payload"].get("analysis_md_path") or "")
            if not (result.get("hot_kernels") or result.get("hot_kernels_top15")):
                result = dict(result)
                result["hot_kernels"] = disk["kernels"]
                log.warning(
                    "record_trace_analyze: envelope carried no hot kernels; "
                    "recovered %d from %s. The analysis succeeded — the result "
                    "envelope lost its payload in transit.",
                    len(disk["kernels"]),
                    disk["path"],
                )
        steady_state_trace = result.get("steady_state_trace") or artifacts.get("tracelens_steady_state_trace") or ""
        summary, kernel_roofline, reusable_ids, withheld_collective = self._build_hot_kernel_summaries(
            result, kernel_roofline_path
        )

        # Project skipped (non-routable) candidates so the LLM sees unoptimizable operators.
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
        if withheld_collective:
            log.warning(
                "kernel targets: withholding %d collective kernel(s) from kernel_opt for the collective lane: %s",
                len(withheld_collective),
                ", ".join(f"{item['kernel_id']}({item['name'][:60]})" for item in withheld_collective),
            )
            warnings_cleaned.append(
                {
                    "code": "collective_lane_withheld_kernels",
                    "detail": (
                        "reserved for the collective lane and removed from the "
                        "kernel_opt target list; unreachable unless that lane runs"
                    ),
                    "kernels": withheld_collective,
                }
            )

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
                # Stored verbatim; the prompt path strips base64 data-URLs.
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
            "steady_state_trace": str(steady_state_trace),
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

        self._append_roofline_snapshot_history(
            payload=payload,
            snapshot_id=snapshot_id,
            ts_iso=ts_iso,
            analysis_md_path=analysis_md_path,
            trace_input=trace_input,
            kernel_roofline_path=kernel_roofline_path,
        )

    def _read_session_roofline_report(
        self,
        payload: dict[str, Any],
        kernel_roofline_path: str,
    ) -> "dict[str, Any] | None":
        """Read the session-level kernel-roofline report written by TraceLens.

        The analyzer writes this report as a side effect of a successful run, so
        it survives a result envelope that lost its payload keys in transit.

        Args:
            payload (dict[str, Any]): The trace_analyze task payload; supplies
                ``roofline_output_name`` when the envelope named no path.
            kernel_roofline_path (str): Report path recovered from the envelope,
                or ``""`` to resolve the session default.

        Returns:
            ``{"path", "payload", "kernels"}`` with kernels ordered by
            descending GPU share, or ``None`` when no report is readable.
        """
        path = Path(kernel_roofline_path) if kernel_roofline_path else None
        if path is None:
            session_dir = getattr(self, "_session_dir", None)
            if not session_dir:
                return None
            name = str((payload or {}).get("roofline_output_name") or "").strip()
            path = Path(session_dir) / "reports" / (name or _DEFAULT_ROOFLINE_REPORT_NAME)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(report, dict):
            return None
        rows = [row for row in (report.get("kernels") or []) if isinstance(row, dict)]
        rows.sort(key=lambda row: float(row.get("gpu_pct") or 0.0), reverse=True)
        return {"path": path, "payload": report, "kernels": rows}

    def _build_hot_kernel_summaries(
        self,
        result: dict[str, Any],
        kernel_roofline_path: str,
    ) -> "tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]":
        """Build ``(summary, kernel_roofline, reusable_ids, withheld)`` from the
        top-N hot kernels, merging the optional per-kernel rocprof roofline sidecar.

        ``reusable_ids`` drives the kernel_opt target list offered to the LLM,
        so collective candidates are withheld: they are owned by the dedicated
        collective lane and ``_batch_kernel_candidates`` drops them, which would
        otherwise dispatch an empty batch for every id picked from this list.
        ``withheld`` names them, because a kernel that no lane will touch must
        not simply vanish from the target list.
        """
        from ..kernel import _kernel_decisions as _m

        hot = result.get("hot_kernels") or []
        summary: list[dict[str, Any]] = []
        kernel_roofline: list[dict[str, Any]] = []
        reusable_ids: list[str] = []
        withheld: list[dict[str, Any]] = []
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
                # TraceLens kernel_category bucket ("" when absent).
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
                "kernel_contract": entry.get("kernel_contract"),
                "is_multigpu": entry.get("is_multigpu") is True,
                # Carries the collective lane's ownership test downstream; without
                # it every reader would re-derive ownership from the name alone.
                "candidate_source": entry.get("candidate_source") or "",
                "recommended_backends": entry.get("recommended_backends") or [],
                "recommended_actions": entry.get("recommended_actions") or [],
                # Vendor-playbook fields (mori's dispatch+combine is the first
                # case -- see _vendor_operator_playbooks.py). Without these,
                # effective_hot_kernel_gpu_pct()/effective_hot_kernel_min_gpu_pct()
                # silently degrade to bare gpu_pct/min_gpu_pct for every caller
                # that gates off this projection (untried_hot_reusable_kernels()
                # is the only one -- _batch_kernel_candidates() reads full
                # candidate dicts off candidates_path instead), losing both the
                # aggregate gate's intended pass (PR #1191 review finding #3)
                # and the playbook's own min_gpu_pct_floor enforcement.
                "patch_strategy": entry.get("patch_strategy") or "",
                "vendor_playbook_group_id": entry.get("vendor_playbook_group_id") or "",
                "vendor_playbook_aggregate_gpu_pct": entry.get("vendor_playbook_aggregate_gpu_pct"),
                "vendor_playbook_min_gpu_pct_floor": entry.get("vendor_playbook_min_gpu_pct_floor"),
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
                if _m.is_collective_candidate(summary_entry):
                    withheld.append(
                        {
                            "kernel_id": str(kid),
                            "name": str(entry.get("name") or ""),
                            "gpu_pct": entry.get("gpu_pct"),
                        }
                    )
                else:
                    reusable_ids.append(str(kid))
        return summary, kernel_roofline, reusable_ids, withheld

    def _append_roofline_snapshot_history(
        self,
        *,
        payload: dict[str, Any],
        snapshot_id: int,
        ts_iso: str,
        analysis_md_path: str,
        trace_input: str,
        kernel_roofline_path: str,
    ) -> None:
        """Append a compact roofline snapshot for report-side comparison;
        best-effort, failures never block the canonical write above."""
        # Append compact history for report-side Roofline Comparison; best-effort.
        try:
            from ..kernel.roofline_snapshot import (
                attach_perfmodel_breakdown,
                build_roofline_snapshot,
            )

            # Stamp decode-roofline ceiling + measured tput.
            from ..kernel.roofline_ceiling import (
                RooflineBreakdown,
                compute_roofline_breakdown_from_state,
            )

            # Resolve which arm this snapshot measures first so the ceiling is
            # anchored to the same arm as achieved. An explicit ``roofline_arm``
            # on the payload wins; absent it, infer from current_best.tput.
            forced_arm = str((payload or {}).get("roofline_arm") or "").strip()
            # Unknown arm values fall through to current_best inference; warn.
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
                # Explicit tag wins: keep the optimized arm even if tput is absent.
                snapshot_arm = "current_best"
                achieved_tput = self._resolve_current_best_achieved_tput()
            elif isinstance(cb_tput, (int, float)) and cb_tput > 0:
                snapshot_arm = "current_best"
                achieved_tput = float(cb_tput)
            else:
                snapshot_arm = "baseline"
                achieved_tput = self._resolve_baseline_achieved_tput()
            # Primary decode ceiling plus memory/compute sides (PerfModel bottom-up).
            breakdown = RooflineBreakdown(0.0, 0.0, 0.0, "unknown")
            try:
                breakdown = compute_roofline_breakdown_from_state(
                    self,
                    arm=snapshot_arm,
                )
            except Exception:  # noqa: BLE001 — ceiling is best-effort
                pass
            peak_tput = float(breakdown.peak_tok_per_sec or 0.0)
            # Scriptable/diffusion has no tok/s decode ceiling; surface the
            # compute-latency roofline (measured per-image e2e latency vs the
            # ideal floor from the sidecar). Best-effort → 0 leaves serving unchanged.
            fw = str(getattr(self, "framework", "") or "")
            e2e_mean_ms, roofline_ideal_ms = self._scriptable_latency_roofline(fw, achieved_tput, kernel_roofline_path)
            history_entry = build_roofline_snapshot(
                snapshot_id=snapshot_id,
                ts=ts_iso,
                analysis_md_path=str(analysis_md_path),
                theoretical_peak_tok_per_sec=peak_tput,
                achieved_tok_per_sec=achieved_tput,
                mem_ceiling_tok_per_sec=float(breakdown.mem_tok_per_sec or 0.0),
                cmp_ceiling_tok_per_sec=float(breakdown.cmp_tok_per_sec or 0.0),
                bound_kind=breakdown.bound_kind,
                throughput_unit=self._roofline_throughput_unit(),
                framework=fw,
                e2e_mean_ms=e2e_mean_ms,
                roofline_ideal_ms=roofline_ideal_ms,
            )
            # Per-op PerfModel breakdown for dashboard visualization.
            attach_perfmodel_breakdown(history_entry, self, arm=snapshot_arm)
            history_entry["trace_input"] = str(trace_input)
            history_entry["macro_cycle"] = int(getattr(self, "macro_cycle", 0) or 0)
            history_entry["analysis_md_path"] = str(analysis_md_path)
            # Sidecar artifact pointer for per-kernel roofline data.
            history_entry["kernel_roofline_path"] = str(kernel_roofline_path)
            if not isinstance(self.roofline_snapshots, list):
                self.roofline_snapshots = []
            self.roofline_snapshots.append(history_entry)
            try:
                from ..kernel.roofline_snapshot import direction_saturation

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
                        bool(prev_sat) and str(prev_sat.get("bound_kind") or "") != str(sat.get("bound_kind") or "")
                    ),
                    "current": sat,
                    "previous": prev_sat,
                }
            except Exception:  # noqa: BLE001 — saturation telemetry is advisory only
                pass
            # R3: retire the pending switch once the top bottleneck has drifted.
            self.maybe_clear_bottleneck_switch_on_drift(
                str(history_entry.get("top_bottleneck") or ""),
            )
            if len(self.roofline_snapshots) > _ROOFLINE_SNAPSHOTS_CAP:
                # Always keep snapshot #1 as the report's baseline anchor.
                base = self.roofline_snapshots[0]
                tail = self.roofline_snapshots[-(_ROOFLINE_SNAPSHOTS_CAP - 1) :]
                self.roofline_snapshots = [base, *tail]
        except Exception:  # noqa: BLE001 — never block record on render concerns
            pass

    def record_conc_sweep(self, result: dict[str, Any]) -> None:
        """Record the concurrency sweep's completion into ``last_conc_sweep``.

        The status field is what lets ``exit_normal_sweep`` return ``sweep_done``:
        the ladder is the only sweep there is, so its terminal state is the
        phase's.

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
        status = str(self.last_conc_sweep.get("status") or "").lower()
        if status in ("succeeded", "partial", "completed") and not self.last_conc_sweep.get("was_skipped"):
            self.last_conc_sweep_watermark = {
                **self.last_conc_sweep,
                "cumulative_gain_validated_at_record": float(getattr(self, "cumulative_gain_validated", 0.0) or 0.0),
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
        from hyperloom.common.gain_math import gain_pct

        entry_gain_pct = gain_pct(tput, base)
        self.gain_per_stack_entry.append(entry_gain_pct)
        return entry_gain_pct

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

    def stamp_deadline_unix(
        self,
        *,
        now_unix: float | None = None,
        budget_minutes: float | None = None,
    ) -> float:
        """Persist the absolute session deadline if a bounded session has none.

        A resume must not reissue a full ``max_minutes`` from the moment
        ``Coordinator.run`` is entered. The first stamp — ``start_ts`` plus the
        budget — is what every remaining-time check reads.

        ``budget_minutes`` is the run() argument before it is stored as
        ``int(max_minutes)``. Tests pass fractional minutes; truncating first
        would make this a no-op and leave the loop with no persisted deadline.

        Args:
            now_unix: Clock used only when ``start_ts`` cannot be parsed;
                defaults to ``time.time()``.
            budget_minutes: Minutes to add to ``start_ts``; ``None`` uses
                :attr:`max_minutes`.

        Returns:
            The unix deadline, or ``0.0`` when the session is unbounded.
        """
        minutes = float(self.max_minutes or 0) if budget_minutes is None else float(budget_minutes)
        existing = float(self.deadline_unix or 0.0)
        if minutes <= 0:
            # A truncated stored budget must not erase a stamp this process or
            # an earlier one already wrote.
            if existing > 0.0:
                return existing
            self.deadline_unix = 0.0
            return 0.0
        if existing > 0.0:
            return existing
        start = to_unix(self.start_ts, None)
        origin = float(start) if start else float(now_unix if now_unix is not None else time.time())
        self.deadline_unix = origin + minutes * 60.0
        return self.deadline_unix

    def remaining_minutes(self, *, now: datetime | None = None) -> float | None:
        """Minutes left in the wall-clock budget; ``None`` when unbounded, else clamped at 0.

        When :attr:`deadline_unix` is stamped, remaining time is derived from
        it so the Coordinator loop, admission, and the grid cannot disagree —
        including when the persisted ``max_minutes`` was truncated to 0.
        Otherwise this falls back to ``max_minutes - elapsed`` so tests that
        inject elapsed without a stamp keep working.

        Args:
            now (datetime | None): Reference time; defaults to the current UTC
                time.

        Returns:
            float | None: Minutes remaining in the budget (clamped at 0.0), or
                ``None`` when the session is unbounded.
        """
        deadline = float(self.deadline_unix or 0.0)
        if deadline > 0.0:
            now_dt = now or datetime.now(timezone.utc)
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=timezone.utc)
            return max(0.0, (deadline - now_dt.timestamp()) / 60.0)
        if not self.max_minutes:
            return None
        return max(0.0, float(self.max_minutes) - self.elapsed_minutes(now=now))

    def monotonic_session_deadline_sec(self) -> float | None:
        """``time.monotonic()`` instant the session budget is spent, or ``None`` if unbounded.

        Converts the persisted unix deadline into the clock the Coordinator
        loop already consults, so a resume cannot pick a fresh full budget.

        Returns:
            A monotonic deadline, or ``None`` when ``max_minutes`` is unset.
        """
        remaining = self.remaining_minutes()
        if remaining is None:
            return None
        return time.monotonic() + remaining * 60.0

    def record_teardown_timing(self, step: str, elapsed_sec: float) -> None:
        """Record one post-deadline teardown step's duration.

        Args:
            step: Name of the teardown step (``coordinator_stop``,
                ``final_json``, ...).
            elapsed_sec: Wall-clock seconds the step took.
        """
        name = str(step or "").strip()
        if not name:
            return
        timings = self.teardown_timings_sec
        if not isinstance(timings, dict):
            self.teardown_timings_sec = {}
            timings = self.teardown_timings_sec
        timings[name] = round(max(0.0, float(elapsed_sec)), 3)
        timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 3)

    def closing_reserve_sec(self) -> float:
        """Seconds held back from every unit of work so CLOSE can still report.

        Resolved through :func:`effective_closing_grace_sec`, the same function
        the Coordinator uses to size the closing phase itself, so the budget
        reserved for that phase and the budget it actually gets are one number.
        A hardcoded reserve told the truth only for sessions of at least 100
        minutes, and charged 120 seconds even to an operator who had disabled
        the closing phase outright.

        Returns:
            float: The closing reserve in seconds; ``0.0`` when the operator
                disabled the closing phase.
        """
        return max(0.0, effective_closing_grace_sec(float(self.max_minutes or 0), self.closing_grace_sec))

    def session_budget_usable_sec(
        self,
        *,
        reserve_sec: float | None = None,
    ) -> float | None:
        """Seconds of wall-clock budget left once the closing reserve is held back.

        The single source for "how much time may a unit of work still claim".
        Admission control (which action may start) and the grid deadline (how
        long a variant may run) both read it, so they cannot disagree about how
        much budget exists.

        Args:
            reserve_sec (float | None): Seconds held back for the CLOSE phase
                and its report; ``None`` takes :meth:`closing_reserve_sec`. An
                explicit ``0`` is honoured as "reserve nothing".

        Returns:
            float | None: Usable seconds (clamped at 0.0), or ``None`` when
                ``max_minutes`` is unset (unbounded budget).
        """
        remaining = self.remaining_minutes()
        if remaining is None:
            return None
        reserve = self.closing_reserve_sec() if reserve_sec is None else float(reserve_sec)
        return max(0.0, remaining * 60.0 - reserve)

    def grid_session_deadline_sec(
        self,
        *,
        reserve_sec: float | None = None,
    ) -> float | None:
        """``time.monotonic()`` deadline for grid variant loops, or ``None`` when the budget is unbounded.

        Reserves ``reserve_sec`` so the CLOSE phase and report still have room
        after the last variant. Grid runners pass this as ``session_deadline_sec``
        so a wall-clock timeout skips the remaining variants instead of draining
        the whole grid.

        Args:
            reserve_sec (float | None): Seconds held back from the raw remaining
                budget; ``None`` takes :meth:`closing_reserve_sec`.

        Returns:
            float | None: A monotonic-clock deadline, or ``None`` when unbounded;
                ``time.monotonic()`` (i.e. already due) once the reserve is gone.
        """
        usable = self.session_budget_usable_sec(reserve_sec=reserve_sec)
        if usable is None:
            return None
        return time.monotonic() + usable

    def optimization_stack_has_unvalidated_keeps(self) -> bool:
        """True iff a new KEEP landed since the last validated measurement (purely a stack-length check vs ``cumulative_gain_validated_stack_len``).

        Returns:
            bool: ``True`` when ``optimization_stack`` is longer than
                ``cumulative_gain_validated_stack_len``.
        """
        return len(self.optimization_stack) > int(self.cumulative_gain_validated_stack_len)


__all__ = ["SharedState", "render_model_arch_compact", "timed_teardown_step"]
