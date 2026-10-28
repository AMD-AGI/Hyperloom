# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SharedState — single-writer (Coordinator) persisted session state, backed by atomic JSON at ``$SESSION_DIR/state.json``."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from hyperloom.common.deadline import Deadline
from hyperloom.common.coerce import first_text, to_float, to_int, to_str_list
from hyperloom.common.env import env_bool
from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.gain_math import gain_pct, gain_pct_or_zero
from hyperloom.common.io import atomic_write_json
from hyperloom.common.perf_metric import (
    AGENTX_KEEP_P50_THRESHOLD_PCT,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    GradedComparison,
    GradingAnchor,
    VERDICT_KEEP,
    VERDICT_REVERT,
    axis_of,
    constraint_veto_reason,
    holds_within_band,
    incomparability_reason,
    output_tput_of,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    total_tput_of,
)
from hyperloom.common.timeutil import now_iso
from hyperloom.plugs import progress as _progress
from hyperloom.common.jsonio import read_json_strict
from hyperloom.common.profile_args import sanitize_profile_server_args
from hyperloom.inference_optimizer.grading import resolved_grading

from .kernel_decision_settings import _DEFAULT_ATTEMPTS_HISTORY
from .failure_evidence import tail_excerpt
from ._shared_state.attempt_audit import _AUDIT_ACTIONS, _KEY_METRIC_MAP
from ._shared_state.enablement_round import EnablementRound
from ._shared_state.phase_state import _control_fields

log = logging.getLogger(__name__)

# Upper bound on retained crash timestamps (trailing-window rate needs only the recent tail).
_CRASH_TIMESTAMP_CAP: int = 200

# escalate_strategy_change hint vocabulary (closed enum; unknown hints are denied). ``skip_to_kernel`` is the
# "exhausted the current lever" signal from FRAMEWORK_AGENT; it advances to KERNEL.
# KERNEL exits automatically when settled with no pending work — the LLM does not self-set a sweep hint.
ESCALATE_HINT_SKIP_TO_KERNEL: str = "skip_to_kernel"
ESCALATE_HINT_SKIP_TO_SWEEP: str = "skip_to_sweep"
ESCALATE_HINT_VOCAB: frozenset[str] = frozenset(
    {
        ESCALATE_HINT_SKIP_TO_KERNEL,
        ESCALATE_HINT_SKIP_TO_SWEEP,
    }
)


def is_valid_escalate_hint(hint: str) -> bool:
    """Return True for any hint Coordinator should act on (closed vocab)."""
    return (hint or "").strip() in ESCALATE_HINT_VOCAB


def first_positive_tput(d: Any) -> float:
    """Return the first positive ``tput``/``output_throughput`` from a dict."""
    src = d if isinstance(d, dict) else {}
    for key in ("tput", "output_throughput"):
        val = src.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return 0.0


def _merged_profile_envs(current_best: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """The current best's env overrides, with the caller's base and variant envs layered on top."""
    raw_current_envs = current_best.get("extra_envs")
    merged_envs = dict(raw_current_envs) if isinstance(raw_current_envs, dict) else {}
    if isinstance(incoming.get("base_extra_envs"), dict):
        merged_envs.update(incoming["base_extra_envs"])
    if isinstance(incoming.get("extra_envs"), dict):
        merged_envs.update(incoming["extra_envs"])
    return merged_envs


def resolve_anchor_with_drift(snapshot_tput: float, state: Any) -> tuple[float, bool]:
    """Grade against the live anchor when a KEEP landed after this task snapshotted its params."""
    live = resolve_grading_anchor_tput(state)
    if live > snapshot_tput:
        return live, snapshot_tput > 0
    return snapshot_tput, False


def resolve_grading_anchor_tput(state: Any) -> float:
    """Output throughput a new candidate is composed on top of."""
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
    keep_threshold_pct: float = 0.0,
    anchor: GradingAnchor = GradingAnchor(),
) -> "GradedComparison":
    """Resolve what a KEEP decision grades: candidate and reference, on one axis, plus the verdict on that pair."""
    # AgentX KEEPs on median interactivity clearing its threshold with the slow tail and output throughput both
    # inside the noise band; anything short of all three is REVERT. A pair that cannot supply the axes degrades
    # together and ``degrade_reason`` says why -- promotion lanes read ``comparable`` and fail closed rather than
    # KEEPing on the diagnostic output figure. ``keep_threshold_pct`` applies to the output axis only.
    # ``anchor`` defaults to the session anchor; explore passes its own because variants stack
    # within a round. Objective and band come from ``resolved_grading``, so both are what the session was seeded
    # with rather than whatever the calling process's environment holds.
    on_intvty, noise_pct = resolved_grading(state)
    # The session's constraints (latency, power) veto a candidate the gain gates would KEEP, on whichever axis graded
    # it. A candidate that already lost carries no veto, so the ledger names the gate that actually refused it.
    degrade_reason = ""
    if on_intvty:
        if anchor.perf is not None:
            ref_perf, reason = anchor.perf, ""
        elif against_baseline:
            ref_perf = perf_snapshot_from_mapping(getattr(state, "baseline_perf", None))
            reason = "" if ref_perf else "baseline_axes_missing"
        else:
            ref_perf, reason = resolve_grading_anchor_perf(state)
        cand_perf = perf_snapshot_from_mapping(measurement)
        if all((ref_perf, cand_perf)):
            gain = gain_pct_or_zero(axis_of(cand_perf, GRADED_INTVTY_P50), axis_of(ref_perf, GRADED_INTVTY_P50))
            # The output guard reads the raw aggregate: the chip count divides both sides of the ratio, so the band
            # holds identically per GPU.
            gates = (
                ("p50_gain_below_threshold", gain >= AGENTX_KEEP_P50_THRESHOLD_PCT),
                ("p90_regressed", holds_within_band(cand_perf, ref_perf, GRADED_INTVTY, noise_pct=noise_pct)),
                ("output_regressed", holds_within_band(cand_perf, ref_perf, GRADED_OUTPUT, noise_pct=noise_pct)),
                (incomparability_reason(cand_perf, ref_perf), False),
            )
            refused = [gate for gate, held in gates if gate and not held]
            sla_veto = "" if refused else constraint_veto_reason(measurement, state)
            return GradedComparison(
                objective=GRADED_INTVTY_P50,
                candidate=axis_of(cand_perf, GRADED_INTVTY_P50),
                reference=axis_of(ref_perf, GRADED_INTVTY_P50),
                verdict=VERDICT_REVERT if any((refused, sla_veto)) else VERDICT_KEEP,
                tput_candidate=total_tput_of(cand_perf),
                tput_reference=total_tput_of(ref_perf),
                veto_reason=sla_veto,
                refused_by=",".join(refused),
            )
        degrade_reason = reason or "candidate_axes_missing"

    if anchor.tput is not None:
        reference = float(anchor.tput)
    elif against_baseline:
        reference = float(getattr(state, "baseline_tput", 0.0))
    else:
        reference = resolve_grading_anchor_tput(state)
    candidate = output_tput_of(measurement)
    # The output-axis figures are diagnostic only. A degraded AgentX pair
    # must not read as a throughput KEEP at the resolver chokepoint.
    gain = gain_pct(candidate, reference)
    cleared = not degrade_reason and gain is not None and gain >= keep_threshold_pct
    sla_veto = constraint_veto_reason(measurement, state) if cleared else ""
    return GradedComparison(
        objective=GRADED_OUTPUT,
        candidate=candidate,
        reference=reference,
        verdict=VERDICT_KEEP if cleared and not sla_veto else VERDICT_REVERT,
        degrade_reason=degrade_reason,
        veto_reason=sla_veto,
    )


def _normalize_envs(value: Any) -> dict[str, str]:
    """Coerce an env mapping to ``str -> str``; a resumed non-dict reads as empty."""
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


# Workload fields a profile task may override; the rest of the profile identity is read off the session.
PROFILE_OVERRIDABLE_KEYS: tuple[str, ...] = (
    "framework",
    "precision",
    "model_path",
    "tp",
    "conc",
    "isl",
    "osl",
    "max_model_len",
)

# ``base_*`` task param -> the ``current_best`` field it mirrors and its normalizer.
_STACK_BASE_FIELDS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("base_extra_args", "extra_server_args", lambda v: str(v or "").strip()),
    ("base_extra_envs", "extra_envs", _normalize_envs),
    ("base_remove_args", "remove_args", to_str_list),
    ("base_unset_envs", "unset_envs", to_str_list),
    ("base_args_mode", "args_mode", lambda v: "replace" if str(v or "").strip().lower() == "replace" else ""),
)


def stack_base_params(current_best: dict[str, Any] | None) -> dict[str, Any]:
    """``base_*`` params projected from the fields ``current_best`` carries."""
    cb = current_best or {}
    return {key: normalize(cb[source]) for key, source, normalize in _STACK_BASE_FIELDS if source in cb}


def inject_stack_base_params(
    params: dict[str, Any],
    state: Any,
    *,
    anchor: bool = False,
    overwrite: bool = False,
) -> None:
    """Seed a task's base config from ``current_best``, in place."""

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


# Integration faults (environment / apply / bench crashes) are distinct from a genuine gate REVERT; a fault means the
# patch was never fairly measured, so it gets its own small retry budget instead of burning the REVERT quota.
_INTEGRATE_FAULT_ERROR_CLASSES = frozenset(
    {
        "missing_integration_inputs",
        "patch_not_applied",
        "apply_failed",
        # The served context cannot host an eval request, so the accuracy gate can never return a verdict under this
        # configuration.
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
# How many hot / skipped kernels ``record_trace_analyze`` keeps in the trace summary (matches the ``*_top15`` field
# names).
_TRACE_HOT_KERNEL_TOP_N = 15

# A projected hot-kernel row carries roofline evidence when any of these is populated.
_ROOFLINE_SIGNAL_KEYS: tuple[str, ...] = (
    "bound_type",
    "arithmetic_intensity",
    "flops_per_byte",
    "efficiency_percent",
    "compute_utilization_pct",
    "bandwidth_utilization_pct",
    "suggestion",
    "roofline_name",
)


def _saturation_delta(sat: dict[str, Any], prev_sat: dict[str, Any], key: str) -> float | None:
    current, previous = sat.get(key), prev_sat.get(key)
    if current is None or previous is None:
        return None
    return round(float(current) - float(previous), 2)


def _skipped_kernels_top(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The hottest skipped (non-routable) candidates, so the LLM sees the operators it cannot optimize."""
    skipped = result.get("skipped_kernels")
    skipped_rows = sorted(
        (e for e in skipped if isinstance(e, dict)) if isinstance(skipped, list) else (),
        key=lambda e: float(e.get("gpu_pct") or 0.0),
        reverse=True,
    )
    return [
        {
            "kernel_id": e.get("kernel_id"),
            "name": e.get("name"),
            "skip_reason": e.get("skip_reason") or "",
            "gpu_pct": e.get("gpu_pct"),
        }
        for e in skipped_rows[:_TRACE_HOT_KERNEL_TOP_N]
    ]


def _hot_kernel_row(entry: dict[str, Any]) -> dict[str, Any]:
    kid = entry.get("kernel_id")
    arithmetic_intensity = entry.get("arithmetic_intensity")
    if arithmetic_intensity is None:
        arithmetic_intensity = entry.get("flops_per_byte")
    efficiency_percent = entry.get("efficiency_percent")
    if efficiency_percent is None:
        efficiency_percent = entry.get("efficiency_pct")
    return {
        "kernel_id": kid,
        "name": entry.get("name"),
        # TraceLens kernel_category bucket ("" when absent).
        "kernel_category": entry.get("kernel_category") or "",
        "gpu_pct": entry.get("gpu_pct"),
        "call_count": entry.get("call_count"),
        "duration_us": entry.get("duration_us"),
        "tracelens_pitem_rank": entry.get("tracelens_pitem_rank"),
        "impact_score": entry.get("impact_score"),
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
        "reusable_native_kernel": bool(entry.get("reusable_native_kernel")),
        "kernel_contract": entry.get("kernel_contract"),
        "is_multigpu": entry.get("is_multigpu") is True,
        # Names the deterministic extractor the row came from.
        "candidate_source": entry.get("candidate_source") or "",
        "recommended_backends": entry.get("recommended_backends") or [],
        "recommended_actions": entry.get("recommended_actions") or [],
        # Vendor-playbook fields; see _vendor_operator_playbooks.py.
        "patch_strategy": entry.get("patch_strategy") or "",
        "vendor_playbook_group_id": entry.get("vendor_playbook_group_id") or "",
        "vendor_playbook_aggregate_gpu_pct": entry.get("vendor_playbook_aggregate_gpu_pct"),
        "vendor_playbook_min_gpu_pct_floor": entry.get("vendor_playbook_min_gpu_pct_floor"),
    }


# Session-level kernel-roofline report the analyzer writes for a non-close run; read back when the trace_analyze
# envelope arrives without its payload keys.
_DEFAULT_ROOFLINE_REPORT_NAME = "kernel_roofline_current.json"

# Global ``last_action_failures`` rolling-log cap.
_DEFAULT_LAST_FAILURES = 30

# Lifecycle-event log cap (fires at every step boundary, so generous but bounded).
_LIFECYCLE_CAP = 500

# Experience KB injection log cap (orchestration and specialist rows share it).
_KB_INJECTIONS_CAP = 20


@dataclass(frozen=True)
class KbInjection:
    """One Experience KB block injected into a prompt, as ``record_experience_kb_injection`` stores it."""

    consumer: str
    read_id: str
    experience_ids: list[str]
    experiences: list[dict[str, Any]]
    prompt_block: str
    domain: str = ""
    gap_canonical_id: str = ""


@dataclass(frozen=True)
class ActionAttempt:
    """One settled attempt of an audited action, as ``record_action_attempt`` ledgers it."""

    task_id: str
    status: str
    decision: str
    result: dict[str, Any] | None
    extras: dict[str, Any] | None = None


# roofline_snapshots history cap (record_trace_analyze).
_ROOFLINE_SNAPSHOTS_CAP = 50


#: top-level state.json schema version, stamped on every save.
LATEST_STATE_SCHEMA_VERSION: int = 6


def effective_closing_grace_sec(
    max_minutes: float | None,
    closing_grace_sec: float | None,
) -> float:
    """Resolve the closing-phase grace window after the wall-clock deadline."""
    if closing_grace_sec is not None:
        return float(closing_grace_sec)
    return min(120.0, (max_minutes or 0.0) * 60.0 * 0.02)


from ._shared_state.elapsed import elapsed_sec_of
from ._shared_state.render import _RenderMixin


from ._shared_state.phase_state import _PhaseStateMixin
from .gaps import GapsStateMixin


@dataclass
class SharedState(_RenderMixin, GapsStateMixin, _PhaseStateMixin):
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
    # Advisory: True when knowingly running a multimodal checkpoint on the text-only path (--allow-mm-text-fallback).
    degraded_mode: bool = False
    # Structured degraded-mode / model-compat warnings (e.g. multimodal text fallback).
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
    # Profile-phase output length (from --profile-osl). 0 = unset (profile defaults to min(osl, 1024)).
    profile_osl: int = 0
    max_model_len: int = 0
    kernel_enabled: bool = True
    # KERNEL-phase backend, recorded once when the session is created: "geak" (default) or "forge".
    kernel_optimizer: str = "geak"
    # Snapshot of the last GEAK e2e run (result.json + final_launch.sh / bench_e2e.sh handles the SWEEP phase reuses).
    geak_result: dict[str, Any] = field(default_factory=dict)
    # Terminal result of the phase-level KernelForge rewrite controller.
    kernel_rewrite_controller_result: dict[str, Any] = field(default_factory=dict)
    # SWEEP-phase concurrency sweep; ``_seed_shared_state`` resolves it at seed time from ``--enable-conc-sweep`` and the
    # benchmark mode.
    conc_sweep_enabled: bool = True
    # Which benchmark workload this session measures: "agentx" (agentic trace replay) or "synthetic" (ISL/OSL).
    benchmark_mode: str = ""
    # Generation counter for AgentX measurements.
    agentx_epoch: int = 0
    # Which AgentX client measured this session: "aiperf" or "mlperf". Empty on
    # sessions that predate the field; resume treats those as aiperf. Compared
    # on resume so the two workloads cannot anchor each other without bumping
    # the measurement epoch (which would invalidate unchanged aiperf sessions).
    agentx_backend: str = ""
    # The grading configuration this session was seeded with: {"objective": GRADED_INTVTY|GRADED_OUTPUT,
    # "noise_pct": float}. Recorded rather than re-derived because the derivation reads HYPERLOOM_PERF_METRIC /
    # HYPERLOOM_PERF_NOISE_PCT, and a resume is a new process: a shell that lost the variable would flip the axis
    # mid-run, and a lost noise band would silently widen a 3.5% guard back to the 5% default. The KEEP/REVERT rule
    # has to be the one the session started with. Empty on sessions predating the field, which fall back to deriving.
    grading: dict[str, Any] = field(default_factory=dict)
    # Stamped once when the run objective is first met.
    target_reached_at: str = ""
    # CONC ladder for conc_sweep, seeded from the workload's own ladder by ``_parse_conc_sweep_concs``.
    conc_sweep_concs: list[int] = field(default_factory=list)
    # Total wall-clock budget (s) for conc_sweep. 0 disables the gate.
    conc_sweep_total_budget_sec: int = 9000
    baseline_tput: float = 0.0
    # Ceiling on mean end-to-end latency (ms) from ``--max-latency-ms``; 0.0 leaves KEEP behaviour unchanged. The
    # only copy of the budget: it is written once at launch and archived with the session, so a resume restores it.
    latency_budget_ms: float = 0.0
    power_budget_w: float = 0.0  # --max-power-w: total W over the serving cards; 0.0 is off. Archived, like latency.
    power_budget_per_gpu_w: dict[str, float] = field(default_factory=dict)  # --max-per-gpu-power-w {GPU id: W}
    # GPU power record: {"declared", "observed": {gpu: {power_cap_w, perf_level, sclk_max_mhz}}, "node" (NPM),
    # "applied", and with --allow-shared-npm-node "shared_npm_node" (the cards left out)}. Asserted on resume.
    gpu_power_settings: dict[str, Any] = field(default_factory=dict)
    # AgentX corpus shape: written at seed from canonical constants. Read by semantic consumers (prompts, manifest,
    # reports) instead of the inert state.isl / state.osl placeholders. Absent on synthetic sessions.
    agentx_corpus_shape: dict[str, Any] = field(default_factory=dict)
    # Baseline AgentX perf snapshot: the slow-tail e2e_norm_intvty_p90 objective plus total_throughput and the
    # reported axes the summary renders.
    baseline_perf: dict[str, Any] = field(default_factory=dict)
    baseline_accuracy: float = 0.0
    baseline_failure_streak: int = 0
    baseline_arg_error_streak: int = 0
    # Signature of the last failure counted in ``baseline_failure_streak``; two in a row that match end the run.
    baseline_last_failure_signature: str = ""
    # Combined backstop counting ANY baseline failure regardless of error_class, catching mixed classes that never
    # trip a per-class streak (anti time-exhaustion).
    baseline_total_failures: int = 0
    # One-shot: a cuda-graph capture failure asks the next baseline to retry with cuda-graph capture disabled.
    baseline_eager_fallback: bool = False
    # Admission for both enablement lanes, from ``--enablement``: ``launch``, ``eval``, ``all`` (default) or ``off`` —
    # with ``off`` a broken baseline fast-fails instead of opening an authoring loop.
    enablement_mode: str = "all"
    # All per-round enablement fields are nested here.
    enablement: EnablementRound = field(default_factory=EnablementRound)
    # Off-loop targeted-build sentinel (task_id/pid/pgid/attempt_root/ aiter_jit_dir/deadline/action); own sentinel,
    # resume-cleared.
    pending_targeted_build: dict = field(default_factory=dict)
    # Baseline-materialized YAML path; injected downstream as ``config_path`` so variants inherit the contract.
    baseline_config_path: str = ""
    baseline_benchmark_script: str | None = None
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
    # Crash-safe rollback/bookkeeping state for the combined Recipe + Kernel PRELUDE validation.
    warm_replay_pending: dict = field(default_factory=dict)
    # Session-authoritative InferenceX checkout.
    active_inferencex_path: str = ""
    # Durable post-save handoffs for section KB staging.
    kb_stage_outbox: list = field(default_factory=list)
    # Owner sections dropped because their persisted artifacts disappeared.
    kb_stage_dead_letter: list = field(default_factory=list)
    # Idempotent terminal Recipe publication state.
    recipe_finalize_status: str = ""
    recipe_finalize_attempts: int = 0
    recipe_finalize_outcome: dict = field(default_factory=dict)
    # One-shot guard for PRELUDE warm-kernel KB read/apply (resume can't re-fire).
    warm_kernel_kb_attempted: bool = False
    # Resolved prior-champion kernel columns (gemm/fusion/rewrite) loaded at PRELUDE from the Recipe ``value.kernel``;
    # read back by the combined promote.
    warm_kernel_kb_plan: list = field(default_factory=list)
    # Baseline COLD (warmup-round) full boot+bench wall-clock; the hard-cap anchor from which ExploreExecutor derives
    # the overtime-kill deadline.
    baseline_runtime_sec: float = 0.0
    # Baseline WARM measure-round wall-clock (client-only, no boot); anchors the explore overtime kill
    # apples-to-apples.
    baseline_warm_runtime_sec: float = 0.0
    # Whether the baseline's hot pass was dropped because the budget could not cover it plus one variant to read
    # against it, leaving the cold pass's depressed figure as the anchor.
    baseline_measure_round_dropped: bool = False
    # The benchmark's own share of the COLD round above, from the server-ready marker onward.
    baseline_post_ready_runtime_sec: float = 0.0
    current_best: dict[str, Any] = field(default_factory=dict)
    # Immutable measurement evidence captured with the current-best promotion.
    current_best_measurement: dict[str, Any] = field(default_factory=dict)
    # Reference launch recipe from the operator's --reference-script: lowest-priority base server args/envs seeding
    # every baseline.
    reference_server_args: str = ""
    reference_envs: dict[str, str] = field(default_factory=dict)
    reference_launch_controls: dict[str, Any] = field(default_factory=dict)
    reference_model: str = ""
    # Operator launch shape, persisted so a bare --resume serves the same contract.
    operator_server_args: str = ""
    # Operator-defined specialist (``--custom-specialist-*``); empty prompt means not configured.
    custom_specialist_prompt: str = ""
    custom_specialist_description: str = ""
    # ``HYPERLOOM_CUSTOM_SPECIALIST_ONLY``: FRAMEWORK_AGENT runs the custom specialist and its bench alone.
    custom_specialist_only: bool = False
    # ``--extra-env NAME=VALUE`` pins.
    operator_extra_env: dict[str, str] = field(default_factory=dict)
    # Operator-supplied custom-workload paths; a promoted enablement runtime replaces framework_repo_path with its checkout.
    bypass_scripts_dir: str = ""
    framework_repo_path: str = ""
    # xDiT model registry name (``--xdit-model-name``).
    xdit_model_name: str = ""
    # ``HYPERLOOM_BENCHMARK_BACKEND`` at seed time (``bypass`` for custom).
    benchmark_backend: str = ""
    # The card's compute-partition shape this session was measured in, as observed at launch: mode, partition count,
    # CU and memory per partition, streams per partition.
    compute_partition: dict[str, Any] = field(default_factory=dict)
    # ``--nodes``, feeding the IR-8 check.
    nodes: int = 1
    # Per-agent Unix timestamp of the most recent completed reactor pass.
    agent_last_active: dict[str, float] = field(default_factory=dict)
    # Warm-recipe replay gates (``--no-warm-replay`` / ``--warm-replay-min-confidence``).
    warm_replay_enabled: bool = True
    warm_replay_min_confidence: float = 0.7
    # Full accepted configuration stack across action families; current_best keeps the materialized full args/env.
    optimization_stack: list[dict[str, Any]] = field(default_factory=list)
    # Index-aligned with ``optimization_stack``: per-entry incremental gain pct; missing => None.
    gain_per_stack_entry: list[float | None] = field(default_factory=list)
    # Total gain over ``baseline_tput``, stamped only from a measurement taken with the whole stack applied;
    # standalone validate_stack denied by PolicyGate.
    cumulative_gain_validated: float = 0.0
    cumulative_gain_validated_ts: str = ""
    # ``optimization_stack`` length at the last validated measurement; longer => new KEEPs need validation.
    cumulative_gain_validated_stack_len: int = 0
    # Bumped by every lift into ``current_best``; stamped onto ``validated_recipe_generation`` by every validation.
    # Unequal => ``current_best`` is not the Recipe the validated gain was measured on, even at the same stack depth.
    working_recipe_generation: int = 0
    validated_recipe_generation: int = 0
    # Resume sentinels.
    pending_integrate: dict[str, Any] = field(default_factory=dict)
    # A GEAK e2e candidate with a self-reported win not yet confirmed by a main-flow rebench; kept OUT of current_best
    # / optimization_stack / the headline gain until validated.
    geak_pending: dict[str, Any] = field(default_factory=dict)
    # Tput watermark for gain-driven roofline refresh; Coordinator re-enqueues at a compound 10% step.
    last_roofline_tput: float = 0.0
    stop_reason: str = ""
    # When the session first stopped, and therefore its end time for consumers.
    stop_ts: str = ""
    # When the current run leg ended without ending the session.
    leg_ended_ts: str = ""
    # When the current run leg began, i.e. the most recent ``--resume``; empty for a session that has only ever run
    # once.
    resumed_ts: str = ""
    # Closing phase — set when wall-clock deadline fires; Coordinator only drains a ``report`` task. Cleared on resume.
    closing_phase: bool = False
    closing_started_unix: float = 0.0
    closing_report_task_id: str = ""
    # True at END of CLOSE 7-step sequencer; cli.finally short-circuits emergency breakdown write. Resume clears it (idempotent).
    close_sequence_done: bool = False
    # Auto-roofline gate (optimisation-phase entry): pending roofline task_id; blocks first-round specialist dispatch until snapshot lands.
    auto_roofline_pending_task_id: str = ""
    crash_count: int = 0
    # Unix timestamps of recent crashes (bounded), used for the trailing-window emergency-stop rate so old crashes age
    # out instead of accumulating forever.
    crash_timestamps: list[float] = field(default_factory=list)
    # Last Coordinator-side exception caught by the tick-loop guard (gives postmortems a traceback).
    last_tick_exception: dict[str, Any] = field(default_factory=dict)
    pruned_families: list[str] = field(default_factory=list)
    start_ts: str = field(default_factory=now_iso)
    max_minutes: int = 0
    # Absolute unix deadline for a bounded session. Stamped once from
    # ``start_ts + max_minutes`` so a resume cannot reissue a full budget.
    # ``0.0`` means unset or unbounded.
    elapsed_charged_sec: float = 0.0
    leg_anchor_unix: float = 0.0
    # Operator's ``--closing-grace-sec``; ``None`` derives it from max_minutes.
    closing_grace_sec: float | None = None
    last_profile_trace: str = ""
    # Launch evidence of the run that produced ``last_profile_trace``; its runtime findings share the trace's shapes.
    last_profile_launch_evidence_path: str = ""
    # ``succeeded``/``recovered``/``failed`` for most recent profile; failed allows re-run with a last_profile_trace.
    last_profile_status: str = ""
    # Workload context captured with ``last_profile_trace``; strict matching prevents consumers from reusing runtime
    # shapes after workload changes.
    last_profile_workload: dict[str, Any] = field(default_factory=dict)
    # ``current_best.action`` in effect when ``last_profile_workload`` was recorded, so the backfill in
    # ``current_profile_workload_context`` can tell a same-arm reuse from a stale one.
    last_profile_workload_action: str = ""
    # Rolling log of PolicyGate denials (newest last, cap 50).
    policy_denial_history: list[dict[str, Any]] = field(default_factory=list)
    # Per-(action_name, rule) consecutive denial counter.
    policy_denial_streak: dict[str, int] = field(default_factory=dict)
    # Server EXTRA_SGLANG_ARGS in effect when last_profile_trace was captured; identical args means the same trace.
    last_profile_args: str = ""
    # Merged host-side rewrite evidence document from the most recent profile (see ``_framework_rewrite_evidence``).
    last_framework_rewrite_evidence: str = ""
    # Why the field above is empty, when it is. "No candidates" and "the probe never ran" both render as no evidence,
    # and only one of them means the workload has nothing left to rewrite; without this the framework phase cannot
    # tell a genuine negative from a broken instrument.
    last_framework_rewrite_evidence_status: str = ""
    # Environment switches behind accepted framework-level source rewrites, registered as search levers.
    authored_framework_levers: list[dict[str, Any]] = field(default_factory=list)

    # Roofline-v2 trace-analyze cache written by record_trace_analyze. roofline_snapshot_id is a property derived from
    # this dict.
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)
    # Append-only compact roofline snapshots for report.py; capped at ``_ROOFLINE_SNAPSHOTS_CAP`` (snapshot #1 always retained as the report's baseline anchor).
    roofline_snapshots: list[dict[str, Any]] = field(default_factory=list)
    # Outer roofline failure counter; bumped on fail, reset on success.
    roofline_failure_streak: int = 0
    # Why this stack can never record GPU kernels; set once from a parsed trace with host ops and no kernels, so the
    # Coordinator stops auto-enqueueing analysis that cannot succeed. Empty means GPU tracing is still presumed usable.
    gpu_trace_unsupported_reason: str = ""
    # Analysis failures that recur on the same input, keyed by that input; ``analysis_failures`` owns the schema.
    analysis_failures: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Feature toggles (mirrored from ``cli.py`` flags at session start).
    framework_agent_phase_enabled: bool = True
    # FRAMEWORK progress: one entry per candidate benchmark; used by breakdown + plateau exit judgment.
    framework_agent_phase_progress: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # One row per measured attempt on any lever; the per-lever dryness judgment
    # reads it. One writer per lever: writeback for config, the dispatcher for
    # patches, both through ``attempt_ledger``.
    attempts: list[dict[str, Any]] = field(default_factory=list)
    # One row per discovery batch; read by the source arm's plateau gate (3 batches <1% => exit).
    framework_agent_batches: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # True when the source arm has no more candidates; read by its plateau gate.
    framework_agent_phase_done: bool = False
    # Consecutive discovery failures; the arm declines only after DISCOVER_FAILURE_RETRY_LIMIT (default 3).
    framework_agent_discover_failures: int = 0
    # Consecutive empty-but-valid discovery batches; tolerate up to DISCOVER_FAILURE_RETRY_LIMIT before exiting.
    framework_agent_empty_discoveries: int = 0
    # Consecutive FRAMEWORK_AGENT phase completions that discovered zero candidates (empty_discovery).
    framework_consecutive_empty_discoveries: int = 0
    # Default True: when PR discovery is empty/exhausted (or the ranker prefers it), the FRAMEWORK pump dispatches a
    # candidate-free authoring specialist that authors a throughput patch from the live source + profile evidence
    # instead of skipping the phase.
    framework_local_explore_enabled: bool = True
    # Maps an authoring specialist task_id -> originating FRAMEWORK candidate id (PR URL), so the authored-outcome
    # bridge can key the progress row on the PR-URL that ``_select_next_framework_agent_candidate`` checks.
    framework_agent_specialist_candidate_map: dict[str, str] = field(
        default_factory=dict,
    )
    # Re-author rounds per candidate id (capped); needs_review verdicts increment.
    specialist_reauthor_attempts: dict[str, int] = field(
        default_factory=dict,
    )
    # Backstop: per-candidate-key count of Critic-review submissions; past the abort threshold the pump force-stamps
    # ``repeated_review_abort`` and stops re-selecting it, bounding one candidate's share of the phase budget.
    framework_agent_review_counts: dict[str, int] = field(
        default_factory=dict,
    )
    # Apply-failure re-author attempts per candidate id, capped by ``_AUTHORED_LANE_MAX_ATTEMPTS``.
    apply_fail_reauthor_attempts: dict[str, int] = field(
        default_factory=dict,
    )
    # Retry contexts queued by the authored lane and drained by ``_drain_apply_fail_retry_pending``.
    apply_fail_retry_pending: list[dict[str, Any]] = field(
        default_factory=list,
    )
    # Default True: Coordinator auto-analysis is ``roofline`` (profile+trace_analyze+analysis.md); False enqueues plain ``profile``.
    enable_roofline: bool = True
    # The concurrency ladder's terminal state; SWEEP→CLOSE exits on it.
    last_conc_sweep: dict[str, Any] = field(default_factory=dict)
    # Durable watermark from the last real conc_sweep measurement; survives the macro-cycle reloop clearing
    # ``last_conc_sweep`` so redundant closeout is skipped when no validated gain landed since the prior conc_sweep.
    last_conc_sweep_watermark: dict[str, Any] = field(default_factory=dict)
    # Wall-clock instant (unix seconds) until which the ``conc_sweep`` SWEEP enqueued is entitled to run: its granted
    # budget plus the lease grace. SWEEP's own budget exits defer to it while no result has landed; 0 means no grant.
    conc_sweep_granted_until_unix: float = 0.0
    # Most recent forge-fusion run result and its e2e integrate result; persisted so resume does not rerun a completed
    # fusion loop or lose the adoption audit.
    last_fusion: dict[str, Any] = field(default_factory=dict)
    last_fusion_integrate: dict[str, Any] = field(default_factory=dict)
    # How many times forge-fusion aborted on infrastructure this session, so KERNEL entry can stop re-arming a cause
    # that does not heal.
    fusion_infra_aborts: int = 0
    # How many times a fusion round left targets its lane ceiling never funded, counted here for the same reason as
    # the aborts above.
    fusion_withheld_retries: int = 0
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
    # Immutable KEEP snapshots awaiting E2E integration, keyed by integration_id.
    pending_kernel_integrations: dict[str, Any] = field(default_factory=dict)
    # Consecutive grid-runner tasks with no new current_best; Robustness nudges Orch off the plateau. Reset on advance.
    params_no_promote_streak: int = 0
    # Cumulative count of explore/integrate_patch rounds that produced at least one valid throughput measurement.
    gain_gated_action_count: int = 0
    # Unified persistent explore-search ledger; ``tested`` keyed by canonical_fingerprint, ``accepted`` holds the round's KEEPs, everything graded down moves to rejected.
    explore_search: dict[str, Any] = field(default_factory=dict)
    # specialist sub-agent rolling state; one entry per config-arm round (round_id, tasks, proposals_total/kept/rejected/skipped, etc.).
    specialist_rounds: list[dict[str, Any]] = field(default_factory=list)
    # Per-kb_anchor coverage counters: config-arm rounds since a specialist was dispatched / since a KEEP landed.
    rounds_since_last_specialist: dict[str, int] = field(default_factory=dict)
    rounds_since_last_keep: dict[str, int] = field(default_factory=dict)
    # Patch verdict ledger keyed by review subject (a specialist task_id, or a candidate id for a PR pre-screen); Critic must approve/advise before PolicyGate allows the integrate_patch delegate.
    specialist_patch_verdicts: dict[str, str] = field(default_factory=dict)
    # Intervention-mix ledger ({change_type∈{config,code_patch,config_attempt,code_patch_attempt}}); Robustness detects config-only loops.
    intervention_mix: list[dict[str, Any]] = field(default_factory=list)
    # Research scout bookkeeping; master switch ``--no-research-scout``; seen_pr_ids shared with FRAMEWORK to avoid re-mining.
    research_scout_enabled: bool = True
    research_scout_interval: int = 3
    # Master switch for advisory "External target gap" prompt block (``--no-target-advisory``); never gates Objective.
    target_advisory_enabled: bool = True
    # Master switch for sedimenting KEEP/REVERT provenance into the persistent recipe; off => recipe stays ephemeral.
    recipe_sediment_enabled: bool = True
    research_scout_seen_pr_ids: list[str] = field(default_factory=list)
    # Round id of last scout dispatch so K-round re-dispatch fires once per qualifying round.
    research_scout_last_round: int = -1
    # Static-recon specialist bookkeeping (explore-opt-5 capability A); master switch ``--no-static-recon``.
    static_recon_enabled: bool = True
    # Research-lane capacity locked at session start (core field; PolicyGate denies mid-session mutation).
    research_lane_capacity: int = 1
    # GPU pool capacity for needs_gpu specialists (0 disables); locked at session start.
    gpu_specialist_capacity: int = 0
    # escalate_strategy_change carry-over: Coordinator writes validated next_action_hint here for compute_next_phase, then clears it either by consuming it (drove a transition) or discarding it (an unrelated transition fired while it was pending).
    pending_escalate_hint: str = ""
    # last hint thrown away by an unrelated transition, never acted on (audit only) for the breakdown.
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
    # Consecutive KERNEL_AGENT ticks that observed no forward motion AND no kernel-lane task in flight.
    kernel_idle_ticks: int = 0
    # Digest of the KERNEL progress signals (integrate verdicts, queued integrations, rejected ids, stack depth,
    # rewrite-controller result, in-flight kernel task ids) seen on the tick that opened the current idle streak.
    kernel_progress_fingerprint: str = ""
    # Unix time the current idle streak opened.
    kernel_idle_since_unix: float = 0.0
    # Unix time an inline kernel request (``integrate``, ``run_optimization``, ...) last reported itself running.
    kernel_inline_step_seen_unix: float = 0.0

    # Monotonic Coordinator tick counter; stable anchor for plateau/phase budget math.
    tick: int = 0

    # Phase state machine fields ``phase`` — run-level pipeline phase
    # (PRELUDE/FRAMEWORK_AGENT/KERNEL_AGENT/SWEEP/CLOSE); Coordinator-only writer.
    phase: str = ""
    # Unix epoch the current phase was entered (budget judge + the prompt's ``entered`` line).
    phase_started_unix: float = 0.0
    # Append-only log of phase transitions (rows from machine_state.make_history_row). Capped at _PHASE_HISTORY_CAP.
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    # Durable per-phase sum of COMPLETED segments, keyed by phase name.
    phase_elapsed_totals: dict[str, float] = field(default_factory=dict)
    # Append-only operator-facing lifecycle log.
    lifecycle: list[dict[str, Any]] = field(default_factory=list)
    # Experience KB blocks injected into FRAMEWORK_AGENT orchestration and specialist prompts: {tick, phase, ts,
    # consumer, domain, gap_canonical_id, read_id, experience_ids, experiences, prompt_block}. Coordinator-only writer.
    experience_kb_injections: list[dict[str, Any]] = field(default_factory=list)
    # Wall-clock budget share per phase: seeded once at phase init from CLI flags/defaults with disabled phases' shares
    # redistributed, kept on resume unless a --*-pct flag is given.
    phase_budget_pct: dict[str, float] = field(default_factory=dict)
    # Cyclic phase machine macro-cycle counter (cycle 0 is the first pass; each SWEEP→FRAMEWORK_AGENT loopback
    # increments it).
    macro_cycle: int = 0
    # Per-cycle budget: wall-clock minutes for ONE macro-cycle.
    cycle_minutes: float = 0.0
    # Global-convergence tracking: validated cumulative gain at the current macro-cycle's start, and the consecutive
    # no-gain cycle streak.
    gain_at_cycle_start: float = 0.0
    no_gain_cycle_streak: int = 0
    # Decision points the PrimaTune predictor has been asked about (``predictor/pump.py``); each is asked once.
    predictor_asked_keys: list[str] = field(default_factory=list)
    # Cyclic bottleneck re-direction: set when a cyclic config plateau winds the cycle down; the next macro-cycle's
    # prompt surfaces a redirect advisory off ``last_cycle_bottleneck``.
    pending_bottleneck_switch: bool = False
    last_cycle_bottleneck: str = ""
    # Latest roofline saturation per specialist-domain family and prev->current cycle bottleneck movement.
    saturated_directions: dict[str, dict[str, Any]] = field(default_factory=dict)
    bottleneck_shift: dict[str, Any] = field(default_factory=dict)
    # Per-cycle advisory focus log; persisted so cycle strategy survives resume.
    cycle_strategy_log: list[dict[str, Any]] = field(default_factory=list)

    # Recipe KB integration fields — Coordinator-only writers.
    recipe_kb_session_id: str = ""
    # Snapshot of ``recipe_kb_t0._cascade_warm_start_search`` output, the ``{workload, hw, tier, confidence, recipe}`` envelope where ``recipe`` is the matched row; empty on first session for a (workload, hw) pair. Bookkeeping, not a prompt input — the model-facing view is ``warm_start_context``.
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    # Snapshot of the recipe row's ``pitfalls`` (negative priors), flat ``{description, severity, ...}`` rows as written by ``Recipe.to_dict``; consumed by the specialist prompt. Resume tolerates older snapshots.
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 snapshot of the recipe row's ``lessons`` (positive priors), flat ``{statement, measured_impact, ...}`` rows, symmetric with warm_start_pitfalls; consumed by the specialist prompt. Empty under --degraded-kb or T0 failure.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    # ISO UTC timestamp of the T0 snapshot; empty under --degraded-kb or T0 failure.
    warm_start_ts: str = ""
    # Model-facing advisory context built by ``recipe_kb_t0``.
    warm_start_context: dict[str, Any] = field(default_factory=dict)

    # structured gaps ledger: dedup'd unresolved bottlenecks (Coordinator-only refresh_gaps); dedup keyed by canonical_id, attempts capped 20/gap, list capped _GAPS_MAX_ENTRIES.
    gaps: list[dict[str, Any]] = field(default_factory=list)

    # The SWEEP handoff turn's result: ``next_cycle_directive``, its ``for_cycle``, and ``parse_error``. Coordinator-only writer.
    orchestration_memory: dict[str, Any] = field(default_factory=dict)

    # Non-field instance attr (set in load_or_init / save): session dir for breakdown instrumentation.
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
    #: ClassVar so it never serialises into state.json.
    PROFILE_WORKLOAD_IDENTITY_KEYS: ClassVar[tuple[str, ...]] = ("benchmark_mode", *PROFILE_OVERRIDABLE_KEYS)

    @classmethod
    def profile_workload_identity(cls, context: Any) -> dict[str, Any]:
        """Project a workload context down to what identifies the profiled run."""
        if not isinstance(context, Mapping):
            return {}
        return {key: context.get(key) for key in cls.PROFILE_WORKLOAD_IDENTITY_KEYS}

    def profile_workload_context(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the normalized workload and runtime identity for a profile trace."""
        from hyperloom.inference_optimizer.grid_server_args import normalize_server_args

        params = overrides if isinstance(overrides, dict) else {}
        workload = {
            name: params[name] if params.get(name) not in (None, "") else getattr(self, name, "")
            for name in PROFILE_OVERRIDABLE_KEYS
        }
        text = {name: str(workload[name] or "").strip() for name in ("framework", "precision", "model_path")}
        model_path = text["model_path"]
        if model_path:
            path = Path(model_path).expanduser()
            if path.exists():
                try:
                    model_path = str(path.resolve())
                except OSError:
                    log.debug(
                        "profile workload path resolution failed for %s; keeping the unresolved path",
                        path,
                        exc_info=True,
                    )
        runtime = {name: params[name] if name in params else params.get(base) for base, name, _ in _STACK_BASE_FIELDS}
        server_args = sanitize_profile_server_args(str(runtime["extra_server_args"] or "").strip())
        joined_args = normalize_server_args(server_args)
        serving_args = str(self.current_best.get("extra_server_args") or "").strip()
        serving_envs = {
            key: value
            for key, value in sorted(_normalize_envs(self.current_best.get("extra_envs")).items())
            if key.strip()
        }
        # benchmark_mode distinguishes AgentX from synthetic traces with the
        # same CONC/TP so a synthetic trace is never reused for an AgentX session.
        context: dict[str, Any] = {
            "benchmark_mode": str(self.benchmark_mode).strip().lower(),
            "framework": text["framework"].lower(),
            "precision": text["precision"].lower(),
            "model_path": model_path,
            **{name: to_int(workload[name], 0) for name in ("tp", "conc", "isl", "osl", "max_model_len")},
            "server_args": joined_args,
            "extra_envs": dict(sorted(_normalize_envs(runtime["extra_envs"]).items())),
            "remove_args": sorted(set(to_str_list(runtime["remove_args"]))),
            "unset_envs": sorted(set(to_str_list(runtime["unset_envs"]))),
            "args_mode": str(runtime["args_mode"] or "append").strip().lower(),
        }
        if any((serving_args, serving_envs)):
            context["serving_config"] = {"extra_server_args": serving_args, "extra_envs": serving_envs}
        return context

    def profile_trace_usable(self) -> bool:
        """Whether the latest profile left a trace; ``recovered`` was salvaged from a run that reported failure."""
        return self.last_profile_status in {"succeeded", "recovered"}

    def profile_trace_matches_workload(self) -> bool:
        """Whether the recorded profile trace is fresh for the current-best workload."""
        recorded = self.last_profile_workload
        if not self.profile_trace_usable() or not isinstance(recorded, dict) or not recorded:
            return False
        # Compare against the *current-best* runtime identity, not the bare context: last_profile_workload is recorded
        # with the real profile params (actual server_args / extra_envs), while profile_workload_context() with no
        # overrides reports server_args="" and skips the current_best runtime backfill, so any workload with server
        # args/extra envs would compare unequal and every fresh profile would be discarded as stale.
        return recorded == self.current_profile_workload_context()

    def record_profile_workload(
        self,
        params: dict[str, Any] | None = None,
        *,
        arm: str = "",
    ) -> dict[str, Any]:
        """Record the workload identity and originating arm for a profile trace."""
        self.last_profile_workload = self.profile_workload_context(params or {})
        if arm == "baseline":
            self.last_profile_workload_action = "baseline"
        else:
            self.last_profile_workload_action = str(self.current_best.get("action") or "")
        self.last_profile_args = str(self.last_profile_workload.get("server_args") or "")
        return self.last_profile_workload

    def current_profile_workload_context(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the profile identity for the active current-best runtime."""
        current_best = self.current_best
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
        has_runtime_override = any(key in current_best or key in incoming for key in runtime_keys)
        recorded = self.last_profile_workload if isinstance(self.last_profile_workload, dict) else {}
        # A bare baseline/profile ``current_best`` carries no runtime fields, so the only record of what the server
        # actually ran with is the last profile.
        recorded_action = first_text(self.last_profile_workload_action)
        if (
            not has_runtime_override
            and current_best.get("action") in {"baseline", "profile"}
            and recorded
            # Legacy sessions predate the recorded action; keep reusing them rather than forcing a reprofile on every
            # run.
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

        merged_envs = _merged_profile_envs(current_best, incoming)
        if merged_envs:
            params["base_extra_envs"] = merged_envs

        for name in PROFILE_OVERRIDABLE_KEYS:
            if name in incoming:
                params[name] = incoming[name]
        return self.profile_workload_context(params)

    # Persistence
    @classmethod
    def state_path(cls, session_dir: Path) -> Path:
        """Return the canonical ``state.json`` path for a session directory."""
        return Path(session_dir) / "state.json"

    @classmethod
    def load_or_init(cls, session_dir: Path) -> "SharedState":
        """Load existing ``state.json`` or return a fresh blank instance."""
        path = cls.state_path(session_dir)
        if not path.exists():
            inst = cls()
        else:
            inst = cls.from_dict(read_json_strict(path, require_dict=True))
        # Remember the session dir for breakdown instrumentation (not serialized).
        inst._session_dir = Path(session_dir)
        return inst

    @classmethod
    def _validate_resume_fields(cls, raw: Mapping[str, Any]) -> None:
        """Reject damaged crash evidence before migrating a persisted state."""
        if "crash_count" in raw:
            count = raw["crash_count"]
            if not isinstance(count, int) or isinstance(count, bool):
                raise ValueError(f"state.json.crash_count must be int, got {type(count).__name__}")
        if "crash_timestamps" in raw:
            timestamps = raw["crash_timestamps"]
            if not isinstance(timestamps, list):
                raise ValueError(f"state.json.crash_timestamps must be list, got {type(timestamps).__name__}")
            for index, timestamp in enumerate(timestamps):
                if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                    raise ValueError(
                        f"state.json.crash_timestamps[{index}] must be a number, got {type(timestamp).__name__}"
                    )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SharedState":
        """Construct state from a mapping, refusing invalid crash evidence with ``ValueError``."""
        cls._validate_resume_fields(raw)
        # Filter to known fields; unknown keys dropped, missing keys default.
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        # The charge anchor belongs to a live leg, not to the file: only
        # ``begin_leg`` arms it, so ``elapsed_charged_sec`` alone survives a load.
        filtered["leg_anchor_unix"] = 0.0
        if not isinstance(filtered.get("specialist_patch_verdicts"), dict):
            filtered["specialist_patch_verdicts"] = {}
        if not isinstance(filtered.get("pending_kernel_integrations"), dict):
            filtered["pending_kernel_integrations"] = {}
        if not isinstance(filtered.get("current_best"), dict):
            filtered["current_best"] = {}
        # Normalize the unified ``explore_search`` ledger at load.
        filtered["explore_search"] = cls._build_explore_search(
            existing=filtered.get("explore_search"),
        )

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
        """Shape the unified ``explore_search`` ledger at load time."""
        from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint as _fp

        existing = existing if isinstance(existing, dict) else {}
        out: dict[str, Any] = dict(existing)
        out.setdefault("tested", {})
        out.setdefault("accepted", [])
        out.setdefault("rejected", [])
        out.setdefault("cursor", len(out.get("tested") or {}))
        out.setdefault("last_round", {})

        # winners_history: normalize persisted rows, sorted by (round_id, ts).
        wh: list[dict[str, Any]] = []
        persisted = existing.get("winners_history")
        for entry in persisted if isinstance(persisted, list) else []:
            if not isinstance(entry, dict):
                continue
            controls = _control_fields(entry)
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

        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize this state to a plain JSON-compatible dict."""
        return asdict(self)

    def save(self, session_dir: Path) -> None:
        """Atomically write ``state.json`` (tmp file + ``os.replace``).

        Serializes via :meth:`to_dict` and writes to a temp file in the
        same directory before an atomic rename, so concurrent readers never
        observe a partial blob. The temp file is cleaned up on failure.

        Charges the elapsed budget first, so every save is a durable record of
        spend and a leg that dies abruptly loses only the time since the last.

        Args:
            session_dir (Path): The session root directory; created if it
                does not already exist.
        """
        self.charge_elapsed()
        # Backfill scriptable/diffusion (xDiT) ``e2el_mean_ms`` from ``tput`` so current_best has the latency metric.
        self._backfill_scriptable_latency()
        path = self.state_path(session_dir)
        atomic_write_json(path, self.to_dict())
        # Author-time breakdown capture: snapshot state-owned sections into the recorder spool right after persisting.
        self._session_dir = Path(session_dir)
        from hyperloom.inference_optimizer.breakdown.recorder import instrument

        instrument.snapshot_state_sections(session_dir, self)
        _progress.emit_budget(self)
        # Derived artifact: re-render current_setting.sh from the current best route so the operator can audit /
        # re-feed it via --reference-script.
        try:
            cb = self.current_best or {}
            if cb:
                from hyperloom.inference_optimizer import reference_script

                text = reference_script.render_reference_script(
                    framework=str(self.framework or os.environ.get("FRAMEWORK", "sglang")),
                    launch=reference_script.LaunchSettings(
                        server_args=str(cb.get("extra_server_args") or ""),
                        envs=dict(cb.get("extra_envs") or {}),
                        overlay_pythonpath=str(cb.get("final_overlay") or ""),
                        unset_envs=to_str_list(cb.get("unset_envs")),
                        remove_args=to_str_list(cb.get("remove_args")),
                        args_mode=str(cb.get("args_mode") or "append"),
                    ),
                    target=reference_script.ServingTarget(
                        model=self.reference_model or os.environ.get("MODEL_PATH"),
                        tp=int(self.tp or 0) or None,
                        max_model_len=int(self.max_model_len or 0) or None,
                        gpu_type=str(self.gpu_type or os.environ.get("GPU_TYPE", "")) or None,
                    ),
                )
                (Path(session_dir) / "current_setting.sh").write_text(
                    text,
                    encoding="utf-8",
                )
        except (OSError, ValueError):
            # Never leave a stale launcher or export a subset of the retained
            # configuration. The complete settings remain in durable state.
            try:
                (Path(session_dir) / "current_setting.sh").unlink(missing_ok=True)
            except OSError:
                log.warning("Could not remove stale current_setting.sh", exc_info=True)
            log.warning("current_setting.sh render failed", exc_info=True)
        # Live status mirror: reflect the persisted snapshot into Langfuse for real-time status.
        try:
            from hyperloom.inference_optimizer.trace.langfuse_emitter import record_status as _lf_record_status

            _lf_record_status(session_dir, self._langfuse_status_summary())
        except Exception:
            log.debug("langfuse status mirror failed", exc_info=True)

    def _backfill_scriptable_latency(self) -> None:
        """Derive ``current_best.e2el_mean_ms`` from ``tput`` for scriptable runs."""
        from hyperloom.inference_optimizer import framework_registry

        fw = str(getattr(self, "framework", "") or "")
        if not framework_registry.is_scriptable(fw):
            return
        cb = self.current_best
        if cb.get("e2el_mean_ms") is not None:
            return
        tput = cb.get("tput")
        if not isinstance(tput, (int, float)) or tput <= 0:
            return
        e2el = framework_registry.primary_metric_value(fw, float(tput))
        if e2el is not None and e2el > 0:
            cb["e2el_mean_ms"] = round(float(e2el), 4)

    def _langfuse_status_summary(self) -> dict[str, Any]:
        """Flatten the state into an OTEL-friendly scalar status snapshot."""
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
            "framework": self.framework or os.environ.get("FRAMEWORK", ""),
            "kb_hit": str((self.warm_start_context or {}).get("status") or ""),
        }
        from ..phases.machine_state import PHASE_FRAMEWORK_AGENT, phase_cumulative_seconds

        session_elapsed_s = max(0.0, self.elapsed_minutes() * 60.0)
        summary["session_elapsed_s"] = int(round(session_elapsed_s))
        explore_elapsed_s = phase_cumulative_seconds(self, phase=PHASE_FRAMEWORK_AGENT)
        summary["explore_elapsed_s"] = int(round(explore_elapsed_s))
        summary["explore_ratio"] = round(explore_elapsed_s / session_elapsed_s, 4) if session_elapsed_s > 0.0 else 0.0
        tput = self.current_best.get("tput")
        if isinstance(tput, (int, float)) and not isinstance(tput, bool):
            summary["current_best_tput"] = round(float(tput), 1)
        summary.update(self._last_lifecycle_summary())
        return {k: v for k, v in summary.items() if isinstance(v, (str, bool, int, float))}

    def _last_lifecycle_summary(self) -> dict[str, Any]:
        """The newest lifecycle row's sequence, status and phase, for whichever of them it recorded."""
        last = self.lifecycle[-1] if self.lifecycle else {}
        summary: dict[str, Any] = {}
        if isinstance(last, dict):
            if last.get("seq") is not None:
                summary["last_seq"] = last.get("seq")
            if last.get("status"):
                summary["last_lifecycle_status"] = str(last.get("status"))
            if last.get("phase"):
                summary["last_lifecycle_phase"] = str(last.get("phase"))
        return summary

    # Mutators (Coordinator only — LLM agents go via intents)
    def add_pruned_family(self, family: str) -> bool:
        """Idempotently mark an action family as pruned."""
        if family in self.pruned_families:
            return False
        self.pruned_families.append(family)
        return True

    def is_pruned(self, family: str) -> bool:
        """Report whether an action family has been pruned."""
        return family in self.pruned_families

    _POLICY_DENIAL_HISTORY_CAP = 50

    def record_policy_denial(
        self,
        *,
        action_name: str,
        rule: str,
        hint: str,
        intent_type: str,
        intent_payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a PolicyGate denial row, stamped with the current tick, and bump the per-(action, rule) streak.

        Records a capped rolling history entry and increments the
        consecutive-denial counter keyed by ``"<action_name>:<rule>"``.

        Args:
            action_name (str): The action the denied intent targeted (empty
                is normalized to ``"*"`` in the streak key).
            rule (str): The PolicyGate rule id that fired.
            hint (str): Human-readable remediation hint surfaced to the LLM.
            intent_type (str): The denied intent's type.
            intent_payload (dict[str, Any] | None): Optional intent payload;
                when present, its sorted keys are recorded for context.

        Returns:
            int: The new consecutive-denial streak value for this
                (action, rule) pair.
        """
        key = f"{action_name or '*'}:{rule}"
        streak = int(self.policy_denial_streak.get(key, 0)) + 1
        self.policy_denial_streak[key] = streak
        entry = {
            "tick": int(self.tick or 0),
            "action_name": action_name or "",
            "rule": rule,
            "hint": hint or "",
            "intent_type": intent_type,
            "streak": streak,
            "ts": now_iso(),
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
        """Clear all consecutive-denial streaks for a given action.

        Drops every ``policy_denial_streak`` entry whose key begins with
        ``"<action_name>:"`` — called when the action finally succeeds so a
        later denial starts a fresh streak.

        Args:
            action_name (str): The action whose streaks should be reset; a
                falsy value is a no-op.
        """
        if not action_name:
            return
        prefix = f"{action_name}:"
        self.policy_denial_streak = {
            k: v for k, v in (self.policy_denial_streak or {}).items() if not k.startswith(prefix)
        }

    # stop_reason ENUM validator
    def set_stop_reason(
        self,
        value: str,
        *,
        strict: bool | None = None,
    ) -> str:
        """Validated writer for :attr:`stop_reason` (Inv-8.3 closed vocab): values outside ``STOP_REASON_VOCAB`` map to ``\"unknown\"`` (lenient) or raise (``strict=True``, default env ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON``). Returns value written."""
        from hyperloom.inference_optimizer.breakdown.stop_reasons import STOP_REASON_VOCAB, is_valid_stop_reason

        text = str(value or "").strip()
        if not text:
            self.stop_reason = ""
            self.stop_ts = ""
            return ""
        if is_valid_stop_reason(text):
            return self._commit_stop_reason(text)
        if strict is None:
            strict = env_bool("INFERENCE_OPTIMIZER_STRICT_STOP_REASON")
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
        """Write a validated stop reason, stamping the end time on the first one."""
        self.stop_reason = reason
        if not self.stop_ts:
            self.stop_ts = now_iso()
        return reason

    # escalate hint plumbing
    def set_pending_escalate_hint(self, hint: str) -> str:
        """Stash the LLM-supplied hint for the next phase compute pass; unknown hints dropped (Inv-8.2: closed vocab). Returns value written."""
        text = str(hint or "").strip()
        if text and not is_valid_escalate_hint(text):
            return ""
        self.pending_escalate_hint = text
        return text

    def consume_pending_escalate_hint(self) -> str:
        """Pop the pending hint because it drove a phase transition; returns cleared hint."""
        hint = (self.pending_escalate_hint or "").strip()
        if not hint:
            return ""
        self.pending_escalate_hint = ""
        return hint

    def discard_pending_escalate_hint(self) -> str:
        """Pop the pending hint because an unrelated transition fired without acting on it; returns cleared hint."""
        hint = (self.pending_escalate_hint or "").strip()
        if not hint:
            return ""
        self.pending_escalate_hint = ""
        self.last_discarded_escalate_hint = hint
        self.last_discarded_escalate_hint_ts = now_iso()
        return hint

    def accepted_baseline_script(self) -> str | None:
        """Benchmark script belonging to the accepted baseline anchor, or None."""
        if self.baseline_benchmark_script is not None:
            return self.baseline_benchmark_script or None
        if self.last_baseline.get("decision") != "promoted":
            return None
        fingerprint = (self.last_baseline.get("extras") or {}).get("fingerprint") or {}
        return str(fingerprint.get("benchmark_script") or "").strip() or None

    def current_top_bottleneck(self) -> str:
        """Return the latest roofline snapshot's ``top_bottleneck`` (\"\" when none)."""
        snaps = self.roofline_snapshots if isinstance(self.roofline_snapshots, list) else []
        if not snaps:
            return ""
        latest = snaps[-1]
        if isinstance(latest, dict):
            return str(latest.get("top_bottleneck") or "")
        return ""

    def current_within_roofline_pct(self) -> float | None:
        """Return the latest snapshot's achieved share of its roofline ceiling."""
        snaps = self.roofline_snapshots if isinstance(self.roofline_snapshots, list) else []
        if not snaps:
            return None
        latest = snaps[-1]
        if not isinstance(latest, dict):
            return None
        value = latest.get("within_roofline_pct")
        if not isinstance(value, (int, float)):
            return None
        return float(value)

    def mark_bottleneck_switch(self, prev_bottleneck: str = "") -> None:
        """Flag that the next macro-cycle should redirect off ``prev_bottleneck`` (R3)."""
        self.pending_bottleneck_switch = True
        pb = (prev_bottleneck or "").strip() or self.current_top_bottleneck()
        if pb:
            self.last_cycle_bottleneck = pb

    def clear_bottleneck_switch(self) -> None:
        """Clear the pending bottleneck-switch handoff (R3)."""
        self.pending_bottleneck_switch = False
        self.last_cycle_bottleneck = ""

    def maybe_clear_bottleneck_switch_on_drift(self, new_top_bottleneck: str) -> bool:
        """Retire a pending switch once the live top bottleneck has drifted (R3)."""
        if not bool(getattr(self, "pending_bottleneck_switch", False)):
            return False
        nt = (new_top_bottleneck or "").strip()
        if nt and nt != (self.last_cycle_bottleneck or ""):
            self.clear_bottleneck_switch()
            return True
        return False

    def record_experience_kb_injection(self, injection: KbInjection) -> bool:
        """Record an injected Experience KB block.

        Orchestration re-reads every tick, so its row is skipped when it injects the same Experiences as the last
        orchestration row; every specialist dispatch is its own injection and always records.
        """
        if injection.consumer == "orchestration":
            last = next(
                (row for row in reversed(self.experience_kb_injections) if row.get("consumer") == injection.consumer),
                {},
            )
            if sorted(last.get("experience_ids") or []) == sorted(injection.experience_ids):
                return False
        self.experience_kb_injections.append(
            {
                "tick": int(self.tick),
                "phase": self.phase,
                "ts": datetime.now(timezone.utc).isoformat(),
                "consumer": injection.consumer,
                "domain": injection.domain,
                "gap_canonical_id": injection.gap_canonical_id,
                "read_id": injection.read_id,
                "experience_ids": list(injection.experience_ids),
                "experiences": [dict(item) for item in injection.experiences],
                "prompt_block": injection.prompt_block,
            }
        )
        del self.experience_kb_injections[:-_KB_INJECTIONS_CAP]
        return True

    def merge_lifecycle_events(self, incoming: Any) -> None:
        """Union ``incoming`` lifecycle rows into this state, ordered by timestamp."""
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
        """Increment the cumulative crash counter and record crash times."""
        self.crash_count += by
        now = time.time()
        for _ in range(max(1, int(by))):
            self.crash_timestamps.append(now)
        if len(self.crash_timestamps) > _CRASH_TIMESTAMP_CAP:
            del self.crash_timestamps[:-_CRASH_TIMESTAMP_CAP]
        return self.crash_count

    def recent_crash_count(self, *, window_sec: float, now: float | None = None) -> int:
        """Count crashes recorded within the trailing ``window_sec`` seconds."""
        ref = time.time() if now is None else now
        cutoff = ref - float(window_sec)
        return sum(1 for t in self.crash_timestamps if t >= cutoff)

    def record_tick_exception(
        self,
        *,
        tick: int,
        stage: str,
        exc: BaseException,
        agent: str = "",
    ) -> dict[str, Any]:
        """Persist a compact Coordinator exception summary for postmortems."""
        entry = {
            "tick": int(tick or 0),
            "ts": now_iso(),
            "stage": str(stage or ""),
            "agent": str(agent or ""),
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[:12000],
        }
        self.last_tick_exception = entry
        return entry

    @staticmethod
    def _is_integrate_fault(result: dict[str, Any]) -> bool:
        """True when an integrate result is an integration *fault*, not a verdict."""
        status = str(result.get("status") or "").strip().lower()
        if status == "failed":
            return True
        err_class = str(result.get("error_class") or "").strip()
        return err_class in _INTEGRATE_FAULT_ERROR_CLASSES

    def record_kernel_integrate_result(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """Record an integrate result and capture the gate's verdict on the KERNEL timeline.

        Every integrate settlement passes through here, so the verdict is captured once rather than at each caller.
        """
        from ..kernel import kernel_decisions as _m

        entry = _m.record_kernel_integrate_result(self, result)
        self._record_integrate_verdict_on_timeline(entry)
        return entry

    def _record_integrate_verdict_on_timeline(self, entry: dict[str, Any] | None) -> None:
        """Mirror one settled integrate verdict onto the KERNEL timeline."""
        if not isinstance(entry, dict):
            return
        from hyperloom.inference_optimizer.breakdown.recorder import kernel_event
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_records import IntegrateVerdict

        last = (entry.get("attempts") or [{}])[-1]
        last = last if isinstance(last, dict) else {}
        rejected = entry.get("rejected")
        cycle = int(getattr(self, "macro_cycle", 0) or 0)
        integration_id = str(entry.get("integration_id") or "")
        kernel_event.record_integrate_verdict(
            macro_cycle=cycle,
            verdict=IntegrateVerdict(
                integration_id=integration_id,
                kernel_id=str(entry.get("kernel_id") or ""),
                decision=str(entry.get("last_decision") or ""),
                status=str(entry.get("last_status") or ""),
                attempt_count=entry.get("attempt_count"),
                fault_count=entry.get("fault_count"),
                gain_pct=entry.get("best_gain_pct"),
                accuracy_pass=last.get("accuracy_pass"),
                validation_tier=str(last.get("validation_tier") or ""),
                patch_path=str(entry.get("patch_path") or ""),
                target_file=str(entry.get("target_file") or ""),
                error_class=str(entry.get("last_error_class") or ""),
                rejected_reason=str(rejected.get("reason") or "") if isinstance(rejected, dict) else "",
                retryable=bool(entry.get("retryable")),
                settled_at=str(entry.get("updated_at") or ""),
                extra_server_args=str(entry.get("extra_server_args") or ""),
                basis=str(entry.get("basis") or ""),
                alignment_status=str(entry.get("alignment_status") or ""),
                # Absent means attributable; a writer that cannot pin the gain on this kernel says so.
                gain_attributed=bool(entry.get("validated", True)),
            ),
        )
        kernel_event.record_integrate_axis(
            macro_cycle=cycle, integration_id=integration_id, graded_objective=str(last.get("graded_objective") or "")
        )

    @property
    def roofline_snapshot_id(self) -> int:
        """Counter of the newest roofline snapshot, or 0 before the first one."""
        raw = (self.last_trace_analyze or {}).get("roofline_snapshot_id")
        return int(raw) if isinstance(raw, int) else 0

    # Per-action audit (kernel parity for non-kernel actions)
    @staticmethod
    def _truncate_excerpt(value: Any, *, limit: int = 1200) -> str | None:
        """Coerce ``value`` to str and trim to ``limit`` chars; None for falsy inputs (renderer shows ``err=(none)``)."""
        if value is None:
            return None
        text = redact_secret_values(str(value))
        if not text:
            return None
        if len(text) <= limit:
            return text
        return text[:limit]

    def _common_result_fields(self, result: dict[str, Any]) -> dict[str, Any]:
        """Build the failure/diagnostic fields shared by the attempt + failure logs."""
        return {
            "error_class": (str(result.get("error_class")) if result.get("error_class") else None),
            "error_excerpt": self._truncate_excerpt(result.get("error")),
            "stderr_tail": tail_excerpt(result.get("error"), limit=1000),
            "stderr_log_path": (str(result.get("stderr_log_path")) if result.get("stderr_log_path") else None),
            "workspace": (str(result.get("workspace")) if result.get("workspace") else None),
            "raw_result_path": (str(result.get("raw_result_path")) if result.get("raw_result_path") else None),
            "reported_success": result.get("reported_success"),
            "variant_name": (str(result.get("variant_name")) if result.get("variant_name") else None),
            "failure_id": result.get("failure_id"),
        }

    def record_action_attempt(self, action: str, attempt: ActionAttempt) -> dict[str, Any] | None:
        """Append one attempt to ``<action>_attempts`` and refresh ``last_<action>``. Entry schema {ts, task_id, status, decision, key_metric, key_metric_kind, workspace, error_class, error_excerpt, stderr_tail, stderr_log_path, raw_result_path, reported_success, variant_name, extras}. Returns the entry, or None when ``action`` not in the audit set (kernel_agent-owned actions use bespoke recorders). Does NOT call :meth:`save`."""
        if action not in _AUDIT_ACTIONS:
            return None
        attempts_attr = f"{action}_attempts"
        last_attr = f"last_{action}"
        result = attempt.result or {}
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
            "ts": now_iso(),
            "task_id": str(attempt.task_id or ""),
            "status": str(attempt.status or ""),
            "decision": str(attempt.decision or ""),
            "key_metric": key_metric,
            "key_metric_kind": metric_kind,
            **self._common_result_fields(result),
            "extras": dict(attempt.extras or {}),
        }
        history: list[dict[str, Any]] = list(getattr(self, attempts_attr) or [])
        history.append(entry)
        if len(history) > _DEFAULT_ATTEMPTS_HISTORY:
            history = history[-_DEFAULT_ATTEMPTS_HISTORY:]
        setattr(self, attempts_attr, history)
        setattr(self, last_attr, dict(entry))
        if action == "baseline":
            from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import record_action_decision

            record_action_decision(
                phase=str(getattr(self, "phase", "") or "unphased"),
                macro_cycle=int(getattr(self, "macro_cycle", 0) or 0),
                task_id=str(entry.get("task_id") or ""),
                decision=str(entry.get("decision") or ""),
            )
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
        entry: dict[str, Any] = {
            "ts": now_iso(),
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
        """Append one structured failure packet to :attr:`failures` (last-wins on ``failure_id``); mirrored to ``<session_dir>/reports/failures/`` so it survives the bounded list."""
        fid = str(fe.get("failure_id") or "")
        history = [e for e in (self.failures or []) if e.get("failure_id") != fid]
        history.append(fe)
        self.failures = history[-_DEFAULT_LAST_FAILURES:]

        session_dir = getattr(self, "_session_dir", None)
        if session_dir:
            from hyperloom.inference_optimizer.session.session_paths import failure_evidence_path

            try:
                atomic_write_json(failure_evidence_path(session_dir, fid), fe)
            except OSError:
                log.debug("failure evidence mirror failed for %s", fid, exc_info=True)

    def find_failure(self, failure_id: str) -> "dict[str, Any] | None":
        """Return the :attr:`failures` entry for ``failure_id``, else ``None``."""
        fid = str(failure_id or "").strip()
        for entry in reversed(self.failures or []):
            if entry.get("failure_id") == fid:
                return entry
        return None

    def failures_for_task(self, task_id: str) -> "list[dict[str, Any]]":
        """Return the ``task_id`` entries from :attr:`failures`, newest first."""
        tid = str(task_id or "").strip()
        return [e for e in reversed(self.failures or []) if e.get("task_id") == tid]

    def _resolve_baseline_achieved_tput(self) -> float:
        """Baseline throughput for a baseline-arm roofline snapshot."""
        if isinstance(self.baseline_tput, (int, float)) and self.baseline_tput > 0:
            return float(self.baseline_tput)
        return first_positive_tput(self.last_baseline)

    def _resolve_current_best_achieved_tput(self) -> float:
        """Optimized-arm throughput for a current_best roofline snapshot."""
        return first_positive_tput(self.current_best)

    def _locate_diffusion_roofline_sidecar(self, kernel_roofline_path: Any) -> Path | None:
        """Locate the ``diffusion_roofline.json`` sidecar for the latest trace run."""
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
        """Resolve the (measured e2e latency, ideal latency floor) ms pair."""
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
                # Priority: full-pipeline analytic ceiling (ms) > trace-summed per-kernel ideal (us).
                approach_a = data.get("analytic_ceiling")
                totals = data.get("totals")
                if isinstance(approach_a, dict) and float(approach_a.get("ideal_ms") or 0.0) > 0:
                    roofline_ideal_ms = float(approach_a["ideal_ms"])
                elif isinstance(totals, dict) and float(totals.get("sigma_ideal_roofline_us") or 0.0) > 0:
                    roofline_ideal_ms = float(totals["sigma_ideal_roofline_us"]) / 1000.0
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                roofline_ideal_ms = 0.0
        return e2e_mean_ms, roofline_ideal_ms

    def _roofline_throughput_unit(self) -> str:
        """Return the throughput unit for roofline snapshots of this workload."""
        from hyperloom.inference_optimizer import framework_registry

        framework = str(getattr(self, "framework", "") or "").strip().lower()
        return framework_registry.throughput_unit(framework)

    def record_trace_analyze(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Write ``last_trace_analyze`` (single writer); ``roofline_snapshot_id`` increments from the previous nested value, restarting from 1 when the cache was cleared."""
        if not isinstance(result, dict):
            return
        result = dict(result)
        # Both callers build the payload as a dict.
        task = payload
        artifacts = result.get("artifact_paths")
        if not isinstance(artifacts, dict):
            artifacts = {}
        candidates_path, kernel_roofline_path = self._recover_from_roofline_report(
            result,
            task,
            candidates_path=result.get("candidates_path") or artifacts.get("kernel_candidates") or "",
            kernel_roofline_path=result.get("kernel_roofline_path") or artifacts.get("kernel_roofline") or "",
        )
        steady_state_trace = result.get("steady_state_trace") or artifacts.get("tracelens_steady_state_trace") or ""
        summary, kernel_roofline, reusable_ids = self._build_hot_kernel_summaries(result)
        warning_rows = result.get("trace_health_warnings")
        warning_rows = warning_rows if isinstance(warning_rows, list) else []
        task_groups = result.get("task_groups")
        # Monotonic snapshot counter: read previous value + 1.
        recorded = self.last_trace_analyze if isinstance(self.last_trace_analyze, dict) else {}
        analysis_md_path = result.get("trace_report_path") or ""
        analysis_md_text = ""
        if analysis_md_path:
            # Stored verbatim; the prompt path strips base64 data-URLs.
            with suppress(OSError, ValueError):
                analysis_md_text = Path(analysis_md_path).read_text(encoding="utf-8", errors="replace")
        self.last_trace_analyze = {
            "trace_input": str(task.get("trace_input") or task.get("trace_dir") or ""),
            "steady_state_trace": str(steady_state_trace),
            "candidates_path": str(candidates_path),
            "kernel_roofline_path": str(kernel_roofline_path),
            "hot_kernels_top15": summary,
            "kernel_roofline_top15": kernel_roofline,
            "skipped_kernels_top": _skipped_kernels_top(result),
            "task_groups": task_groups if isinstance(task_groups, list) else [],
            "reusable_native_kernel_ids": reusable_ids,
            "trace_health_warnings": [dict(e) for e in warning_rows if isinstance(e, dict) and e.get("code")],
            "analysis_md_path": str(analysis_md_path),
            "analysis_md_text": analysis_md_text,
            "roofline_snapshot_id": to_int(recorded.get("roofline_snapshot_id"), 0) + 1,
            "roofline_baseline_gain_at_snapshot": float(self.cumulative_gain_validated),
            "ts": now_iso(),
        }

        self._append_roofline_snapshot_history(payload)

    def _recover_from_roofline_report(
        self,
        result: dict[str, Any],
        task: dict[str, Any],
        *,
        candidates_path: Any,
        kernel_roofline_path: Any,
    ) -> tuple[Any, Any]:
        """Fill what the envelope lost from the session roofline report; the candidates and roofline paths after it."""
        disk = self._read_session_roofline_report(task, kernel_roofline_path)
        if not disk:
            return candidates_path, kernel_roofline_path
        report = disk["payload"]
        kernel_roofline_path = kernel_roofline_path or str(disk["path"])
        candidates_path = candidates_path or str(report.get("kernel_candidates_path") or "")
        result["trace_report_path"] = result.get("trace_report_path") or str(report.get("analysis_md_path") or "")
        if not (result.get("hot_kernels") or result.get("hot_kernels_top15")):
            result["hot_kernels"] = disk["kernels"]
            log.warning(
                "record_trace_analyze: envelope carried no hot kernels; "
                "recovered %d from %s. The analysis succeeded — the result "
                "envelope lost its payload in transit.",
                len(disk["kernels"]),
                disk["path"],
            )
        return candidates_path, kernel_roofline_path

    def _read_session_roofline_report(
        self,
        payload: dict[str, Any],
        kernel_roofline_path: str,
    ) -> "dict[str, Any] | None":
        """Read the session-level kernel-roofline report written by TraceLens."""
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
    ) -> "tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]":
        """Build ``(summary, kernel_roofline, reusable_ids)`` from the top-N hot kernels."""
        hot = result.get("hot_kernels")
        summary: list[dict[str, Any]] = []
        kernel_roofline: list[dict[str, Any]] = []
        reusable_ids: list[str] = []
        for entry in hot[:_TRACE_HOT_KERNEL_TOP_N] if isinstance(hot, list) else []:
            if not isinstance(entry, dict):
                continue
            row = _hot_kernel_row(entry)
            summary.append(row)
            if any(row.get(key) not in (None, "", []) for key in _ROOFLINE_SIGNAL_KEYS):
                kernel_roofline.append(dict(row))
            if row["reusable_native_kernel"] and row["kernel_id"]:
                reusable_ids.append(str(row["kernel_id"]))
        return summary, kernel_roofline, reusable_ids

    def _append_roofline_snapshot_history(self, payload: dict[str, Any]) -> None:
        """Append a compact roofline snapshot of ``last_trace_analyze`` for report-side comparison.

        Runs after the canonical write above, so a failure here cannot undo it.
        """
        from hyperloom.inference_optimizer.roofline_ceiling import compute_roofline_breakdown_from_state
        from hyperloom.inference_optimizer.roofline_snapshot import (
            RooflineReading,
            attach_perfmodel_breakdown,
            build_roofline_snapshot,
        )

        latest = self.last_trace_analyze
        snapshot_id = latest["roofline_snapshot_id"]
        analysis_md_path = str(latest["analysis_md_path"])
        kernel_roofline_path = str(latest["kernel_roofline_path"])
        # Resolve which arm this snapshot measures first so the ceiling is anchored to the same arm as achieved.
        forced_arm = str((payload or {}).get("roofline_arm") or "").strip()
        # Unknown arm values fall through to current_best inference; warn.
        if forced_arm and forced_arm not in ("baseline", "current_best"):
            log.warning(
                "record_trace_analyze: ignoring unknown roofline_arm=%r; falling back to current_best inference",
                forced_arm,
            )
            forced_arm = ""
        cb_tput = self.current_best.get("tput")
        snapshot_arm = forced_arm or ("current_best" if first_positive_tput({"tput": cb_tput}) else "baseline")
        achieved_tput = (
            self._resolve_current_best_achieved_tput()
            if snapshot_arm == "current_best"
            else self._resolve_baseline_achieved_tput()
        )
        # Scriptable/diffusion has no tok/s decode ceiling; surface the compute-latency roofline (measured
        # per-image e2e latency vs the ideal floor from the sidecar).
        fw = str(getattr(self, "framework", "") or "")
        e2e_mean_ms, roofline_ideal_ms = self._scriptable_latency_roofline(fw, achieved_tput, kernel_roofline_path)
        history_entry = build_roofline_snapshot(
            snapshot_id=snapshot_id,
            ts=latest["ts"],
            analysis_md_path=analysis_md_path,
            reading=RooflineReading(
                # Primary decode ceiling plus memory/compute sides (PerfModel bottom-up).
                ceiling=compute_roofline_breakdown_from_state(self, arm=snapshot_arm),
                achieved_tok_per_sec=achieved_tput,
                throughput_unit=self._roofline_throughput_unit(),
                e2e_mean_ms=e2e_mean_ms,
                roofline_ideal_ms=roofline_ideal_ms,
            ),
            framework=fw,
        )
        # Per-op PerfModel breakdown for dashboard visualization.
        attach_perfmodel_breakdown(history_entry, self, arm=snapshot_arm)
        history_entry["trace_input"] = str(latest["trace_input"])
        history_entry["macro_cycle"] = int(getattr(self, "macro_cycle", 0) or 0)
        history_entry["analysis_md_path"] = analysis_md_path
        # Sidecar artifact pointer for per-kernel roofline data.
        history_entry["kernel_roofline_path"] = kernel_roofline_path
        if not isinstance(self.roofline_snapshots, list):
            self.roofline_snapshots = []
        self.roofline_snapshots.append(history_entry)
        self._record_bottleneck_saturation(history_entry, snapshot_id)
        # R3: retire the pending switch once the top bottleneck has drifted.
        self.maybe_clear_bottleneck_switch_on_drift(
            str(history_entry.get("top_bottleneck") or ""),
        )
        if len(self.roofline_snapshots) > _ROOFLINE_SNAPSHOTS_CAP:
            # Always keep snapshot #1 as the report's baseline anchor.
            base = self.roofline_snapshots[0]
            tail = self.roofline_snapshots[-(_ROOFLINE_SNAPSHOTS_CAP - 1) :]
            self.roofline_snapshots = [base, *tail]

    def _record_bottleneck_saturation(self, history_entry: dict[str, Any], snapshot_id: Any) -> None:
        """Stamp the saturation of this snapshot's dominant direction and how it shifted since the previous cycle."""
        from hyperloom.inference_optimizer.roofline_snapshot import direction_saturation

        sat = direction_saturation(history_entry)
        hint = sat.get("domain_hint") if isinstance(sat.get("domain_hint"), dict) else {}
        domain_key = str(hint.get("domain") or sat.get("direction") or "unknown")
        latest_cycle = int(history_entry.get("macro_cycle") or 0)
        prev_cycle_snapshot: dict[str, Any] = {}
        for prev in reversed(self.roofline_snapshots[:-1]):
            if int(prev.get("macro_cycle") or 0) < latest_cycle:
                prev_cycle_snapshot = prev
                break
        if not prev_cycle_snapshot and len(self.roofline_snapshots) >= 2:
            prev_cycle_snapshot = self.roofline_snapshots[-2]
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
            "from": prev_sat.get("direction", ""),
            "to": sat.get("direction", ""),
            "from_domain": (prev_sat.get("domain_hint") or {}).get("domain", ""),
            "to_domain": domain_key,
            "prev_cycle": prev_cycle_snapshot.get("macro_cycle"),
            "cycle": latest_cycle,
            "within_delta": _saturation_delta(sat, prev_sat, "within_pct"),
            "gap_delta": _saturation_delta(sat, prev_sat, "gap_pct"),
            "bound_kind_changed": (
                bool(prev_sat) and str(prev_sat.get("bound_kind") or "") != str(sat.get("bound_kind") or "")
            ),
            "current": sat,
            "previous": prev_sat,
        }

    def record_conc_sweep(self, result: dict[str, Any]) -> None:
        """Record the concurrency sweep's completion into ``last_conc_sweep``."""
        if not isinstance(result, dict):
            return
        self.last_conc_sweep = {
            "ts": now_iso(),
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
        """Bump the Coordinator tick counter."""
        self.tick = int(self.tick or 0) + 1
        return self.tick

    def append_stack_gain_entry(self, *, new_tput: float) -> None:
        """Mirror an optimization_stack append into gain_per_stack_entry as ``(new_tput-baseline_tput)/baseline_tput*100``."""
        self.gain_per_stack_entry.append(gain_pct(to_float(new_tput, 0.0), to_float(self.baseline_tput, 0.0)))

    # Session budget: elapsed is summed forward across legs.
    def begin_leg(self, *, now_unix: float | None = None) -> None:
        """Open a run leg: start charging elapsed time from this instant.

        A leg is one process's turn at the session. Time between legs is not
        charged; every second inside one is, folded into
        :attr:`elapsed_charged_sec` by :meth:`charge_elapsed`. How the previous
        leg ended is deliberately not consulted, so no leg can begin by handing
        itself a fresh budget.

        Args:
            now_unix: Wall instant the leg starts; defaults to ``time.time()``.
        """
        self.leg_anchor_unix = float(time.time() if now_unix is None else now_unix)

    def charge_elapsed(self, *, now_unix: float | None = None) -> float:
        """Fold the time since the last charge into the session's elapsed total.

        Called on every state save. A no-op until :meth:`begin_leg` opens a
        leg, so reading and re-saving a state bills the session nothing.

        Args:
            now_unix: Wall instant to charge up to; defaults to ``time.time()``.

        Returns:
            float: The new :attr:`elapsed_charged_sec`.
        """
        anchor = self.leg_anchor_unix
        if anchor <= 0.0:
            return self.elapsed_charged_sec
        now = float(time.time() if now_unix is None else now_unix)
        self.elapsed_charged_sec += max(0.0, now - anchor)
        self.leg_anchor_unix = now
        return self.elapsed_charged_sec

    def elapsed_minutes(self, *, now: datetime | None = None) -> float:
        """Minutes of budget this session has consumed, summed over every leg.

        The charged total plus whatever the live leg has run since the last
        charge.

        Args:
            now (datetime | None): Reference time; defaults to the current UTC
                time.

        Returns:
            float: Minutes consumed; never negative.
        """
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        return self.elapsed_sec_of(vars(self), now_unix=now_dt.timestamp()) / 60.0

    elapsed_sec_of = staticmethod(elapsed_sec_of)

    def extend_budget_minutes(self, minutes: float) -> float:
        """Grant more wall-clock budget to this session.

        Args:
            minutes: Minutes to add; non-positive is a no-op.

        Returns:
            float: The session's budget in minutes after the grant; ``0.0``
                when the session is unbounded and nothing was granted.
        """
        added = float(minutes)
        if added <= 0.0:
            return float(self.max_minutes)
        if not self.max_minutes:
            # Granting an unbounded session a budget would bound it.
            return 0.0
        self.max_minutes = int(float(self.max_minutes) + added)
        return float(self.max_minutes)

    def remaining_minutes(self, *, now: datetime | None = None) -> float | None:
        """Minutes left in the wall-clock budget; ``None`` when unbounded, else clamped at 0.

        :meth:`session_deadline` keeps the sign, for a caller that must tell
        "just expired" from "expired an hour ago".

        Args:
            now (datetime | None): Reference time; defaults to the current UTC
                time.

        Returns:
            float | None: Minutes remaining in the budget (clamped at 0.0), or
                ``None`` when the session is unbounded.
        """
        if not self.max_minutes:
            return None
        return max(0.0, float(self.max_minutes) - self.elapsed_minutes(now=now))

    def session_deadline(self, *, now: datetime | None = None) -> Deadline | None:
        """The absolute instant this session's budget runs out.

        ``None`` means unbounded, and only that: an exhausted session returns a
        :class:`Deadline` already in the past.

        Args:
            now (datetime | None): Reference time for the elapsed sum; defaults
                to the current UTC time.

        Returns:
            Deadline | None: The stop instant, or ``None`` when unbounded.
        """
        if not self.max_minutes:
            return None
        spent_min = self.elapsed_minutes(now=now)
        return Deadline.after((float(self.max_minutes) - spent_min) * 60.0)

    def closing_reserve_sec(self) -> float:
        """Seconds held back from every unit of work so CLOSE can still report."""
        return max(0.0, effective_closing_grace_sec(float(self.max_minutes or 0), self.closing_grace_sec))

    def session_budget_usable_sec(
        self,
        *,
        reserve_sec: float | None = None,
    ) -> float | None:
        """Seconds of wall-clock budget left once the closing reserve is held back."""
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
        """``time.monotonic()`` deadline for grid variant loops, or ``None`` when the budget is unbounded."""
        usable = self.session_budget_usable_sec(reserve_sec=reserve_sec)
        if usable is None:
            return None
        return time.monotonic() + usable

    @property
    def has_validated_gain_stamp(self) -> bool:
        """Whether a cumulative gain has been validated against the current anchor."""
        return bool(
            self.cumulative_gain_validated_ts
            or self.cumulative_gain_validated_stack_len
            or self.cumulative_gain_validated
        )

    def optimization_stack_has_unvalidated_keeps(self) -> bool:
        """True iff ``current_best`` changed since the last validated measurement (stack grew or a lift landed)."""
        return len(self.optimization_stack) > int(self.cumulative_gain_validated_stack_len) or int(
            self.working_recipe_generation
        ) != int(self.validated_recipe_generation)


__all__ = ["SharedState"]
