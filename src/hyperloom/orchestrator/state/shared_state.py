# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SharedState — single-writer (Coordinator) persisted session state, backed by atomic JSON at ``$SESSION_DIR/state.json``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from hyperloom_common.coerce import to_float
from hyperloom.common.gain_math import gain_pct, gain_pct_or_zero
from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.common.perf_metric import (
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
    snapshot_axis,
    total_tput_of,
)
from hyperloom_common.timeutil import now_iso
from hyperloom_common.jsonio import read_json_strict
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.inference_optimizer.grid_server_args import strip_benchmark_harness_flags

from .kernel_decision_settings import _DEFAULT_ATTEMPTS_HISTORY
from ._shared_state.attempt_audit import _AUDIT_ACTIONS, _KEY_METRIC_MAP
from ._shared_state.budget import _BudgetStateMixin
from ._shared_state.enablement_round import EnablementRound
from ._shared_state.roofline import _RooflineStateMixin, first_positive_tput

log = logging.getLogger(__name__)


# The two hints the FRAMEWORK_AGENT exit reads. ``skip_to_kernel`` is the model's "this lever is exhausted", sent as
# ``escalate_strategy_change``; ``skip_to_sweep`` is the Coordinator's own, set when a custom-specialist-only cycle
# has run all its work.
ESCALATE_HINT_SKIP_TO_KERNEL: str = "skip_to_kernel"
ESCALATE_HINT_SKIP_TO_SWEEP: str = "skip_to_sweep"


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
    # AgentX KEEPs on the P50 interactivity objective clearing its bar with each of its guards inside the noise band;
    # anything short of all of them is REVERT. A pair that cannot supply the axes degrades together and
    # ``degrade_reason`` says why -- promotion lanes read ``comparable`` and fail closed rather than KEEPing on the
    # diagnostic output figure. ``keep_threshold_pct`` applies to the output axis only. ``anchor`` defaults to the
    # session anchor; explore passes its own because variants stack within a round. Objective and band come from
    # ``resolved_grading``, so both are what the session was seeded with rather than whatever the calling process's
    # environment holds.
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
            objective = AgentXObjective.P50_INTERACTIVITY
            gain = gain_pct_or_zero(axis_of(cand_perf, objective.metric), axis_of(ref_perf, objective.metric))
            # The chip count divides both sides of each throughput ratio, so the band holds identically per GPU.
            holds = partial(holds_within_band, cand_perf, ref_perf, noise_pct=noise_pct)
            gates = (
                (objective.below_threshold, gain >= objective.bar_pct),
                *((guard.refusal, holds(snapshot_axis(guard.metric))) for guard in objective.guards),
                (incomparability_reason(cand_perf, ref_perf), False),
            )
            refused = [gate for gate, held in gates if gate and not held]
            sla_veto = "" if refused else constraint_veto_reason(measurement, state)
            return GradedComparison(
                objective=objective.metric,
                candidate=axis_of(cand_perf, objective.metric),
                reference=axis_of(ref_perf, objective.metric),
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


from ._shared_state.render import _RenderMixin


from ._shared_state.phase_state import _PhaseStateMixin
from .gaps import GapsStateMixin
from ._shared_state.profile_workload import _STACK_BASE_FIELDS, _ProfileWorkloadMixin
from ._shared_state.action_records import _ActionRecordsMixin
from ._shared_state.run_signals import _RunSignalsMixin
from ._shared_state.persistence import _PersistenceMixin


@dataclass
class SharedState(
    _ProfileWorkloadMixin,
    _ActionRecordsMixin,
    _RunSignalsMixin,
    _PersistenceMixin,
    _RenderMixin,
    GapsStateMixin,
    _PhaseStateMixin,
    _RooflineStateMixin,
    _BudgetStateMixin,
):
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
    # The last GEAK e2e run's result.json, with the verdict the rebench settled on it.
    geak_result: dict[str, Any] = field(default_factory=dict)
    # Terminal result of the phase-level KernelForge rewrite controller.
    kernel_rewrite_controller_result: dict[str, Any] = field(default_factory=dict)
    # Whether the session runs its concurrency sweep once it closes (``--enable-conc-sweep``, ``hyperloom.orchestrator.conc_sweep``).
    conc_sweep_enabled: bool = False
    # Which benchmark workload this session measures: "agentx" (agentic trace replay) or "synthetic" (ISL/OSL).
    benchmark_mode: str = ""
    # The DATASET_PATH file the session's fixed rounds send in place of random prompts, as given; empty when none.
    # The path and not a digest of the file: its prompts are counted where a point is rendered and sent where the round
    # runs, which need not be this host.
    # Compared on resume, and no store keyed on ISL/OSL files a session that records one.
    prompt_dataset: str = ""
    # Generation of the measurement path this session was measured on; resume refuses one below the build's epoch.
    measurement_epoch: int = 0
    # Which AgentX client measured this session: "inferencex" or "mlperf"; empty outside AgentX. Compared on resume so
    # the two workloads cannot anchor each other without bumping the measurement epoch (which would invalidate
    # unchanged sessions of the other client).
    agentx_backend: str = ""
    agentx_accuracy: bool = True  # Native AgentX GSM8K round; off only by ``--no-agentx-accuracy`` at seed.
    # The grading configuration this session was seeded with: {"objective": GRADED_INTVTY|GRADED_OUTPUT,
    # "noise_pct": float}. Recorded rather than re-derived because the derivation reads HYPERLOOM_PERF_METRIC /
    # HYPERLOOM_PERF_NOISE_PCT, and a resume is a new process: a shell that lost the variable would flip the axis
    # mid-run, and a lost noise band would silently widen a 3.5% guard back to the 5% default. The KEEP/REVERT rule
    # has to be the one the session started with. Empty on sessions predating the field, which fall back to deriving.
    grading: dict[str, Any] = field(default_factory=dict)
    # Stamped once when the run objective is first met.
    target_reached_at: str = ""
    # The operator's conc-sweep points (``--conc-sweep-concs``), read only for a base no InferenceX recipe gave.
    conc_sweep_concs: list[int] = field(default_factory=list)
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
    # The eval definition digest ``baseline_accuracy`` was taken under; "" for a score no inference_suite round took.
    baseline_accuracy_definition_digest: str = ""
    baseline_failure_streak: int = 0
    # The live leg's share of ``baseline_failure_streak``; ``begin_leg`` and every reset of the streak zero it.
    baseline_failures_this_leg: int = 0
    baseline_arg_error_streak: int = 0  # Nothing advances it; old sessions' state.json and close-out records carry it.
    # Combined backstop counting ANY baseline failure regardless of error_class, catching mixed classes that never
    # trip a per-class streak (anti time-exhaustion).
    baseline_total_failures: int = 0
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
    # Baseline WARM measure-round wall-clock (client-only, no boot); anchors the explore overtime kill apples-to-apples.
    baseline_warm_runtime_sec: float = 0.0
    # Whether the baseline's hot pass was dropped because the budget could not cover it plus one variant to read
    # against it, leaving the cold pass's depressed figure as the anchor.
    baseline_measure_round_dropped: bool = False
    # The COLD round above from the server-ready marker onward (benchmark and accuracy eval), and that eval's own time.
    baseline_post_ready_runtime_sec: float = 0.0
    baseline_eval_runtime_sec: float = 0.0
    # Wall-clock of the slowest roofline (profile + TraceLens) this session finished; CLOSE sizes its own from it.
    roofline_runtime_sec: float = 0.0
    current_best: dict[str, Any] = field(default_factory=dict)
    # Immutable measurement evidence captured with the current-best promotion.
    current_best_measurement: dict[str, Any] = field(default_factory=dict)
    # Digest of the session's frozen ``base_launch.json``; empty when the session froze none.
    base_launch_digest: str = ""
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
    # The card's compute-partition shape this session was measured in, as observed at launch: mode, partition count,
    # CU and memory per partition, streams per partition.
    compute_partition: dict[str, Any] = field(default_factory=dict)
    # ``--nodes``, feeding the IR-8 check.
    nodes: int = 1
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
    # Session budget: seconds charged by closed legs, plus the wall instant the live leg opened (``0.0`` between legs).
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
    # Unified persistent explore-search ledger; ``tested`` keyed by canonical_fingerprint, everything graded down moves to rejected.
    explore_search: dict[str, Any] = field(default_factory=dict)
    # specialist sub-agent rolling state; one entry per config-arm round (round_id, tasks, proposals_total/kept/rejected/skipped, etc.).
    specialist_rounds: list[dict[str, Any]] = field(default_factory=list)
    # Intervention-mix ledger ({change_type∈{config,code_patch,config_attempt,code_patch_attempt}}); Robustness detects config-only loops.
    intervention_mix: list[dict[str, Any]] = field(default_factory=list)
    # Research scout bookkeeping; master switch ``--no-research-scout``.
    research_scout_enabled: bool = True
    research_scout_interval: int = 3
    # Master switch for advisory "External target gap" prompt block (``--no-target-advisory``); never gates Objective.
    target_advisory_enabled: bool = True
    # Master switch for sedimenting KEEP/REVERT provenance into the persistent recipe; off => recipe stays ephemeral.
    recipe_sediment_enabled: bool = True
    # Round id of last scout dispatch so K-round re-dispatch fires once per qualifying round.
    research_scout_last_round: int = -1
    # Static-recon specialist bookkeeping (explore-opt-5 capability A); master switch ``--no-static-recon``.
    static_recon_enabled: bool = True
    # Research-lane capacity locked at session start (core field; PolicyGate denies mid-session mutation).
    research_lane_capacity: int = 1
    # GPU pool capacity for needs_gpu specialists (0 disables); locked at session start.
    gpu_specialist_capacity: int = 0
    # The hint the FRAMEWORK_AGENT exit reads on the next phase decision; every phase transition clears it.
    pending_escalate_hint: str = ""
    # per-phase plateau threshold overrides locked at session start (CLI flags); empty => library defaults.
    plateau_overrides: dict[str, Any] = field(default_factory=dict)
    # E2E integrate bookkeeping keyed by kernel_id+patch_path+args; prevents re-validating the same patch after NEEDS_REVIEW/REVERT.
    kernel_integrate_attempts: dict[str, Any] = field(default_factory=dict)
    # Crash-safe stack-validation checkpoints (CYCLE_END-entry combo E2E).
    pending_stack_validation_result: dict[str, Any] = field(default_factory=dict)
    pending_stack_validation_apply_results: list[dict[str, Any]] = field(default_factory=list)
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
    # (PRELUDE/FRAMEWORK_AGENT/KERNEL_AGENT/CYCLE_END/CLOSE); Coordinator-only writer.
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
    # Cyclic phase machine macro-cycle counter (cycle 0 is the first pass; each CYCLE_END→FRAMEWORK_AGENT loopback
    # increments it).
    macro_cycle: int = 0
    # The macro_cycle whose CYCLE_END entry work finished; CYCLE_END exits only once it names the current cycle.
    cycle_end_settled_cycle: int = -1
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

    # The CYCLE_END handoff turn's result: ``next_cycle_directive``, its ``for_cycle``, and ``parse_error``. Coordinator-only writer.
    orchestration_memory: dict[str, Any] = field(default_factory=dict)

    # Non-field instance attr (set in load_or_init / save): session dir for breakdown instrumentation.
    _session_dir = None

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
    def from_dict(cls, raw: dict[str, Any]) -> "SharedState":
        """Construct state from a mapping, refusing invalid crash evidence with ``ValueError``."""
        cls._validate_resume_fields(raw)
        # Filter to known fields; unknown keys dropped, missing keys default.
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        # The charge anchor belongs to a live leg, not to the file: only
        # ``begin_leg`` arms it, so ``elapsed_charged_sec`` alone survives a load.
        filtered["leg_anchor_unix"] = 0.0
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

    @property
    def current_best_launch_args(self) -> str | None:
        """The complete server args the current best was launched with, as a serving recipe.

        Read from the promoted measurement's launch evidence: what the launch path composed from every layer --
        session base, operator ``--server-args``, the optimizer's delta -- where ``extra_server_args`` holds the
        delta alone. ``None`` when that launch was not recorded.
        """
        evidence = ((self.current_best or {}).get("measurement") or {}).get("launch_evidence") or {}
        launched = evidence.get("requested_server_args")
        return None if launched is None else strip_benchmark_harness_flags(launched)

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

    def accepted_baseline_script(self) -> str | None:
        """Benchmark script belonging to the accepted baseline anchor, or None."""
        if self.baseline_benchmark_script is not None:
            return self.baseline_benchmark_script or None
        if self.last_baseline.get("decision") != "promoted":
            return None
        fingerprint = (self.last_baseline.get("extras") or {}).get("fingerprint") or {}
        return str(fingerprint.get("benchmark_script") or "").strip() or None

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

    def record_action_attempt(self, action: str, attempt: ActionAttempt) -> dict[str, Any] | None:
        """Append one attempt to ``<action>_attempts`` and refresh ``last_<action>``. Entry schema {ts, task_id, status, decision, key_metric, key_metric_kind, workspace, error_class, error_excerpt, stderr_tail, stderr_log_path, raw_result_path, reported_success, variant_name, extras}. Returns the entry, or None when ``action`` not in the audit set (kernel_agent-owned actions use bespoke recorders). Does NOT call :meth:`save`."""
        if action not in _AUDIT_ACTIONS:
            return None
        attempts_attr = f"{action}_attempts"
        last_attr = f"last_{action}"
        result = attempt.result or {}
        metric_key, metric_kind = _KEY_METRIC_MAP[action]
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

    # No action-score API; ``increment_tick`` is a pure monotonic counter for plateau/phase budget math.
    def increment_tick(self) -> int:
        """Bump the Coordinator tick counter."""
        self.tick = int(self.tick or 0) + 1
        return self.tick

    def append_stack_gain_entry(self, *, new_tput: float) -> None:
        """Mirror an optimization_stack append into gain_per_stack_entry as ``(new_tput-baseline_tput)/baseline_tput*100``."""
        self.gain_per_stack_entry.append(gain_pct(to_float(new_tput, 0.0), to_float(self.baseline_tput, 0.0)))

    def closing_reserve_sec(self) -> float:
        """Seconds held back from every unit of work so CLOSE can still report."""
        return max(0.0, effective_closing_grace_sec(float(self.max_minutes or 0), self.closing_grace_sec))

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
