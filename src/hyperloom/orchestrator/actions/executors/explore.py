# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ExploreExecutor."""

from __future__ import annotations
from hyperloom.orchestrator.actions.executors._workload_envs import prepare_grid_benchmark_config
from hyperloom.common.benchmark_config import is_magpie_agentx
import functools
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hyperloom.common.coerce import to_str_list
from hyperloom.common.gain_math import gain_pct
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    VERDICT_REVERT,
    agentx_active,
    constraint_fields,
    graded_axes_of,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    stamp_output_per_gpu,
)
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...state.failure_evidence import (
    UNMEASURED_OUTCOMES,
    FAILURE_STAGE_DECISION,
    FAILURE_STAGE_WARMUP,
    classify_failure_attribution,
    make_failure_id,
    tail_excerpt,
)
from ...state.shared_state import (
    first_positive_tput,
    resolve_anchor_with_drift,
    resolve_graded_comparison,
    stack_base_params,
)
from ..stop_attribution import (
    StoppedByTheRun,
    stopped_by_the_run_class,
    variant_budget_stop,
)
from ._accuracy_gate import (
    accuracy_passed,
    parse_eval_results,
)
from hyperloom.inference_optimizer.canonical_fingerprint import workload_signature
from .measured_variants import GAIN_BELOW_THRESHOLD, MeasuredOn, measured_verdict, stack_of_measured
from ._grid_base import (
    TS_FAILED,
    TS_KILLED_OVERTIME,
    TS_SKIPPED_DEDUP,
)
from .benchmark_result import double_run_requested
from ._grid_runner import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    _MN_BACKENDS_PRIORITY,
    _MN_PARAMS_PRIORITY,
    GridVariant,
    SessionDirField,
    _build_variant_yaml,
    _num_gpus_for_config,
    apply_aiter_moe_pin_filter,
    apply_compatibility_filter,
    apply_multi_node_invalid_variants,
    apply_user_skip_list,
    reorder_grid_for_multi_node,
    resolve_skip_spec,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
    session_grid_bounds,
)
from hyperloom.inference_optimizer.grid_server_args import (
    compose_server_args,
    server_args_env_name,
)
from ._ray_serving import maybe_serving_lease
from ._spec_decode_context_copy import CONTEXT_COPY_REFUSAL

from ._launch_restatement import observed_launch_after
from ._server_lifecycle import (
    resolve_lifecycle_params,
    teardown_lifecycle_server,
)
from ._variant_grid import (
    _framework_lever_attributions,
    _grid_variants_from_payload,
    _seed_grid_payload,
    _seed_lever_variants,
    _skip_baseline_noop,
    _stamp_canonical_fingerprints,
    _variant_control_fields,
    canonical_fingerprint_of,
    ObservedLaunch,
    filter_baseline_noop_variants,
    filter_context_copy_drafts,
    filter_operator_pinned_envs,
    framework_lever_grid,
    observed_launch_from_state,
)
from ._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
)


log = logging.getLogger(__name__)


_now_iso = functools.partial(now_iso, "auto")

STACK_REVALIDATE_SOURCE: str = "stack_revalidate"

# ``skipped_dup`` reason for a variant whose launch the session already measured on the same stack.
ALREADY_MEASURED: str = "already_measured"


def is_stack_revalidation(params: dict | None) -> bool:
    return str((params or {}).get("source") or "") == STACK_REVALIDATE_SOURCE


# Audit/provenance metadata stashed on a GridVariant that must survive being rebuilt into a derived variant.
_CARRIED_VARIANT_ATTRS: tuple[str, ...] = (
    "provenance",
    "reasoning_origin",
    "scope",
    "overlay_pythonpath",
    "accepted_kernels",
    "experience_citations",
)


def _decision_fields(gv: Any) -> dict[str, Any]:
    """Why this variant was proposed, as authored at action time, for every row that records it."""
    return {
        "note": gv.note,
        "reasoning_origin": str(getattr(gv, "reasoning_origin", "") or ""),
        "experience_citations": list(getattr(gv, "experience_citations", []) or []),
    }


def _variant_with_eval_off(gv: Any) -> Any:
    """Return a copy of ``gv`` with ``RUN_EVAL`` forced off.

    The variant layer is the last one :func:`run_grid` applies over the
    accumulated stack, so forcing the key here holds whatever the stack carries.
    """
    envs = dict(gv.extra_envs)
    envs["RUN_EVAL"] = "false"
    return _carry_variant_metadata(
        gv,
        GridVariant(
            name=gv.name,
            extra_server_args=gv.extra_server_args,
            extra_envs=envs,
            note=gv.note,
            remove_args=to_str_list(getattr(gv, "remove_args", [])),
            unset_envs=to_str_list(getattr(gv, "unset_envs", [])),
            args_mode=str(getattr(gv, "args_mode", "append") or "append"),
        ),
    )


def _carry_variant_metadata(src: Any, dst: Any) -> Any:
    """Copy the carried audit metadata from ``src`` onto ``dst``."""
    for attr in _CARRIED_VARIANT_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    return dst


def _keep_uses_effective_args(
    benchmark: dict[str, Any], base_mode: str, variant_mode: str, remove_args: list[str], unset_envs: list[str]
) -> bool:
    """Native KEEP edits remain relative to the verified recipe unless explicitly replaced."""
    return (
        base_mode == "replace"
        or variant_mode == "replace"
        or (not is_magpie_agentx(benchmark) and bool(remove_args or unset_envs))
    )


class _RoundRefusal(Exception):
    """The round has nothing it can measure; ``result`` is what the action publishes instead."""

    def __init__(self, error_class: str, error: str, workspace: str = "") -> None:
        super().__init__(error)
        self.result: dict[str, Any] = {"status": "failed", "error_class": error_class, "error": error}
        if workspace:
            self.result["workspace"] = workspace


class _Workload(NamedTuple):
    """The materialized benchmark config every variant of a round launches from."""

    config_path: Path
    output_root: Path
    model_path: str
    gpu_type: str
    benchmark_script: str | None
    result_dir: str | None


class _Launch(NamedTuple):
    """What the materialized YAML already asks the server for."""

    framework: str
    cfg: dict[str, Any]
    yaml_envs: Mapping[str, Any]
    inherited_args: str


class _RoundBaseline(NamedTuple):
    """The stack and the figures a round opened on, as the task's params described them."""

    tput: float
    extra_args: str
    extra_envs: dict[str, str]
    remove_args: list[str]
    unset_envs: list[str]
    args_mode: str
    accuracy: float


class _RoundSetup(NamedTuple):
    """What the task asked this round to do, resolved."""

    params: dict[str, Any]
    shared_state: Any
    workload: _Workload
    launch: _Launch
    baseline: _RoundBaseline


class _ServerReuse(NamedTuple):
    """Whether a variant's decision round can re-attach to the server its warmup left hot.

    Multi-node, a non-builtin script or a profiler-on round is ineligible and
    falls back to a cold decision round.
    """

    eligible: bool
    framework: str
    port: int

    @classmethod
    def of(cls, lifecycle: Mapping[str, Any]) -> "_ServerReuse":
        """Read eligibility off the materialized YAML's ``server_lifecycle`` block."""
        return cls(
            bool(lifecycle.get("eligible")),
            str(lifecycle.get("framework") or ""),
            int(lifecycle.get("port") or 0),
        )

    def lifecycle_params(self, slot: Path) -> dict[str, Any] | None:
        """The ``server_lifecycle`` block a variant's warmup and decision rounds share."""
        return {"cleanup": False, "pid_dir": str(slot), "port": self.port} if self.eligible else None


class _RoundLaunch(NamedTuple):
    """One benchmark round of one variant, as :func:`run_grid` takes it."""

    variant: GridVariant
    slot: Path
    lifecycle: dict[str, Any] | None
    expected_sec: float | None
    server_ready: bool
    ready_server_log: str | None


class _VariantRun(NamedTuple):
    """One variant of a round, composed as it will actually launch."""

    idx: int
    gv: GridVariant
    fingerprint: str
    provenance: str
    scope: str
    control_fields: dict[str, Any]
    slot: Path
    remove_args: list[str]
    unset_envs: list[str]
    extra_envs: dict[str, str]
    launched_gv: GridVariant
    decision_gv: GridVariant
    lifecycle: dict[str, Any] | None


class _GradedVerdict(NamedTuple):
    """What the graded comparison made of one variant's measurement."""

    outcome: str
    reason: str
    gain: float | None
    graded_gain: float | None


class _AccuracyVerdict(NamedTuple):
    """What the accuracy gate made of one variant's eval output."""

    passed: bool
    value: Any
    reference: float | None
    gated: bool


class _VariantVerdict(NamedTuple):
    """Everything a round decided about one variant it measured."""

    graded: Any
    measured: dict[str, Any]
    outcome: str
    reason: str
    gain: float | None
    graded_gain: float | None
    gates: list[dict[str, Any]]
    accuracy: _AccuracyVerdict
    tput: float | None


class _KeepComposition(NamedTuple):
    """The stack a KEEP leaves behind for the rest of the round."""

    effective_args: str
    stack_args: str
    control_fields: dict[str, Any]
    persist_effective: bool


@dataclass
class _VariantStack:
    """The configuration each variant launches on top of, as the round has built it.

    Every KEEP folds its variant in, so a variant later in the round launches on --
    and is graded against -- a different stack than the round opened with.
    """

    extra_args: str
    extra_envs: dict[str, str]
    remove_args: list[str]
    unset_envs: list[str]
    args_mode: str
    tput: float
    perf: Any


@dataclass
class _RoundLedger:
    """What a round records about every variant it touched.

    ``apply_explore_search_update`` accumulates these over the durable ledger.
    """

    skipped: list[dict[str, Any]] = field(default_factory=list)
    winners: list[dict[str, Any]] = field(default_factory=list)
    tested: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    name_index: dict[str, Any] = field(default_factory=dict)
    winners_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _RoundBudget:
    """What a round may spend on each variant, and what stopped it when it ran out.

    A variant that never ran is not a variant that failed, so the round reports
    the stop rather than a bare, unattributed failure.
    """

    session_deadline_sec: float | None
    phase_deadline: Any
    warmup_expected_sec: float | None
    decision_expected_sec: float | None
    use_warm_decision: bool
    stop: StoppedByTheRun | None = None
    detail: str = ""
    untested: int = 0


def _float_param(params: Mapping[str, Any], key: str) -> float:
    """A wall-clock task param in seconds, or ``0.0`` when it is absent or not a number."""
    raw = params.get(key)
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _refresh_anchor_params(params: dict[str, Any], shared_state: Any) -> None:
    """Re-read the anchor and the stack it was measured on, as a pair."""
    snapshot_tput = float(params.get("base_tput") or 0.0)
    anchor, anchor_drifted = (
        (snapshot_tput, False)
        if params.get("source") == STACK_REVALIDATE_SOURCE
        else resolve_anchor_with_drift(snapshot_tput, shared_state)
    )
    if anchor <= snapshot_tput:
        return
    if anchor_drifted:
        log.warning("explore: anchor drift %.1f -> %.1f; re-reading base args", snapshot_tput, anchor)
    params["base_tput"] = anchor
    current_best = getattr(shared_state, "current_best", None)
    # Only current_best carries args; a baseline_tput anchor leaves the params stack (seeded from the baseline
    # record) authoritative.
    if first_positive_tput(current_best) > 0:
        params.update(stack_base_params(current_best))


def _round_baseline(params: Mapping[str, Any], shared_state: Any) -> _RoundBaseline:
    """The stack and the figures the round opens on."""
    # The measured baseline outranks a proposed one.
    accuracy = float(getattr(shared_state, "baseline_accuracy", 0.0) or 0.0) if shared_state is not None else 0.0
    if accuracy <= 0:
        accuracy = float(params.get("accuracy_baseline") or 0.0) or float(params.get("baseline_accuracy") or 0.0)
    return _RoundBaseline(
        tput=float(params.get("base_tput") or 0.0),
        extra_args=str(params.get("base_extra_args") or "").strip(),
        extra_envs=dict(params.get("base_extra_envs") or {}),
        remove_args=to_str_list(params.get("base_remove_args")),
        unset_envs=to_str_list(params.get("base_unset_envs")),
        args_mode=str(params.get("base_args_mode") or "append").strip().lower(),
        accuracy=accuracy,
    )


def _launch_defaults(config_path: Path, base_args_mode: str) -> _Launch:
    """Resolve the framework and the server args the materialized YAML already carries."""
    cfg: dict[str, Any] = {}
    try:
        with config_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        framework = str((cfg.get("benchmark") or {}).get("framework") or "").lower()
        yaml_envs = (cfg.get("benchmark") or {}).get("envs") or {}
        inherited_args = str(yaml_envs.get(server_args_env_name(framework)) or "").strip()
    except (OSError, yaml.YAMLError) as exc:
        log.warning("explore: could not resolve framework from %s: %s", config_path, exc)
        return _Launch("", cfg, {}, "")
    return _Launch(framework, cfg, yaml_envs, "" if base_args_mode == "replace" else inherited_args)


def _round_budget(params: Mapping[str, Any], extra: Mapping[str, Any], *, reuse_eligible: bool) -> _RoundBudget:
    """What one variant of this round is expected to cost, and how long the round may keep spending."""
    baseline_runtime_sec = _float_param(params, "baseline_runtime_sec")
    baseline_warm_runtime_sec = _float_param(params, "baseline_warm_runtime_sec")
    use_warm_decision = reuse_eligible and double_run_requested(params)
    decision_anchor_sec = (
        baseline_warm_runtime_sec if (use_warm_decision and baseline_warm_runtime_sec > 0) else baseline_runtime_sec
    )
    session_deadline_sec, session_expected_sec = session_grid_bounds(extra.get("shared_state") or extra.get("state"))
    return _RoundBudget(
        session_deadline_sec=session_deadline_sec,
        phase_deadline=extra.get("phase_deadline"),
        warmup_expected_sec=(baseline_runtime_sec if baseline_runtime_sec > 0 else None) or session_expected_sec,
        decision_expected_sec=(decision_anchor_sec if decision_anchor_sec > 0 else None) or session_expected_sec,
        use_warm_decision=use_warm_decision,
    )


def _round_grading_anchor_perf(shared_state: Any) -> Any:
    """The figure every variant of this round is graded against, on the interactivity axis."""
    running_base_perf, anchor_reason = resolve_grading_anchor_perf(shared_state)
    if running_base_perf is None:
        # AgentX grading needs both axes on the round anchor. Without them every
        # variant fails closed below; output figures are diagnostic only and must
        # not become throughput KEEPs.
        log.info("explore: AgentX grading unavailable for this round (%s)", anchor_reason)
    return running_base_perf


def _measured_axes(r: Any, shared_state: Any) -> dict[str, Any]:
    """The axes the graded comparison rules on, as one variant measured them."""
    measured = {
        GRADED_OUTPUT: r.output_throughput,
        "input_throughput": r.input_throughput,
        "total_throughput": r.total_token_throughput,
        GRADED_INTVTY: r.intvty_p90,
        "tpot_p90_ms": r.tpot_p90_ms,
        GRADED_INTVTY_P50: r.intvty_p50,
        GRADED_DURATION: r.duration_seconds,
        GRADED_ERROR_RATE: r.request_error_rate,
        **constraint_fields(r),
    }
    stamp_output_per_gpu(measured, getattr(shared_state, "tp", None))
    return measured


def _graded_axes_moved(graded: Any) -> str:
    """How far each graded axis moved, for the refusal the ledger renders."""
    return (
        f"intvty {graded.reference:.1f}->{graded.candidate:.1f} "
        f"tput {graded.tput_reference:.1f}->{graded.tput_candidate:.1f}"
    )


def _refused_by(graded: Any) -> str:
    """Which guard refused the variant."""
    if graded.veto_reason:
        return graded.veto_reason
    if graded.graded_on_intvty:
        return f"{graded.refused_by} ({_graded_axes_moved(graded)})"
    return GAIN_BELOW_THRESHOLD


def _graded_decision(graded: Any, r: Any, variant_name: str) -> _GradedVerdict:
    """Rule on a variant's measurement, before the accuracy gate sees it."""
    if r.status != "succeeded":
        return _GradedVerdict(TS_FAILED, (r.error or "")[-1200:] or "no_measurement", None, None)
    if graded.degrade_reason:
        # Same fail-closed rule as ``lift_to_current_best``: an AgentX session
        # that could not grade on interactivity does not KEEP on output
        # throughput instead.
        log.info("explore: variant %r not comparable (%s)", variant_name, graded.degrade_reason)
        return _GradedVerdict(TS_FAILED, graded.degrade_reason, None, None)
    if graded.verdict == VERDICT_REVERT:
        moved = gain_pct(graded.candidate, graded.reference) if graded.reference > 0 else None
        return _GradedVerdict("REVERT", _refused_by(graded), None, moved)
    return _GradedVerdict(TS_FAILED, "", gain_pct(graded.candidate, graded.reference), None)


def _graded_gate_row(graded: Any, reason: str, keep_threshold_pct: float) -> dict[str, Any]:
    """The graded comparison's own verdict, as the decision-gate row it is."""
    on_intvty = graded.graded_on_intvty
    return {
        "gate": (
            graded.vetoed_by
            if graded.veto_reason
            else ("graded_axes" if (on_intvty or graded.degrade_reason) else "keep_threshold")
        ),
        "passed": (False if graded.degrade_reason else graded.verdict != VERDICT_REVERT),
        "observed": (graded.candidate if on_intvty else gain_pct(graded.candidate, graded.reference)),
        "threshold": graded.reference if on_intvty else keep_threshold_pct,
        "reason": reason,
    }


def _accuracy_refusal(value: Any) -> str:
    """Why the accuracy gate refused the variant."""
    return "accuracy_unavailable" if value is None else "accuracy_drop"


def _accuracy_verdict(slot: Path, *, framework: str, benchmark_mode: str, reference: float) -> _AccuracyVerdict:
    """Rule on the variant's eval output, when there is a reference to rule it against."""
    from hyperloom.inference_optimizer import framework_registry

    scriptable = framework_registry.is_scriptable(framework)
    if not (scriptable or reference > 0):
        return _AccuracyVerdict(True, None, None, False)
    value = parse_eval_results(slot, framework=framework, benchmark_mode=benchmark_mode).get("accuracy")
    if not isinstance(value, (int, float)):
        return _AccuracyVerdict(False, value, None, True)
    against = 1.0 if scriptable else reference
    return _AccuracyVerdict(accuracy_passed(against, float(value)), value, against, True)


def _accuracy_gate_row(accuracy: _AccuracyVerdict) -> dict[str, Any]:
    """The accuracy gate's verdict, as the decision-gate row it is."""
    return {
        "gate": "accuracy",
        "passed": accuracy.passed,
        "observed": accuracy.value if isinstance(accuracy.value, (int, float)) else None,
        "threshold": accuracy.reference,
        "reason": "" if accuracy.passed else _accuracy_refusal(accuracy.value),
    }


def _variant_identity(run: _VariantRun) -> dict[str, Any]:
    """Which variant a ledger row is about, as every row that records one states it."""
    return {
        "fingerprint": run.fingerprint,
        "name": run.gv.name,
        "extra_server_args": run.gv.extra_server_args,
        "extra_envs": dict(run.gv.extra_envs),
        **run.control_fields,
        **_decision_fields(run.gv),
    }


def _tested_row(explore_round: "_ExploreRound", run: _VariantRun, r: Any, verdict: _VariantVerdict) -> dict[str, Any]:
    """The ledger row for a variant this round measured and ruled on."""
    return {
        **_variant_identity(run),
        "outcome": verdict.outcome,
        "status": r.status,
        "tput": verdict.tput,
        "decision_tput": verdict.tput,
        **graded_axes_of(verdict.measured),
        "gain_pct": verdict.gain,
        "graded_gain_pct": verdict.graded_gain,
        "graded_objective": verdict.graded.objective,
        # Which gate refused the variant. The prompt ledger renders this row, and without the reason a
        # REVERT on the tail guard is indistinguishable from one the benchmark never measured.
        "reason": verdict.reason,
        "base_tput": explore_round.stack.tput,
        "round_id": explore_round.round_id,
        "ts": _now_iso(),
        "provenance": run.provenance,
        "scope": run.scope,
        "workload_signature": explore_round.workload_signature,
        "framework": explore_round.launch.framework,
        "workspace": r.workspace,
        "error_class": r.error_class or "",
        "server_log_path": r.server_log_path,
        "launch_evidence": dict(r.launch_evidence or {}),
        "launch_evidence_path": r.launch_evidence_path,
        "stage": FAILURE_STAGE_DECISION,
        "measured_against": explore_round.measured_against(),
        "gates": verdict.gates,
        # What stood behind an adoption. A session with no baseline accuracy
        # gates nothing, so its KEEPs rest on throughput alone -- a weaker claim
        # than one an accuracy gate ruled on, and the two must not read alike.
        # Only a KEEP carries it, matching the field's adoption-scoped meaning
        # elsewhere: on a reverted row "accuracy_pass" would name the gate that
        # refused it. What ruled against those is in ``gates``.
        "validation_basis": (
            ("accuracy_pass" if verdict.accuracy.gated else "keep_verdict_unscored")
            if verdict.outcome == "KEEP"
            else ""
        ),
    }


def _warmup_failure_row(explore_round: "_ExploreRound", run: _VariantRun, w: Any) -> dict[str, Any]:
    """The ledger row for a variant whose warmup round failed before anything graded it."""
    return {
        **_variant_identity(run),
        "outcome": TS_FAILED,
        "status": getattr(w, "status", "failed") if w is not None else "failed",
        "tput": None,
        "gain_pct": None,
        "base_tput": explore_round.stack.tput,
        "round_id": explore_round.round_id,
        "ts": _now_iso(),
        "provenance": run.provenance,
        "workload_signature": explore_round.workload_signature,
        "framework": explore_round.launch.framework,
        "reason": "warmup_failed",
        "error_class": w.error_class if w is not None else "",
        "server_log_path": w.server_log_path if w is not None else None,
        "launch_evidence": dict(w.launch_evidence or {}) if w is not None else {},
        "launch_evidence_path": w.launch_evidence_path if w is not None else None,
        "stage": FAILURE_STAGE_WARMUP,
        "error_excerpt": tail_excerpt(w.error) if w is not None else None,
        "workspace": w.workspace if w is not None else None,
        "raw_result_path": w.raw_result_path if w is not None else None,
        # It launched on a stack even though it never reached a gate, so no gates
        # are recorded and nothing validated it.
        "measured_against": explore_round.measured_against(),
    }


def _keep_entry(
    explore_round: "_ExploreRound", run: _VariantRun, r: Any, verdict: _VariantVerdict, comp: _KeepComposition
) -> dict[str, Any]:
    """The adoption record for a variant this round KEPT."""
    return {
        "fingerprint": run.fingerprint,
        "name": run.gv.name,
        "candidate_extra_server_args": run.gv.extra_server_args,
        "candidate_extra_envs": dict(run.gv.extra_envs or {}),
        "recipe_delta": {
            "extra_server_args": run.gv.extra_server_args,
            "extra_envs": dict(run.gv.extra_envs or {}),
            **run.control_fields,
        },
        "extra_server_args": comp.stack_args,
        "effective_extra_server_args": comp.effective_args,
        "extra_envs": dict(run.extra_envs),
        **comp.control_fields,
        **_decision_fields(run.gv),
        "provenance": run.provenance,
        # Names of the authored kernels this config carried, when an overlay was loaded.
        "accepted_kernels": list(getattr(run.gv, "accepted_kernels", []) or []),
        "gain_pct": verdict.gain,
        "graded_objective": verdict.graded.objective,
        # The verdict this KEEP rests on. ``None`` means the variant was not gated (not
        # high-risk, or no baseline) rather than that it scored nothing.
        "accuracy": verdict.accuracy.value,
        "tput": verdict.tput,
        "decision_tput": verdict.tput,
        # The axes this KEEP was graded on travel with it: current_best becomes the next round's
        # anchor, and an anchor without them degrades the session.
        **graded_axes_of(verdict.measured),
        # Promotion re-checks the budgets against this row, not the round's measurement.
        **constraint_fields(r),
        "single_workspace": r.workspace,
        "launch_evidence": dict(r.launch_evidence or {}),
        "launch_evidence_path": r.launch_evidence_path,
        "round_id": explore_round.round_id,
        "accepted_at_round": explore_round.round_id,
        "ts": _now_iso(),
    }


@dataclass
class _ExploreRound:
    """One explore round: the grid it measures, the stack it builds, the ledger it writes."""

    round_id: str
    workload: _Workload
    launch: _Launch
    baseline: _RoundBaseline
    shared_state: Any
    runnable: list[GridVariant]
    lever_payload: list[dict[str, Any]]
    stack: _VariantStack
    ledger: _RoundLedger
    budget: _RoundBudget
    reuse: _ServerReuse
    keep_threshold_pct: float
    grade_on_intvty: bool
    on_the_synthetic_benchmark: bool
    is_revalidation: bool
    workload_signature: str
    observed: ObservedLaunch
    # The launch the round opened on, and the fingerprints it has since put on the
    # server. A variant only skipped -- a restatement of the stack, or a verdict the
    # session already holds -- has not claimed the launch, so its twin still gets a turn.
    opening_stack: dict[str, Any] = field(init=False)
    launched_fps: set[str] = field(default_factory=set)
    judged_without_measuring: int = 0
    serving_lease: Any = None

    def __post_init__(self) -> None:
        self.opening_stack = self.launched_stack()

    def launched_stack(self) -> dict[str, Any]:
        """The stack as it launches: its args with the recipe's inherited ones and the removals applied."""
        stack = stack_of_measured(self.measured_against())
        stack["extra_server_args"] = compose_server_args(
            inherited_args=self.launch.inherited_args,
            base_extra_args=self.stack.extra_args,
            remove_args=self.stack.remove_args,
            args_mode=self.stack.args_mode,
        )
        return stack

    def measured_against(self) -> dict[str, Any]:
        """The stack this variant launched on top of, as it stands now."""
        return {
            "throughput": self.stack.tput,
            "accuracy": self.baseline.accuracy or None,
            "extra_server_args": self.stack.extra_args,
            "extra_envs": dict(self.stack.extra_envs),
            "remove_args": list(self.stack.remove_args),
            "unset_envs": list(self.stack.unset_envs),
            "args_mode": self.stack.args_mode,
        }

    def variant_cost_sec(self) -> float:
        """What admitting one more variant costs in wall-clock."""
        if self.budget.decision_expected_sec is None:
            from ._subprocess_kill import resolve_benchmark_timeouts

            return resolve_benchmark_timeouts()[1]
        warmup_sec = float(self.budget.warmup_expected_sec or 0.0) if self.budget.use_warm_decision else 0.0
        return float(self.budget.decision_expected_sec) + warmup_sec

    def stop_before_variant(self, stop: StoppedByTheRun, *, idx: int, needed_sec: float) -> None:
        """Record that the session budget cannot pay for the rest of the grid."""
        self.budget.stop = stop
        self.budget.detail = stop.never_started
        self.budget.untested = len(self.runnable) - idx
        log.warning(
            "explore: cannot fit another variant (%s, needs %.0fs); stopping after %d/%d variant(s)",
            stop.error_class,
            needed_sec,
            idx,
            len(self.runnable),
        )

    def stopped_by_the_run(self, result: Any, *, variant: GridVariant, idx: int, round_label: str) -> bool:
        """Whether the run stopped this round, and record it if it did."""
        stopped = stopped_by_the_run_class(getattr(result, "error_class", "") if result is not None else "")
        if stopped is None:
            return False
        self.budget.stop = stopped
        self.budget.detail = stopped.interrupted
        self.budget.untested = len(self.runnable) - idx
        log.warning(
            "explore: the %s round of variant %s was stopped by the run (%s); it and the "
            "%d variant(s) after it stay out of the ledger so a resume can retry them",
            round_label,
            variant.name,
            stopped.error_class,
            self.budget.untested - 1,
        )
        return True

    def launched_already(self, gv: GridVariant) -> bool:
        """Whether this round already launched ``gv``'s fingerprint, recording the duplicate if so."""
        fp = canonical_fingerprint_of(gv)
        if fp not in self.launched_fps:
            return False
        self.ledger.skipped.append({"name": gv.name, "fingerprint": fp, "reason": "round_dup"})
        return True

    def record_launch(self, gv: GridVariant) -> None:
        """Claim the launch for ``gv``'s fingerprint, so a later twin of it is a round duplicate."""
        self.launched_fps.add(canonical_fingerprint_of(gv))

    def restates_the_stack(self, gv: GridVariant) -> bool:
        """Whether ``gv`` only restates the launch the stack runs as it stands, recording the skip if so."""
        config, env = observed_launch_after(
            self.observed.config,
            self.observed.env,
            self.opening_stack,
            self.launched_stack(),
            self.launch.framework,
        )
        _, restated = filter_baseline_noop_variants(
            [gv],
            framework=self.launch.framework,
            build_variant_yaml=_build_variant_yaml,
            base_yaml_path=self.workload.config_path,
            base_extra_args=self.stack.extra_args,
            base_extra_envs=dict(self.stack.extra_envs),
            base_remove_args=list(self.stack.remove_args),
            base_unset_envs=list(self.stack.unset_envs),
            base_args_mode=self.stack.args_mode,
            model_path=self.workload.model_path,
            gpu_type=self.workload.gpu_type,
            benchmark_script=self.workload.benchmark_script,
            observed=ObservedLaunch(config, env),
        )
        if not restated:
            return False
        _skip_baseline_noop(gv.name, restated[0][1], self.ledger.skipped)
        self.judged_without_measuring += 1
        return True

    def already_measured(self, gv: GridVariant) -> bool:
        """Whether the session ledger already holds a verdict for ``gv`` on this stack, recording the skip if so."""
        if self.shared_state is None or self.is_revalidation or getattr(gv, "overlay_pythonpath", ""):
            return False
        fp = canonical_fingerprint_of(gv)
        prior = measured_verdict(
            getattr(self.shared_state, "explore_search", None),
            fp,
            variant={"extra_args": gv.extra_server_args, "extra_envs": gv.extra_envs, **_variant_control_fields(gv)},
            now=MeasuredOn(
                stack_of_measured(self.measured_against()), self.workload_signature, self.keep_threshold_pct
            ),
        )
        if prior is None:
            return False
        detail = (
            f"measured as {prior.get('name') or fp} in {prior.get('round_id') or 'an earlier round'} on this "
            f"stack: {prior.get('outcome')}"
        )
        log.info("explore: skipping already-measured variant %s (%s)", gv.name, detail)
        self.ledger.skipped.append({"name": gv.name, "fingerprint": fp, "reason": ALREADY_MEASURED, "detail": detail})
        self.judged_without_measuring += 1
        return True

    def context_copy_refusal(self, gv: GridVariant) -> str:
        """Why this draft's copied context is refused on the stack as it stands, or ``""``."""
        if not self.on_the_synthetic_benchmark or self.is_revalidation:
            return ""
        _, copy_refused = filter_context_copy_drafts(
            [gv],
            build_variant_yaml=_build_variant_yaml,
            base_yaml_path=self.workload.config_path,
            base_extra_args=self.stack.extra_args,
            model_path=self.workload.model_path,
            gpu_type=self.workload.gpu_type,
            benchmark_script=self.workload.benchmark_script,
            base_args_mode=self.stack.args_mode,
            base_extra_envs=dict(self.stack.extra_envs),
            base_remove_args=list(self.stack.remove_args),
            base_unset_envs=list(self.stack.unset_envs),
        )
        return copy_refused[0][1] if copy_refused else ""

    def teardown_variant_server(self, slot: Path) -> None:
        """Reap one variant's persistent server; idempotent, and a no-op when reuse was ineligible."""
        if self.reuse.eligible:
            teardown_lifecycle_server(
                pid_dir=slot,
                framework=self.reuse.framework,
                port=self.reuse.port,
            )

    def record_warmup_failure(self, run: _VariantRun, w: Any) -> None:
        """Record a variant whose warmup round failed, so the round skips its decision round."""
        log.warning(
            "explore: variant %s warmup round failed (%s); skipping decision round.",
            run.gv.name,
            (getattr(w, "error", "") or "")[-200:] if w is not None else "no_result",
        )
        self.ledger.tested[run.fingerprint] = _warmup_failure_row(self, run, w)
        if run.gv.name:
            self.ledger.name_index[run.gv.name] = run.fingerprint
        self.ledger.rejected.append(
            {
                **_variant_identity(run),
                "reason": "warmup_failed",
                "gain_pct": None,
                "tput": None,
                "round_id": self.round_id,
                "ts": _now_iso(),
                "provenance": run.provenance,
                "stage": FAILURE_STAGE_WARMUP,
                "error_excerpt": tail_excerpt(w.error) if w is not None else None,
                "workspace": w.workspace if w is not None else None,
            }
        )

    def record_measurement(self, run: _VariantRun, r: Any, verdict: _VariantVerdict) -> None:
        """Record what this round measured and decided about one variant."""
        self.ledger.tested[run.fingerprint] = _tested_row(self, run, r, verdict)
        if run.gv.name:
            self.ledger.name_index[run.gv.name] = run.fingerprint

    def reject(self, run: _VariantRun, r: Any, verdict: _VariantVerdict) -> None:
        """Record a variant this round measured and refused."""
        self.ledger.rejected.append(
            {
                **_variant_identity(run),
                "reason": verdict.reason or "not_keep",
                "gain_pct": verdict.gain,
                "tput": verdict.tput,
                "round_id": self.round_id,
                "ts": _now_iso(),
                "provenance": run.provenance,
                "error_class": r.error_class or "",
                "server_log_path": r.server_log_path,
                "workspace": r.workspace,
            }
        )

    def adopt(self, run: _VariantRun, r: Any, verdict: _VariantVerdict) -> None:
        """Fold a KEEP'd variant onto the stack the rest of the round launches on."""
        comp = self._keep_composition(run)
        self.ledger.winners.append(_keep_entry(self, run, r, verdict, comp))
        self.stack = _VariantStack(
            extra_args=comp.stack_args,
            extra_envs=dict(run.extra_envs),
            remove_args=list(run.remove_args),
            unset_envs=list(run.unset_envs),
            args_mode="replace" if comp.persist_effective else "append",
            tput=verdict.tput if verdict.tput and verdict.tput > 0 else self.stack.tput,
            # Only a comparable intvty KEEP advances the round anchor. Degraded
            # measurements never reach here.
            perf=perf_snapshot_from_mapping(verdict.measured) if self.grade_on_intvty else self.stack.perf,
        )
        self.ledger.winners_history.append(
            {
                "round_id": self.round_id,
                "variant_name": run.gv.name,
                "fingerprint": run.fingerprint,
                "gain_pct": verdict.gain,
                "extra_args": run.gv.extra_server_args,
                "extra_envs": dict(run.gv.extra_envs),
                **run.control_fields,
                "provenance": run.provenance,
                "scope": run.scope,
                "ts": _now_iso(),
            }
        )

    def _keep_composition(self, run: _VariantRun) -> _KeepComposition:
        """Compose the server args and controls a KEEP'd variant leaves on the stack."""
        gv = run.gv
        effective_args = compose_server_args(
            inherited_args=self.launch.inherited_args,
            base_extra_args=self.stack.extra_args,
            variant_extra_args=gv.extra_server_args,
            remove_args=run.remove_args,
            args_mode="replace" if self.stack.args_mode == "replace" else gv.args_mode,
        )
        stack_args = compose_server_args(
            base_extra_args=self.stack.extra_args,
            variant_extra_args=gv.extra_server_args,
            remove_args=gv.remove_args,
            args_mode=gv.args_mode,
        )
        control_fields = dict(run.control_fields)
        if run.remove_args:
            control_fields["remove_args"] = list(run.remove_args)
        if run.unset_envs:
            control_fields["unset_envs"] = list(run.unset_envs)
        persist_effective = _keep_uses_effective_args(
            self.launch.cfg.get("benchmark") or {},
            self.stack.args_mode,
            gv.args_mode,
            run.remove_args,
            run.unset_envs,
        )
        if persist_effective:
            control_fields["args_mode"] = "replace"
        return _KeepComposition(
            effective_args,
            effective_args if persist_effective else stack_args,
            control_fields,
            persist_effective,
        )

    def answered_its_grid(self) -> bool:
        """Whether the round has a verdict for the grid it was given, winner or not."""
        measured = any(t.get("outcome") in ("KEEP", "REVERT", TS_KILLED_OVERTIME) for t in self.ledger.tested.values())
        answered_otherwise = bool(self.runnable) and self.judged_without_measuring == len(self.runnable)
        return measured or bool(self.ledger.winners) or answered_otherwise

    async def measure_its_grid(self) -> None:
        """Measure every variant the round can still afford, in the order the grid gives them."""
        if not self.runnable:
            return
        self.serving_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(self.workload.config_path))
        try:
            for idx, gv in enumerate(self.runnable):
                if self.launched_already(gv) or self.restates_the_stack(gv) or self.already_measured(gv):
                    continue
                needed_sec = self.variant_cost_sec()
                stop = variant_budget_stop(self.budget.session_deadline_sec, self.budget.phase_deadline, needed_sec)
                if stop is not None:
                    self.stop_before_variant(stop, idx=idx, needed_sec=needed_sec)
                    break
                refusal = self.context_copy_refusal(gv)
                if refusal:
                    log.warning("explore: refusing variant %s (%s)", gv.name, refusal)
                    self.ledger.skipped.append({"name": gv.name, "reason": CONTEXT_COPY_REFUSAL, "detail": refusal})
                    continue
                self.record_launch(gv)
                run = self._stage_variant(idx, gv)
                try:
                    await self._measure_variant(run)
                finally:
                    self.teardown_variant_server(run.slot)
                if self.budget.stop is not None:
                    break
        finally:
            if self.serving_lease is not None:
                self.serving_lease.close()

    def _stage_variant(self, idx: int, gv: GridVariant) -> _VariantRun:
        """Compose the variant as it will actually launch, on the stack as it stands now."""
        stack = self.stack
        if stack.args_mode == "replace":
            remove_args = to_str_list(getattr(gv, "remove_args", []))
        else:
            remove_args = list(dict.fromkeys(stack.remove_args + to_str_list(getattr(gv, "remove_args", []))))
        unset_envs = list(dict.fromkeys(stack.unset_envs + to_str_list(getattr(gv, "unset_envs", []))))
        extra_envs = dict(stack.extra_envs)
        for key in gv.unset_envs:
            extra_envs.pop(key, None)
        extra_envs.update(gv.extra_envs)
        run_gv = _carry_variant_metadata(
            gv,
            GridVariant(
                name=gv.name,
                extra_server_args=gv.extra_server_args,
                extra_envs=extra_envs,
                note=gv.note,
                remove_args=list(gv.remove_args),
                unset_envs=unset_envs,
                args_mode=str(getattr(gv, "args_mode", "append") or "append"),
            ),
        )
        # The decision round skips eval when a warmup preceded it: it is timed
        # against a throughput-only anchor, and ``parse_eval_results`` falls back
        # to the score the warmup took. Without a warmup there is nothing to fall
        # back to, so the decision round keeps its own eval.
        slot = self.workload.output_root / f"v{idx:02d}_{_safe(gv.name)}"
        slot.mkdir(parents=True, exist_ok=True)
        return _VariantRun(
            idx=idx,
            gv=gv,
            fingerprint=canonical_fingerprint_of(gv),
            provenance=getattr(gv, "provenance", "llm_direct"),
            scope=str(getattr(gv, "scope", "") or ""),
            control_fields=_variant_control_fields(gv),
            slot=slot,
            remove_args=remove_args,
            unset_envs=unset_envs,
            extra_envs=extra_envs,
            launched_gv=run_gv,
            decision_gv=_variant_with_eval_off(run_gv) if self.budget.use_warm_decision else run_gv,
            lifecycle=self.reuse.lifecycle_params(slot),
        )

    async def _launch_round(self, launch: _RoundLaunch) -> list[Any]:
        """Run one benchmark round of one variant, on the stack as it stands now."""
        workload = self.workload
        stack = self.stack
        return await run_grid(
            base_yaml_path=workload.config_path,
            base_extra_args=stack.extra_args,
            grid=[launch.variant],
            output_root=launch.slot,
            model_path=workload.model_path,
            gpu_type=workload.gpu_type,
            benchmark_script=workload.benchmark_script,
            result_dir=workload.result_dir,
            server_lifecycle=launch.lifecycle,
            base_args_mode=stack.args_mode,
            base_extra_envs=dict(stack.extra_envs),
            base_remove_args=list(stack.remove_args),
            base_unset_envs=list(stack.unset_envs),
            server_already_ready=launch.server_ready,
            ready_server_log=launch.ready_server_log,
            serving_lease=self.serving_lease,
            session_deadline_sec=self.budget.session_deadline_sec,
            variant_expected_sec=launch.expected_sec,
        )

    async def _warm_the_server(self, run: _VariantRun) -> Any:
        """Run the warmup round that leaves the server hot for the decision round."""
        warmup_slot = run.slot / "warmup_round"
        warmup_slot.mkdir(parents=True, exist_ok=True)
        warmup_results = await self._launch_round(
            _RoundLaunch(
                variant=run.launched_gv,
                slot=warmup_slot,
                lifecycle=run.lifecycle,
                expected_sec=self.budget.warmup_expected_sec,
                server_ready=False,
                ready_server_log=None,
            ),
        )
        return warmup_results[0] if warmup_results else None

    async def _measure_variant(self, run: _VariantRun) -> None:
        """Measure one variant and record what the round decided about it."""
        warm = self.budget.use_warm_decision
        ready_server_log = None
        if warm:
            w = await self._warm_the_server(run)
            if self.stopped_by_the_run(w, variant=run.gv, idx=run.idx, round_label="warmup"):
                return
            if w is None or getattr(w, "status", "") != "succeeded":
                self.record_warmup_failure(run, w)
                return
            ready_server_log = w.server_log_path
        results = await self._launch_round(
            _RoundLaunch(
                variant=run.decision_gv,
                slot=run.slot,
                lifecycle=run.lifecycle,
                expected_sec=self.budget.decision_expected_sec,
                server_ready=warm,
                ready_server_log=ready_server_log,
            ),
        )
        if not results:
            # run_grid returns one result per grid entry.
            log.warning("explore: variant %s produced no result", run.gv.name)
            return
        r = results[0]
        if self.stopped_by_the_run(r, variant=run.gv, idx=run.idx, round_label="decision"):
            return
        verdict = self._judge_variant(run, r)
        self.record_measurement(run, r, verdict)
        if verdict.outcome == "KEEP":
            self.adopt(run, r, verdict)
        else:
            self.reject(run, r, verdict)

    def _judge_variant(self, run: _VariantRun, r: Any) -> _VariantVerdict:
        """Rule on one variant: the graded axes first, then the accuracy gate."""
        measured = _measured_axes(r, self.shared_state)
        graded = resolve_graded_comparison(
            self.shared_state,
            measured,
            keep_threshold_pct=self.keep_threshold_pct,
            anchor_perf=self.stack.perf,
            anchor_tput=self.stack.tput,
        )
        decision = _graded_decision(graded, r, run.gv.name)
        # Each gate's verdict as it rules, in the order it ruled. Read off the
        # outcome afterwards, "REVERT" cannot say which gate ended the arc, and a
        # gate that never ran is indistinguishable from one that ruled against
        # the variant. Without a measurement nothing rules at all, which is why
        # no row is appended then.
        gates: list[dict[str, Any]] = []
        if r.status == "succeeded":
            gates.append(_graded_gate_row(graded, decision.reason, self.keep_threshold_pct))
        accuracy = _AccuracyVerdict(True, None, None, False)
        outcome, reason = decision.outcome, decision.reason
        if outcome == TS_FAILED and not reason:
            accuracy = _accuracy_verdict(
                run.slot,
                framework=self.launch.framework,
                benchmark_mode=str(getattr(self.shared_state, "benchmark_mode", "") or ""),
                reference=self.baseline.accuracy,
            )
            if accuracy.gated:
                gates.append(_accuracy_gate_row(accuracy))
            outcome = "KEEP" if accuracy.passed else "REVERT"
            reason = "" if accuracy.passed else _accuracy_refusal(accuracy.value)
        return _VariantVerdict(
            graded=graded,
            measured=measured,
            outcome=outcome,
            reason=reason,
            gain=decision.gain,
            graded_gain=decision.graded_gain,
            gates=gates,
            accuracy=accuracy,
            tput=r.output_throughput,
        )

    def budget_error(self, status: str) -> dict[str, Any]:
        """Why a round that measured nothing measured nothing, when the run stopped it."""
        if status != "failed" or self.budget.stop is None:
            return {}
        return {
            "error_class": self.budget.stop.error_class,
            "error": (
                f"{self.budget.detail}; {self.budget.untested} variant(s) went unmeasured "
                "and stay out of the ledger so a resume can retry them"
            ),
        }

    def search_update(self) -> dict[str, Any]:
        """This round's ledger writes, as ``apply_explore_search_update`` accumulates them."""
        ledger = self.ledger
        rejected_dedup: dict[str, dict[str, Any]] = {}
        for entry in ledger.rejected:
            fp = str(entry.get("fingerprint") or "")
            if not fp:
                continue
            rejected_dedup[fp] = entry
        killed_overtime_fps = [
            str(te.get("fingerprint") or "") for te in ledger.tested.values() if te.get("outcome") == TS_KILLED_OVERTIME
        ]
        return {
            "schema_version": 1,
            "tested": ledger.tested,
            "rejected": list(rejected_dedup.values()),
            "name_index": ledger.name_index,
            "winners_history": ledger.winners_history,
            "domains_round_summary": [],
            "last_round": {
                "round_id": self.round_id,
                "base_tput": self.baseline.tput,
                "base_extra_args": self.baseline.extra_args,
                "tested": [w["fingerprint"] for w in ledger.winners] + [lr["fingerprint"] for lr in ledger.rejected],
                "round_winners": [w["fingerprint"] for w in ledger.winners],
                "killed_overtime": killed_overtime_fps,
                "skipped_dup": ledger.skipped,
                "ts": _now_iso(),
            },
        }


class ExploreExecutor:
    """ActionRunner for the merged ``explore`` action."""

    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the explore executor and its gating thresholds."""
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = session_dir
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the merged ``explore`` action for one task."""
        try:
            explore_round = self._open_round(ctx)
        except _RoundRefusal as refusal:
            return refusal.result
        await explore_round.measure_its_grid()
        return self._round_result(explore_round, task_id=str(ctx.task.task_id))

    def _open_round(self, ctx) -> _ExploreRound:
        """Resolve what this round will measure and the stack it opens on."""
        params = dict(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}
        ss = extra.get("shared_state") or extra.get("state")
        workload = self._materialize_workload(params, extra, ss, task_id=str(ctx.task.task_id))
        _refresh_anchor_params(params, ss)
        baseline = _round_baseline(params, ss)
        setup = _RoundSetup(params, ss, workload, _launch_defaults(workload.config_path, baseline.args_mode), baseline)
        # Read before the grid filters materialize anything: the signature is the
        # workload the round opened on.
        ws_sig = workload_signature()
        lever_payload = framework_lever_grid(ss)
        ledger = _RoundLedger()
        observed = ObservedLaunch(*observed_launch_from_state(ss))
        runnable = self._runnable_variants(setup, self._proposed_grid(setup, lever_payload), ledger, observed)
        grade_on_intvty, _ = resolved_grading(ss)
        anchor_perf = _round_grading_anchor_perf(ss) if grade_on_intvty else None
        reuse = _ServerReuse.of(resolve_lifecycle_params(workload.config_path))
        # Seeded by the Coordinator from the durable cursor: this executor holds no
        # history of its own, so it cannot count the rounds before this one.
        round_id_seed = int(params.get("explore_search_cursor") or 0) + 1
        return _ExploreRound(
            round_id=f"explore-{round_id_seed:03d}",
            workload=workload,
            launch=setup.launch,
            baseline=baseline,
            shared_state=ss,
            runnable=runnable,
            lever_payload=lever_payload,
            stack=_VariantStack(
                extra_args=baseline.extra_args,
                extra_envs=dict(baseline.extra_envs),
                remove_args=list(dict.fromkeys(baseline.remove_args)),
                unset_envs=list(dict.fromkeys(baseline.unset_envs)),
                args_mode=baseline.args_mode,
                tput=baseline.tput,
                perf=anchor_perf,
            ),
            ledger=ledger,
            budget=_round_budget(params, extra, reuse_eligible=reuse.eligible),
            reuse=reuse,
            keep_threshold_pct=float(params.get("keep_threshold_pct", self.keep_threshold_pct)),
            grade_on_intvty=grade_on_intvty,
            on_the_synthetic_benchmark=not agentx_active(benchmark_mode=getattr(ss, "benchmark_mode", "")),
            is_revalidation=is_stack_revalidation(params),
            workload_signature=ws_sig,
            observed=observed,
        )

    def _materialize_workload(
        self, params: dict[str, Any], extra: Mapping[str, Any], shared_state: Any, *, task_id: str
    ) -> _Workload:
        """Re-materialize the benchmark config so variant YAMLs honour the operator's workload."""
        config_path = Path(params.get("config_path") or self.default_config_path or default_baseline_config())
        if not config_path.exists():
            raise _RoundRefusal("missing_config", f"config not found: {config_path}")
        output_root = Path(
            params.get("output_dir") or extra.get("workspace") or runs_dir(self.session_dir, "explore", task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)
        model_path = resolve_session_model_path(
            params=params,
            state_model_path=str(getattr(shared_state, "model_path", "") or "") if shared_state else "",
            for_serving=True,
        )
        gpu_type = str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        try:
            benchmark_script = sanitize_script_name(params.get("benchmark_script"))
            result_dir = sanitize_result_dir(params.get("result_dir"))
        except ValueError as exc:
            raise _RoundRefusal("bad_param", str(exc)) from exc
        try:
            config_path = prepare_grid_benchmark_config(
                config_path,
                output_root,
                model_path=model_path or None,
                gpu_type=gpu_type or None,
                benchmark_script=benchmark_script,
                out_name="explore_base.with_envs.yaml",
                grading=getattr(shared_state, "grading", None),
            )
        except FrameworkScriptMismatchError as exc:
            raise _RoundRefusal("framework_script_mismatch", str(exc)) from exc
        return _Workload(config_path, output_root, model_path, gpu_type, benchmark_script, result_dir)

    def _proposed_grid(self, setup: _RoundSetup, lever_payload: list[dict[str, Any]]) -> list[GridVariant]:
        """The variants this round was asked to measure, before any filter sees them."""
        grid_payload = setup.params.get("grid") or []
        if not isinstance(grid_payload, list):
            grid_payload = []
        # Framework-rewrite levers first.
        grid_payload = _seed_lever_variants(grid_payload, lever_payload)
        if not grid_payload:
            grid_payload = _seed_grid_payload(
                setup.params, framework=setup.launch.framework, yaml_envs=setup.launch.yaml_envs
            )
        if not isinstance(grid_payload, list) or not grid_payload:
            raise _RoundRefusal(
                "empty_grid",
                (
                    "explore: params.grid must be a non-empty list of variant "
                    "dicts. The Orchestration prompt "
                    "should fill this from specialist proposals / default_grid."
                ),
                workspace=setup.workload.output_root.as_posix(),
            )
        grid = _grid_variants_from_payload(grid_payload)
        if setup.launch.framework == "custom":
            grid, pinned_dropped = filter_operator_pinned_envs(grid, setup.launch.yaml_envs)
            for name, reason in pinned_dropped:
                log.warning("explore: dropping variant %s (%s)", name, reason)
        if not grid:
            raise _RoundRefusal(
                "empty_grid",
                "explore: params.grid contains no valid variants",
                workspace=setup.workload.output_root.as_posix(),
            )
        return grid

    def _runnable_variants(
        self, setup: _RoundSetup, grid: list[GridVariant], ledger: _RoundLedger, observed: ObservedLaunch
    ) -> list[GridVariant]:
        """The variants worth launching, in the order the round will take them."""
        _stamp_canonical_fingerprints(
            grid,
            base_remove_args=setup.baseline.remove_args,
            base_unset_envs=setup.baseline.unset_envs,
            base_args_mode=setup.baseline.args_mode,
        )
        shaped = self._shaped_for_the_cluster(setup, list(grid), ledger) if grid else []
        runnable = self._past_the_opening_restatements(setup, shaped, ledger, observed)
        log.info(
            "explore grid: payload=%d → runnable=%d (leading baseline_noop=%d)",
            len(grid),
            len(runnable),
            len(shaped) - len(runnable),
        )
        return runnable

    @staticmethod
    def _past_the_opening_restatements(
        setup: _RoundSetup, runnable: list[GridVariant], ledger: _RoundLedger, observed: ObservedLaunch
    ) -> list[GridVariant]:
        """The grid from its first variant that changes something, dropping the restatements ahead of it."""
        if not runnable:
            return runnable
        kept, restated = filter_baseline_noop_variants(
            runnable,
            framework=setup.launch.framework,
            build_variant_yaml=_build_variant_yaml,
            base_yaml_path=setup.workload.config_path,
            base_extra_args=setup.baseline.extra_args,
            base_extra_envs=setup.baseline.extra_envs,
            base_remove_args=setup.baseline.remove_args,
            base_unset_envs=setup.baseline.unset_envs,
            base_args_mode=setup.baseline.args_mode,
            model_path=setup.workload.model_path,
            gpu_type=setup.workload.gpu_type,
            benchmark_script=setup.workload.benchmark_script,
            observed=observed,
        )
        kept_ids = {id(gv) for gv in kept}
        leading = next((i for i, gv in enumerate(runnable) if id(gv) in kept_ids), len(runnable))
        for name, reason in restated[:leading]:
            _skip_baseline_noop(name, reason, ledger.skipped)
        return runnable[leading:]

    @staticmethod
    def _shaped_for_the_cluster(
        setup: _RoundSetup, runnable: list[GridVariant], ledger: _RoundLedger
    ) -> list[GridVariant]:
        """Drop the variants this cluster, model or operator will not run, then order what is left."""
        runnable, mn_dropped = apply_multi_node_invalid_variants(runnable)
        # Honour an operator-pinned SGLANG_USE_AITER=0: drop variants that would re-enable the (hang-prone) aiter
        # MoE runner.
        runnable, aiter_dropped = apply_aiter_moe_pin_filter(runnable)
        # xDiT do-not-set list, plus flags the model class does not support.
        runnable, compat_dropped = apply_compatibility_filter(
            runnable,
            framework=setup.launch.framework,
            model_path=setup.workload.model_path,
            gpu_type=setup.workload.gpu_type or str(getattr(setup.shared_state, "gpu_type", "") or ""),
            stack_fingerprint=getattr(setup.shared_state, "stack_fingerprint_meta", None),
            base_server_args=compose_server_args(
                inherited_args=setup.launch.inherited_args,
                base_extra_args=setup.baseline.extra_args,
                remove_args=setup.baseline.remove_args,
                args_mode=setup.baseline.args_mode,
            ),
        )
        # Operator-supplied --skip-variants patterns.
        runnable, skip_dropped = apply_user_skip_list(runnable, skip_spec=resolve_skip_spec(setup.params))
        for dropped in (*mn_dropped, *aiter_dropped, *compat_dropped, *skip_dropped):
            ledger.skipped.append(
                {
                    "name": dropped.get("name", ""),
                    "reason": dropped.get("source", "grid_invalid"),
                    "detail": dropped.get("reason", ""),
                }
            )
        return reorder_grid_for_multi_node(
            runnable,
            priority_tags=_MN_PARAMS_PRIORITY + _MN_BACKENDS_PRIORITY,
        )

    def _round_result(self, explore_round: _ExploreRound, *, task_id: str) -> dict[str, Any]:
        """The action result: what the round measured, kept and refused."""
        ledger = explore_round.ledger
        per_variant_outcomes = self._per_variant_outcomes(
            ledger.tested, ledger.rejected, ledger.skipped, task_id=task_id
        )
        lever_attributions = _framework_lever_attributions(per_variant_outcomes, explore_round.lever_payload)
        if lever_attributions:
            log.info(
                "explore: attributed %d framework rewrite lever(s): %s",
                len(lever_attributions),
                ", ".join(f"{a['switch']}={a['gain_pct']:+.2f}%" for a in lever_attributions),
            )
        best_winner = max(
            ledger.winners,
            key=lambda w: float(w.get("gain_pct") or 0.0),
            default=None,
        )
        best_gain_pct = float(best_winner.get("gain_pct") or 0.0) if best_winner else 0.0
        status = "succeeded" if explore_round.answered_its_grid() else "failed"
        return {
            "status": status,
            **explore_round.budget_error(status),
            "session_budget_untested": explore_round.budget.untested,
            "base_tput": explore_round.baseline.tput,
            "running_base_tput": explore_round.stack.tput,
            # Each KEEP advances the stack, so this is the final one.
            "output_throughput": float(explore_round.stack.tput) if ledger.winners else None,
            "best_variant": best_winner,
            "best_gain_pct": best_gain_pct,
            "winners": ledger.winners,
            "losers": ledger.rejected,
            "skipped_dup": ledger.skipped,
            # flat per-variant outcomes.
            "per_variant_outcomes": per_variant_outcomes,
            "framework_lever_attributions": lever_attributions,
            "explore_search_update": explore_round.search_update(),
            "round_id": explore_round.round_id,
            "workspace": explore_round.workload.output_root.as_posix(),
            "framework": explore_round.launch.framework,
            # gain_pct for the audit trail (best gain of the batch).
            "gain_pct": best_gain_pct,
            "explore_grid_exhausted": not explore_round.runnable,
        }

    def _per_variant_outcomes(
        self,
        tested_update: Mapping[str, Any],
        rejected_update: Any,
        skipped_dup: Any,
        *,
        task_id: str,
    ) -> list[dict[str, Any]]:
        """Describe what each variant of this round did, flat, for the fact-write hook."""
        reasons_by_fp: dict[str, str] = {
            str(r.get("fingerprint") or ""): str(r.get("reason") or "") for r in rejected_update
        }
        per_variant_outcomes: list[dict[str, Any]] = []
        for fp_key, te in tested_update.items():
            outcome = str(te.get("outcome") or "")
            if outcome not in (
                "KEEP",
                "REVERT",
                TS_FAILED,
                TS_KILLED_OVERTIME,
            ):
                continue
            per_variant_outcomes.append(
                self._tested_variant_row(fp_key, te, outcome=outcome, task_id=task_id, reasons_by_fp=reasons_by_fp)
            )
        for sd in skipped_dup:
            per_variant_outcomes.append(
                {
                    "variant_name": str(sd.get("name") or ""),
                    "outcome": TS_SKIPPED_DEDUP,
                    "fingerprint": str(sd.get("fingerprint") or ""),
                    "provenance": "",
                    "metrics": {},
                    "reason": str(sd.get("reason") or ""),
                }
            )
        return per_variant_outcomes

    def _tested_variant_row(
        self,
        fp_key: str,
        te: Mapping[str, Any],
        *,
        outcome: str,
        task_id: str,
        reasons_by_fp: Mapping[str, str],
    ) -> dict[str, Any]:
        """Describe one variant this round actually measured."""
        failure_attribution = ""
        if outcome in UNMEASURED_OUTCOMES:
            failure_attribution = classify_failure_attribution(
                error_class=te.get("error_class"),
                error_excerpt=te.get("error_excerpt"),
                reason=reasons_by_fp.get(fp_key, ""),
                explicit=te.get("failure_attribution"),
            )
        return {
            "variant_name": str(te.get("name") or ""),
            "outcome": outcome,
            "fingerprint": fp_key,
            "failure_id": make_failure_id(
                task_id=task_id,
                fingerprint=fp_key,
                variant_name=str(te.get("name") or ""),
            ),
            "stage": str(te.get("stage") or FAILURE_STAGE_DECISION),
            "error_excerpt": te.get("error_excerpt"),
            "provenance": str(te.get("provenance") or ""),
            "scope": str(te.get("scope") or ""),
            "metrics": _measured_metrics(te),
            "reason": reasons_by_fp.get(fp_key, ""),
            "error_class": str(te.get("error_class") or ""),
            "failure_attribution": failure_attribution,
            "server_log_path": te.get("server_log_path"),
            "workspace": te.get("workspace"),
            "raw_result_path": te.get("raw_result_path"),
            "variant": _variant_knobs(te),
            # The verdicts and the stack behind them, as the round
            # ruled. Absent keys mean the variant never got that far:
            # no gate ruled on a warmup failure, and nothing validated
            # one that was killed before it was graded.
            "measured_against": te.get("measured_against") or {},
            "gates": [gate for gate in (te.get("gates") or []) if isinstance(gate, dict)],
            "validation_basis": str(te.get("validation_basis") or ""),
        }


def _measured_metrics(te: Mapping[str, Any]) -> dict[str, Any]:
    """The figures a ledger row has, for the per-variant outcome the fact-write hook reads."""
    reported = (
        "tput",
        "gain_pct",
        "base_tput",
        "estimated_output_throughput",
        "runtime_sec",
        "wall_clock_ratio_vs_baseline",
    )
    return {key: te[key] for key in reported if te.get(key) is not None}


def _variant_knobs(te: Mapping[str, Any]) -> dict[str, Any]:
    """The knobs a variant was launched with, for the journal's ``classify_change_kind``."""
    return {
        "name": str(te.get("name") or ""),
        "extra_server_args": str(te.get("extra_server_args") or ""),
        "extra_envs": dict(te.get("extra_envs") or {}),
        "remove_args": list(te.get("remove_args") or []),
        "unset_envs": list(te.get("unset_envs") or []),
        "args_mode": str(te.get("args_mode") or "append"),
        "note": str(te.get("note") or ""),
        "reasoning_origin": str(te.get("reasoning_origin") or ""),
        "experience_citations": list(te.get("experience_citations") or []),
    }


def _safe(name: str) -> str:
    """Filesystem-safe slug for variant directory names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


explore_executor = ExploreExecutor()


__all__ = [
    "ExploreExecutor",
    "explore_executor",
]
