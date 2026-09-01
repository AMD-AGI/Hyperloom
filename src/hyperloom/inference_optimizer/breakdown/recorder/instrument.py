# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time instrumentation for ``session_breakdown.json``.

These helpers are called from the producing code (the Coordinator's
``SharedState``) to record breakdown facts where they are born, instead of
having the exporter re-walk artifacts later.

Every helper is best-effort: all failures are swallowed (logged at debug).
Payloads are shaped to the matching ``schema.py`` TypedDict.

Coverage in this module, in four groups:

* Coordinator state snapshots -- ``session`` / ``workload`` / ``final`` /
  ``explore_search`` / ``sweep`` singletons, plus ``optimization_stack`` /
  ``roofline`` items keyed by a stable id and one ``phase_timeline`` event
  per recorded action attempt.
* Kernel-agent lifecycle (``PRODUCER_KERNEL_AGENT``) -- discovery ->
  dispatch -> backend result -> micro/E2E, plus GEAK and GEMM-tuning
  invocations.
* Critic / robustness items (producers ``critic`` / ``robustness``), read
  from the agent workdir before pruning.
* The canonical v4 entity/event streams (subject / operation / measurement /
  adoption / artifact / trace_event / phase_transition / run_snapshot).

Several recorders here read just-written agent artifacts from disk. The
authoritative public surface is the re-export list in ``recorder/__init__``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.coerce import to_float
from hyperloom.common.jsonio import read_json
from hyperloom.common.timeutil import iso_z, now_iso

from ..agent_ownership import (
    UNATTRIBUTED,
    agent_from_lever,
    agent_from_phase,
    patch_author,
    patch_lever_kind,
)
from ..critic_reviews import normalize_framework_reviews
from .trace import trace_skip

log = logging.getLogger(__name__)

PRODUCER_COORDINATOR = "coordinator"
PRODUCER_KERNEL_AGENT = "kernel-agent"

# kernel-agent backend -> invocation section.
_GEAK_BACKENDS = frozenset({"geak"})
_FORGE_BACKENDS = frozenset({"forge"})

_FAILED_STATUSES = frozenset({"failed", "error", "crashed", "timeout"})
_VALID_PRODUCER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Statuses an action executor uses to state an adoption verdict itself. These
# outrank a caller-supplied routing label when the two disagree.
_EXECUTOR_ADOPTION_VERDICTS = frozenset(
    {
        "KEPT",
        "KEPT_INERT",
        "REVERTED",
        "REJECTED",
        "ACCURACY_UNAVAILABLE_REJECT",
    }
)


def _now_iso_safe() -> str:
    """Return the current UTC time as an ISO-8601 string (``""`` on failure).

    Returns:
        The current UTC time as a microsecond-precision ISO-8601 string, or
        ``""`` if the clock read fails.
    """
    try:
        return now_iso(timespec="microseconds")
    except Exception:  # noqa: BLE001
        return ""


def _recorder(session_dir: Path | str, producer: str):
    """Return the process-cached recorder for ``session_dir`` and ``producer``.

    Args:
        session_dir (Path | str): the session directory backing the recorder.
        producer (str): the breakdown producer label owning the fragments.

    Returns:
        The process-cached recorder for the ``(session_dir, producer)`` pair.
    """
    from .recorder import get_recorder

    return get_recorder(session_dir, producer=producer)


def _rel(path: Path, session_dir: Path | str) -> str:
    """Render ``path`` relative to ``session_dir`` (falls back to str).

    Args:
        path (Path): the path to render.
        session_dir (Path | str): the session directory to relativize against.

    Returns:
        str: ``path`` relative to ``session_dir``, or the plain string form when
            it is not under the session dir.
    """
    try:
        return str(Path(path).relative_to(Path(session_dir)))
    except (ValueError, TypeError):
        return str(path)


def _stable_id(prefix: str, *parts: Any) -> str:
    """Build a readable, collision-resistant id from author-time values."""
    raw_parts: list[str] = []
    for part in parts:
        if isinstance(part, Mapping):
            text = json.dumps(dict(part), sort_keys=True, separators=(",", ":"), default=str)
        else:
            text = str(part or "")
        if text:
            raw_parts.append(text)
    raw = "|".join(raw_parts) or "unknown"
    readable = re.sub(r"[^A-Za-z0-9._:-]+", "-", raw).strip("-")[:96] or "unknown"
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{prefix}:{readable}:{digest}"


def _measurement_occurrence(*run_identity: Any, value: Any = None) -> str:
    """Tell measuring something again apart from measuring it once.

    A measurement id is derived from its operation, and an operation is
    deliberately stable across a subject's retries -- one kernel, one
    operation, however many times it is tried. Measurements inherited that
    collapsing, so re-measuring a kernel wrote over the very numbers an
    earlier adoption had been decided on, leaving the adoption citing evidence
    that no longer agreed with it.

    ``run_identity`` names the run that produced the reading -- an attempt id,
    a benchmark report path, the timestamp the producer stamped on the entry.
    That is what should be passed. Readings taken by one run share a key, which
    is correct: they are one act of measuring, and a key drawn from the run is
    unmoved when one of its metrics is later filled in or corrected. A key
    drawn from the values is not, and re-recording a run to add its end-to-end
    numbers used to re-id the untouched micro reading beside them, splitting
    one reading into two and reporting a re-measure that never happened.

    ``value`` is a last resort for callers with no run to name, and only ever
    this metric's own reading -- never a sibling's, which is what made the key
    move. It can only tell apart readings that differ.

    Neither form may be a counter the recorder keeps: several producers replay
    their records from state after a resume, so an id has to be reproducible
    from what is being recorded rather than from how many parts already exist.
    The plain ordinal a reader wants is assigned at assembly instead, where the
    whole set is in hand at once.
    """
    parts = [str(part).strip() for part in run_identity if part is not None and str(part).strip()]
    if not parts:
        numeric = to_float(value)
        rendered = f"{numeric:.10g}" if numeric is not None else str(value or "")
        parts = [rendered] if rendered.strip() else []
    if not parts:
        return ""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:10]


def _operation_status(status: Any) -> str:
    """Normalize action status while preserving terminal failures."""
    value = str(status or "").strip().lower()
    if value in {"kept", "keep", "promoted", "succeeded", "success", "completed", "complete", "ok"}:
        return "succeeded"
    if value in {"running", "started", "pending"}:
        return "running"
    if value in {"skipped", "discarded", "reverted", "revert", "rejected", "no_promote"}:
        return value
    return "failed" if value in _FAILED_STATUSES or value else "unknown"


_AGENT_BY_ACTION = {
    "explore": "explore",
    "replay_warm_recipe": "warm_replay",
    "warm_replay": "warm_replay",
    "baseline": "coordinator",
    "profile": "coordinator",
    "roofline": "coordinator",
    "sweep": "coordinator",
    "conc_sweep": "coordinator",
    "validate": "coordinator",
    "validate_stack": "coordinator",
    "trace_analyze": "coordinator",
    "critic": "critic",
    "robustness": "robustness",
}


def _resolve_agent(
    action: str,
    *,
    result: Mapping[str, Any] | None = None,
    phase: str = "",
) -> str:
    """Name the agent that owns this unit of work, at the moment it settles.

    Ownership is a fact the producer knows and the exporter cannot recover: by
    the time a delayed ``integrate_patch`` lands, the active phase has usually
    moved on. Returning ``unattributed`` means the producer genuinely has no
    evidence of an owner, which is a reportable data gap rather than a guess.
    """
    name = str(action or "").strip().lower()
    result = result or {}

    if name.startswith("integrate_patch") or name == "integrate":
        return patch_author(result)

    direct = _AGENT_BY_ACTION.get(name)
    if direct:
        return direct
    if name.startswith("kernel_opt") or name in {"geak_e2e", "gemm_tuning", "fusion", "kernel_optimization"}:
        return "kernel_agent"

    return (
        agent_from_lever(patch_lever_kind(result))
        or agent_from_phase(result.get("source_phase"))
        or agent_from_phase(phase)
        or UNATTRIBUTED
    )


def _action_operation_id(action: str, entry: Mapping[str, Any]) -> str:
    """Return the resume-stable operation id for an action event."""
    extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
    task_id = str(entry.get("task_id") or "").strip()
    if task_id:
        return _stable_id("op", action, task_id)
    key = next(
        (
            extras.get(name)
            for name in ("round_id", "round", "tick", "fingerprint", "candidate_id", "key")
            if extras.get(name) not in (None, "")
        ),
        "",
    )
    return _stable_id("op", action, key or entry.get("ts") or entry)


def _record_action_artifacts(
    session_dir: Path | str,
    *,
    operation_id: str,
    result: Mapping[str, Any],
    producer: str,
) -> list[str]:
    """Record path references already present in an action result."""
    artifact_ids: list[str] = []
    scalar_keys = (
        "workspace",
        "output_dir",
        "report_path",
        "benchmark_report_path",
        "raw_result_path",
        "stderr_log_path",
        "analysis_md_path",
        "kernel_roofline_path",
        "main_trace_path",
        "trace_dir",
        "report_json_path",
        "report_csv_path",
        "config_path",
        "materialized_config",
        "decision_path",
    )
    list_keys = ("trace_files", "patches_applied", "patches_reverted", "artifacts")
    refs: list[tuple[str, Any]] = [(key, result.get(key)) for key in scalar_keys]
    for key in list_keys:
        values = result.get(key)
        if isinstance(values, (list, tuple)):
            refs.extend((key, value) for value in values)
    seen: set[str] = set()
    for index, (kind, path) in enumerate(refs):
        if not isinstance(path, (str, Path)) or not str(path).strip():
            continue
        rendered = str(path)
        if rendered in seen:
            continue
        seen.add(rendered)
        artifact_id = _stable_id("artifact", operation_id, kind, rendered)
        artifact_ids.append(artifact_id)
        record_artifact(
            session_dir,
            artifact_id=artifact_id,
            operation_id=operation_id,
            producer_operation_id=operation_id,
            kind=kind,
            name=Path(rendered).name or kind,
            path=rendered,
            status="available",
            producer=producer,
            consumers=[operation_id],
            coverage={"source": "structured_result"},
        )
    return artifact_ids


def _record_action_measurements(
    session_dir: Path | str,
    *,
    action: str,
    operation_id: str,
    result: Mapping[str, Any],
    entry: Mapping[str, Any],
    producer: str,
) -> list[str]:
    """Record scalar and per-point measurements from a structured result."""
    measurement_ids: list[str] = []
    metric_fields = (
        ("output_throughput", "throughput", str(result.get("throughput_unit") or "tok/s")),
        ("throughput", "throughput", str(result.get("throughput_unit") or "tok/s")),
        ("best_tput", "throughput", str(result.get("throughput_unit") or "tok/s")),
        ("throughput_after", "throughput", str(result.get("throughput_unit") or "tok/s")),
        ("baseline_tput", "baseline_throughput", str(result.get("throughput_unit") or "tok/s")),
        ("achieved_tok_per_sec", "achieved_throughput", "tok/s"),
        ("theoretical_peak_tok_per_sec", "theoretical_peak_throughput", "tok/s"),
        ("accuracy", "accuracy", str(result.get("accuracy_unit") or "score")),
        ("ttft_mean_ms", "ttft_mean", "ms"),
        ("tpot_mean_ms", "tpot_mean", "ms"),
        ("e2el_mean_ms", "e2el_mean", "ms"),
        ("best_gain_pct", "gain", "percent"),
        ("delta_pct", "gain", "percent"),
        ("actual_gain_pct", "gain", "percent"),
        ("within_roofline_pct", "roofline_utilization", "percent"),
        ("snapshot_id", "snapshot", "id"),
    )
    dimensions = {
        key: result.get(key)
        for key in ("conc", "isl", "osl", "tp", "ep", "framework", "precision")
        if result.get(key) is not None
    }
    common = {
        "operation_id": operation_id,
        "status": _operation_status(entry.get("status") or result.get("status")),
        "measured_at": str(entry.get("ts") or result.get("ts") or ""),
        "producer": producer,
        "dimensions": dimensions,
        "metric_basis": str(result.get("metric_basis") or result.get("accuracy_source") or "partial:not_provided"),
        "harness": (
            result.get("harness")
            or result.get("bench_client")
            or result.get("benchmark_script")
            or result.get("materialized_config")
            or result.get("config_path")
            or {"status": "partial", "reason": "not_provided"}
        ),
        "workload": (
            dict(result.get("workload") or {})
            if isinstance(result.get("workload"), Mapping)
            else {**dimensions, "lock": "partial", "status": "partial"}
        ),
        "samples": list(result.get("samples") or []) if isinstance(result.get("samples"), (list, tuple)) else [],
        "aggregation": result.get("aggregation") or "result_scalar",
    }
    # Every metric below is read from one execution of the action, so they share
    # the stamp that execution carries: a later pass that fills in one of them
    # must not re-id the ones it did not touch.
    occurrence_run = str(common["measured_at"] or "")
    seen_names: set[str] = set()
    for field, name, unit in metric_fields:
        value = result.get(field)
        if value is None or name in seen_names:
            continue
        seen_names.add(name)
        occurrence = _measurement_occurrence(occurrence_run, value=value)
        measurement_id = _stable_id("measurement", operation_id, name, occurrence)
        measurement_ids.append(measurement_id)
        record_measurement(
            session_dir,
            measurement_id=measurement_id,
            kind=name,
            name=name,
            value=value,
            unit=unit,
            source={
                "field": field,
                "action": action,
                "role": "baseline" if field == "baseline_tput" else "final" if name == "throughput" else "derived",
            },
            metric_basis=(
                str(result.get("metric_basis") or "output")
                if name in {"throughput", "baseline_throughput", "achieved_throughput"}
                else common["metric_basis"]
            ),
            metadata={"completeness": "complete" if common["samples"] else "partial"},
            **{key: value for key, value in common.items() if key != "metric_basis"},
        )

    point_groups: list[tuple[str, Any]] = [
        ("sweep", result.get("all_variants")),
        ("comparison", result.get("comparison")),
    ]
    for group, points in point_groups:
        if not isinstance(points, list):
            continue
        for index, point in enumerate(points):
            if not isinstance(point, Mapping):
                continue
            point_key = point.get("variant_name") or point.get("name") or point.get("conc") or index
            for field, name, unit in (
                ("output_throughput_tok_s", "throughput", "tok/s"),
                ("output_throughput", "throughput", "tok/s"),
                ("throughput", "throughput", "tok/s"),
                ("speedup", "speedup", "ratio"),
                ("ttft_mean_ms", "ttft_mean", "ms"),
                ("tpot_mean_ms", "tpot_mean", "ms"),
                ("e2el_mean_ms", "e2el_mean", "ms"),
            ):
                if point.get(field) is None:
                    continue
                measurement_id = _stable_id(
                    "measurement",
                    operation_id,
                    group,
                    point_key,
                    name,
                    _measurement_occurrence(occurrence_run, value=point.get(field)),
                )
                measurement_ids.append(measurement_id)
                record_measurement(
                    session_dir,
                    measurement_id=measurement_id,
                    operation_id=operation_id,
                    kind=name,
                    name=name,
                    value=point.get(field),
                    unit=unit,
                    status=str(point.get("status") or common["status"]),
                    measured_at=common["measured_at"],
                    producer=producer,
                    dimensions={
                        key: point.get(key)
                        for key in ("variant_name", "conc", "isl", "osl")
                        if point.get(key) is not None
                    },
                    metric_basis=str(result.get("metric_basis") or ""),
                    harness=common["harness"],
                    workload=common["workload"],
                    samples=[],
                    aggregation="point",
                    source={"field": field, "group": group},
                )
    return measurement_ids


def _record_adoption_transition(
    session_dir: Path | str,
    *,
    adoption_id: str,
    operation_id: str,
    adopted: bool,
    producer: str,
    reason: str,
    **fields: Any,
) -> None:
    """Upsert the current state of one canonical adoption."""
    now = str(fields.pop("transitioned_at", "") or _now_iso_safe())
    record_adoption(
        session_dir,
        adoption_id=adoption_id,
        producer=producer,
        operation_id=operation_id,
        status="adopted" if adopted else "revoked",
        decision="KEEP" if adopted else "REVERT",
        validated=adopted,
        reason=reason,
        **({"adopted_at": now} if adopted else {"revoked_at": now}),
        **fields,
    )


def _mirror_action_v4(
    session_dir: Path | str,
    *,
    action: str,
    entry: Mapping[str, Any],
    result: Mapping[str, Any],
    phase: str,
    macro_cycle: int,
    tick: int,
    producer: str,
) -> None:
    """Mirror one settled non-kernel action into canonical v4 streams."""
    identity_entry = dict(entry)
    identity_extras = dict(entry.get("extras") or {}) if isinstance(entry.get("extras"), Mapping) else {}
    if not entry.get("task_id") and not any(
        identity_extras.get(name) not in (None, "")
        for name in ("round_id", "round", "tick", "fingerprint", "candidate_id", "key")
    ):
        identity_extras["tick"] = tick
    identity_entry["extras"] = identity_extras
    operation_id = _action_operation_id(action, identity_entry)
    status = _operation_status(entry.get("status") or result.get("status"))
    decision = str(entry.get("decision") or result.get("decision") or result.get("status") or "")
    extras = dict(entry.get("extras") or {}) if isinstance(entry.get("extras"), Mapping) else {}
    task_id = str(entry.get("task_id") or "")
    subject_id = _stable_id("subject", action, task_id or extras or result.get("candidate") or result)
    subject_type = {
        "baseline": "workload",
        "profile": "profile",
        "roofline": "roofline_snapshot",
        "explore": "variant",
        "sweep": "sweep",
        "conc_sweep": "concurrency_sweep",
    }.get(action, action)
    subject_attributes: dict[str, Any] = {}
    if isinstance(result.get("candidate"), Mapping):
        subject_attributes["candidate"] = dict(result["candidate"])
    if isinstance(result.get("best_variant"), Mapping):
        subject_attributes["variant"] = dict(result["best_variant"])
    elif isinstance(result.get("best_winner"), Mapping):
        subject_attributes["variant"] = dict(result["best_winner"])
    if action in {"profile", "roofline"}:
        subject_attributes.update(
            {
                key: result.get(key)
                for key in (
                    "snapshot_id",
                    "roofline_arm",
                    "trace_health",
                    "hot_kernels",
                    "top_bottleneck",
                )
                if result.get(key) is not None
            }
        )
    record_subject(
        session_dir,
        subject_id=subject_id,
        subject_type=subject_type,
        role="target",
        name=str(extras.get("candidate_id") or extras.get("variant_name") or extras.get("best_variant_name") or action),
        attributes=subject_attributes,
        producer=producer,
    )
    measurement_ids = _record_action_measurements(
        session_dir,
        action=action,
        operation_id=operation_id,
        result=result,
        entry=entry,
        producer=producer,
    )
    artifact_ids = _record_action_artifacts(
        session_dir,
        operation_id=operation_id,
        result=result,
        producer=producer,
    )
    ended_at = str(entry.get("ts") or result.get("ts") or _now_iso_safe())
    substeps: list[dict[str, Any]] = []
    if action == "baseline":
        substeps.append(
            {
                "substep_id": _stable_id("substep", operation_id, "benchmark"),
                "kind": "benchmark",
                "name": "baseline benchmark",
                "status": status,
                "ended_at": ended_at,
                "measurements": [mid for mid in measurement_ids if "throughput" in mid],
                "artifacts": artifact_ids,
            }
        )
        eval_status = (
            "skipped"
            if result.get("run_eval_disabled")
            else (
                "succeeded" if result.get("accuracy") is not None else ("failed" if status == "failed" else "partial")
            )
        )
        substeps.append(
            {
                "substep_id": _stable_id("substep", operation_id, "eval"),
                "kind": "evaluation",
                "name": "accuracy evaluation",
                "status": eval_status,
                "ended_at": ended_at,
                "measurements": [mid for mid in measurement_ids if "accuracy" in mid],
                "metadata": {
                    "accuracy_source": result.get("accuracy_source"),
                    "run_eval_disabled": result.get("run_eval_disabled"),
                },
            }
        )
    elif action == "roofline":
        for name in ("profile", "trace_analyze"):
            substeps.append(
                {
                    "substep_id": _stable_id("substep", operation_id, name),
                    "kind": name,
                    "name": name,
                    "status": status if str(result.get("phase") or "") in ("", name) else "succeeded",
                    "ended_at": ended_at,
                    "artifacts": artifact_ids,
                }
            )
    elif action == "profile":
        substeps.append(
            {
                "substep_id": _stable_id("substep", operation_id, "profile"),
                "kind": "profile",
                "name": "profile capture",
                "status": status,
                "ended_at": ended_at,
                "measurements": measurement_ids,
                "artifacts": artifact_ids,
                "metadata": {"trace_health": result.get("trace_health")},
            }
        )
    gates: list[dict[str, Any]] = []
    agent = _resolve_agent(action, result=result, phase=phase)
    # ``or`` would let a real 0.0% fall through to ``best_gain_pct``; a measured
    # zero is a verdict, not a missing value.
    _raw_gain = result.get("delta_pct")
    if _raw_gain is None:
        _raw_gain = result.get("best_gain_pct")
    gain_pct = to_float(_raw_gain)
    keep_threshold_pct = to_float(result.get("keep_threshold_pct"))
    decision_reason = str(result.get("decision_reason") or result.get("reason") or "")
    if keep_threshold_pct is not None:
        # Without the threshold the verdict is unfalsifiable after the fact:
        # "+0.55%" alone never explains why the run declined to keep it.
        gates.append(
            {
                "gate_id": _stable_id("gate", operation_id, "keep_threshold"),
                "kind": "keep_threshold",
                "name": "keep_threshold",
                "status": (
                    "passed"
                    if gain_pct is not None and gain_pct >= keep_threshold_pct
                    else "failed"
                    if gain_pct is not None
                    else "partial"
                ),
                "decision": ("allow" if gain_pct is not None and gain_pct >= keep_threshold_pct else "deny"),
                "reason": decision_reason,
                "evaluated_at": ended_at,
                "inputs": {"keep_threshold_pct": keep_threshold_pct},
                "evidence": {
                    "gain_pct": gain_pct,
                    "keep_threshold_pct": keep_threshold_pct,
                    "base_tput": to_float(result.get("base_tput")),
                    "output_throughput": to_float(result.get("output_throughput")),
                },
            }
        )
    adoption_ids: list[str] = []
    verdict = str(entry.get("decision") or result.get("decision") or result.get("status") or "").strip().upper()
    # The executor owns the adoption verdict. A caller's coarse routing label
    # ("discarded") must never overwrite a KEEP the executor already committed.
    executor_verdict = str(result.get("status") or "").strip().upper()
    if executor_verdict in _EXECUTOR_ADOPTION_VERDICTS:
        verdict = executor_verdict
    adoptable_actions = {
        "explore",
        "integrate",
        "integrate_patch",
        "replay_warm_recipe",
        "warm_replay",
        "warm_recipe",
    }
    keep_verdict = verdict in {"KEEP", "KEPT", "KEPT_INERT", "PROMOTED", "ADOPTED"}
    revert_verdict = verdict in {
        "REVERT",
        "REVERTED",
        "REJECTED",
        "FAILED",
        "ACCURACY_UNAVAILABLE_REJECT",
    }
    validated = result.get("validated", result.get("accuracy_pass"))
    # An accuracy gate that ran but returned no verdict is a first-class
    # outcome, not a failure: a session with no eval configured, or a baseline
    # accuracy of zero, reaches here with ``None``. Reading that as "did not
    # pass" would zero out the whole ledger for such a run. The KEEP stands;
    # what gets recorded alongside it is that nothing checked the accuracy.
    validation_passed = keep_verdict if validated is None else bool(validated)
    validation_basis = "keep_verdict_unscored" if validated is None else "accuracy_pass"
    if action in adoptable_actions and ((keep_verdict and validation_passed) or revert_verdict):
        adoption_id = _stable_id("adoption", operation_id)
        adoption_ids.append(adoption_id)
        # Enablement and inert keeps are genuine adoptions that must not be
        # counted as gain: the code lands, the measured delta is not its own.
        attribution_eligible = result.get("attribution_eligible")
        if attribution_eligible is None:
            attribution_eligible = not (
                bool(result.get("enablement")) or bool(result.get("baseline_enablement")) or verdict == "KEPT_INERT"
            )
        _record_adoption_transition(
            session_dir,
            adoption_id=adoption_id,
            operation_id=operation_id,
            adopted=keep_verdict and validation_passed,
            reason=decision_reason or verdict,
            transitioned_at=ended_at,
            subject={"subject_id": subject_id, "subject_type": subject_type},
            artifact_ids=artifact_ids,
            measurement_ids=measurement_ids,
            kind=action,
            agent=agent,
            attribution_eligible=bool(attribution_eligible),
            gain_pct=gain_pct,
            throughput_before=to_float(result.get("base_tput")),
            throughput_after=to_float(result.get("output_throughput") or result.get("tput")),
            configuration=dict(result.get("configuration") or {}),
            validation_basis=validation_basis,
            producer=producer,
            metadata={
                "keep_threshold_pct": keep_threshold_pct,
                "executor_status": str(result.get("status") or ""),
                "provenance": str(result.get("provenance") or ""),
                "domain": str(result.get("domain") or ""),
            },
        )
    record_operation(
        session_dir,
        operation_id=operation_id,
        root_operation_id=operation_id,
        kind="composite" if action in {"baseline", "roofline"} else action,
        name=action,
        phase=phase,
        macro_cycle=int(macro_cycle or 0),
        status=status,
        source="author_time_writeback",
        executor_class="deterministic",
        purpose="discovery"
        if action in {"sweep", "conc_sweep"}
        else ("validation" if action == "baseline" else "optimization"),
        scope=str(extras.get("scope") or ""),
        agent=agent,
        strategy_group=action,
        strategy=str(extras.get("provenance") or result.get("provenance") or result.get("strategy") or action),
        producer=producer,
        sequence=int(tick or 0),
        ended_at=ended_at,
        subject={"subject_id": subject_id, "subject_type": subject_type},
        substeps=substeps,
        gates=gates,
        decisions=[
            {
                "decision_id": _stable_id("decision", operation_id),
                "kind": action,
                "verdict": verdict or decision,
                "reason": decision_reason,
                "decided_at": ended_at,
                "component": producer,
                "evidence": {
                    **extras,
                    "gain_pct": gain_pct,
                    "keep_threshold_pct": keep_threshold_pct,
                    "executor_status": str(result.get("status") or ""),
                },
            }
        ],
        outputs=dict(result),
        error=result.get("error"),
        measurement_refs=measurement_ids,
        artifact_refs=artifact_ids,
        adoption_refs=adoption_ids,
        extensions={"task_id": task_id, "tick": tick},
        metadata={"extras": extras},
    )
    transition_id = _stable_id(
        "transition",
        operation_id,
        f"macro_cycle:{int(macro_cycle or 0)}",
        f"tick:{int(tick or 0)}",
        status,
        decision,
        ended_at,
    )
    record_phase_transition(
        session_dir,
        transition_id=transition_id,
        operation_id=operation_id,
        phase=phase,
        action=action,
        status=status,
        decision=decision,
        ts=ended_at,
        producer=producer,
    )
    record_trace_event(
        session_dir,
        trace_event_id=_stable_id("trace", operation_id, status),
        operation_id=operation_id,
        kind="operation_finalized",
        phase=phase,
        status=status,
        decision=decision,
        ts=ended_at,
        producer=producer,
    )


def record_phase_event(
    session_dir: Path | str | None,
    *,
    action: str,
    entry: dict[str, Any],
    result: Mapping[str, Any] | None = None,
    phase: str = "",
    macro_cycle: int = 0,
    tick: int = 0,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record one ``phase_timeline`` event from a ``record_action_attempt``
    entry, and mirror the same attempt into the canonical v4 streams via
    ``_mirror_action_v4``.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        action (str): the action name the event is keyed by.
        entry (dict[str, Any]): the ``record_action_attempt`` entry to project
            into a phase_timeline payload.
        result (Mapping[str, Any] | None): the settled action result, mirrored
            into v4 only.
        phase (str): phase label for the v4 mirror; falls back to
            ``entry["phase"]`` when empty.
        macro_cycle (int): macro cycle for the v4 mirror.
        tick (int): used to synthesize an operation identity when the entry
            carries no task_id or round key.
        producer (str): the breakdown producer label (defaults to the
            Coordinator).
    """
    if not session_dir or not isinstance(entry, dict):
        trace_skip(reason="no session_dir" if not session_dir else "entry is not a dict", section="phase_transitions")
        return
    try:
        task_id = str(entry.get("task_id") or "")
        payload = {
            "ts": str(entry.get("ts") or ""),
            "action": str(action or ""),
            "task_id": task_id,
            "status": str(entry.get("status") or ""),
            "decision": str(entry.get("decision") or ""),
            "key_metric": to_float(entry.get("key_metric")),
            "key_metric_kind": entry.get("key_metric_kind"),
            "workspace": entry.get("workspace"),
            "error_class": entry.get("error_class"),
            "extras": dict(entry.get("extras") or {}),
        }
        # Stable key per (action, task) so a re-recorded attempt overwrites.
        key = f"{action}-{task_id}" if task_id else None
        _recorder(session_dir, producer).record_item(
            "phase_timeline",
            payload,
            key=key,
        )
        _mirror_action_v4(
            session_dir,
            action=action,
            entry=entry,
            result=dict(result or {}),
            phase=phase or str(entry.get("phase") or ""),
            macro_cycle=macro_cycle,
            tick=tick,
            producer=producer,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_phase_event failed", exc_info=True)
        trace_skip(reason="writer raised", section="phase_transitions", error=exc)


def record_action_operation(
    session_dir: Path | str | None,
    *,
    action: str,
    task_id: str,
    status: str,
    decision: str,
    result: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
    phase: str = "",
    macro_cycle: int = 0,
    tick: int = 0,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Mirror an action into v4 without changing legacy action ledgers."""
    if not session_dir:
        trace_skip(reason="no session_dir", section="operations")
        return
    try:
        _mirror_action_v4(
            session_dir,
            action=action,
            entry={
                "ts": str((result or {}).get("ts") or _now_iso_safe()),
                "task_id": str(task_id or ""),
                "status": str(status or ""),
                "decision": str(decision or ""),
                "extras": dict(extras or {}),
            },
            result=dict(result or {}),
            phase=phase,
            macro_cycle=macro_cycle,
            tick=tick,
            producer=producer,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_action_operation failed", exc_info=True)
        trace_skip(reason="writer raised", section="operations", error=exc)


def snapshot_state_sections(
    session_dir: Path | str | None,
    state: Any,
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Snapshot every state-owned breakdown section from a live ``SharedState``.

    Singletons overwrite the producer's own file; event-stream items are keyed
    by a stable id so repeated snapshots are idempotent. Best-effort per
    section: one failing section never blocks the others.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        state (Any): the live ``SharedState`` snapshotted into each section.
        producer (str): the breakdown producer label (defaults to the
            Coordinator).
    """
    if not session_dir or state is None:
        trace_skip(reason="no session_dir" if not session_dir else "no state", section="run_snapshot")
        return
    rec = None
    try:
        rec = _recorder(session_dir, producer)
    except Exception as exc:  # noqa: BLE001
        log.debug("recorder unavailable", exc_info=True)
        trace_skip(reason="writer raised", section="run_snapshot", error=exc)
        return

    for name, fn in (
        ("session", _snapshot_session),
        ("explore_search", _snapshot_explore_search),
        ("optimization_stack", _snapshot_optimization_stack),
        ("roofline", _snapshot_roofline),
    ):
        try:
            fn(rec, state)
        except Exception as exc:  # noqa: BLE001
            log.debug("snapshot section %s failed", name, exc_info=True)
            trace_skip(reason="writer raised", section=name, error=exc)
    try:
        _snapshot_v4_run(rec, state)
    except Exception as exc:  # noqa: BLE001
        log.debug("snapshot v4 run failed", exc_info=True)
        trace_skip(reason="writer raised", section="run_snapshot", error=exc)


def _snapshot_v4_run(rec, st: Any) -> None:
    """Write the complete currently available v4 run snapshot from memory."""
    current_best = dict(getattr(st, "current_best", None) or {})
    stack = list(getattr(st, "optimization_stack", None) or [])
    model_info = dict(getattr(st, "model_info", None) or {})
    model_arch = dict(getattr(st, "model_arch", None) or {})
    model = {
        **model_arch,
        **model_info,
        "name": str(getattr(st, "model_name", "") or ""),
        "path": str(getattr(st, "model_path", "") or ""),
        "class": str(getattr(st, "model_class", "") or ""),
        "type": str(getattr(st, "model_type", "") or model_info.get("model_type") or ""),
        "architectures": list(getattr(st, "model_architectures", None) or model_info.get("architectures") or []),
    }
    workload = {
        "framework": str(getattr(st, "framework", "") or ""),
        "model_name": str(getattr(st, "model_name", "") or ""),
        "model_path": str(getattr(st, "model_path", "") or ""),
        "gpu_type": str(getattr(st, "gpu_type", "") or ""),
        "precision": str(getattr(st, "precision", "") or ""),
        "tp": int(getattr(st, "tp", 0) or 0),
        "ep": int(getattr(st, "ep", 0) or 0),
        "conc": int(getattr(st, "conc", 0) or 0),
        "isl": int(getattr(st, "isl", 0) or 0),
        "osl": int(getattr(st, "osl", 0) or 0),
        "max_model_len": int(getattr(st, "max_model_len", 0) or 0),
        "objective": {
            "target_gain_pct": getattr(st, "target_gain_pct", None),
            "target_tput": getattr(st, "target_tput", None),
        },
    }
    stop_reason = str(getattr(st, "stop_reason", "") or "")
    outcome_status = "running"
    if stop_reason:
        outcome_status = (
            "failed"
            if any(marker in stop_reason.lower() for marker in ("failed", "error", "crash", "abort"))
            else "completed"
        )
    rec.record_upsert_singleton(
        "run_snapshot",
        {
            "run": {
                "session_id": str(getattr(st, "session_id", "") or ""),
                "claw_session_id": str(getattr(st, "claw_session_id", "") or ""),
                "sandbox_user_id": str(getattr(st, "sandbox_user_id", "") or ""),
                "started_at": str(getattr(st, "start_ts", "") or ""),
                "phase": str(getattr(st, "phase", "") or ""),
                "macro_cycle": int(getattr(st, "macro_cycle", 0) or 0),
                "tick": int(getattr(st, "tick", 0) or 0),
                "max_minutes": int(getattr(st, "max_minutes", 0) or 0),
                "stop_reason": stop_reason,
            },
            "workload": workload,
            "model": model,
            "versions": dict(getattr(st, "versions", None) or getattr(st, "tool_versions", None) or {}),
            "outcome": {
                "status": outcome_status,
                "stop_reason": stop_reason,
                "baseline_throughput": to_float(getattr(st, "baseline_tput", None)),
                "baseline_accuracy": to_float(getattr(st, "baseline_accuracy", None)),
                "current_best": current_best,
                "cumulative_gain_validated_pct": to_float(getattr(st, "cumulative_gain_validated", None)),
                "optimization_stack_size": len(stack),
            },
        },
    )


def _snapshot_session(rec, st: Any) -> None:
    """Snapshot the ``session`` singleton from ``st`` (no-op without a session id).

    A session that has stopped carries ``ended_at_utc``, taken from the state's
    own stop timestamp: without it the exporter has no end to measure against
    and reports the run as still going. ``start_ts`` is what the exported
    elapsed time is measured from; a resume re-anchors it on the new leg only
    when the previous one crashed or stopped for a recorded reason, so after a
    clean stop it still names the original start. The manifest-derived fields
    the live state cannot know (image, host, pid) are filled in by the
    collector at export.

    Args:
        rec: the recorder used to write the singleton.
        st (Any): the live ``SharedState`` to snapshot.
    """
    session_id = str(getattr(st, "session_id", "") or "")
    if not session_id:
        return
    stop_reason = str(getattr(st, "stop_reason", "") or "")
    rec.record_singleton(
        "session",
        {
            "session_id": session_id,
            "claw_session_id": getattr(st, "claw_session_id", "") or "",
            "sandbox_user_id": getattr(st, "sandbox_user_id", "") or "",
            "start_ts": str(getattr(st, "start_ts", "") or ""),
            # A resumed run clears its reason but not necessarily the stale
            # timestamp, so the pair is only ever emitted together.
            "ended_at_utc": iso_z(getattr(st, "stop_ts", "")) if stop_reason else "",
            "stop_reason": stop_reason,
            "max_minutes": int(getattr(st, "max_minutes", 0) or 0),
            "tick_count": int(getattr(st, "tick", 0) or 0),
            "phase": str(getattr(st, "phase", "") or ""),
        },
    )


def _snapshot_explore_search(rec, st: Any) -> None:
    """Snapshot the ``explore_search`` singleton from ``st`` (no-op when empty).

    Augments the base search dict with the no-promote streak, discovered
    flags, and the ledger-owned synergy list.

    Args:
        rec: the recorder used to write the singleton.
        st (Any): the live ``SharedState`` to snapshot.
    """
    search = dict(getattr(st, "explore_search", None) or {})
    if not search:
        return
    search["no_promote_streak"] = int(getattr(st, "params_no_promote_streak", 0) or 0)
    search["discovered_flags"] = dict(getattr(st, "discovered_flags", None) or {})
    search["synergy_attempted"] = list(search.get("synergy_attempted") or [])
    rec.record_singleton("explore_search", search)


def _snapshot_optimization_stack(rec, st: Any) -> None:
    """Snapshot each ``optimization_stack`` entry from ``st`` as a keyed item.

    Backfills a missing per-entry ``gain_pct`` from ``st.gain_per_stack_entry``
    when available; each item is keyed by its stack index for idempotency.

    Args:
        rec: the recorder used to write the items.
        st (Any): the live ``SharedState`` to snapshot.
    """
    stack = getattr(st, "optimization_stack", None) or []
    gains = getattr(st, "gain_per_stack_entry", None) or []
    for i, entry in enumerate(stack):
        if not isinstance(entry, dict):
            continue
        payload = dict(entry)
        if payload.get("gain_pct") is None and i < len(gains):
            payload["gain_pct"] = to_float(gains[i])
        rec.record_item("optimization_stack", payload, key=str(i))


def _snapshot_roofline(rec, st: Any) -> None:
    """Snapshot each ``roofline`` snapshot from ``st`` as a keyed item.

    Each item is keyed by its snapshot id (falling back to the list index) for
    idempotency.

    Args:
        rec: the recorder used to write the items.
        st (Any): the live ``SharedState`` to snapshot.
    """
    snapshots = getattr(st, "roofline_snapshots", None) or []
    for idx, snap in enumerate(snapshots):
        if not isinstance(snap, dict):
            continue
        sid = str(snap.get("snapshot_id") or snap.get("id") or idx)
        rec.record_item("roofline", snap, key=sid)


def _best_attempt_id(
    attempts: list[Any],
    verification: dict[str, Any],
) -> str:
    """Pick the adopted attempt id: verification hint, else highest speedup.

    Mirrors the collector's selection so the kernel-level decision lands on the
    same attempt the breakdown would attribute it to.

    Args:
        attempts (list[Any]): the per-backend attempt rows.
        verification (dict[str, Any]): the verification block carrying the
            ``best_attempt_id`` / ``best_backend`` hints.

    Returns:
        str: the adopted attempt id (verification hint, else highest speedup),
            or ``""`` when there are no attempt rows.
    """
    rows = [a for a in attempts if isinstance(a, dict)]
    if not rows:
        return ""
    want_id = str(verification.get("best_attempt_id") or "")
    if want_id:
        return want_id
    want_backend = str(verification.get("best_backend") or "").lower()
    candidates = rows
    if want_backend:
        backend_rows = [a for a in rows if str(a.get("backend") or "").lower() == want_backend]
        if backend_rows:
            candidates = backend_rows

    def _spd(a: dict[str, Any]) -> float:
        """Return an attempt's micro/plain speedup (``-inf`` when absent).

        Args:
            a: An attempt record mapping.

        Returns:
            The attempt's ``micro_speedup`` (or ``speedup``) as a float, or
            ``-inf`` when neither is present.
        """
        v = to_float(a.get("micro_speedup") or a.get("speedup"))
        return v if v is not None else float("-inf")

    best = max(candidates, key=_spd)
    return str(best.get("attempt_id") or best.get("id") or "")


def _invocation_section(backend: str) -> str | None:
    """Map a kernel-agent backend to its invocation section name.

    Args:
        backend (str): the backend name (geak / forge / ...).

    Returns:
        str | None: the matching invocation section, or ``None`` when the backend
            has no invocation lane.
    """
    b = str(backend or "").lower()
    if b in _GEAK_BACKENDS:
        return "geak_invocations"
    if b in _FORGE_BACKENDS:
        return "forge_invocations"
    return None


def _session_key(session_dir: Path | str) -> str:
    """Return the stable session component used by canonical kernel ids."""
    try:
        return Path(session_dir).resolve().name
    except (TypeError, ValueError, OSError):
        return str(session_dir or "unknown")


_KERNEL_ROUTE_CONTEXT: dict[str, dict[str, str]] = {}


def _kernel_route_context_key(session_dir: Path | str) -> str:
    """Return the in-process context key for one session."""
    return _session_key(session_dir)


def _kernel_selection_operation_id(
    session_dir: Path | str,
    *,
    macro_cycle: int | None = None,
    run_discriminator: str = "",
) -> str:
    """Return the current cycle/run-specific strategy-selection operation id."""
    context_key = _kernel_route_context_key(session_dir)
    if macro_cycle is None and not run_discriminator:
        current = _KERNEL_ROUTE_CONTEXT.get(context_key, {}).get("selection_id")
        if current:
            return current
        run_discriminator = f"runtime:{_now_iso_safe()}"
    discriminator = f"macro_cycle:{int(macro_cycle)}" if macro_cycle is not None else f"run:{run_discriminator}"
    return _stable_id("op", "kernel_optimizer_selection", context_key, discriminator)


# Canonical kernel routes: the route operation every kernel record hangs under, and the strategy
# stamped on the kernel operation itself. Two entries today, and the pair is derived rather than
# written at each call site because the forge identity used to be a literal in nine places — a tenth
# reader was one edit away from silently parenting somebody else's kernels under Forge.
#
# A GEAK kernel replayed from its kernel_journey must NOT hang under the Forge route: the tree would
# then assert it ran beneath a route that never dispatched it, and a reader walking parents to answer
# "which optimizer produced this?" gets the wrong answer. The `geak` route operation already exists
# (see the GEAK dispatch writer below); this makes the replay reachable to it.
CANONICAL_KERNEL_ROUTES: dict[str, tuple[str, str]] = {
    # route_strategy -> (route operation name, strategy stamped on the kernel operation)
    "kernel_agent_forge": ("kernel_agent_forge", "forge"),
    "geak": ("geak", "geak"),
}


def _canonical_route(route_strategy: str | None) -> tuple[str, str]:
    """(route operation name, kernel strategy) for a canonical route.

    Unknown values fall back to the forge pair, which is what every caller got before routes were
    named, so an unrecognised string cannot silently drop a kernel off the streams.
    """
    return CANONICAL_KERNEL_ROUTES.get(str(route_strategy or ""), CANONICAL_KERNEL_ROUTES["kernel_agent_forge"])


def _kernel_route_operation_id(
    session_dir: Path | str,
    strategy: str,
    *,
    macro_cycle: int | None = None,
    run_discriminator: str = "",
) -> str:
    """Return the current cycle/run-specific competing route operation id."""
    context_key = _kernel_route_context_key(session_dir)
    if macro_cycle is None and not run_discriminator:
        current = _KERNEL_ROUTE_CONTEXT.get(context_key, {}).get(f"route:{strategy}")
        if current:
            return current
        run_discriminator = f"runtime:{_now_iso_safe()}"
    discriminator = f"macro_cycle:{int(macro_cycle)}" if macro_cycle is not None else f"run:{run_discriminator}"
    route_id = _stable_id(
        "op",
        "kernel_optimizer_run",
        context_key,
        discriminator,
        strategy,
    )
    _KERNEL_ROUTE_CONTEXT.setdefault(context_key, {})[f"route:{strategy}"] = route_id
    return route_id


def _kernel_route_subject_id(
    session_dir: Path | str,
    strategy: str,
    *,
    route_operation_id: str = "",
) -> str:
    """Return the stable subject id for one competing kernel route."""
    route_id = route_operation_id or _kernel_route_operation_id(session_dir, strategy)
    return _stable_id("subject", "kernel_optimizer_route", route_id, strategy)


def _kernel_subject_id(session_dir: Path | str, kernel_id: str) -> str:
    """Return the stable subject id for one native kernel."""
    return _stable_id("subject", "kernel", _session_key(session_dir), kernel_id)


def _kernel_operation_id(session_dir: Path | str, kernel_id: str) -> str:
    """Return the stable native per-kernel operation id."""
    return _stable_id("op", "kernel_optimization", _session_key(session_dir), kernel_id)


def _measurement_metadata(
    source: str,
    *,
    harness: Any = None,
    workload: Any = None,
    samples: Any = None,
    aggregation: Any = None,
) -> dict[str, Any]:
    """Build explicit measurement provenance, including partial metadata."""
    harness_value = harness if harness not in (None, "", {}) else {"status": "partial", "reason": "not_provided"}
    if isinstance(workload, Mapping) and workload:
        workload_value = dict(workload)
        workload_value.setdefault(
            "lock",
            "complete"
            if any(workload_value.get(key) is not None for key in ("model", "tp", "conc", "isl", "osl"))
            else "partial",
        )
        if workload_value["lock"] == "partial":
            workload_value.setdefault("status", "partial")
    else:
        workload_value = {"lock": "partial", "status": "partial", "reason": "not_provided"}
    sample_value = list(samples) if isinstance(samples, (list, tuple)) else []
    aggregation_value = (
        aggregation
        if aggregation not in (None, "", {})
        else {"method": "reported_value", "status": "partial", "sample_count": len(sample_value) or None}
    )
    return {
        "source": source,
        "harness": harness_value,
        "workload": workload_value,
        "samples": sample_value,
        "aggregation": aggregation_value,
        "metadata": {"completeness": "complete" if sample_value else "partial"},
    }


def record_kernel_strategy_selection(
    session_dir: Path | str | None,
    *,
    selected_strategy: str,
    actual_path: str,
    candidates: list[str] | tuple[str, ...] = ("geak", "kernel_agent_forge"),
    macro_cycle: int | None = None,
    run_discriminator: str = "",
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record the GEAK/native XOR decision and the selected route only."""
    if not session_dir:
        trace_skip(reason="no session_dir", section="operations")
        return
    selected = str(selected_strategy or "").strip()
    paths = ["geak", "kernel_agent_forge"]
    for value in candidates:
        candidate = str(value).strip()
        if candidate and candidate not in paths:
            paths.append(candidate)
    if selected not in paths:
        paths.append(selected)
    now = _now_iso_safe()
    discriminator = str(run_discriminator or "")
    if macro_cycle is None and not discriminator:
        discriminator = f"runtime:{now}"
    selection_id = _kernel_selection_operation_id(
        session_dir,
        macro_cycle=macro_cycle,
        run_discriminator=discriminator,
    )
    route_id = _kernel_route_operation_id(
        session_dir,
        selected,
        macro_cycle=macro_cycle,
        run_discriminator=discriminator,
    )
    context = _KERNEL_ROUTE_CONTEXT.setdefault(_kernel_route_context_key(session_dir), {})
    previous_selection_id = context.get("selection_id", "")
    previous_route_id = context.get("active_route_id", "")
    previous_strategy = context.get("active_strategy", "")
    version_key = f"selection_version:{selection_id}"
    selection_version = int(context.get(version_key, "0") or 0) + 1
    if (
        previous_selection_id == selection_id
        and previous_route_id
        and previous_route_id != route_id
        and previous_strategy != selected
    ):
        record_operation(
            session_dir,
            operation_id=previous_route_id,
            producer=producer,
            status="superseded",
            ended_at=now,
            extensions={
                "route_competition": {
                    "active": False,
                    "selected": False,
                    "superseded_by": route_id,
                    "selection_version": selection_version,
                }
            },
        )
    context["selection_id"] = selection_id
    context["active_route_id"] = route_id
    context["active_strategy"] = selected
    context[version_key] = str(selection_version)
    context["discriminator"] = f"macro_cycle:{int(macro_cycle)}" if macro_cycle is not None else f"run:{discriminator}"
    record_operation(
        session_dir,
        operation_id=selection_id,
        producer=producer,
        kind="strategy_selection",
        name="kernel_optimizer_strategy_selection",
        phase="KERNEL_AGENT",
        scope="phase",
        strategy_group="kernel_optimizer",
        strategy=selected,
        status="succeeded",
        executor_class="deterministic",
        macro_cycle=macro_cycle,
        started_at=now,
        ended_at=now,
        outputs={
            "candidates": paths,
            "selected_strategy": selected,
            "actual_path": str(actual_path or selected),
            "xor": True,
            "selection_version": selection_version,
        },
        decisions=[
            {
                "decision_id": _stable_id(
                    "decision",
                    selection_id,
                    "selected",
                    selection_version,
                ),
                "kind": "strategy_selection",
                "verdict": selected,
                "reason": "resolved_in_memory_kernel_configuration",
                "decided_at": now,
                "evidence": {"actual_path": str(actual_path or selected), "candidates": paths},
            }
        ],
    )
    subject_id = _kernel_route_subject_id(
        session_dir,
        selected,
        route_operation_id=route_id,
    )
    subject = {
        "subject_id": subject_id,
        "subject_type": "kernel_optimizer_route",
        "role": "selected",
        "name": selected,
        "attributes": {"strategy_group": "kernel_optimizer"},
    }
    record_subject(session_dir, subject, producer=producer, subject_id=subject_id)
    record_operation(
        session_dir,
        operation_id=route_id,
        producer=producer,
        kind="kernel_optimizer_run",
        name=selected,
        phase="KERNEL_AGENT",
        scope="run",
        strategy_group="kernel_optimizer",
        strategy=selected,
        status="running",
        executor_class="llm_tool" if selected == "geak" else "deterministic",
        parent_operation_id=selection_id,
        root_operation_id=selection_id,
        macro_cycle=macro_cycle,
        subject=subject,
        inputs={"selected_strategy": selected, "actual_path": str(actual_path or selected)},
        extensions={
            "route_competition": {
                "selected": True,
                "active": True,
                "xor": True,
                "selection_version": selection_version,
            }
        },
    )


def record_native_kernel_run_start(
    session_dir: Path | str | None,
    *,
    payload: Mapping[str, Any] | None = None,
    macro_cycle: int | None = None,
    route_operation_id: str = "",
    route_strategy: str = "kernel_agent_forge",
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Upsert the selected native Kernel Agent plus Forge route as running."""
    route_name, _ = _canonical_route(route_strategy)
    if not session_dir:
        trace_skip(reason="no session_dir", section="kernel_journey")
        return
    current_route_id = _KERNEL_ROUTE_CONTEXT.get(
        _kernel_route_context_key(session_dir),
        {},
    ).get(f"route:{route_name}", "")
    route_id = route_operation_id or _kernel_route_operation_id(
        session_dir,
        route_name,
        macro_cycle=macro_cycle,
        run_discriminator=""
        if current_route_id
        else str((payload or {}).get("run_id") or (payload or {}).get("task_id") or ""),
    )
    record_operation(
        session_dir,
        operation_id=route_id,
        producer=producer,
        kind="kernel_optimizer_run",
        name=route_name,
        phase="KERNEL_AGENT",
        scope="run",
        strategy_group="kernel_optimizer",
        strategy=route_name,
        executor_class="llm_tool" if route_name == "geak" else "deterministic",
        status="running",
        started_at=_now_iso_safe(),
        inputs=dict(payload or {}),
        subject={
            "subject_id": _kernel_route_subject_id(
                session_dir,
                route_name,
                route_operation_id=route_id,
            ),
            "subject_type": "kernel_optimizer_route",
            "role": "selected",
            # From route_name, like the operation above it. Left literal, a GEAK replay produced
            # operation.name=geak next to subject.name=kernel_agent_forge — one record naming two
            # different optimizers, and the subject is what identity lookups resolve against.
            "name": route_name,
        },
    )


def record_native_kernel_run_result(
    session_dir: Path | str | None,
    *,
    result: Mapping[str, Any],
    macro_cycle: int | None = None,
    route_operation_id: str = "",
    route_strategy: str = "kernel_agent_forge",
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Finalize the native Kernel Agent plus Forge route from its result."""
    route_name, _ = _canonical_route(route_strategy)
    if not session_dir:
        trace_skip(reason="no session_dir", section="kernel_journey")
        return
    value = dict(result or {})
    current_route_id = _KERNEL_ROUTE_CONTEXT.get(
        _kernel_route_context_key(session_dir),
        {},
    ).get(f"route:{route_name}", "")
    route_id = route_operation_id or _kernel_route_operation_id(
        session_dir,
        route_name,
        macro_cycle=macro_cycle,
        run_discriminator="" if current_route_id else str(value.get("run_id") or value.get("task_id") or ""),
    )
    result_status = _operation_status(value.get("status"))
    batch_results = value.get("batch_results") if isinstance(value.get("batch_results"), list) else []
    if batch_results:
        terminal = [_operation_status(item.get("status")) for item in batch_results if isinstance(item, Mapping)]
        result_status = "succeeded" if any(item == "succeeded" for item in terminal) else result_status
    record_operation(
        session_dir,
        operation_id=route_id,
        producer=producer,
        kind="kernel_optimizer_run",
        name=route_name,
        phase="KERNEL_AGENT",
        scope="run",
        strategy_group="kernel_optimizer",
        strategy=route_name,
        executor_class="llm_tool" if route_name == "geak" else "deterministic",
        status=result_status,
        ended_at=_now_iso_safe(),
        outputs={
            "status": value.get("status"),
            "batch_mode": value.get("batch_mode"),
            "batch_kernel_ids": value.get("batch_kernel_ids"),
            "backend_order": value.get("backend_order"),
            "result_kernel_id": value.get("kernel_id"),
        },
        error=value.get("error") or value.get("error_class"),
    )


def _geak_result_artifacts(
    session_dir: Path | str,
    operation_id: str,
    result: Mapping[str, Any],
    producer: str,
) -> list[str]:
    """Record GEAK artifact references already carried by the result."""
    artifact_ids: list[str] = []
    for field in (
        "eval_dir",
        "report_path",
        "final_launch_script",
        "bench_script",
        "final_patch",
        "kernel_journey_path",
    ):
        path = result.get(field)
        if not path:
            continue
        artifact_id = _stable_id("artifact", operation_id, field, path)
        record_artifact(
            session_dir,
            artifact_id=artifact_id,
            producer=producer,
            operation_id=operation_id,
            producer_operation_id=operation_id,
            kind=field,
            path=str(path),
            present=None,
            coverage={"status": "reference_only"},
        )
        artifact_ids.append(artifact_id)
    return artifact_ids


def record_geak_operation(
    session_dir: Path | str | None,
    *,
    stage: str,
    result: Mapping[str, Any] | None = None,
    status: str | None = None,
    validated: bool = False,
    measured_tput: Any = None,
    validation_source: str = "",
    macro_cycle: int | None = None,
    route_operation_id: str = "",
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Upsert the GEAK route across runner, candidate, rebench, and final validation."""
    if not session_dir:
        trace_skip(reason="no session_dir", section="operations")
        return
    value = dict(result or {})
    context = _KERNEL_ROUTE_CONTEXT.get(_kernel_route_context_key(session_dir), {})
    current_route_id = context.get("route:geak", "")
    run_discriminator = "" if current_route_id else str(value.get("run_id") or value.get("task_id") or "")
    route_id = route_operation_id or _kernel_route_operation_id(
        session_dir,
        "geak",
        macro_cycle=macro_cycle,
        run_discriminator=run_discriminator,
    )
    selection_id = _kernel_selection_operation_id(
        session_dir,
        macro_cycle=macro_cycle,
        run_discriminator=run_discriminator,
    )
    now = _now_iso_safe()
    normalized_status = _operation_status(status or value.get("status") or "running")
    if stage in {"runner_started", "candidate", "rebench_started", "geak_harness_fallback"}:
        normalized_status = "running"
    if validated:
        normalized_status = "succeeded"
    substep = {
        "substep_id": _stable_id("substep", route_id, stage),
        "kind": stage,
        "name": stage,
        "status": "succeeded" if validated else normalized_status,
        "ended_at": now if normalized_status != "running" or validated else "",
        "metadata": {
            "validation_source": validation_source or None,
            "final_validation": bool(validated),
        },
    }
    measurement_refs: list[str] = []
    # The bench run these numbers were read from. A rebench of the same route
    # writes a new report and so is kept as its own reading, while re-recording
    # a stage that already reported lands back on the reading it first wrote.
    occurrence_run = str(
        value.get("report_path") or value.get("eval_dir") or value.get("run_id") or value.get("task_id") or ""
    )
    workload = value.get("workload") if isinstance(value.get("workload"), Mapping) else {}
    harness = value.get("bench_client") or value.get("harness") or "geak_e2e"
    samples = value.get("samples") or value.get("throughput_samples")
    aggregation = value.get("aggregation")
    for label, raw, source_name, headline in (
        ("baseline", value.get("baseline_throughput_tok_s") or value.get("ref_tput"), "geak_runner", False),
        (
            "final",
            measured_tput if measured_tput is not None else value.get("final_throughput_tok_s"),
            validation_source or "geak_runner",
            bool(validated),
        ),
    ):
        numeric = to_float(raw)
        if numeric is None:
            continue
        measurement_id = _stable_id(
            "measurement",
            route_id,
            label,
            source_name,
            _measurement_occurrence(occurrence_run, value=numeric),
        )
        metadata = _measurement_metadata(
            source_name,
            harness=validation_source if headline and validation_source else harness,
            workload=workload,
            samples=samples,
            aggregation=aggregation,
        )
        record_measurement(
            session_dir,
            measurement_id=measurement_id,
            producer=producer,
            operation_id=route_id,
            kind="throughput",
            name=f"{label}_throughput",
            value=numeric,
            unit="tok/s",
            status="validated" if headline else "provisional",
            measured_at=now,
            metric_basis="output",
            dimensions={"role": label, "headline_eligible": headline},
            **metadata,
        )
        measurement_refs.append(measurement_id)
    artifact_refs = _geak_result_artifacts(session_dir, route_id, value, producer)
    parity = value.get("output_parity")
    alignment = value.get("alignment_metrics") if isinstance(value.get("alignment_metrics"), Mapping) else {}
    final_validation = (
        value.get("final_validation")
        if isinstance(value.get("final_validation"), Mapping)
        else value.get("validation")
        if isinstance(value.get("validation"), Mapping)
        else {}
    )
    internal_refs = list(value.get("internal_refs") or []) if isinstance(value.get("internal_refs"), list) else []
    for field in ("accepted_kernels", "accepted_heads", "kernels_attempted"):
        refs = value.get(field)
        if not isinstance(refs, list):
            continue
        for index, ref in enumerate(refs):
            ref_value = dict(ref) if isinstance(ref, Mapping) else {"value": ref}
            ref_key = ref_value.get("kernel_id") or ref_value.get("name") or ref_value.get("value") or index
            internal_refs.append(
                {
                    "relation_id": _stable_id("relation", route_id, "geak_internal", field, ref_key),
                    "kind": field,
                    "ref": ref_value,
                    "provisional": not validated,
                }
            )
    gates: list[dict[str, Any]] = []
    if parity is not None:
        parity_ok = _to_bool(parity)
        gates.append(
            {
                "gate_id": _stable_id("gate", route_id, "output_parity"),
                "kind": "correctness",
                "name": "output_parity",
                "status": "passed" if parity_ok is True else "failed" if parity_ok is False else "partial",
                "decision": "allow" if parity_ok is True else "deny" if parity_ok is False else "review",
                "evidence": {"output_parity": parity},
            }
        )
    if validated or final_validation:
        gates.append(
            {
                "gate_id": _stable_id("gate", route_id, "final_validation"),
                "kind": "final_validation",
                "name": "orchestrator_final_validation",
                "status": "passed" if validated else "partial",
                "decision": "allow" if validated else "review",
                "evaluated_at": now,
                "evidence": {
                    "source": validation_source or None,
                    "measured_tput": to_float(measured_tput),
                    "details": dict(final_validation),
                },
            }
        )
    timing_fields: dict[str, Any] = {}
    if stage == "runner_started":
        timing_fields["started_at"] = now
    if normalized_status != "running":
        timing_fields["ended_at"] = now
    record_operation(
        session_dir,
        operation_id=route_id,
        producer=producer,
        kind="kernel_optimizer_run",
        name="geak",
        phase="KERNEL_AGENT",
        scope="run",
        strategy_group="kernel_optimizer",
        strategy="geak",
        executor_class="llm_tool",
        status=normalized_status,
        parent_operation_id=selection_id,
        root_operation_id=selection_id,
        subject={
            "subject_id": _kernel_route_subject_id(
                session_dir,
                "geak",
                route_operation_id=route_id,
            ),
            "subject_type": "kernel_optimizer_route",
            "role": "selected",
            "name": "geak",
        },
        substeps=[substep],
        gates=gates,
        outputs={
            "status": value.get("status"),
            "returncode": value.get("returncode"),
            "accepted_config": value.get("accepted_config"),
            "throughput_speedup": value.get("throughput_speedup"),
            "final_validation_precedence": "orchestrator_final_validation",
        },
        measurement_refs=measurement_refs,
        artifact_refs=artifact_refs,
        extensions={
            "geak": {
                "candidate_state": "validated" if validated else "provisional",
                "output_parity": parity,
                "alignment": dict(alignment),
                "final_validation": dict(final_validation),
                "internal_refs": internal_refs,
            }
        },
        error=value.get("error") or value.get("error_class"),
        **timing_fields,
    )
    # A route is execution context, not an optimization attempt. Its final
    # validation belongs in the gates/measurements above; a KEEP adoption is
    # emitted by ``record_geak_e2e_attempt`` (aggregate route win) or
    # ``record_kernel_e2e`` (an attributable per-kernel win). Attaching an
    # adoption here creates an orphan from the optimization ledger because
    # ``kernel_optimizer_run`` is intentionally not an attempt kind.


def record_geak_e2e_attempt(
    session_dir: Path | str | None,
    *,
    kind: str,
    throughput_before: float,
    throughput_after: float,
    baseline_tput: float | None = None,
    gain_pct: float | None = None,
    attribution_eligible: bool = True,
    macro_cycle: int | None = None,
    accepted_config: Mapping[str, Any] | None = None,
    provenance: str = "",
    occurrence: Any = None,
    result: Mapping[str, Any] | None = None,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record one validated GEAK route-level win as a countable attempt.

    ``record_geak_operation`` describes the GEAK route, but route operations
    are intentionally excluded from the canonical optimization ledger.  This
    companion record carries the validated before/after pair on an attempt kind
    the ledger counts, so the GEAK dashboard bucket receives the gain without
    relying on per-kernel attribution.

    Args:
        result: GEAK's ``result.json`` payload, read only for the artifact
            paths (report, eval dir, journey, patch) attached to the adoption.
            Without them the ledger's keep names a gain with nothing on disk
            to audit it against.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section="operations")
        return
    before = to_float(throughput_before)
    after = to_float(throughput_after)
    if not (before and after and before > 0 and after > 0):
        trace_skip(reason="no throughput pair to attribute", section="operations")
        return
    try:
        attempt_kind = kind if kind in {"kernel_optimization", "gemm_tuning"} else "kernel_optimization"
        cycle = int(macro_cycle) if macro_cycle is not None else 0
        now = _now_iso_safe()
        route_id = _kernel_route_operation_id(session_dir, "geak", macro_cycle=macro_cycle)
        recorded_occurrence = occurrence if occurrence is not None else f"{before}->{after}"
        # The measured pair is part of the identity: one macro cycle can promote
        # twice (a rebench that beats an earlier promotion), and keying on the
        # cycle alone would merge the second win onto the first and lose its
        # gain. Re-writing the SAME pair still collapses, which is what keeps
        # the writer idempotent.
        operation_id = _stable_id(
            "op",
            "geak_e2e_attempt",
            _session_key(session_dir),
            f"macro_cycle:{cycle}",
            recorded_occurrence,
        )
        subject = {
            "subject_id": _kernel_route_subject_id(session_dir, "geak", route_operation_id=route_id),
            "subject_type": "kernel_optimizer_route",
            "role": "selected",
            "name": "geak",
        }
        measurement_refs: list[str] = []
        for name, numeric in (("baseline_throughput", before), ("final_throughput", after)):
            is_accounting_anchor = name == "baseline_throughput"
            measurement_id = _stable_id(
                "measurement",
                operation_id,
                name,
                _measurement_occurrence(recorded_occurrence, value=numeric),
            )
            record_measurement(
                session_dir,
                measurement_id=measurement_id,
                producer=producer,
                operation_id=operation_id,
                subject=subject,
                kind="throughput",
                name=name,
                value=numeric,
                unit="tok/s",
                # ``before`` is the residual ledger anchor, not a throughput
                # sample taken by the GEAK harness.  It can be synthesized as
                # ``pre_geak + claimed_kernel_delta`` so calling it validated
                # would put a fictitious measurement on the canonical stream.
                status="derived" if is_accounting_anchor else "validated",
                measured_at=now,
                metric_basis="output",
                dimensions={
                    "role": "baseline" if is_accounting_anchor else "final",
                    **(
                        {
                            "derived": True,
                            "derivation": "geak_route_residual_anchor",
                        }
                        if is_accounting_anchor
                        else {}
                    ),
                },
                **_measurement_metadata("geak_e2e_orchestrator", harness="geak_e2e"),
            )
            measurement_refs.append(measurement_id)
        artifact_refs = (
            _geak_result_artifacts(session_dir, operation_id, result, producer) if isinstance(result, Mapping) else []
        )
        record_operation(
            session_dir,
            operation_id=operation_id,
            producer=producer,
            kind=attempt_kind,
            name="geak_e2e",
            phase="KERNEL_AGENT",
            macro_cycle=cycle,
            scope="run",
            strategy_group="kernel_optimizer",
            strategy="geak",
            executor_class="llm_tool",
            status="succeeded",
            parent_operation_id=route_id,
            root_operation_id=route_id,
            subject=subject,
            outputs={
                "decision": "KEEP",
                "validated": True,
                "source": "geak_e2e",
                "baseline_tput": to_float(baseline_tput),
                "accepted_config": dict(accepted_config or {}),
            },
            measurement_refs=measurement_refs,
            artifact_refs=artifact_refs,
            ended_at=now,
        )
        adoption_id = _stable_id("adoption", operation_id, "geak_e2e")
        _record_adoption_transition(
            session_dir,
            adoption_id=adoption_id,
            producer=producer,
            operation_id=operation_id,
            adopted=True,
            attribution_eligible=bool(attribution_eligible),
            reason=provenance or "geak_e2e_validated",
            subject=subject,
            transitioned_at=now,
            measurement_ids=measurement_refs,
            artifact_ids=artifact_refs,
            kind=attempt_kind,
            gain_pct=to_float(gain_pct),
            throughput_before=before,
            throughput_after=after,
            configuration=dict(accepted_config or {}),
            validation_basis="e2e_validation",
            metadata={"validation_tier": "geak_e2e_orchestrator"},
        )
        record_operation(
            session_dir,
            operation_id=operation_id,
            producer=producer,
            adoption_refs=[adoption_id],
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_geak_e2e_attempt failed", exc_info=True)
        trace_skip(reason="writer raised", section="operations", error=exc)


def _record_geak_internal_ref(
    session_dir: Path | str,
    *,
    kernel_id: str,
    stage: str,
    payload: Mapping[str, Any],
    producer: str,
) -> None:
    """Keep GEAK-internal kernels as route extensions, never native children."""
    route_id = _kernel_route_operation_id(session_dir, "geak")
    relation_id = _stable_id("relation", route_id, "geak_internal", kernel_id, stage)
    record_operation(
        session_dir,
        operation_id=route_id,
        producer=producer,
        extensions={
            "geak": {
                "internal_refs": [
                    {
                        "relation_id": relation_id,
                        "kernel_id": kernel_id,
                        "stage": stage,
                        "evidence": dict(payload),
                        "provisional": True,
                    }
                ]
            }
        },
    )


def record_gemm_tuning_operation(
    session_dir: Path | str | None,
    *,
    payload: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
    macro_cycle: int | None = None,
    attempt_discriminator: str = "",
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record the independent Kernel-phase GEMM tuning run and KEEP adoption."""
    if not session_dir:
        trace_skip(reason="no session_dir", section="operations")
        return
    inputs = dict(payload or {})
    value = dict(result or {})
    task_id = str(inputs.get("task_id") or value.get("task_id") or "kernel_entry_gemm_tuning")
    cycle = (
        int(macro_cycle) if macro_cycle is not None else int(inputs.get("macro_cycle") or value.get("macro_cycle") or 0)
    )
    attempt_key = str(
        attempt_discriminator
        or inputs.get("attempt_id")
        or inputs.get("run_id")
        or value.get("attempt_id")
        or value.get("run_id")
        or task_id
    )
    operation_id = _stable_id(
        "op",
        "gemm_tuning",
        _session_key(session_dir),
        f"macro_cycle:{cycle}",
        f"attempt:{attempt_key}",
    )
    backend = str(value.get("backend") or inputs.get("gemm_tuning_backend") or "forge").lower()
    now = _now_iso_safe()
    status = "running" if result is None else _operation_status(value.get("status"))
    attempts: list[dict[str, Any]] = []
    attempt_rows = value.get("tuners_run") if isinstance(value.get("tuners_run"), list) else []
    for index, attempt in enumerate(attempt_rows):
        if not isinstance(attempt, Mapping):
            continue
        attempts.append(
            {
                "attempt_id": _stable_id("attempt", operation_id, attempt.get("tuner") or index),
                "backend": str(attempt.get("tuner") or backend),
                "status": _operation_status(attempt.get("status")),
                "outputs": dict(attempt),
            }
        )
    if value.get("fallback_backend"):
        attempts.append(
            {
                "attempt_id": _stable_id("attempt", operation_id, "fallback", value.get("fallback_backend")),
                "backend": str(value.get("fallback_backend")),
                "status": status,
                "metadata": {
                    "requested_backend": value.get("requested_backend"),
                    "fallback_reason": value.get("fallback_reason"),
                },
            }
        )
    measurement_refs: list[str] = []
    # The tuning attempt reads the kernel-time ratio once; the end-to-end
    # numbers beside it are filled in later, by a validation run of its own.
    # Keying each to the run it came from is what keeps that second pass from
    # re-issuing the ratio it never re-measured.
    e2e_run = str(value.get("final_report_path") or value.get("workspace") or "")
    for name, raw, basis, unit, occurrence_run in (
        ("best_speedup", value.get("best_speedup"), "kernel_time_ratio", "ratio", attempt_key),
        ("baseline_throughput", value.get("baseline_tput"), "output", "tok/s", e2e_run),
        ("final_throughput", value.get("new_tput") or value.get("final_throughput"), "output", "tok/s", e2e_run),
        ("e2e_gain_pct", value.get("e2e_gain_pct"), "output", "percent", e2e_run),
    ):
        numeric = to_float(raw)
        if numeric is None:
            continue
        occurrence = _measurement_occurrence(occurrence_run, value=numeric)
        measurement_id = _stable_id("measurement", operation_id, name, occurrence)
        record_measurement(
            session_dir,
            measurement_id=measurement_id,
            producer=producer,
            operation_id=operation_id,
            kind="gemm_tuning",
            name=name,
            value=numeric,
            unit=unit,
            status="validated" if value.get("e2e_validated") else "provisional",
            measured_at=now,
            metric_basis=basis,
            dimensions={"engine": backend},
            **_measurement_metadata(
                "gemm_tuning_result",
                harness=value.get("harness") or value.get("benchmark_script"),
                workload=value.get("workload"),
                samples=value.get("samples"),
                aggregation=value.get("aggregation"),
            ),
        )
        measurement_refs.append(measurement_id)
    artifact_refs: list[str] = []
    artifact_values: list[tuple[str, Any]] = [
        ("workspace", value.get("workspace")),
        ("tuned_file", value.get("tuned_file")),
        ("final_report_path", value.get("final_report_path")),
        ("benchmark_script", value.get("benchmark_script")),
    ]
    artifacts = value.get("artifacts")
    if isinstance(artifacts, Mapping):
        artifact_values.extend((str(name), path) for name, path in artifacts.items())
    for kind, path in artifact_values:
        if not path:
            continue
        artifact_id = _stable_id("artifact", operation_id, kind, path)
        record_artifact(
            session_dir,
            artifact_id=artifact_id,
            producer=producer,
            operation_id=operation_id,
            producer_operation_id=operation_id,
            kind=kind,
            path=str(path),
            coverage={"status": "reference_only"},
        )
        artifact_refs.append(artifact_id)
    decision = str(value.get("decision") or "").upper()
    timing_fields: dict[str, Any] = {"started_at": now} if result is None else {"ended_at": now}
    record_operation(
        session_dir,
        operation_id=operation_id,
        producer=producer,
        kind="gemm_tuning",
        name="gemm_tuning",
        phase="KERNEL_AGENT",
        macro_cycle=cycle,
        scope="run",
        strategy_group="gemm_engine",
        strategy=backend,
        executor_class="deterministic",
        status=status,
        inputs=inputs,
        outputs={
            "decision": decision,
            "engine": value.get("engine") or backend,
            "requested_backend": value.get("requested_backend"),
            "fallback_backend": value.get("fallback_backend"),
            "fallback_reason": value.get("fallback_reason"),
            "recommended_env": value.get("recommended_env"),
        },
        attempts=attempts,
        measurement_refs=measurement_refs,
        artifact_refs=artifact_refs,
        extensions={"gemm": {"e2e_validated": value.get("e2e_validated"), "result": value}},
        error=value.get("error") or value.get("error_class"),
        **timing_fields,
    )
    adoption_id = _stable_id("adoption", operation_id, "keep")
    if result is None:
        trace_skip(
            reason="no result to adopt on",
            section="adoptions",
            entity=adoption_id,
            producer=producer,
        )
        return
    e2e_keep = decision == "KEEP" and value.get("e2e_validated") is True
    if not e2e_keep and decision not in {"REVERT", "REJECTED"}:
        # A KEEP that end-to-end validation has not confirmed is not an
        # adoption. The operation is on the ledger either way, so without this
        # the missing adoption reads exactly like one that failed to write.
        trace_skip(
            reason=f"decision {decision!r} is not an e2e-validated keep or a revert",
            section="adoptions",
            entity=adoption_id,
            producer=producer,
        )
        return
    _record_adoption_transition(
        session_dir,
        adoption_id=adoption_id,
        producer=producer,
        operation_id=operation_id,
        adopted=e2e_keep,
        reason=str(value.get("decision_reason") or ("gemm_e2e_keep" if e2e_keep else "gemm_e2e_revert")),
        transitioned_at=now,
        measurement_ids=measurement_refs,
        artifact_ids=artifact_refs,
        kind="gemm_tuning",
        gain_pct=to_float(value.get("e2e_gain_pct")),
        configuration=dict(value.get("recommended_env") or value.get("extra_envs") or {}),
        validation_basis="e2e_validation",
    )
    record_operation(session_dir, operation_id=operation_id, producer=producer, adoption_refs=[adoption_id])


def record_collective_promotion(
    session_dir: Path | str | None,
    *,
    integration_id: str,
    kernel_id: str = "",
    baseline_tput: float | None = None,
    new_tput: float | None = None,
    gain_pct: float | None = None,
    patch_path: str = "",
    target_file: str = "",
    backend: str = "forge",
    collective_op: str = "",
    world_size: Any = None,
    kernel_speedup: Any = None,
    configuration: Mapping[str, Any] | None = None,
    ts: str | None = None,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record an end-to-end validated collective KEEP as it is promoted.

    The collective lane reaches its verdict through its own recovery path
    rather than the kernel integrate queue, so none of the kernel recorders
    fire for it. Left unrecorded, the read model cannot see the change at all:
    the patch lands, the workload moves, and the whole gain reports as
    belonging to no step. Recorded here, it is one attempt of kind
    ``kernel_collective`` with the adoption that credits it.

    Args:
        session_dir: The session directory; a falsy value is a no-op.
        integration_id: The integrate this promotion settled, which is what
            keeps two promotions of the same kernel apart.
        kernel_id: The kernel the collective change targets.
        baseline_tput: Session baseline throughput the gain is stated against.
        new_tput: Throughput measured after the change landed.
        gain_pct: The end-to-end gain the integrate measured.
        patch_path: The applied patch.
        target_file: The file the patch changed.
        backend: The engine that produced the change.
        collective_op: The collective operation optimized, for evidence.
        world_size: The world size it was measured at, for evidence.
        kernel_speedup: The kernel-time ratio behind the end-to-end figure.
        configuration: Environment carried by the change.
        ts: Author-time stamp the caller already minted for this promotion.
        producer: The breakdown producer label.
    """
    if not session_dir or not integration_id:
        trace_skip(
            reason="no session_dir" if not session_dir else "no integration_id",
            section="kernel_collective",
            entity=kernel_id,
        )
        return
    try:
        now = str(ts or _now_iso_safe())
        operation_id = _stable_id(
            "op",
            "kernel_collective",
            _session_key(session_dir),
            f"integration:{integration_id}",
        )
        subject = {
            "subject_id": _kernel_subject_id(session_dir, str(kernel_id or integration_id)),
            "subject_type": "kernel",
            "role": "optimization_target",
            "name": str(kernel_id or "forge_collective"),
        }
        measurement_refs: list[str] = []
        for name, raw, unit, basis in (
            ("baseline_throughput", baseline_tput, "tok/s", "output"),
            ("final_throughput", new_tput, "tok/s", "output"),
            ("e2e_gain_pct", gain_pct, "percent", "output"),
            ("best_speedup", kernel_speedup, "ratio", "kernel_time_ratio"),
        ):
            numeric = to_float(raw)
            if numeric is None:
                continue
            # Keyed by the integrate these came off, so re-running the lane
            # measures beside these rather than over them.
            occurrence = _measurement_occurrence(integration_id, value=numeric)
            measurement_id = _stable_id("measurement", operation_id, name, occurrence)
            record_measurement(
                session_dir,
                measurement_id=measurement_id,
                producer=producer,
                operation_id=operation_id,
                kind="kernel_collective",
                name=name,
                value=numeric,
                unit=unit,
                status="validated",
                measured_at=now,
                metric_basis=basis,
                dimensions={"engine": backend, "collective_op": str(collective_op or "")},
                **_measurement_metadata("collective_integrate_e2e"),
            )
            measurement_refs.append(measurement_id)
        artifact_refs: list[str] = []
        for kind, path in (("patch", patch_path), ("target_file", target_file)):
            if not path:
                continue
            artifact_id = _stable_id("artifact", operation_id, kind, path)
            record_artifact(
                session_dir,
                artifact_id=artifact_id,
                producer=producer,
                operation_id=operation_id,
                producer_operation_id=operation_id,
                subject=subject,
                kind=kind,
                path=str(path),
                coverage={"status": "reference_only"},
            )
            artifact_refs.append(artifact_id)
        adoption_id = _stable_id("adoption", operation_id, "collective")
        _record_adoption_transition(
            session_dir,
            adoption_id=adoption_id,
            producer=producer,
            operation_id=operation_id,
            adopted=True,
            reason="collective_integrate_e2e_passed",
            transitioned_at=now,
            subject=subject,
            measurement_ids=measurement_refs,
            artifact_ids=artifact_refs,
            kind="kernel_collective",
            agent="kernel_agent",
            gain_pct=to_float(gain_pct),
            throughput_before=to_float(baseline_tput),
            throughput_after=to_float(new_tput),
            configuration=dict(configuration or {}),
            validation_basis="e2e_validation",
            metadata={"validation_tier": "collective_integrate_e2e"},
        )
        record_operation(
            session_dir,
            operation_id=operation_id,
            producer=producer,
            kind="kernel_collective",
            name=str(kernel_id or "forge_collective"),
            phase="KERNEL_AGENT",
            scope="kernel",
            strategy_group="kernel_optimizer",
            strategy=backend,
            executor_class="deterministic",
            status="succeeded",
            ended_at=now,
            subject=subject,
            outputs={
                "integrated": True,
                "decision": "KEEP",
                "validated": True,
                "collective_op": str(collective_op or ""),
                "world_size": world_size,
                "integration_id": str(integration_id),
            },
            measurement_refs=measurement_refs,
            artifact_refs=artifact_refs,
            adoption_refs=[adoption_id],
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_collective_promotion failed", exc_info=True)
        trace_skip(reason="writer raised", section="kernel_collective", error=exc)


def record_kernel_invocations(
    session_dir: Path | str | None,
    result: dict[str, Any],
    *,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record kernel backend invocations from an in-process kernel-agent result.

    Reads ``result['attempts']`` (per-backend ladder) so backend-level failures
    are captured even when the kernel-agent crashed before persisting the
    on-disk source. When the whole invocation failed before any backend ran
    (pre-dispatch gating), a single ``FAILED`` marker is recorded so the failure
    stays visible in the invocation view.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        result (dict[str, Any]): the in-process kernel-agent result carrying the
            per-backend ``attempts`` ladder, verification, and proposal.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not isinstance(result, dict):
        trace_skip(reason="no session_dir" if not session_dir else "result is not a dict", section="kernel_invocations")
        return
    try:
        rec = _recorder(session_dir, producer)
        kid = str(result.get("kernel_id") or "")
        run_id = str(result.get("run_id") or result.get("session_id") or "")
        attempts = result.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        backend_names = {
            str(attempt.get("backend") or "").strip().lower()
            for attempt in attempts
            if isinstance(attempt, Mapping) and attempt.get("backend")
        }
        route_strategy = (
            "geak_internal"
            if backend_names == {"geak"} or (not backend_names and str(result.get("backend") or "").lower() == "geak")
            else "kernel_agent_forge"
        )
        record_kernel_backend_result(
            session_dir,
            result,
            route_strategy=route_strategy,
            producer=producer,
        )
        verification = result.get("verification") or {}
        proposal = result.get("proposal") or {}
        kernel_decision = str(proposal.get("decision") or "").upper()
        best_attempt_id = _best_attempt_id(attempts, verification)

        recorded_any = False
        for att in attempts:
            if not isinstance(att, dict):
                continue
            backend = str(att.get("backend") or "").lower()
            section = _invocation_section(backend)
            if section is None:
                continue
            status = str(att.get("status") or "").lower()
            decision = str(att.get("decision") or "").upper()
            if not decision and status in _FAILED_STATUSES:
                decision = "FAILED"
            attempt_id = str(att.get("attempt_id") or att.get("id") or "")
            # Stamp the kernel-level decision onto the adopted (best) attempt.
            if kernel_decision and attempt_id and attempt_id == best_attempt_id:
                decision = kernel_decision
            optimized = att.get("optimized_path") or att.get("optimized_file")
            payload = {
                "kernel_id": kid,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "ts": str(att.get("ts") or att.get("started_at") or ""),
                "backend": backend,
                "decision": decision,
                "status": status,
                "micro_speedup": to_float(att.get("micro_speedup") or att.get("speedup")),
                "optimized_files": [str(optimized)] if optimized else [],
                "error": att.get("error") or att.get("error_message"),
            }
            key = attempt_id or f"{kid}-{backend}"
            rec.record_item(section, payload, key=key)
            recorded_any = True

        if recorded_any:
            return

        # No per-backend attempts: capture a pre-dispatch / infra failure so
        # the invocation view still shows it (root cause of invisible failures).
        status = str(result.get("status") or "").lower()
        err_class = str(result.get("error_class") or "")
        decision = str((result.get("proposal") or {}).get("decision") or "").upper()
        failed = status in _FAILED_STATUSES or (decision == "REVERT" and bool(err_class))
        if not failed:
            return
        backend = str(result.get("backend") or "").lower()
        section = _invocation_section(backend)
        if section is None:
            # The backend could not be determined; do not fabricate a GEAK
            # invocation. The failure stays visible via the kernel_dispatch /
            # kernel_backend_result journey lanes, and this says which lane to
            # go looking in rather than leaving the invocation view short by
            # one with no explanation.
            trace_skip(
                reason=f"backend {backend!r} names no invocation stream",
                section="kernel_invocations",
                entity=kid,
                producer=producer,
            )
            return
        payload = {
            "kernel_id": kid,
            "attempt_id": "",
            "run_id": run_id,
            "backend": backend,
            "decision": "FAILED",
            "status": status or "failed",
            "error": result.get("error") or err_class or None,
            "error_class": err_class or None,
            # Distinguishes a pre-dispatch gating failure from a backend that ran and failed.
            "pre_dispatch_failure": True,
        }
        rec.record_item(section, payload, key=f"{kid}-predispatch" if kid else None)
    except Exception as exc:  # noqa: BLE001
        log.debug("record_kernel_invocations failed", exc_info=True)
        trace_skip(reason="writer raised", section="kernel_invocations", error=exc)


def _to_bool(value: Any) -> bool | None:
    """Coerce a loosely-typed truthy/falsy value to ``bool``.

    Args:
        value (Any): the value to interpret (a bool, or a string like
            ``"true"`` / ``"failed"`` / ``"ok"``).

    Returns:
        bool | None: the interpreted boolean, or ``None`` when ``value`` is
            None or not a recognized truthy/falsy token.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "pass", "passed", "ok"):
        return True
    if s in ("false", "0", "no", "fail", "failed"):
        return False
    return None


# Cache of resolved tool metadata, keyed by ``tool:root_dir`` (one-shot probe per key).
_TOOL_META_CACHE: dict[str, dict[str, Any]] = {}

# Per-tool "authoritative version" recipe. ``root_env`` holds the install root
# (used for the commit probe and git-based version strategies). ``version``
# picks how the human version is derived:
#   * "git_describe" -> ``git describe --tags --always --dirty`` of the root
#   * "git_short"    -> ``git rev-parse --short HEAD`` of the root (== commit)
#   * ("cmd", argv)  -> first line of ``argv --version`` style CLI output
#   * ("dist", names)-> importlib.metadata version of the first matching dist
_TOOL_PROVENANCE: dict[str, dict[str, Any]] = {
    "tracelens": {"root_env": "TRACELENS_ROOT", "version": "git_describe"},
    # The whole-pipeline GEAK e2e optimizer. Its checkout lives under $GEAK_ROOT
    # and its version is that repo's git SHA.
    "geak": {"root_env": "GEAK_ROOT", "version": "git_short"},
    # forge (the Kernel-Forge autonomous loop) ships inside this distribution,
    # so there is no checkout to ``git rev-parse``: its version is Hyperloom's.
    # The "forge" key stays -- downstream provenance JSON reads it by name.
    "forge": {"root_env": "", "version": ("dist", ("hyperloom-inference_optimizer",))},
    "claude": {"root_env": "", "version": ("cmd", ("claude", "--version"))},
    "codex": {"root_env": "", "version": ("cmd", ("codex", "--version"))},
    "inferencex": {"root_env": "INFERENCEX_PATH", "version": "git_short"},
    "kernel_agent": {"root_env": "HYPERLOOM_KERNEL_AGENT_ROOT", "version": "git_short"},
}


def _run_first_line(argv: list[str]) -> str:
    """Run ``argv`` and return the trimmed first output line (never raises).

    Args:
        argv (list[str]): the command argv to run.

    Returns:
        str: the trimmed first line of output (capped at 120 chars), or ``""``
            on failure / non-zero exit.
    """
    import subprocess  # local: keep module import cost off the common path

    try:
        out = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return ""
    if out.returncode != 0:
        return ""
    text = (out.stdout or "").strip() or (out.stderr or "").strip()
    return text.splitlines()[0].strip()[:120] if text else ""


def _git_short_commit(root: Path) -> str:
    """Best-effort ``git rev-parse --short HEAD`` for ``root`` (never raises).

    Args:
        root (Path): the repo root to inspect.

    Returns:
        str: the short commit hash, or ``""`` when it cannot be resolved.
    """
    return _run_first_line(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
    )


def _git_describe(root: Path) -> str:
    """Best-effort ``git describe --tags --always --dirty`` (never raises).

    Args:
        root (Path): the repo root to inspect.

    Returns:
        str: the ``git describe`` output, or ``""`` when it cannot be resolved.
    """
    return _run_first_line(
        ["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
    )


def _dist_version(names: tuple[str, ...]) -> str:
    """First resolvable ``importlib.metadata`` version among ``names`` ("" if none).

    Args:
        names (tuple[str, ...]): candidate distribution names to resolve in
            order.

    Returns:
        str: the first resolvable distribution version (rejecting a stale
            ``0.0.0``), or ``""`` when none resolve.
    """
    try:
        from importlib.metadata import version as _dist_ver
    except Exception:  # noqa: BLE001
        return ""
    for name in names:
        try:
            v = str(_dist_ver(name) or "").strip()
        except Exception:  # noqa: BLE001
            continue
        # Reject a stale 0.0.0 masquerade.
        if v and v != "0.0.0":
            return v
    return ""


def _probe_tool_version(strategy: Any, root_dir: str) -> str:
    """Resolve a tool's human version per its provenance ``strategy``.

    Args:
        strategy (Any): the provenance strategy (``"git_describe"`` /
            ``"git_short"`` / a ``("cmd", argv)`` or ``("dist", names)`` tuple).
        root_dir (str): the tool install root for git-based strategies.

    Returns:
        str: the resolved version string, or ``""`` when it cannot be derived.
    """
    try:
        if strategy == "git_describe":
            return _git_describe(Path(root_dir)) if root_dir else ""
        if strategy == "git_short":
            return _git_short_commit(Path(root_dir)) if root_dir else ""
        if isinstance(strategy, tuple) and len(strategy) == 2:
            kind, arg = strategy
            if kind == "cmd":
                return _run_first_line(list(arg))
            if kind == "dist":
                return _dist_version(tuple(arg))
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _tool_metadata(
    tool: str,
    *,
    root: str | None = None,
    root_env: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Resolve ``{tool, root_dir, commit, version}`` for an external tool.

    Root resolution: explicit ``root`` > caller ``root_env`` > the tool's
    registered ``root_env``. ``commit`` is a cached ``git rev-parse`` of the
    root. ``version`` is the caller-supplied value, else a cached per-tool probe
    following ``_TOOL_PROVENANCE``. Best-effort: never raises into the optimizer.

    Args:
        tool (str): the external tool name (keys into ``_TOOL_PROVENANCE``).
        root (str | None): an explicit install root, highest precedence.
        root_env (str | None): a caller-supplied env var naming the root.
        version (str | None): a caller-supplied version, preferred over the
            probe.

    Returns:
        dict[str, Any]: the resolved ``{tool, root_dir, commit, version}``
            metadata.
    """
    import os

    key = str(tool or "").lower()
    hint = _TOOL_PROVENANCE.get(key, {})
    root_dir = str(
        root or os.environ.get(root_env or "", "") or os.environ.get(str(hint.get("root_env") or ""), "")
    ).strip()
    cache_key = f"{key}:{root_dir}"
    cached = _TOOL_META_CACHE.get(cache_key)
    if cached is None:
        commit = ""
        if root_dir:
            try:
                if Path(root_dir).is_dir():
                    commit = _git_short_commit(Path(root_dir))
            except Exception:  # noqa: BLE001
                commit = ""
        probed = _probe_tool_version(hint.get("version"), root_dir) if hint else ""
        cached = {
            "tool": tool,
            "root_dir": root_dir,
            "commit": commit,
            "_probed_version": probed,
        }
        _TOOL_META_CACHE[cache_key] = cached
    meta = {
        "tool": cached["tool"],
        "root_dir": cached["root_dir"],
        "commit": cached["commit"],
    }
    meta["version"] = str(version or "") or str(cached.get("_probed_version") or "")
    return meta


def _normalize_hot_kernel(k: dict[str, Any]) -> dict[str, Any]:
    """Project a raw hot-kernel candidate onto the discovery shape.

    Args:
        k (dict[str, Any]): the raw hot-kernel candidate dict.

    Returns:
        dict[str, Any]: the candidate projected onto the normalized discovery
            shape.
    """
    return {
        "kernel_id": str(k.get("kernel_id") or k.get("id") or ""),
        "name": str(k.get("name") or k.get("kernel_name") or ""),
        "gpu_pct": to_float(k.get("gpu_pct") or k.get("gpu_percent")),
        "time_ms": to_float(k.get("time_ms") or k.get("duration_ms")),
        "bound_type": str(k.get("bound_type") or k.get("bottleneck") or ""),
        "arithmetic_intensity": to_float(k.get("arithmetic_intensity")),
        "flops_per_byte": to_float(k.get("flops_per_byte")),
        "efficiency_percent": to_float(k.get("efficiency_percent")),
        "reusable_native_kernel": bool(k.get("reusable_native_kernel") or False),
        "source_file": k.get("source_file"),
        "recommended_backends": list(k.get("recommended_backends") or []),
        "selected_for_optimization": bool(k.get("selected_for_optimization") or False),
    }


def record_kernel_discovery(
    session_dir: Path | str | None,
    *,
    source: str,
    status: str,
    hot_kernels: list[Any] | None = None,
    scan: dict[str, Any] | None = None,
    tool: str | None = None,
    tool_root: str | None = None,
    tool_root_env: str | None = None,
    tool_version: str | None = None,
    duration_sec: Any = None,
    error: str | None = None,
    route_strategy: str = "kernel_agent_forge",
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record one hot-kernel discovery run (stage 1 of ``kernel_journey``).

    One item per discovery invocation, keyed by the candidates/report path so a
    re-run with the same artifact overwrites rather than duplicates. Carries the
    full hot-kernel list the run surfaced.

    ``source`` is the discovery *route* label the dashboard groups by. ``tool``
    is the underlying tool whose authoritative version lands in the top-level
    ``versions`` map; it defaults to ``source`` but is decoupled because routes
    can share one toolchain (e.g. ``bypass`` reuses the TraceLens toolchain).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        source (str): the discovery route label the dashboard groups by.
        status (str): the discovery run status.
        hot_kernels (list[Any] | None): the hot-kernel candidates the run
            surfaced.
        scan (dict[str, Any] | None): scan metadata (carries the
            candidates/report path used as the idempotency key).
        tool (str | None): the underlying tool whose version is recorded
            (defaults to ``source``).
        tool_root (str | None): an explicit tool install root.
        tool_root_env (str | None): an env var naming the tool root.
        tool_version (str | None): a caller-supplied tool version.
        duration_sec (Any): the run duration in seconds.
        error (str | None): an error string when the run failed.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section="kernel_discovery")
        return
    try:
        kernels = [_normalize_hot_kernel(k) for k in (hot_kernels or []) if isinstance(k, dict)]
        scan = dict(scan or {})
        payload = {
            "source": str(source or ""),
            "status": str(status or ""),
            "ts": _now_iso_safe(),
            "duration_sec": to_float(duration_sec),
            "scan": scan,
            "hot_kernel_count": len(kernels),
            "hot_kernels": kernels,
            "error": error,
        }
        key = str(scan.get("candidates_path") or scan.get("trace_report_path") or "") or None
        _recorder(session_dir, producer).record_item(
            "kernel_discovery",
            payload,
            key=key,
        )
        # The discovery tool's authoritative version lands in the top-level
        # ``versions`` map, following the underlying ``tool``.
        record_tool_version(
            session_dir,
            tool=(tool or source),
            root=tool_root,
            root_env=tool_root_env,
            version=tool_version,
            producer=producer,
        )
        route = str(route_strategy or "kernel_agent_forge")
        if route == "legacy_only":
            # The legacy route stays out of the canonical streams by design, so
            # a session run on it has no operations at all. That is the same
            # shape as a session whose records were lost, and this is what
            # tells the two apart.
            trace_skip(
                reason="legacy_only route is not on the canonical streams",
                section="operations",
                entity="kernel_discovery",
                producer=producer,
            )
            return
        if route == "geak_internal":
            for kernel in kernels:
                kid = str(kernel.get("kernel_id") or "")
                if kid:
                    _record_geak_internal_ref(
                        session_dir,
                        kernel_id=kid,
                        stage="discovery",
                        payload=kernel,
                        producer=producer,
                    )
        else:
            record_native_kernel_run_start(
                session_dir,
                payload={"discovery_source": source, "tool": tool or source},
                route_strategy=route,
                producer=producer,
            )
            route_name, _ = _canonical_route(route)
            route_id = _kernel_route_operation_id(session_dir, route_name)
            record_operation(
                session_dir,
                operation_id=route_id,
                producer=producer,
                substeps=[
                    {
                        "substep_id": _stable_id("substep", route_id, "discovery"),
                        "kind": "kernel_discovery",
                        "name": "kernel_discovery",
                        "status": _operation_status(status),
                        "metadata": {
                            "source": source,
                            "tool": tool or source,
                            "hot_kernel_count": len(kernels),
                            "scan": scan,
                        },
                    }
                ],
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_kernel_discovery failed", exc_info=True)
        trace_skip(reason="writer raised", section="kernel_discovery", error=exc)


def record_tool_version(
    session_dir: Path | str | None,
    *,
    tool: str,
    root: str | None = None,
    root_env: str | None = None,
    version: str | None = None,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record one external tool's authoritative version into ``versions``.

    Idempotent per tool name (last write wins). Resolves ``{tool, root_dir,
    commit, version}`` via the tool provenance registry and spools it as one
    ``versions`` item; the assembler folds the substream into the top-level
    ``versions`` map. Best-effort: never raises into the optimizer.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        tool (str): the external tool name; a falsy value is a no-op.
        root (str | None): an explicit tool install root.
        root_env (str | None): an env var naming the tool root.
        version (str | None): a caller-supplied version, preferred over the
            probe.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not tool:
        trace_skip(reason="no session_dir" if not session_dir else "no tool", section="versions")
        return
    try:
        meta = _tool_metadata(
            tool,
            root=root,
            root_env=root_env,
            version=version,
        )
        _recorder(session_dir, producer).record_item(
            "versions",
            meta,
            key=str(tool).lower(),
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_tool_version failed", exc_info=True)
        trace_skip(reason="writer raised", section="versions", error=exc)


def record_kernel_dispatch(
    session_dir: Path | str | None,
    *,
    kernel_id: str,
    dispatched: bool,
    backends: list[str] | None = None,
    skip_reason: str = "",
    orchestration_commit: str = "",
    task_group: str | None = None,
    route_strategy: str = "kernel_agent_forge",
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record the dispatch decision for one kernel (stage 2 of ``kernel_journey``).

    Idempotent per ``kernel_id`` (last decision wins). ``dispatched`` is False
    for kernels gated out before any backend ran, with ``skip_reason`` holding
    the gate (non_reusable_kernel / missing_source / budget_exhausted / ...).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        kernel_id (str): the kernel id the decision is keyed by; a falsy value
            is a no-op.
        dispatched (bool): whether the kernel was dispatched to a backend.
        backends (list[str] | None): the backends the kernel was dispatched to.
        skip_reason (str): the gate that blocked dispatch when ``dispatched`` is
            False.
        orchestration_commit (str): the orchestration commit at dispatch time.
        task_group (str | None): the task group label.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not kernel_id:
        trace_skip(reason="no session_dir" if not session_dir else "no kernel_id", section="kernel_dispatch")
        return
    try:
        payload = {
            "kernel_id": str(kernel_id),
            "dispatched": bool(dispatched),
            "backends": [str(b) for b in (backends or [])],
            "skip_reason": str(skip_reason or ""),
            "orchestration_commit": str(orchestration_commit or ""),
            "task_group": task_group,
            "ts": _now_iso_safe(),
        }
        _recorder(session_dir, producer).record_item(
            "kernel_dispatch",
            payload,
            key=str(kernel_id),
        )
        if str(route_strategy or "") == "legacy_only":
            trace_skip(
                reason="legacy_only route is not on the canonical streams",
                section="operations",
                entity=str(kernel_id),
                producer=producer,
            )
            return
        if str(route_strategy or "") == "geak_internal":
            _record_geak_internal_ref(
                session_dir,
                kernel_id=str(kernel_id),
                stage="dispatch",
                payload=payload,
                producer=producer,
            )
            # Recorded, but on the GEAK-internal reference stream rather than
            # as an operation, so this kernel is absent from the ledger by
            # routing rather than by loss.
            trace_skip(
                reason="geak_internal route recorded as an internal reference",
                section="operations",
                entity=str(kernel_id),
                producer=producer,
            )
            return
        record_native_kernel_run_start(session_dir, route_strategy=route_strategy, producer=producer)
        subject_id = _kernel_subject_id(session_dir, str(kernel_id))
        operation_id = _kernel_operation_id(session_dir, str(kernel_id))
        subject = {
            "subject_id": subject_id,
            "subject_type": "kernel",
            "role": "optimization_target",
            "name": str(kernel_id),
            "attributes": {"task_group": task_group},
        }
        record_subject(session_dir, subject, subject_id=subject_id, producer=producer)
        record_operation(
            session_dir,
            operation_id=operation_id,
            producer=producer,
            kind="kernel_optimization",
            name=str(kernel_id),
            phase="KERNEL_AGENT",
            scope="kernel",
            strategy_group="kernel_backend",
            strategy=_canonical_route(route_strategy)[1],
            executor_class="llm_tool",
            status="running" if dispatched else "skipped",
            started_at=_now_iso_safe(),
            parent_operation_id=_kernel_route_operation_id(session_dir, _canonical_route(route_strategy)[0]),
            root_operation_id=_kernel_route_operation_id(session_dir, _canonical_route(route_strategy)[0]),
            subject=subject,
            inputs={
                "backends": [str(value) for value in (backends or [])],
                "task_group": task_group,
                "orchestration_commit": orchestration_commit,
            },
            gates=[
                {
                    "gate_id": _stable_id("gate", operation_id, "dispatch"),
                    "kind": "dispatch",
                    "name": "kernel_dispatch",
                    "status": "passed" if dispatched else "failed",
                    "decision": "allow" if dispatched else "deny",
                    "reason": str(skip_reason or ""),
                    "evidence": {"backends": [str(value) for value in (backends or [])]},
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_kernel_dispatch failed", exc_info=True)
        trace_skip(reason="writer raised", section="kernel_dispatch", error=exc)


def record_kernel_backend_result(
    session_dir: Path | str | None,
    result: dict[str, Any],
    *,
    route_strategy: str = "kernel_agent_forge",
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record per-backend attempts for one kernel (stage 3 of ``kernel_journey``).

    One item per attempt, keyed by ``attempt_id`` (falls back to
    ``run_id-backend``) so retries across runs are preserved rather than
    collapsed. Mirrors the attempt ladder in ``result['attempts']`` and carries
    the per-attempt timing + tool metadata when the kernel-agent surfaced them.

    An attempt row as the backend writes it holds only how the run went --
    ``status`` / ``optimized_path`` / ``error``. The verdict
    (``proposal.decision``), the compile/correctness gates and the source
    artifact are computed once per kernel and live beside ``attempts`` on the
    result, so they are folded onto the attempt ``verification`` adopted and
    onto no other. A losing backend keeps its own status, an unknown gate
    stays unknown, and a failed attempt with no verdict of its own is recorded
    as ``FAILED``.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        result (dict[str, Any]): the kernel-agent result carrying the
            per-backend ``attempts`` ladder, verification, and metadata.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not isinstance(result, dict):
        trace_skip(
            reason="no session_dir" if not session_dir else "result is not a dict", section="kernel_backend_result"
        )
        return
    try:
        rec = _recorder(session_dir, producer)
        kid = str(result.get("kernel_id") or "")
        run_id = str(result.get("run_id") or result.get("session_id") or "")
        attempts = result.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        result_meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        # The kernel-level micro_speedup (best across attempts) lives in
        # ``verification``; stamp it onto the adopted (best) attempt.
        verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
        best_attempt_id = _best_attempt_id(attempts, verification)
        kernel_micro_speedup = to_float(verification.get("micro_speedup"))
        # The verdict and the verification evidence are kernel-level facts: a
        # backend attempt carries only status/optimized_path/error, while
        # ``proposal.decision``, the compile/correctness gates and the source
        # artifact all live beside ``attempts`` on the result. They belong to
        # the attempt verification actually adopted, so they are stamped onto
        # that one and no other -- the same rule
        # :func:`record_kernel_invocations` applies to the invocation lanes.
        # Without an explicit ``best_attempt_id`` nothing was adopted, so no
        # attempt inherits them (``_best_attempt_id``'s speedup fallback would
        # otherwise hand a failed kernel's REVERT to an arbitrary row).
        proposal = result.get("proposal") if isinstance(result.get("proposal"), dict) else {}
        kernel_decision = str(proposal.get("decision") or "").upper()
        adopted_attempt_id = str(verification.get("best_attempt_id") or "")
        kernel_artifact_path = str(verification.get("best_artifact_path") or "")
        legacy_only = str(route_strategy or "") == "legacy_only"
        geak_internal = str(route_strategy or "") == "geak_internal"
        if geak_internal:
            if kid:
                _record_geak_internal_ref(
                    session_dir,
                    kernel_id=kid,
                    stage="backend_result",
                    payload={
                        "run_id": run_id,
                        "attempts": attempts,
                        "verification": verification,
                        "proposal": result.get("proposal"),
                    },
                    producer=producer,
                )
        elif kid and not legacy_only:
            record_native_kernel_run_start(session_dir, route_strategy=route_strategy, producer=producer)
            subject_id = _kernel_subject_id(session_dir, kid)
            subject = {
                "subject_id": subject_id,
                "subject_type": "kernel",
                "role": "optimization_target",
                "name": kid,
            }
            record_subject(session_dir, subject, subject_id=subject_id, producer=producer)
        operation_id = _kernel_operation_id(session_dir, kid) if kid and not geak_internal and not legacy_only else ""
        canonical_attempts: list[dict[str, Any]] = []
        canonical_gates: list[dict[str, Any]] = []
        measurement_refs: list[str] = []
        artifact_refs: list[str] = []
        recorded_any = False
        for att in attempts:
            if not isinstance(att, dict):
                continue
            backend = str(att.get("backend") or "").lower()
            attempt_id = str(att.get("attempt_id") or att.get("id") or "")
            optimized = att.get("optimized_path") or att.get("optimized_file")
            att_meta = att.get("metadata") if isinstance(att.get("metadata"), dict) else {}
            micro_speedup = to_float(att.get("micro_speedup") or att.get("speedup"))
            if (
                micro_speedup is None
                and kernel_micro_speedup is not None
                and attempt_id
                and attempt_id == best_attempt_id
            ):
                micro_speedup = kernel_micro_speedup
            is_adopted = bool(attempt_id) and attempt_id == adopted_attempt_id
            status_lower = str(att.get("status") or "").lower()
            decision = str(att.get("decision") or "").upper()
            if not decision and status_lower in _FAILED_STATUSES:
                decision = "FAILED"
            if is_adopted and kernel_decision:
                decision = kernel_decision
            compile_passed = _to_bool(att.get("compile_passed"))
            correctness_passed = _to_bool(att.get("correctness_passed"))
            correctness_source = att.get("correctness_source")
            if is_adopted:
                if compile_passed is None:
                    compile_passed = _to_bool(verification.get("compile_passed"))
                if correctness_passed is None:
                    correctness_passed = _to_bool(verification.get("correctness_passed"))
                correctness_source = correctness_source or verification.get("correctness_source")
            payload = {
                "kernel_id": kid,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "backend": backend,
                "model": att.get("model"),
                "ts": str(att.get("ts") or att.get("started_at") or att.get("created_at") or ""),
                "status": status_lower,
                "decision": decision,
                "micro_speedup": micro_speedup,
                "compile_passed": compile_passed,
                "correctness_passed": correctness_passed,
                "correctness_source": str(correctness_source) if correctness_source else None,
                # The source artifact the kernel was carried to integrate with.
                # ``optimized_files`` is the attempt's own output path, which
                # for a real backend run is its stdout log -- not the rewrite.
                "best_artifact_path": kernel_artifact_path if is_adopted else "",
                "optimized_files": [str(optimized)] if optimized else [],
                "error": att.get("error") or att.get("error_message"),
                "error_class": str(att.get("error_type") or "") or None,
                "duration_sec": to_float(att.get("duration_sec") or att.get("elapsed_sec") or att.get("elapsed_s")),
            }
            key = attempt_id or (f"{run_id}-{backend}" if run_id else None)
            rec.record_item("kernel_backend_result", payload, key=key)
            recorded_any = True
            if operation_id:
                canonical_attempt_id = attempt_id or _stable_id("attempt", operation_id, run_id, backend)
                # Resolved above: kernel-level evidence only reaches the
                # adopted attempt, so a losing backend's gates stay unknown
                # rather than inheriting the winner's.
                canonical_correctness_source = correctness_source or "partial:not_provided"
                canonical_attempts.append(
                    {
                        "attempt_id": canonical_attempt_id,
                        "backend": backend,
                        "status": _operation_status(att.get("status")),
                        "started_at": str(att.get("started_at") or att.get("created_at") or ""),
                        "ended_at": str(att.get("ended_at") or ""),
                        "outputs": {
                            "decision": decision,
                            "compile_passed": compile_passed,
                            "correctness_passed": correctness_passed,
                            "correctness_source": canonical_correctness_source,
                            "verification_status": verification.get("status"),
                        },
                        "error": att.get("error") or att.get("error_message"),
                    }
                )
                for gate_name, gate_value, gate_kind in (
                    ("compile", compile_passed, "compile"),
                    ("correctness", correctness_passed, "correctness"),
                ):
                    canonical_gates.append(
                        {
                            "gate_id": _stable_id("gate", operation_id, canonical_attempt_id, gate_name),
                            "kind": gate_kind,
                            "name": gate_name,
                            "status": "passed"
                            if gate_value is True
                            else "failed"
                            if gate_value is False
                            else "partial",
                            "decision": "allow" if gate_value is True else "deny" if gate_value is False else "review",
                            "evidence": {"source": canonical_correctness_source, "value": gate_value},
                        }
                    )
                numeric_speedup = to_float(att.get("micro_speedup") or att.get("speedup"))
                if numeric_speedup is not None:
                    measurement_id = _stable_id(
                        "measurement",
                        operation_id,
                        canonical_attempt_id,
                        "micro_speedup",
                        _measurement_occurrence(
                            att.get("ended_at") or att.get("started_at"),
                            value=numeric_speedup,
                        ),
                    )
                    record_measurement(
                        session_dir,
                        measurement_id=measurement_id,
                        producer=producer,
                        operation_id=operation_id,
                        subject=subject,
                        kind="kernel_benchmark",
                        name="micro_speedup",
                        value=numeric_speedup,
                        unit="ratio",
                        status="provisional",
                        metric_basis="kernel_time_ratio",
                        dimensions={"backend": backend, "attempt_id": canonical_attempt_id},
                        **_measurement_metadata(
                            "kernel_agent_result",
                            harness=att.get("harness") or att.get("benchmark_file"),
                            workload=att.get("workload"),
                            samples=att.get("samples"),
                            aggregation=att.get("aggregation"),
                        ),
                    )
                    measurement_refs.append(measurement_id)
                if optimized:
                    artifact_id = _stable_id("artifact", operation_id, canonical_attempt_id, optimized)
                    record_artifact(
                        session_dir,
                        artifact_id=artifact_id,
                        producer=producer,
                        operation_id=operation_id,
                        producer_operation_id=operation_id,
                        subject=subject,
                        kind="optimized_kernel",
                        path=str(optimized),
                        coverage={"status": "reference_only"},
                    )
                    artifact_refs.append(artifact_id)
            # The backend's authoritative version lands in the top-level ``versions`` map.
            if backend:
                record_tool_version(
                    session_dir,
                    tool=backend,
                    root=str(att_meta.get("root_dir") or result_meta.get("root_dir") or "") or None,
                    version=str(att_meta.get("version") or result_meta.get("version") or "") or None,
                    producer=producer,
                )

        if operation_id:
            proposal = result.get("proposal") if isinstance(result.get("proposal"), Mapping) else {}
            decision = str(proposal.get("decision") or "").upper()
            operation_status = _operation_status(result.get("status"))
            if decision in {"KEEP", "REVERT", "PARTIAL", "NEEDS_REVIEW"}:
                operation_status = "succeeded" if decision == "KEEP" else decision.lower()
            canonical_gates.append(
                {
                    "gate_id": _stable_id("gate", operation_id, "verification"),
                    "kind": "verification",
                    "name": "kernel_verification",
                    "status": str(verification.get("status") or ("passed" if decision == "KEEP" else "partial")),
                    "decision": "allow" if decision == "KEEP" else "review",
                    "evidence": {
                        "compile_passed": _to_bool(verification.get("compile_passed")),
                        "correctness_passed": _to_bool(verification.get("correctness_passed")),
                        "correctness_source": verification.get("correctness_source") or "partial:not_provided",
                        "best_attempt_id": best_attempt_id,
                        "best_backend": verification.get("best_backend"),
                    },
                }
            )
            record_operation(
                session_dir,
                operation_id=operation_id,
                producer=producer,
                kind="kernel_optimization",
                name=kid,
                phase="KERNEL_AGENT",
                scope="kernel",
                strategy_group="kernel_backend",
                strategy=str(verification.get("best_backend") or result.get("backend") or "forge"),
                executor_class="llm_tool",
                status=operation_status,
                ended_at=_now_iso_safe() if operation_status != "running" else "",
                parent_operation_id=_kernel_route_operation_id(session_dir, _canonical_route(route_strategy)[0]),
                root_operation_id=_kernel_route_operation_id(session_dir, _canonical_route(route_strategy)[0]),
                subject=subject,
                attempts=canonical_attempts,
                gates=canonical_gates,
                decisions=[
                    {
                        "decision_id": _stable_id("decision", operation_id, "proposal"),
                        "kind": "kernel_proposal",
                        "verdict": decision or "unknown",
                        "reason": "; ".join(str(value) for value in (proposal.get("reasons") or [])),
                        "evidence": dict(proposal),
                    }
                ],
                outputs={"verification": dict(verification), "proposal": dict(proposal)},
                measurement_refs=measurement_refs,
                artifact_refs=artifact_refs,
                extensions={"forge": {"correctness_source": verification.get("correctness_source")}},
                error=result.get("error") or result.get("error_class"),
            )

        if recorded_any or not kid:
            return

        # No per-backend attempts: capture a pre-dispatch / infra failure as a
        # synthetic FAILED attempt so kernel_journey shows the failure too.
        status = str(result.get("status") or "").lower()
        err_class = str(result.get("error_class") or "")
        decision = str((result.get("proposal") or {}).get("decision") or "").upper()
        failed = status in _FAILED_STATUSES or (decision == "REVERT" and bool(err_class))
        if not failed:
            return
        # Never default an unattributable failure to GEAK; record it as
        # "unknown" so GEAK's failure count is not inflated.
        backend = str(result.get("backend") or "").lower() or "unknown"
        payload = {
            "kernel_id": kid,
            "attempt_id": "",
            "run_id": run_id,
            "backend": backend,
            "model": None,
            "ts": _now_iso_safe(),
            "status": status or "failed",
            "decision": "FAILED",
            "micro_speedup": None,
            "compile_passed": None,
            "correctness_passed": None,
            "optimized_files": [],
            "error": result.get("error") or err_class or None,
            "error_class": err_class or None,
            "duration_sec": None,
            # Distinguishes a pre-dispatch gating failure from a backend that ran and failed.
            "pre_dispatch_failure": True,
        }
        rec.record_item(
            "kernel_backend_result",
            payload,
            key=f"{kid}-predispatch",
        )
        if operation_id:
            record_operation(
                session_dir,
                operation_id=operation_id,
                producer=producer,
                kind="kernel_optimization",
                name=kid,
                phase="KERNEL_AGENT",
                scope="kernel",
                strategy_group="kernel_backend",
                strategy=backend,
                executor_class="llm_tool",
                status="failed",
                ended_at=_now_iso_safe(),
                parent_operation_id=_kernel_route_operation_id(session_dir, _canonical_route(route_strategy)[0]),
                root_operation_id=_kernel_route_operation_id(session_dir, _canonical_route(route_strategy)[0]),
                attempts=[
                    {
                        "attempt_id": _stable_id("attempt", operation_id, "predispatch"),
                        "backend": backend,
                        "status": "failed",
                        "error": payload.get("error"),
                        "metadata": {"pre_dispatch_failure": True},
                    }
                ],
                error=payload.get("error") or payload.get("error_class"),
            )
        if backend != "unknown":
            record_tool_version(
                session_dir,
                tool=backend,
                root=str(result_meta.get("root_dir") or "") or None,
                version=str(result_meta.get("version") or "") or None,
                producer=producer,
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_kernel_backend_result failed", exc_info=True)
        trace_skip(reason="writer raised", section="kernel_backend_result", error=exc)


def record_kernel_e2e(
    session_dir: Path | str | None,
    *,
    kernel_id: str,
    integrated: bool,
    e2e_gain_pct: Any = None,
    validated: bool | None = None,
    decision: str = "",
    patch_path: str | None = None,
    target_file: str | None = None,
    extra_server_args: str = "",
    result: Mapping[str, Any] | None = None,
    route_strategy: str = "kernel_agent_forge",
    validation_tier: str = "",
    occurrence: Any = None,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record the latest end-to-end integrate outcome for one kernel (stage 4).

    Idempotent per ``kernel_id`` using overwrite-on-rewrite semantics: a later
    final-validation verdict replaces the provisional candidate verdict rather
    than appending a duplicate. ``e2e_gain_pct`` is the validated end-to-end
    gain at integrate (negative => regressed and reverted).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        kernel_id (str): the kernel id the outcome is keyed by; a falsy value is
            a no-op.
        integrated (bool): whether the kernel change was integrated.
        e2e_gain_pct (Any): the validated end-to-end gain percent at integrate.
        validated (bool | None): whether the gain was validated.
        decision (str): the integrate decision (KEEP / REVERT / ...).
        patch_path (str | None): the applied patch path.
        target_file (str | None): the integrated target file.
        extra_server_args (str): extra server args carried by the change.
        occurrence (Any): which integrate of this kernel these numbers were read
            from, as named by the caller; defaults to the integration id, then
            to the benchmark the integrate was graded on. Keeps a later
            re-measure from landing on the readings an earlier KEEP was decided
            on, and is recorded so a replay after resume reproduces the ids.
        producer (str): the breakdown producer label (defaults to the
            kernel-agent).
    """
    if not session_dir or not kernel_id:
        trace_skip(reason="no session_dir" if not session_dir else "no kernel_id", section="kernel_e2e")
        return
    try:
        evidence = dict(result or {})
        # Which integrate these numbers were read from. ``integration_id`` names
        # it whenever the kernel-patch queue issued one, but an env-only
        # integrate carries no artifact, so it never enters that queue and never
        # gets an id -- it is graded on its runtime bundle alone. Naming those by
        # the benchmark they were graded on keeps them apart, and a replay hands
        # back the same report path rather than a fresh count. Relative, since
        # this is recorded: the run is what identifies it, not where it ran.
        graded_on = str(evidence.get("report_path") or evidence.get("workspace") or "")
        recorded_occurrence = (
            occurrence
            if occurrence is not None
            else str(evidence.get("integration_id") or "")
            or (_rel(Path(graded_on), session_dir) if graded_on else "")
            or None
        )
        payload = {
            "kernel_id": str(kernel_id),
            "integrated": bool(integrated),
            "e2e_gain_pct": to_float(e2e_gain_pct),
            "validated": bool(validated) if validated is not None else None,
            "decision": str(decision or "").upper(),
            "patch_path": patch_path,
            "target_file": target_file,
            "extra_server_args": str(extra_server_args or ""),
            "ts": _now_iso_safe(),
            # Carried on the record so the replay paths that re-record this
            # outcome from state pass the same value back instead of counting
            # a fresh occurrence on every pass.
            "occurrence": recorded_occurrence,
        }
        for field in (
            "self_reported_e2e_gain_pct",
            "revalidation_measured_tput",
            "revalidation_current_best_tput",
            "revalidation_provenance",
            "rejection_reason",
        ):
            if evidence.get(field) is not None:
                payload[field] = evidence[field]
        if validation_tier:
            payload["validation_tier"] = validation_tier
        _recorder(session_dir, producer).record_item(
            "kernel_e2e",
            payload,
            key=str(kernel_id),
        )
        if str(route_strategy or "") == "legacy_only":
            # No operation and no adoption for this integrate. On a KEEP that
            # is a change the workload carries with nothing on the ledger
            # claiming it, which is precisely the shape a lost write leaves.
            trace_skip(
                reason="legacy_only route is not on the canonical streams",
                section="adoptions",
                entity=str(kernel_id),
                producer=producer,
            )
            return
        if str(route_strategy or "") == "geak_internal":
            _record_geak_internal_ref(
                session_dir,
                kernel_id=str(kernel_id),
                stage="e2e",
                payload={**payload, "result": evidence},
                producer=producer,
            )
            trace_skip(
                reason="geak_internal route recorded as an internal reference",
                section="adoptions",
                entity=str(kernel_id),
                producer=producer,
            )
            return
        operation_id = _kernel_operation_id(session_dir, str(kernel_id))
        subject = {
            "subject_id": _kernel_subject_id(session_dir, str(kernel_id)),
            "subject_type": "kernel",
            "role": "optimization_target",
            "name": str(kernel_id),
        }
        decision_value = str(decision or "").upper()
        measurement_refs: list[str] = []
        # One integrate of one kernel is one occurrence. Re-integrating the
        # same kernel later measures it again, and those numbers must not
        # displace the ones an earlier KEEP was decided on. All three readings
        # below come off that single benchmark, so they share its key.
        for name, raw, role in (
            ("baseline_throughput", evidence.get("base_tput"), "baseline"),
            ("final_throughput", evidence.get("new_tput"), "final"),
            ("e2e_gain_pct", e2e_gain_pct, "delta"),
        ):
            numeric = to_float(raw)
            if numeric is None:
                continue
            measurement_id = _stable_id(
                "measurement",
                operation_id,
                "integrate",
                name,
                _measurement_occurrence(recorded_occurrence, value=numeric),
            )
            record_measurement(
                session_dir,
                measurement_id=measurement_id,
                producer=producer,
                operation_id=operation_id,
                subject=subject,
                kind="e2e",
                name=name,
                value=numeric,
                unit="percent" if role == "delta" else "tok/s",
                status="validated" if validated is True else "provisional",
                metric_basis="output",
                **({"occurrence": recorded_occurrence} if recorded_occurrence is not None else {}),
                dimensions={"role": role, "baseline_source": "integrate_input", "final_source": "integrate_rebaseline"},
                **_measurement_metadata(
                    "integrate_handler",
                    harness=evidence.get("harness") or "orchestrator_baseline",
                    workload=evidence.get("workload"),
                    samples=evidence.get("samples"),
                    aggregation=evidence.get("aggregation"),
                ),
            )
            measurement_refs.append(measurement_id)
        parity = evidence.get("parity") or evidence.get("output_parity") or evidence.get("accuracy_pass")
        decision_reason = str(evidence.get("decision_reason") or "")
        final_validated = bool(validated) and decision_value in {"KEEP", "ADOPTED"}
        gate = {
            "gate_id": _stable_id("gate", operation_id, "integrate_e2e"),
            "kind": "e2e",
            "name": "integrate_e2e",
            "status": "passed"
            if final_validated
            else "failed"
            if decision_value in {"REVERT", "REJECTED"}
            else "partial",
            "decision": "allow"
            if final_validated
            else "deny"
            if decision_value in {"REVERT", "REJECTED"}
            else "review",
            "reason": decision_reason,
            "evidence": {
                "parity": parity,
                "validated": validated,
                "validation_tier": validation_tier or evidence.get("validation_tier"),
                "decision_reason": decision_reason,
                "rebuild_check": evidence.get("rebuild_check"),
                "source_import_confirmed": evidence.get("source_import_confirmed"),
            },
        }
        artifact_refs: list[str] = []
        for kind, path in (
            ("patch", patch_path),
            ("target_file", target_file),
            ("report", evidence.get("report_path")),
        ):
            if not path:
                continue
            artifact_id = _stable_id("artifact", operation_id, kind, path)
            record_artifact(
                session_dir,
                artifact_id=artifact_id,
                producer=producer,
                operation_id=operation_id,
                producer_operation_id=operation_id,
                subject=subject,
                kind=kind,
                path=str(path),
                coverage={"status": "reference_only"},
            )
            artifact_refs.append(artifact_id)
        adoption_refs: list[str] = []
        if final_validated or decision_value in {"REVERT", "REJECTED"}:
            adoption_id = _stable_id("adoption", operation_id, "integrate")
            # ATTRIBUTABLE ONLY WITH A THROUGHPUT PAIR. `e2e_gain_pct` is a percentage the executor
            # measured against whatever baseline it happened to hold at the time. The collector
            # turns an adoption into points of the ONE session baseline by walking the throughput
            # chain; with no pair to anchor it, it can only sum the local percentages, and
            # percentages taken against different denominators do not add. Replaying real GEAK
            # journeys showed the cost of pretending otherwise: 36 keeps whose local deltas summed
            # to +348.6 pp of a session that did not move that far.
            # The adoption is still written, so the keep stays visible and countable; only its
            # contribution to any total is withheld.
            before_tput = to_float(evidence.get("base_tput"))
            after_tput = to_float(evidence.get("new_tput"))
            has_pair = bool(before_tput and after_tput and before_tput > 0 and after_tput > 0)
            _record_adoption_transition(
                session_dir,
                adoption_id=adoption_id,
                producer=producer,
                operation_id=operation_id,
                adopted=final_validated,
                attribution_eligible=has_pair,
                reason=decision_reason or ("integrate_e2e_passed" if final_validated else "integrate_e2e_failed"),
                subject=subject,
                artifact_ids=artifact_refs,
                measurement_ids=measurement_refs,
                kind="kernel_optimization",
                gain_pct=to_float(e2e_gain_pct),
                # Frozen inline, not just referenced: measurement ids are stable
                # per kernel, so a later attempt on the same kernel overwrites
                # the very numbers this adoption was decided on.
                throughput_before=before_tput,
                throughput_after=after_tput,
                configuration={
                    "patch_path": patch_path,
                    "target_file": target_file,
                    "extra_server_args": str(extra_server_args or ""),
                },
                validation_basis="e2e_validation",
                metadata={
                    "validation_tier": validation_tier or "integrate_e2e",
                    **({} if has_pair else {"non_attributable_reason": "no_throughput_pair"}),
                },
            )
            adoption_refs.append(adoption_id)
        record_operation(
            session_dir,
            operation_id=operation_id,
            producer=producer,
            kind="kernel_optimization",
            name=str(kernel_id),
            phase="KERNEL_AGENT",
            scope="kernel",
            executor_class="llm_tool",
            status="succeeded"
            if final_validated
            else "reverted"
            if decision_value in {"REVERT", "REJECTED"}
            else "needs_review",
            ended_at=_now_iso_safe(),
            parent_operation_id=_kernel_route_operation_id(session_dir, _canonical_route(route_strategy)[0]),
            root_operation_id=_kernel_route_operation_id(session_dir, _canonical_route(route_strategy)[0]),
            subject=subject,
            gates=[gate],
            outputs={
                "integrated": bool(integrated),
                "decision": decision_value,
                "validated": validated,
                "validation_tier": validation_tier or evidence.get("validation_tier"),
            },
            measurement_refs=measurement_refs,
            artifact_refs=artifact_refs,
            adoption_refs=adoption_refs,
            extensions={
                "integrate": {
                    "parity": parity,
                    "decision_reason": decision_reason,
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_kernel_e2e failed", exc_info=True)
        trace_skip(reason="writer raised", section="kernel_e2e", error=exc)


def record_specialist_round(
    session_dir: Path | str | None,
    entry: dict[str, Any],
    *,
    phase: str = "",
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Record one ``specialist_runs`` round (idempotent by ``round_id``).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        entry (dict[str, Any]): the specialist round entry (keyed by
            ``round_id``); an empty/non-dict value is a no-op.
        phase (str): the runtime phase used when the entry does not already
            declare ``source_phase``. A specialist runs in more than one phase,
            so this cannot be a constant.
        producer (str): the breakdown producer label (defaults to the
            Coordinator).
    """
    if not session_dir or not isinstance(entry, dict) or not entry:
        trace_skip(reason="no session_dir" if not session_dir else "empty entry", section="specialist_rounds")
        return
    try:
        source_phase = str(entry.get("source_phase") or phase or entry.get("phase") or "").strip().upper()
        recorded_entry = dict(entry)
        if source_phase:
            recorded_entry.setdefault("source_phase", source_phase)
        key = str(recorded_entry.get("round_id") or "") or None
        _recorder(session_dir, producer).record_item(
            "specialist_runs",
            recorded_entry,
            key=key,
        )
        round_id = str(recorded_entry.get("round_id") or key or recorded_entry.get("task_id") or "unknown")
        operation_id = _stable_id("op", "specialist", round_id)
        round_subject_id = _stable_id("subject", "specialist-round", round_id)
        domains = list(recorded_entry.get("domains") or [])
        if recorded_entry.get("domain"):
            domains.append(str(recorded_entry.get("domain")))
        domains.extend(str(tag) for tag in (recorded_entry.get("tags") or []) if str(tag))
        domains = list(dict.fromkeys(domain for domain in domains if domain))
        record_subject(
            session_dir,
            subject_id=round_subject_id,
            subject_type="specialist_round",
            role="proposal_source",
            name=round_id,
            attributes={
                "domains": domains,
                "proposals_total": recorded_entry.get("proposals_total"),
            },
            producer=producer,
        )
        domain_subjects: list[dict[str, Any]] = []
        for domain in domains:
            domain_id = _stable_id("subject", "specialist-domain", round_id, domain)
            record_subject(
                session_dir,
                subject_id=domain_id,
                subject_type="specialist_domain",
                role="proposal_domain",
                name=str(domain),
                attributes={"round_id": round_id},
                producer=producer,
            )
            domain_subjects.append({"subject_id": domain_id, "subject_type": "specialist_domain"})
        proposal_subjects: list[dict[str, Any]] = []
        proposals = recorded_entry.get("proposal_set")
        if isinstance(proposals, list):
            for index, proposal in enumerate(proposals):
                if not isinstance(proposal, Mapping):
                    continue
                proposal_key = (
                    proposal.get("proposal_id") or proposal.get("fingerprint") or proposal.get("name") or index
                )
                proposal_id = _stable_id("subject", "specialist-proposal", round_id, proposal_key)
                record_subject(
                    session_dir,
                    subject_id=proposal_id,
                    subject_type="variant",
                    role="proposal",
                    name=str(proposal.get("name") or proposal_key),
                    attributes=dict(proposal),
                    producer=producer,
                )
                proposal_subjects.append({"subject_id": proposal_id, "subject_type": "variant", "role": "proposal"})
        record_operation(
            session_dir,
            operation_id=operation_id,
            root_operation_id=operation_id,
            kind="specialist",
            name=f"specialist round {round_id}",
            phase=source_phase,
            status="succeeded" if recorded_entry.get("completed_at") else "partial",
            source="specialist_recorder_hook",
            executor_class="llm_agent",
            purpose="proposal",
            scope=str(recorded_entry.get("scope") or ""),
            strategy_group="specialist",
            strategy="multi_domain",
            producer=producer,
            ended_at=str(recorded_entry.get("completed_at") or recorded_entry.get("dispatched_at") or ""),
            subject={"subject_id": round_subject_id, "subject_type": "specialist_round"},
            subjects=domain_subjects + proposal_subjects,
            outputs=recorded_entry,
            adoption_refs=[],
            extensions={"downstream_relation": "proposal_only"},
        )
        record_trace_event(
            session_dir,
            trace_event_id=_stable_id("trace", operation_id, "completed"),
            operation_id=operation_id,
            kind="specialist_proposals_recorded",
            status="succeeded" if entry.get("completed_at") else "partial",
            ts=str(entry.get("completed_at") or entry.get("dispatched_at") or ""),
            producer=producer,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_specialist_round failed", exc_info=True)
        trace_skip(reason="writer raised", section="specialist_rounds", error=exc)


def record_critic_iteration(
    session_dir: Path | str | None,
    *,
    iter_n: int,
    request: dict[str, Any] | None = None,
    judge_bundle: dict[str, Any] | None = None,
    review: dict[str, Any] | None,
    emit: dict[str, Any] | None,
    workdir: Path | str | None,
    kb_priors: dict[str, Any] | None = None,
    producer: str = "critic",
) -> None:
    """Record one ``critic_robustness.critic_iterations`` item.

    Recorded per-iteration under a session-unique identity so workdir pruning
    and resume-time turn-index reuse never erase history; payload mirrors
    ``collectors.collect_critic_robustness`` and retains normalized Framework
    review rows for the V6 timeline.

    ``kb_priors`` (when provided) carries the per-iteration KB integration
    trace: whether the historical priors were used, the request, the response,
    and whether the final verdict referenced them. Omitted from the payload
    when empty so historical items are unchanged.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        iter_n (int): the process-local critic iteration number.
        request (dict[str, Any] | None): the critic request payload.
        judge_bundle (dict[str, Any] | None): the proposal bundle reviewed.
        review (dict[str, Any] | None): the critic review payload.
        emit (dict[str, Any] | None): the critic emit payload.
        workdir (Path | str | None): the critic backend workdir holding the
            per-iteration artifact files.
        kb_priors (dict[str, Any] | None): the per-iteration historical KB
            priors trace; omitted when empty.
        producer (str): the breakdown producer label (defaults to ``critic``).
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section="critic_iterations")
        return
    try:
        review = review if isinstance(review, dict) else {}
        emit = emit if isinstance(emit, dict) else {}
        wd = Path(workdir) if workdir else None
        request = request if isinstance(request, dict) else read_json(wd / "request.json", default={}) if wd else {}
        judge_bundle = (
            judge_bundle
            if isinstance(judge_bundle, dict)
            else read_json(wd / "judge_bundle.json", default={})
            if wd
            else {}
        )
        payload = {
            "iter": int(iter_n),
            "ts": str(emit.get("ts") or review.get("ts") or ""),
            "topic": str(emit.get("topic") or review.get("topic") or ""),
            "verdict": str(review.get("verdict") or emit.get("verdict") or ""),
            "summary": str(review.get("summary") or emit.get("summary") or "")[:500],
            "request_path": _rel(wd / "request.json", session_dir) if wd else None,
            "judge_bundle_path": _rel(wd / "judge_bundle.json", session_dir) if wd else None,
            "emit_path": _rel(wd / "emit.json", session_dir) if wd else None,
            "review_path": _rel(wd / "review.json", session_dir) if wd else None,
            "kb_writes": list(emit.get("kb_writes") or []) if isinstance(emit.get("kb_writes"), list) else [],
        }
        request_context = request.get("context") if isinstance(request.get("context"), dict) else {}
        phase = str(request_context.get("phase") or "").strip().upper()
        if phase:
            payload["phase"] = phase
        try:
            macro_cycle = int(request_context["macro_cycle"])
        except (KeyError, TypeError, ValueError):
            macro_cycle = None
        if macro_cycle is not None:
            payload["macro_cycle"] = macro_cycle
        framework_reviews = normalize_framework_reviews(
            request=request,
            judge_bundle=judge_bundle,
            review=review,
            emit=emit,
            review_path=_rel(wd / "review.json", session_dir).replace("\\", "/") if wd else None,
        )
        if framework_reviews:
            payload["framework_reviews"] = framework_reviews
        if isinstance(kb_priors, dict) and kb_priors:
            payload["kb_priors"] = kb_priors
        iteration_id = _stable_id(
            "critic-iteration",
            iter_n,
            payload.get("ts"),
            [row.get("proposal_msg_id") for row in framework_reviews],
            payload.get("topic"),
            request,
            judge_bundle,
            review,
            emit,
        )
        payload["iteration_id"] = iteration_id
        _recorder(session_dir, producer).record_item(
            "critic_iterations",
            payload,
            key=iteration_id,
        )
        operation_id = _stable_id("op", iteration_id)
        artifact_refs: list[str] = []
        for name in ("request_path", "judge_bundle_path", "emit_path", "review_path"):
            path = payload.get(name)
            if not path:
                continue
            artifact_id = _stable_id("artifact", operation_id, name, path)
            artifact_refs.append(artifact_id)
            record_artifact(
                session_dir,
                artifact_id=artifact_id,
                operation_id=operation_id,
                producer_operation_id=operation_id,
                kind=name,
                path=str(path),
                present=True,
                status="available",
                producer=producer,
            )
        record_operation(
            session_dir,
            operation_id=operation_id,
            root_operation_id=operation_id,
            kind="critic",
            name=str(payload.get("topic") or "critic review"),
            status="succeeded",
            source="critic_recorder_hook",
            executor_class="llm_agent",
            purpose="review",
            producer=producer,
            ended_at=str(payload.get("ts") or ""),
            decisions=[
                {
                    "decision_id": _stable_id("decision", operation_id),
                    "kind": "critic",
                    "verdict": str(payload.get("verdict") or ""),
                    "reason": str(payload.get("summary") or ""),
                    "decided_at": str(payload.get("ts") or ""),
                    "component": producer,
                }
            ],
            artifact_refs=artifact_refs,
            outputs=payload,
        )
        record_trace_event(
            session_dir,
            trace_event_id=_stable_id("trace", operation_id, "reviewed"),
            operation_id=operation_id,
            kind="critic_reviewed",
            verdict=str(payload.get("verdict") or ""),
            ts=str(payload.get("ts") or ""),
            producer=producer,
        )
        for index, write in enumerate(payload.get("kb_writes") or []):
            if not isinstance(write, Mapping):
                continue
            write_key = (
                write.get("write_id") or write.get("point_id") or write.get("edge_id") or write.get("kind") or index
            )
            write_operation_id = _stable_id("op", "kb-write", iteration_id, write_key)
            result_payload = write.get("result") if isinstance(write.get("result"), Mapping) else {}
            write_status = _operation_status(result_payload.get("status") or write.get("status") or "succeeded")
            record_operation(
                session_dir,
                operation_id=write_operation_id,
                root_operation_id=operation_id,
                parent_operation_id=operation_id,
                kind="kb_write",
                name=str(write.get("kind") or write_key),
                status=write_status,
                source="critic_commit_review",
                executor_class="deterministic",
                purpose="knowledge_write",
                producer=producer,
                ended_at=str(payload.get("ts") or ""),
                inputs=dict(write),
                outputs=dict(result_payload),
            )
            record_trace_event(
                session_dir,
                trace_event_id=_stable_id("trace", write_operation_id, write_status),
                operation_id=write_operation_id,
                parent_operation_id=operation_id,
                kind="kb_write_finalized",
                status=write_status,
                ts=str(payload.get("ts") or ""),
                producer=producer,
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_critic_iteration failed", exc_info=True)
        trace_skip(reason="writer raised", section="critic_iterations", error=exc)


def record_robustness_signal(
    session_dir: Path | str | None,
    *,
    workdir: Path | str | None,
    producer: str = "robustness",
) -> None:
    """Record one ``critic_robustness.robustness_signals`` item.

    Reads ``signal.json`` / ``action.json`` from the just-written ``workdir``
    (idempotent on the workdir name) so the signal is captured before the
    robustness backend prunes old workdirs; payload mirrors the collector.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        workdir (Path | str | None): the just-written robustness workdir holding
            ``signal.json`` / ``action.json`` (idempotency key); a falsy value
            is a no-op.
        producer (str): the breakdown producer label (defaults to
            ``robustness``).
    """
    if not session_dir or not workdir:
        trace_skip(reason="no session_dir" if not session_dir else "no workdir", section="robustness_signals")
        return
    try:
        wd = Path(workdir)
        signal_data = read_json(wd / "signal.json", default={}, require_dict=True)
        action_data = read_json(wd / "action.json", default={}, require_dict=True)
        payload = {
            "ts": str(signal_data.get("ts") or action_data.get("ts") or ""),
            "signal": str(signal_data.get("signal") or signal_data.get("kind") or ""),
            "action": str(action_data.get("action") or action_data.get("kind") or ""),
            "workdir": _rel(wd, session_dir),
        }
        _recorder(session_dir, producer).record_item(
            "robustness_signals",
            payload,
            key=wd.name,
        )
        operation_id = _stable_id("op", "robustness", wd.name)
        record_operation(
            session_dir,
            operation_id=operation_id,
            root_operation_id=operation_id,
            kind="robustness",
            name=str(payload.get("signal") or "robustness signal"),
            status="succeeded" if payload.get("action") else "partial",
            source="robustness_recorder_hook",
            executor_class="llm_agent",
            purpose="recovery",
            producer=producer,
            ended_at=str(payload.get("ts") or ""),
            outputs=payload,
            extensions={"metadata_completeness": "partial" if not payload.get("signal") else "available"},
        )
        record_trace_event(
            session_dir,
            trace_event_id=_stable_id("trace", operation_id, "handled"),
            operation_id=operation_id,
            kind="robustness_signal_handled",
            signal=payload.get("signal"),
            action=payload.get("action"),
            ts=str(payload.get("ts") or ""),
            producer=producer,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_robustness_signal failed", exc_info=True)
        trace_skip(reason="writer raised", section="robustness_signals", error=exc)


def record_singleton_section(
    session_dir: Path | str | None,
    section: str,
    payload: dict[str, Any],
    *,
    producer: str,
) -> None:
    """Record a producer-owned singleton section (report summaries, etc.).

    Args:
        session_dir (Path | str | None): the session directory; a falsy value is
            a no-op.
        section (str): the singleton section name to record.
        payload (dict[str, Any]): the section payload; an empty/non-dict value
            is a no-op.
        producer (str): the breakdown producer label that owns the section.
    """
    if not session_dir or not isinstance(payload, dict) or not payload:
        trace_skip(reason="no session_dir" if not session_dir else "empty payload", section=section)
        return
    try:
        _recorder(session_dir, producer).record_singleton(section, payload)
        if section.startswith("kb_") or section in {"warm_replay", "kb_provenance"}:
            operation_id = _stable_id(
                "op",
                section,
                payload.get("task_id") or payload.get("id") or payload.get("ts") or payload,
            )
            record_operation(
                session_dir,
                operation_id=operation_id,
                root_operation_id=operation_id,
                kind=section,
                name=section,
                status=_operation_status(payload.get("status") or "succeeded"),
                source="recorder_hook",
                executor_class="deterministic",
                purpose="knowledge",
                producer=producer,
                ended_at=str(payload.get("ts") or ""),
                outputs=payload,
            )
            record_trace_event(
                session_dir,
                trace_event_id=_stable_id("trace", operation_id, "recorded"),
                operation_id=operation_id,
                kind=f"{section}_recorded",
                status=_operation_status(payload.get("status") or "succeeded"),
                ts=str(payload.get("ts") or ""),
                producer=producer,
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_singleton_section %s failed", section, exc_info=True)
        trace_skip(reason="writer raised", section=section, error=exc)


def _v4_payload(
    payload: Mapping[str, Any] | None,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine a mapping payload with keyword fields for a v4 helper."""
    value = dict(payload) if isinstance(payload, Mapping) else {}
    value.update(fields)
    return value


def _valid_v4_call(
    session_dir: Path | str | None,
    producer: str,
    payload: Mapping[str, Any],
    *,
    section: str = "",
) -> bool:
    """Return whether a v4 helper call has usable routing metadata.

    A caller that fails this is asking to record something and getting
    nothing, which is the shape of gap the export can only report as an
    absence, so say which of the three requirements was missing.
    """
    if not session_dir:
        reason = "no session_dir"
    elif not payload:
        reason = "empty payload"
    elif not _VALID_PRODUCER.fullmatch(str(producer or "")):
        reason = f"invalid producer {producer!r}"
    else:
        return True
    trace_skip(reason=reason, section=section, producer=str(producer or ""))
    return False


def _record_v4_entity(
    session_dir: Path | str | None,
    *,
    section: str,
    payload: Mapping[str, Any] | None,
    fields: Mapping[str, Any],
    id_field: str,
    entity_id: str | None,
    producer: str,
) -> None:
    """Best-effort writer for one stable v4 entity."""
    value = _v4_payload(payload, fields)
    stable_id = str(entity_id or value.get(id_field) or "").strip()
    if stable_id:
        value[id_field] = stable_id
    if not _valid_v4_call(session_dir, producer, value, section=section):
        return
    if not stable_id:
        # An entity is merged by its id, so one without an id has nothing to
        # be merged into and cannot be written at all.
        trace_skip(reason=f"no {id_field}", section=section, producer=producer)
        return
    try:
        _recorder(session_dir, producer).record_upsert_item(
            section,
            value,
            key=stable_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_%s failed", section, exc_info=True)
        trace_skip(
            reason="writer raised",
            section=section,
            producer=producer,
            entity=stable_id,
            error=exc,
        )


def _record_v4_event(
    session_dir: Path | str | None,
    *,
    section: str,
    payload: Mapping[str, Any] | None,
    fields: Mapping[str, Any],
    key_fields: tuple[str, ...],
    producer: str,
) -> None:
    """Best-effort writer for a v4 event with an optional stable key."""
    value = _v4_payload(payload, fields)
    if not _valid_v4_call(session_dir, producer, value, section=section):
        return
    key = next((str(value.get(name) or "").strip() for name in key_fields if value.get(name)), "")
    try:
        recorder = _recorder(session_dir, producer)
        if key:
            recorder.record_upsert_item(section, value, key=key)
        else:
            recorder.record_item(section, value)
    except Exception as exc:  # noqa: BLE001
        log.debug("record_%s failed", section, exc_info=True)
        trace_skip(
            reason="writer raised",
            section=section,
            producer=producer,
            entity=key,
            error=exc,
        )


def record_run_snapshot(
    session_dir: Path | str | None,
    snapshot: Mapping[str, Any] | None = None,
    *,
    producer: str = PRODUCER_COORDINATOR,
    **fields: Any,
) -> None:
    """Record a partial v4 run snapshot using author-time facts only."""
    value = _v4_payload(snapshot, fields)
    if not _valid_v4_call(session_dir, producer, value, section="run_snapshot"):
        return
    try:
        _recorder(session_dir, producer).record_upsert_singleton("run_snapshot", value)
    except Exception as exc:  # noqa: BLE001
        log.debug("record_run_snapshot failed", exc_info=True)
        trace_skip(
            reason="writer raised",
            section="run_snapshot",
            producer=producer,
            error=exc,
        )


def record_phase_transition(
    session_dir: Path | str | None,
    transition: Mapping[str, Any] | None = None,
    *,
    producer: str = PRODUCER_COORDINATOR,
    **fields: Any,
) -> None:
    """Record one v4 phase transition."""
    value = _v4_payload(transition, fields)
    if not value.get("transition_id") and not value.get("event_id"):
        value["transition_id"] = _stable_id(
            "transition",
            value.get("operation_id") or "",
            f"macro_cycle:{value.get('macro_cycle')}",
            f"tick:{value.get('tick')}",
            f"event:{value.get('event_sequence')}",
            value.get("from_phase") or "",
            value.get("phase") or value.get("to_phase") or "",
            value.get("ts") or _now_iso_safe(),
        )
    _record_v4_event(
        session_dir,
        section="phase_transitions",
        payload=value,
        fields={},
        key_fields=("transition_id", "event_id"),
        producer=producer,
    )


def record_subject(
    session_dir: Path | str | None,
    subject: Mapping[str, Any] | None = None,
    *,
    subject_id: str | None = None,
    producer: str = PRODUCER_COORDINATOR,
    **fields: Any,
) -> None:
    """Upsert one v4 subject by stable ``subject_id``."""
    _record_v4_entity(
        session_dir,
        section="subjects",
        payload=subject,
        fields=fields,
        id_field="subject_id",
        entity_id=subject_id,
        producer=producer,
    )


_AGENT_BY_PRODUCER = {
    PRODUCER_KERNEL_AGENT: "kernel_agent",
    "critic": "critic",
    "robustness": "robustness",
    "framework-kb": "framework_agent",
}

_AGENT_BY_OPERATION_KIND = {
    "kernel_optimization": "kernel_agent",
    "kernel_optimizer_run": "kernel_agent",
    "kernel_optimizer_selection": "kernel_agent",
    "strategy_selection": "kernel_agent",
    "gemm_tuning": "kernel_agent",
    # A specialist round is discovery, not an attempt: it carries no gain and
    # no adoption, so this names the agent whose activity it was.
    "specialist": "explore",
    "critic": "critic",
    "kb_write": "critic",
    "robustness": "robustness",
}


def _default_operation_agent(
    value: Mapping[str, Any],
    producer: str,
) -> str:
    """Fall back to the owning agent implied by the producer and work kind.

    Explicit ``agent=`` at the call site always wins; this only keeps operations
    recorded by paths that predate the field from landing without an owner.
    """
    kind = str(value.get("kind") or "").strip().lower()
    by_kind = _AGENT_BY_OPERATION_KIND.get(kind)
    if by_kind:
        return by_kind
    by_producer = _AGENT_BY_PRODUCER.get(str(producer or "").strip().lower())
    if by_producer:
        return by_producer
    return _resolve_agent(
        str(value.get("name") or kind),
        result=value.get("outputs") if isinstance(value.get("outputs"), Mapping) else None,
        phase=str(value.get("phase") or ""),
    )


def record_operation(
    session_dir: Path | str | None,
    operation: Mapping[str, Any] | None = None,
    *,
    operation_id: str | None = None,
    producer: str = PRODUCER_COORDINATOR,
    **fields: Any,
) -> None:
    """Upsert one v4 operation by stable ``operation_id``."""
    value = _v4_payload(operation, fields)
    # Only stamp an owner when this call actually defines the operation; a
    # partial upsert that just patches one field must not overwrite the agent
    # its defining call already recorded.
    if not value.get("agent") and (value.get("kind") or value.get("name")):
        value["agent"] = _default_operation_agent(value, producer)
    _record_v4_entity(
        session_dir,
        section="operations",
        payload=value,
        fields={},
        id_field="operation_id",
        entity_id=operation_id,
        producer=producer,
    )


def record_measurement(
    session_dir: Path | str | None,
    measurement: Mapping[str, Any] | None = None,
    *,
    measurement_id: str | None = None,
    producer: str = PRODUCER_COORDINATOR,
    **fields: Any,
) -> None:
    """Upsert one v4 measurement by stable ``measurement_id``."""
    _record_v4_entity(
        session_dir,
        section="measurements",
        payload=measurement,
        fields=fields,
        id_field="measurement_id",
        entity_id=measurement_id,
        producer=producer,
    )


def record_adoption(
    session_dir: Path | str | None,
    adoption: Mapping[str, Any] | None = None,
    *,
    adoption_id: str | None = None,
    producer: str = PRODUCER_COORDINATOR,
    **fields: Any,
) -> None:
    """Upsert one v4 adoption by stable ``adoption_id``."""
    value = _v4_payload(adoption, fields)
    if not value.get("agent") and value.get("kind"):
        value["agent"] = _default_operation_agent(value, producer)
    if value.get("attribution_eligible") is None and value.get("kind"):
        value["attribution_eligible"] = True
    status = str(value.get("status") or "").lower()
    decision = str(value.get("decision") or "").upper()
    if status == "adopted" or (decision == "KEEP" and value.get("validated") is True):
        value["status"] = "adopted"
        value["decision"] = "KEEP"
        value["validated"] = True
    elif status in {"revoked", "reverted"} or decision == "REVERT":
        value["status"] = "revoked"
        value["decision"] = "REVERT"
        value["validated"] = False
    _record_v4_entity(
        session_dir,
        section="adoptions",
        payload=value,
        fields={},
        id_field="adoption_id",
        entity_id=adoption_id,
        producer=producer,
    )


def record_artifact(
    session_dir: Path | str | None,
    artifact: Mapping[str, Any] | None = None,
    *,
    artifact_id: str | None = None,
    producer: str = PRODUCER_COORDINATOR,
    **fields: Any,
) -> None:
    """Upsert one v4 artifact reference by stable ``artifact_id``."""
    _record_v4_entity(
        session_dir,
        section="artifacts",
        payload=artifact,
        fields=fields,
        id_field="artifact_id",
        entity_id=artifact_id,
        producer=producer,
    )


def record_session_validation(
    session_dir: Path | str | None,
    *,
    baseline_tput: float | None,
    validated_tput: float | None,
    validated_gain_pct: float | None,
    stack_len: int,
    source: str,
    measurement_basis: str,
    ts: str | None = None,
    producer: str = PRODUCER_COORDINATOR,
) -> str | None:
    """Record what the session was measured to have gained, as it was decided.

    Without this the breakdown's session total is the sum of its own ledger,
    so the ledger can never be found to disagree with the end-to-end
    measurement the run actually promoted -- the two numbers come from the
    same addition. Recording the promoted figure at the moment it is promoted
    gives the export something independent to reconcile against.

    Each promotion is its own record rather than an overwrite of one, so a
    session leaves behind the checkpoints it passed through. The id is drawn
    from the stack length and the timestamp, never from the value: two
    checkpoints that happen to measure the same number are still two
    checkpoints.

    Args:
        session_dir: The hyperloom session directory.
        baseline_tput: Throughput the gain is measured against.
        validated_tput: Throughput just measured.
        validated_gain_pct: The promoted gain, in percent of baseline.
        stack_len: Adopted-stack length this figure was validated at.
        source: The path that promoted it, e.g. ``integrate_patch``.
        measurement_basis: ``e2e_rebench`` when the throughput was measured
            end to end, ``e2e_decision_round`` when it is an explore round's own
            grading measurement, ``derived_speedup`` when it was inferred from a
            micro-benchmark's speedup.
        ts: Author-time stamp; defaults to now.
        producer: Recorder producer label.

    Returns:
        The operation id, or ``None`` when nothing could be recorded.
    """
    if not session_dir or validated_gain_pct is None:
        return None
    stamp = str(ts or _now_iso_safe())
    operation_id = _stable_id("op", "session_validation", stack_len, stamp)
    occurrence = _measurement_occurrence(operation_id)
    measurement_ids: list[str] = []
    for name, value, unit in (
        ("baseline_throughput", baseline_tput, "tok/s"),
        ("throughput", validated_tput, "tok/s"),
        ("gain", validated_gain_pct, "percent"),
    ):
        if value is None:
            continue
        measurement_id = _stable_id("measurement", operation_id, name, occurrence)
        measurement_ids.append(measurement_id)
        record_measurement(
            session_dir,
            measurement_id=measurement_id,
            operation_id=operation_id,
            kind=name,
            name=name,
            value=float(value),
            unit=unit,
            measured_at=stamp,
            metric_basis=measurement_basis,
            source={"action": "session_validation", "role": name},
            producer=producer,
        )
    record_operation(
        session_dir,
        operation_id=operation_id,
        kind="session_validation",
        name="cumulative_gain_validated",
        agent="coordinator",
        status="succeeded",
        started_at=stamp,
        ended_at=stamp,
        measurement_refs=measurement_ids,
        outputs={
            "validated_at_stack_len": stack_len,
            "source": source,
            "measurement_basis": measurement_basis,
            "validated_gain_pct": float(validated_gain_pct),
        },
        producer=producer,
    )
    return operation_id


def record_trace_event(
    session_dir: Path | str | None,
    event: Mapping[str, Any] | None = None,
    *,
    producer: str = PRODUCER_COORDINATOR,
    **fields: Any,
) -> None:
    """Record one v4 trace event, idempotent when an event id is supplied."""
    _record_v4_event(
        session_dir,
        section="trace_events",
        payload=event,
        fields=fields,
        key_fields=("trace_event_id", "event_id", "span_id"),
        producer=producer,
    )


__all__ = [
    "PRODUCER_COORDINATOR",
    "PRODUCER_KERNEL_AGENT",
    "record_critic_iteration",
    "record_action_operation",
    "record_adoption",
    "record_artifact",
    "record_kernel_backend_result",
    "record_kernel_discovery",
    "record_kernel_dispatch",
    "record_kernel_e2e",
    "record_kernel_invocations",
    "record_kernel_strategy_selection",
    "record_native_kernel_run_start",
    "record_native_kernel_run_result",
    "record_geak_e2e_attempt",
    "record_geak_operation",
    "record_gemm_tuning_operation",
    "record_phase_event",
    "record_phase_transition",
    "record_measurement",
    "record_operation",
    "record_robustness_signal",
    "record_session_validation",
    "record_singleton_section",
    "record_specialist_round",
    "record_subject",
    "record_tool_version",
    "record_trace_event",
    "record_run_snapshot",
    "snapshot_state_sections",
]
