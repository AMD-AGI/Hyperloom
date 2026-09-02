# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical optimization projection for ``session_breakdown.json``.

Every field here traces to something a producer recorded when it happened.
There is no reconstruction from business state: a session whose recorder
streams are empty is reported as such rather than re-derived from
``state.json``, because a projection assembled after the fact cannot tell a
run that adopted nothing from a run whose records never landed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..agent_ownership import UNATTRIBUTED, patch_author


#: Matches the major version of the breakdown envelope
#: (``hyperloom.session_breakdown.v5.0``), so one number answers both
#: questions. ``5`` means the recorder-first shape: ``attempts``,
#: ``summary_by_agent``, and the reconciliation fields under ``validation``.
OPTIMIZATIONS_SCHEMA_VERSION = 5

_SOURCES = (
    "warm_replay",
    "primatune",
    "explore",
    "framework_agent",
    "kernel_agent",
    "unattributed",
)

# Operation kinds that represent one attempt at making the workload faster.
# Everything else the recorder captures (discovery, review, routing, baseline
# measurement) is context, not an attempt.
_ATTEMPT_KINDS = frozenset(
    {
        "kernel_optimization",
        "kernel_collective",
        "gemm_tuning",
        "integrate_patch",
        "framework_agent",
        "explore",
        "replay_warm_recipe",
    }
)

_KEEP_DECISIONS = frozenset({"KEEP", "KEPT", "KEPT_INERT", "ADOPT", "ADOPTED", "PROMOTED"})
_REVERT_DECISIONS = frozenset({"REVERT", "REVERTED", "REJECTED", "FAILED", "ACCURACY_UNAVAILABLE_REJECT"})

# Measurement names, not the result fields they came from: the recorder maps
# ``output_throughput`` and ``delta_pct`` onto ``throughput`` and ``gain``
# before writing, so matching on the field names would match nothing.
_THROUGHPUT_AFTER_NAMES = frozenset({"final_throughput", "throughput"})
_THROUGHPUT_BEFORE_NAMES = frozenset({"baseline_throughput"})
_GAIN_NAMES = frozenset({"e2e_gain_pct", "gain"})


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_ts(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _empty_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {source: {"keeps": 0, "total_gain_pct": 0.0} for source in _SOURCES}
    summary["kernel_agent"]["by_backend"] = {
        "geak": {"keeps": 0, "total_gain_pct": 0.0, "non_attributable_keeps": 0},
        "forge": {"keeps": 0, "total_gain_pct": 0.0, "non_attributable_keeps": 0},
        "unattributed": {"keeps": 0, "total_gain_pct": 0.0, "non_attributable_keeps": 0},
    }
    return summary


def _collect_backend_attempts(
    session_id: str,
    geak_invocations: list[dict[str, Any]],
    forge_invocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize every backend invocation into one chronological attempt list."""

    attempts: list[dict[str, Any]] = []
    for default_backend, invocations in (
        ("geak", geak_invocations),
        ("forge", forge_invocations),
    ):
        for raw in invocations:
            if not isinstance(raw, dict):
                continue
            duration = next(
                (
                    _to_float(raw.get(field))
                    for field in (
                        "duration_sec",
                        "elapsed_sec",
                        "elapsed_time_sec",
                        "elapsed",
                    )
                    if _to_float(raw.get(field)) is not None
                ),
                None,
            )
            attempts.append(
                {
                    "attempt_id": str(raw.get("attempt_id") or ""),
                    "run_id": str(raw.get("run_id") or ""),
                    "kernel_id": str(raw.get("kernel_id") or ""),
                    "backend": str(raw.get("backend") or default_backend),
                    "decision": str(raw.get("decision") or "").upper(),
                    "ts": str(raw.get("ts") or ""),
                    "duration_sec": duration,
                    "micro_speedup": _to_float(raw.get("micro_speedup")),
                    "compile_passed": raw.get("compile_passed"),
                    "correctness_passed": raw.get("correctness_passed"),
                    "error_class": (str(raw.get("error_class")) if raw.get("error_class") else None),
                    "error": str(raw.get("error")) if raw.get("error") else None,
                    "result_path": (str(raw.get("result_path")) if raw.get("result_path") else None),
                    "verification_path": (str(raw.get("verification_path")) if raw.get("verification_path") else None),
                }
            )

    attempts.sort(
        key=lambda row: (
            str(row.get("kernel_id") or ""),
            _parse_ts(row.get("ts")) if _parse_ts(row.get("ts")) is not None else float("inf"),
            str(row.get("backend") or ""),
            str(row.get("attempt_id") or ""),
        )
    )
    sequence_by_kernel: dict[str, int] = {}
    for attempt in attempts:
        kernel_id = str(attempt.get("kernel_id") or "")
        sequence_by_kernel[kernel_id] = sequence_by_kernel.get(kernel_id, 0) + 1
        attempt["sequence"] = sequence_by_kernel[kernel_id]
        if not attempt.get("attempt_id"):
            attempt["attempt_id"] = (
                f"{session_id}:kernel-attempt:"
                f"{kernel_id or 'unknown'}:{attempt.get('backend') or 'unknown'}:"
                f"{attempt['sequence']}"
            )
    return attempts


def _empty_kind_summary() -> dict[str, Any]:
    return {"keeps": 0, "total_gain_pct": 0.0}


def _work_kind(operation: dict[str, Any]) -> str:
    """Return the operation's real work kind.

    ``composite`` is a container label the recorder uses for multi-step actions;
    the action name underneath it is what identifies the work.
    """
    kind = str(operation.get("kind") or "").strip().lower()
    if kind in {"", "composite"}:
        return str(operation.get("name") or "").strip().lower()
    return kind


_AGENT_BY_RECORDED_KIND = {
    "kernel_optimization": "kernel_agent",
    "kernel_collective": "kernel_agent",
    "gemm_tuning": "kernel_agent",
    "framework_agent": "framework_agent",
    "explore": "explore",
    "replay_warm_recipe": "warm_replay",
}


def _attempt_agent(operation: dict[str, Any], adoption: dict[str, Any]) -> str:
    """Return the owning agent, preferring what the producer actually recorded.

    Sessions recorded before producers stamped ``agent`` still need a bucket, so
    fall back to the work kind and, for patch application, the authoring markers
    the executor left in its own result.
    """
    recorded = str(operation.get("agent") or adoption.get("agent") or "").strip()
    if recorded:
        return recorded

    kind = _work_kind(operation)
    by_kind = _AGENT_BY_RECORDED_KIND.get(kind)
    if by_kind:
        return by_kind

    if kind == "integrate_patch":
        outputs = operation.get("outputs") if isinstance(operation.get("outputs"), dict) else {}
        return patch_author(outputs)
    return UNATTRIBUTED


def _last_decision(operation: dict[str, Any]) -> dict[str, Any]:
    decisions = [row for row in operation.get("decisions") or [] if isinstance(row, dict)]
    return decisions[-1] if decisions else {}


def _threshold_from_operation(operation: dict[str, Any]) -> tuple[float | None, str]:
    """Pull the keep threshold out of whichever gate or decision recorded it.

    Which of the four places it came from is part of the answer. A threshold on
    the gate is the bar that gate actually ruled against; one on the operation's
    outputs is whatever the executor was configured with, which is not
    necessarily what ruled. Reporting only the number lets a rejection be
    explained by a bar that never applied to it.

    Returns:
        The threshold and the place it was recorded, or ``(None, "")``.
    """
    for gate in operation.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        for origin, holder in (("gate.inputs", gate.get("inputs")), ("gate.evidence", gate.get("evidence"))):
            if isinstance(holder, dict):
                value = _to_float(holder.get("keep_threshold_pct"))
                if value is not None:
                    return value, origin
    evidence = _last_decision(operation).get("evidence")
    if isinstance(evidence, dict):
        value = _to_float(evidence.get("keep_threshold_pct"))
        if value is not None:
            return value, "decision.evidence"
    outputs = operation.get("outputs")
    if isinstance(outputs, dict):
        value = _to_float(outputs.get("keep_threshold_pct"))
        if value is not None:
            return value, "outputs"
    return None, ""


def _gate_rows(operation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in operation.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        rows.append(
            {
                "kind": str(gate.get("kind") or ""),
                "name": str(gate.get("name") or ""),
                "status": str(gate.get("status") or ""),
                "decision": str(gate.get("decision") or ""),
                "reason": str(gate.get("reason") or ""),
            }
        )
    return rows


def _sub_attempt_rows(operation: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the per-backend tries nested inside one optimization attempt."""
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(operation.get("attempts") or []):
        if not isinstance(value, dict):
            continue
        outputs = value.get("outputs") if isinstance(value.get("outputs"), dict) else {}
        error = value.get("error")
        rows.append(
            {
                "attempt_id": str(value.get("attempt_id") or ""),
                "sequence": value.get("sequence") or index + 1,
                "backend": str(value.get("backend") or ""),
                "status": str(value.get("status") or ""),
                "decision": str(outputs.get("decision") or "").upper(),
                "started_at": str(value.get("started_at") or ""),
                "duration_sec": _duration_between(
                    value.get("started_at"),
                    value.get("ended_at"),
                ),
                "micro_speedup": _to_float(outputs.get("micro_speedup")),
                "compile_passed": outputs.get("compile_passed"),
                "correctness_passed": outputs.get("correctness_passed"),
                "error": str(error) if error and not isinstance(error, dict) else None,
            }
        )
    return rows


def _first_recorded(*candidates: tuple[str, Any]) -> tuple[Any, str]:
    """Return the first candidate a producer actually recorded, and its origin.

    These chains exist because several producers record the same fact in
    different places. Which place a value came from is what says whether it is
    the verdict an executor stated or a status inferred from the operation
    around it, and losing that distinction is how a fallback becomes
    indistinguishable from the real thing.

    Args:
        candidates: ``(origin, value)`` pairs, most authoritative first.

    Returns:
        The first non-empty value with the origin it came from; ``(None, "")``
        when nothing was recorded.
    """
    for origin, value in candidates:
        if value is not None and value != "":
            return value, origin
    return None, ""


def _recorded_attempt_row(
    operation: dict[str, Any],
    *,
    adoption: dict[str, Any],
    measurement_by_id: dict[str, dict[str, Any]],
    artifact_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Project one recorded operation into a single canonical attempt row."""
    operation_id = str(operation.get("operation_id") or "")
    outputs = operation.get("outputs") if isinstance(operation.get("outputs"), dict) else {}
    decision_row = _last_decision(operation)
    subject = operation.get("subject") if isinstance(operation.get("subject"), dict) else {}

    recorded_ids = [str(mid) for mid in operation.get("measurement_refs") or []]
    pinned_ids = [str(mid) for mid in adoption.get("measurement_ids") or []]
    if pinned_ids:
        # The adoption named the measurements it was decided on. Re-measuring
        # the same subject now records new ones beside these instead of over
        # them, so these still say what they said when the call was made.
        measurement_ids = pinned_ids
        measurement_source = "adoption_pinned"
    else:
        measurement_ids = _latest_measurement_per_name(recorded_ids, measurement_by_id)
        measurement_source = "latest_occurrence"
    occurrence_index = _occurrence_index(recorded_ids, measurement_by_id)
    measured_before = None
    measured_after = None
    measured_gain = None
    # Which measurement name each of the three came off. Every one of them
    # accepts more than one name because producers stamp them differently, and
    # the names are not synonyms in every context: a reading taken as
    # ``final_throughput`` and one taken as ``throughput`` were taken by
    # different producers under their own conventions. When the chain below
    # multiplies these together, an alias that meant something slightly
    # different propagates into the cumulative figure rather than staying on
    # its own row, so the name is kept beside the value.
    measured_before_name = ""
    measured_after_name = ""
    measured_gain_name = ""
    # Readings that matched the same role under a different name. Only the
    # first is used; a second one that disagrees means the role is ambiguous.
    alias_conflicts: list[str] = []
    measurements: list[dict[str, Any]] = []
    for measurement_id in measurement_ids:
        measurement = measurement_by_id.get(str(measurement_id))
        if not measurement:
            continue
        name = str(measurement.get("name") or "").strip().lower()
        value = _to_float(measurement.get("value"))
        measurements.append(
            {
                "name": name,
                "value": measurement.get("value"),
                "unit": str(measurement.get("unit") or ""),
                # Which reading of this name it is, oldest first, so a reader
                # can tell a re-measure from the one that was decided on
                # without having to compare ids.
                "occurrence": occurrence_index.get(str(measurement_id), 0),
                "occurrences_of_name": sum(
                    1
                    for mid, _ in occurrence_index.items()
                    if str((measurement_by_id.get(mid) or {}).get("name") or "").strip().lower() == name
                ),
            }
        )
        if value is None:
            continue
        if name in _THROUGHPUT_AFTER_NAMES:
            if measured_after is None:
                measured_after, measured_after_name = value, name
            elif name != measured_after_name and _disagrees((measured_after, value)):
                alias_conflicts.append(f"throughput_after:{measured_after_name}/{name}")
        elif name in _THROUGHPUT_BEFORE_NAMES:
            if measured_before is None:
                measured_before, measured_before_name = value, name
            elif name != measured_before_name and _disagrees((measured_before, value)):
                alias_conflicts.append(f"throughput_before:{measured_before_name}/{name}")
        elif name in _GAIN_NAMES:
            if measured_gain is None:
                measured_gain, measured_gain_name = value, name
            elif name != measured_gain_name and _disagrees((measured_gain, value)):
                alias_conflicts.append(f"local_gain:{measured_gain_name}/{name}")

    # Values frozen on the adoption outrank the referenced measurements, which
    # archives recorded before per-occurrence ids may have since overwritten.
    frozen_before = _to_float(adoption.get("throughput_before"))
    frozen_after = _to_float(adoption.get("throughput_after"))
    throughput_before = frozen_before if frozen_before is not None else measured_before
    throughput_after = frozen_after if frozen_after is not None else measured_after
    if measurement_source == "adoption_pinned" and _disagrees(
        (frozen_before, measured_before),
        (frozen_after, measured_after),
    ):
        # The adoption pinned these readings, yet they no longer say what it
        # was decided on: they were written over before ids carried an
        # occurrence. The frozen values still stand; the citation does not.
        measurement_source = "adoption_pinned_stale"

    artifact_ids = list(operation.get("artifact_refs") or [])
    artifact_ids += [aid for aid in adoption.get("artifact_ids") or [] if aid not in artifact_ids]
    artifacts: list[dict[str, str]] = []
    for artifact_id in artifact_ids:
        artifact = artifact_by_id.get(str(artifact_id))
        if not artifact:
            continue
        path = str(artifact.get("path") or artifact.get("uri") or "")
        if not path:
            continue
        artifacts.append({"kind": str(artifact.get("kind") or ""), "path": path})

    decision, decision_source = _first_recorded(
        ("adoption.decision", adoption.get("decision")),
        ("decision.verdict", decision_row.get("verdict")),
        ("outputs.decision", outputs.get("decision")),
        ("outputs.status", outputs.get("status")),
        ("operation.status", operation.get("status")),
    )
    decision = str(decision or "").strip().upper()
    adopted = (
        bool(adoption)
        and adoption.get("validated") is True
        and str(adoption.get("decision") or "").upper() in _KEEP_DECISIONS
    )
    evidence = decision_row.get("evidence") if isinstance(decision_row.get("evidence"), dict) else {}
    # Deliberately never named ``gain_pct``: this is what the executor measured
    # against its own starting point, which is not the session baseline once
    # anything has been adopted. Summing these across attempts is wrong, and a
    # shared field name is all it takes for someone to try.
    local_gain_pct, local_gain_source = _first_recorded(
        ("adoption.gain_pct", _to_float(adoption.get("gain_pct"))),
        ("decision.evidence.gain_pct", _to_float(evidence.get("gain_pct"))),
        ("outputs.delta_pct", _to_float(outputs.get("delta_pct"))),
        (f"measurement.{measured_gain_name}" if measured_gain_name else "measurement", measured_gain),
    )
    keep_threshold_pct, keep_threshold_source = _threshold_from_operation(operation)

    return {
        "attempt_id": operation_id,
        "agent": _attempt_agent(operation, adoption),
        "agent_method": (
            "recorded" if str(operation.get("agent") or adoption.get("agent") or "").strip() else "derived"
        ),
        "producer": str(operation.get("producer") or ""),
        "kind": _work_kind(operation),
        "name": str(operation.get("name") or subject.get("name") or ""),
        "subject": {
            "type": str(subject.get("subject_type") or ""),
            "name": str(subject.get("name") or ""),
        },
        "kernel_id": (
            str(subject.get("name") or "") if "kernel" in str(subject.get("subject_type") or "").lower() else None
        ),
        "backend": str(operation.get("strategy") or ""),
        "phase": str(operation.get("phase") or ""),
        "macro_cycle": operation.get("macro_cycle"),
        "started_at": str(operation.get("started_at") or ""),
        "ended_at": str(operation.get("ended_at") or ""),
        "duration_sec": _duration_between(
            operation.get("started_at"),
            operation.get("ended_at"),
        ),
        "status": str(operation.get("status") or ""),
        "decision": decision,
        # Where each of these came from. A verdict an executor stated and a
        # status inferred from the operation around it are different claims,
        # and the value alone cannot tell them apart.
        "decision_source": decision_source,
        "decision_reason": str(adoption.get("reason") or decision_row.get("reason") or outputs.get("reason") or ""),
        "keep_threshold_pct": keep_threshold_pct,
        # A bar recorded on the gate is the one that gate ruled against; one
        # recorded on the outputs is the executor's configuration, which need
        # not be what applied.
        "keep_threshold_source": keep_threshold_source,
        "adopted": adopted,
        # What the operation says happened to the workload, as distinct from
        # what the adoption stream credits. The two are written by one call and
        # dropped independently, so they can disagree.
        "integrated": (bool(outputs.get("integrated")) if outputs.get("integrated") is not None else None),
        # What stood behind the verdict: an accuracy gate that ruled, an
        # end-to-end re-measurement, or a KEEP nothing checked the accuracy of.
        "validation_basis": str(adoption.get("validation_basis") or ""),
        "attribution_eligible": (bool(adoption.get("attribution_eligible", True)) if adoption else None),
        "local_gain_pct": round(local_gain_pct, 6) if local_gain_pct is not None else None,
        "local_gain_source": local_gain_source,
        "throughput_before": throughput_before,
        "throughput_after": throughput_after,
        # ``adoption`` means the number was frozen when the call was made;
        # ``measurement.<name>`` means it was read back afterwards, off a
        # reading recorded under that name, and could since have moved.
        "throughput_before_source": (
            "adoption"
            if frozen_before is not None
            else f"measurement.{measured_before_name}"
            if measured_before is not None
            else ""
        ),
        "throughput_after_source": (
            "adoption"
            if frozen_after is not None
            else f"measurement.{measured_after_name}"
            if measured_after is not None
            else ""
        ),
        # Roles that more than one recorded name laid claim to, with readings
        # that do not agree. The first name won; this says the choice was not
        # free.
        "alias_conflicts": alias_conflicts,
        "adoption_id": str(adoption.get("adoption_id") or "") or None,
        "gates": _gate_rows(operation),
        "backend_attempts": _sub_attempt_rows(operation),
        "measurements": measurements,
        # Which reading of a repeatedly measured subject these numbers came
        # from, and how many readings the operation has in total. Without this
        # a re-measured kernel looks the same as one measured once.
        "measurement_source": measurement_source,
        "measurement_occurrences": sum(1 for mid in recorded_ids if mid in measurement_by_id),
        "artifacts": artifacts,
    }


def _disagrees(*pairs: tuple[float | None, float | None]) -> bool:
    """Report whether a frozen value and its cited measurement have parted ways.

    Compared in relative terms so a re-serialized float never counts, while a
    genuine re-measurement always does.
    """
    for frozen, measured in pairs:
        if frozen is None or measured is None or not frozen:
            continue
        if abs(measured - frozen) / abs(frozen) > 1e-6:
            return True
    return False


def _occurrence_index(
    measurement_ids: list[str],
    measurement_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Number an operation's readings of each metric, oldest first.

    Recorded ids have to be derived from what is being recorded rather than
    from a counter, since several producers replay their records after a
    resume. That makes them stable but unreadable, so the plain ordinal a
    reader wants is assigned here, where the whole set is in hand at once.
    """
    ordered: dict[str, list[tuple[float, str]]] = {}
    for measurement_id in measurement_ids:
        measurement = measurement_by_id.get(str(measurement_id))
        if not measurement:
            continue
        name = str(measurement.get("name") or "").strip().lower()
        taken_at = _parse_ts(measurement.get("measured_at"))
        ordered.setdefault(name, []).append((taken_at if taken_at is not None else float("inf"), str(measurement_id)))
    index: dict[str, int] = {}
    for rows in ordered.values():
        # Ties fall back to the id so the numbering is at least deterministic
        # for readings whose timestamps are identical or missing.
        for position, (_, measurement_id) in enumerate(sorted(rows)):
            index[measurement_id] = position
    return index


def _latest_measurement_per_name(
    measurement_ids: list[str],
    measurement_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """Keep one measurement per name: the most recent time it was taken.

    An operation accumulates a reference for every occurrence of every metric
    it measured, because retrying a subject no longer overwrites the earlier
    numbers. With no adoption naming which occurrence a decision used, the
    newest one is the operation's current state.
    """
    newest: dict[str, tuple[float, str]] = {}
    for measurement_id in measurement_ids:
        measurement = measurement_by_id.get(str(measurement_id))
        if not measurement:
            continue
        name = str(measurement.get("name") or "").strip().lower()
        taken_at = _parse_ts(measurement.get("measured_at")) or float("-inf")
        current = newest.get(name)
        if current is None or taken_at >= current[0]:
            newest[name] = (taken_at, str(measurement_id))
    chosen = {measurement_id for _, measurement_id in newest.values()}
    return [measurement_id for measurement_id in measurement_ids if measurement_id in chosen]


def _recorded_baseline_throughput(
    operations: list[dict[str, Any]],
    measurement_by_id: dict[str, dict[str, Any]],
) -> float | None:
    """Return the session baseline throughput the recorder measured.

    The baseline is where the session started, so when it has been measured
    more than once the earliest reading is the one every gain in the report is
    stated against. Taking the newest would quietly move the denominator under
    figures that were already published.
    """
    earliest: tuple[float, float] | None = None
    for operation in operations:
        if not isinstance(operation, dict) or _work_kind(operation) != "baseline":
            continue
        for measurement_id in operation.get("measurement_refs") or []:
            measurement = measurement_by_id.get(str(measurement_id))
            if not measurement:
                continue
            # Producers differ on which of the two names they stamp on a
            # baseline run; on a baseline operation both mean the same thing.
            if str(measurement.get("name") or "").strip().lower() not in {
                "throughput",
                "baseline_throughput",
            }:
                continue
            value = _to_float(measurement.get("value"))
            if not value:
                continue
            taken_at = _parse_ts(measurement.get("measured_at")) or float("inf")
            if earliest is None or taken_at < earliest[0]:
                earliest = (taken_at, value)
    return earliest[1] if earliest else None


def _recorded_session_validation(
    operations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the last gain the run itself promoted as validated.

    This is the only figure in the section that does not come from the ledger,
    which is exactly why it is worth having: a total computed by summing the
    ledger can never be found to disagree with it. Sessions recorded before
    producers wrote this have none, and the caller falls back to the sum.

    Args:
        operations: The recorded operation stream.

    Returns:
        The newest session-validation record's outputs, or ``None``.
    """
    latest: tuple[int, float, dict[str, Any]] | None = None
    for operation in operations:
        if not isinstance(operation, dict) or _work_kind(operation) != "session_validation":
            continue
        outputs = operation.get("outputs") if isinstance(operation.get("outputs"), dict) else {}
        if _to_float(outputs.get("validated_gain_pct")) is None:
            continue
        # Stack length first: it is the run's own ordering of these
        # checkpoints, and it survives clocks that do not move monotonically.
        stack_len = int(_to_float(outputs.get("validated_at_stack_len")) or 0)
        at = _parse_ts(operation.get("ended_at") or operation.get("started_at")) or 0.0
        if latest is None or (stack_len, at) >= (latest[0], latest[1]):
            latest = (stack_len, at, outputs)
    return latest[2] if latest else None


def _summarize_by_agent(
    attempts: list[dict[str, Any]],
    gain_by_attempt: dict[str, float],
) -> dict[str, Any]:
    """Aggregate attempts into the per-agent view (first layer of the report).

    ``gain_by_attempt`` holds each adopted attempt's baseline-relative
    contribution, so per-agent totals add up to
    ``validation.attributed_total_gain_pct``. They fall short of the session's
    end-to-end gain by whatever no attempt accounts for, which is reported
    separately as ``validation.unattributed_gain_pct``.
    """
    summary: dict[str, Any] = {}
    for attempt in attempts:
        agent = str(attempt.get("agent") or "unattributed")
        bucket = summary.setdefault(
            agent,
            {
                "attempts": 0,
                "keeps": 0,
                "reverts": 0,
                "attributable_gain_pct": 0.0,
                "non_attributable_keeps": 0,
                "by_kind": {},
            },
        )
        bucket["attempts"] += 1
        kind_bucket = bucket["by_kind"].setdefault(
            str(attempt.get("kind") or "unknown"),
            {"attempts": 0, "keeps": 0, "attributable_gain_pct": 0.0},
        )
        kind_bucket["attempts"] += 1
        decision = str(attempt.get("decision") or "").upper()
        if attempt.get("adopted"):
            bucket["keeps"] += 1
            kind_bucket["keeps"] += 1
            if attempt.get("attribution_eligible") is False:
                bucket["non_attributable_keeps"] += 1
            else:
                gain = gain_by_attempt.get(str(attempt.get("attempt_id") or ""), 0.0)
                bucket["attributable_gain_pct"] += gain
                kind_bucket["attributable_gain_pct"] += gain
        elif decision in _REVERT_DECISIONS:
            bucket["reverts"] += 1
    for bucket in summary.values():
        bucket["attributable_gain_pct"] = round(float(bucket["attributable_gain_pct"]), 6)
        for kind_bucket in bucket["by_kind"].values():
            kind_bucket["attributable_gain_pct"] = round(
                float(kind_bucket["attributable_gain_pct"]),
                6,
            )
    return summary


def _duration_between(started_at: Any, ended_at: Any) -> float | None:
    started = _parse_ts(started_at)
    ended = _parse_ts(ended_at)
    if started is None or ended is None or ended < started:
        return None
    return round(ended - started, 6)


def _collect_gemm_tuning_runs(
    operations: list[dict[str, Any]],
    adoptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adoption_by_operation = {
        str(adoption.get("operation_id") or ""): adoption
        for adoption in adoptions
        if isinstance(adoption, dict) and adoption.get("operation_id")
    }
    runs: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("kind") != "gemm_tuning":
            continue
        extensions = operation.get("extensions") if isinstance(operation.get("extensions"), dict) else {}
        gemm_extension = extensions.get("gemm") if isinstance(extensions.get("gemm"), dict) else {}
        result = dict(gemm_extension.get("result")) if isinstance(gemm_extension.get("result"), dict) else {}
        outputs = operation.get("outputs") if isinstance(operation.get("outputs"), dict) else {}
        adoption = adoption_by_operation.get(
            str(operation.get("operation_id") or ""),
            {},
        )
        result.setdefault(
            "engine",
            outputs.get("engine") or operation.get("strategy") or "",
        )
        result.setdefault("status", operation.get("status") or "")
        result.setdefault("decision", outputs.get("decision") or "")
        result.setdefault("source", operation.get("producer") or "")
        result.setdefault(
            "ts",
            operation.get("ended_at") or operation.get("started_at") or "",
        )
        result.setdefault(
            "duration_sec",
            _duration_between(
                operation.get("started_at"),
                operation.get("ended_at"),
            ),
        )
        result.setdefault(
            "adopted",
            adoption.get("validated") is True
            and str(adoption.get("decision") or "").upper() in {"KEEP", "ADOPT", "ADOPTED"},
        )
        if result.get("gain_pct") is None:
            result["gain_pct"] = _to_float(adoption.get("gain_pct"))
        if "candidates" not in result:
            result["candidates"] = [
                dict(attempt.get("outputs") or {})
                for attempt in operation.get("attempts") or []
                if isinstance(attempt, dict)
            ]
        runs.append(result)
    return runs


def collect_recorded_optimizations(
    session_id: str,
    operations: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    adoptions: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    geak_invocations: list[dict[str, Any]],
    forge_invocations: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the optimization read model straight from author-time records.

    Every field here traces to something a producer recorded when it happened,
    so ownership, verdicts, and the thresholds behind them survive export
    instead of being re-inferred from business state after the fact.
    """

    measurement_by_id = {
        str(row.get("measurement_id") or ""): row
        for row in measurements
        if isinstance(row, dict) and row.get("measurement_id")
    }
    artifact_by_id = {
        str(row.get("artifact_id") or ""): row for row in artifacts if isinstance(row, dict) and row.get("artifact_id")
    }
    adoption_by_operation: dict[str, dict[str, Any]] = {}
    for row in adoptions:
        if not isinstance(row, dict):
            continue
        operation_id = str(row.get("operation_id") or "")
        if not operation_id:
            continue
        current = adoption_by_operation.get(operation_id)
        # An operation can transition (adopted then revoked); the last word wins.
        if current is None or str(row.get("adopted_at") or row.get("revoked_at") or "") >= str(
            current.get("adopted_at") or current.get("revoked_at") or ""
        ):
            adoption_by_operation[operation_id] = row

    # The ledger walks operations, so an adoption whose operation is not among
    # them contributes nothing and says nothing. Both streams are written by
    # separate calls that can fail separately, which is exactly how one lands
    # without the other.
    kind_by_operation = {
        str(operation.get("operation_id") or ""): _work_kind(operation)
        for operation in operations
        if isinstance(operation, dict) and operation.get("operation_id")
    }

    attempts: list[dict[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if _work_kind(operation) not in _ATTEMPT_KINDS:
            continue
        attempts.append(
            _recorded_attempt_row(
                operation,
                adoption=adoption_by_operation.get(
                    str(operation.get("operation_id") or ""),
                    {},
                ),
                measurement_by_id=measurement_by_id,
                artifact_by_id=artifact_by_id,
            )
        )
    attempts.sort(
        key=lambda row: (
            _parse_ts(row.get("ended_at") or row.get("started_at")) or float("inf"),
            str(row.get("attempt_id") or ""),
        )
    )

    orphan_adoptions: list[str] = []
    off_ledger_adoptions: list[str] = []
    for row in adoptions:
        if not isinstance(row, dict):
            continue
        adoption_id = str(row.get("adoption_id") or "")
        kind = kind_by_operation.get(str(row.get("operation_id") or ""))
        if kind is None:
            orphan_adoptions.append(adoption_id)
        elif kind not in _ATTEMPT_KINDS:
            off_ledger_adoptions.append(f"{adoption_id}({kind})")
    if orphan_adoptions:
        warnings.append(
            "optimizations: adoptions reference operations that were never "
            f"recorded, so their gain is absent from the ledger: "
            f"{sorted(set(orphan_adoptions))[:5]}"
        )
    if off_ledger_adoptions:
        # A kind missing from ``_ATTEMPT_KINDS`` is how a newly added optimizer
        # silently stays out of the accounting.
        warnings.append(
            "optimizations: adoptions belong to operation kinds the ledger "
            "does not count as attempts: "
            f"{sorted(set(off_ledger_adoptions))[:5]}"
        )

    # The mirror image of an orphan adoption, and the more damaging of the two.
    # An operation saying the change was integrated is the workload having
    # moved; with no adoption to credit it, the gain walk below skips the step
    # entirely, yet the next adopted step still starts from the higher figure.
    # The difference lands in ``unattributed_gain_pct``, where it is
    # indistinguishable from drift nobody caused -- a plausible number in place
    # of a missing record. Both rows come from one call through a writer that
    # swallows its own failures, so losing one and keeping the other is
    # reachable rather than theoretical.
    unclaimed_integrations = [
        str(attempt["attempt_id"])
        for attempt in attempts
        if attempt.get("integrated") is True and not attempt.get("adoption_id")
    ]
    if unclaimed_integrations:
        warnings.append(
            f"optimizations: {len(unclaimed_integrations)} operation(s) record "
            "an integrated change with no adoption crediting it, so their gain "
            "is booked as unattributed rather than to the step that earned it: "
            f"{sorted(unclaimed_integrations)[:5]}"
        )

    alias_conflicts = sorted({conflict for attempt in attempts for conflict in attempt.get("alias_conflicts") or []})
    if alias_conflicts:
        # Two names for one role, disagreeing. Whichever was read first won,
        # and the chain arithmetic carries that choice into every later step.
        warnings.append(
            "optimizations: measurements recorded under different names for "
            f"the same role disagree, and the first read won: {alias_conflicts[:5]}"
        )

    # Reported gain is measured against the session baseline, so each adopted
    # step contributes the percentage points it added to the cumulative figure.
    # An executor's own local gain is relative to whatever it started from,
    # which is not the baseline once anything has already been adopted.
    #
    # ``entries`` is the gain ledger over ``attempts``: one row per adopted and
    # attributable attempt, carrying only what the chain arithmetic needs.
    # Everything descriptive stays on the attempt and is reachable through
    # ``adopted_attempt_id``.
    baseline_tput = _recorded_baseline_throughput(operations, measurement_by_id)
    entries: list[dict[str, Any]] = []
    cumulative = 0.0
    # Throughput the next adopted step is expected to start from: the baseline
    # for the first one, then wherever the previous one left off. A step that
    # starts somewhere else means something moved the workload without being
    # adopted, and that movement belongs to nobody. Crediting it to the next
    # step is how a kernel's reported gain silently absorbs an earlier patch.
    expected_before = baseline_tput
    unattributed = 0.0
    attributed = 0.0
    for attempt in attempts:
        if not attempt.get("adopted") or attempt.get("attribution_eligible") is False:
            continue
        local_gain = _to_float(attempt.get("local_gain_pct"))
        throughput_before = _to_float(attempt.get("throughput_before"))
        throughput_after = _to_float(attempt.get("throughput_after"))
        drift = 0.0
        chain_continuous = True
        if baseline_tput and throughput_after:
            started_from = throughput_before or expected_before or baseline_tput
            drift = (started_from - expected_before) / baseline_tput * 100.0 if expected_before else 0.0
            # Percentage points of the baseline this step added. Stated this
            # way the rows sum exactly, with no chaining subtleties to get
            # wrong, and any drift stays outside the sum.
            gain = (throughput_after - started_from) / baseline_tput * 100.0
            gain_method = "baseline_chain"
            expected_before = throughput_after
        elif baseline_tput and expected_before and local_gain is not None:
            # The step ran and moved the workload; only its finishing
            # throughput went unrecorded. A local gain is by definition
            # measured against where the step started, which is where the
            # previous one left off, so the missing reading can be put back
            # and the chain carried on.
            #
            # Adding the local figure to the running total directly would be
            # a unit error as well as a broken chain: it is a percentage of
            # this step's own starting point, not percentage points of the
            # baseline.
            projected_after = expected_before * (1.0 + local_gain / 100.0)
            gain = (projected_after - expected_before) / baseline_tput * 100.0
            gain_method = "local_gain_projected"
            expected_before = projected_after
            # If the local figure was measured against something else, the
            # next step's drift is what says so.
            chain_continuous = False
        else:
            gain = local_gain
            gain_method = "recorded_adoption" if local_gain is not None else "missing"
            # Nothing left to chain from: crediting the next step's head start
            # to whoever follows is the error this guards against.
            expected_before = None
            chain_continuous = False
        unattributed += drift
        cumulative += drift + (gain or 0.0)
        # Summed unrounded, so the reported gap equals the drift exactly rather
        # than trailing it by a rounding step.
        attributed += gain or 0.0
        stack_index = len(entries)
        entries.append(
            {
                "id": f"{session_id}:optimization:{stack_index}",
                "stack_index": stack_index,
                "adopted_attempt_id": attempt["attempt_id"],
                "adoption_id": attempt.get("adoption_id"),
                "source": attempt["agent"],
                "source_method": attempt["agent_method"],
                "optimization_kind": attempt["kind"],
                "name": attempt["name"],
                "backend": attempt.get("backend") or None,
                # Gain against the session baseline: the only figure that may
                # be summed, and the one ``cumulative_gain_pct`` is built from.
                "gain_pct": round(gain, 6) if gain is not None else None,
                "gain_method": gain_method,
                # False when this step's finishing throughput was never
                # recorded, so the drift across it could not be measured.
                "chain_continuous": chain_continuous,
                # The executor's own figure, carried so the two are visibly
                # different numbers rather than one ambiguous field.
                "local_gain_pct": (round(local_gain, 6) if local_gain is not None else None),
                "cumulative_gain_pct": round(cumulative, 6),
                "throughput_after": throughput_after,
                "validated": True,
                "ts": attempt.get("ended_at") or "",
            }
        )

    # Below this the drift is float noise from re-serialized throughputs, well
    # under any measurement's own repeatability.
    if abs(unattributed) > 0.01:
        warnings.append(
            "optimizations: "
            f"{unattributed:+.6f}pp of the session's gain sits between adopted "
            "steps and belongs to no attempt; it is reported as "
            "validation.unattributed_gain_pct rather than credited to the step "
            "that follows it"
        )
    discontinuous = [str(entry["adopted_attempt_id"]) for entry in entries if not entry.get("chain_continuous")]
    if discontinuous:
        warnings.append(
            f"optimizations: {len(discontinuous)} adopted step(s) recorded no "
            "finishing throughput, so the drift across them could not be "
            f"measured: {sorted(discontinuous)[:5]}"
        )

    summary_by_agent = _summarize_by_agent(
        attempts,
        {str(entry["adopted_attempt_id"]): _to_float(entry.get("gain_pct")) or 0.0 for entry in entries},
    )
    backend_attempts = _collect_backend_attempts(
        session_id,
        geak_invocations,
        forge_invocations,
    )

    summary_by_source = _empty_summary()
    summary_by_kind: dict[str, dict[str, Any]] = {}
    for entry in entries:
        source = str(entry["source"])
        bucket = summary_by_source.setdefault(source, {"keeps": 0, "total_gain_pct": 0.0})
        gain = _to_float(entry.get("gain_pct")) or 0.0
        bucket["keeps"] += 1
        bucket["total_gain_pct"] = round(float(bucket["total_gain_pct"]) + gain, 6)
        if source == "kernel_agent":
            backend = str(entry.get("backend") or "unattributed")
            by_backend = bucket.setdefault("by_backend", {})
            if backend not in by_backend:
                backend = "unattributed"
            backend_bucket = by_backend.setdefault(
                backend,
                {"keeps": 0, "total_gain_pct": 0.0},
            )
            backend_bucket["keeps"] += 1
            backend_bucket["total_gain_pct"] = round(
                float(backend_bucket["total_gain_pct"]) + gain,
                6,
            )
        kind_bucket = summary_by_kind.setdefault(
            str(entry.get("optimization_kind") or "unknown"),
            _empty_kind_summary(),
        )
        kind_bucket["keeps"] += 1
        kind_bucket["total_gain_pct"] = round(
            float(kind_bucket["total_gain_pct"]) + gain,
            6,
        )

    non_attributable = [
        attempt for attempt in attempts if attempt.get("adopted") and attempt.get("attribution_eligible") is False
    ]
    if non_attributable:
        withheld = sorted(str(attempt.get("attempt_id") or "<unknown>") for attempt in non_attributable)
        # Name the cut explicitly: a bare 5-element list reads as the complete
        # set, so someone chasing the sixth withheld adoption never learns it
        # exists.
        shown = f"{withheld[:5]}" + (f" (first 5 of {len(withheld)})" if len(withheld) > 5 else "")
        warnings.append(
            "optimizations: "
            f"{len(non_attributable)} kept adoption(s) have no attributable "
            f"throughput pair, so their gain was withheld: {shown}"
        )
    # `entries` is the GAIN ledger and deliberately holds only attributable keeps, so a backend whose
    # keeps are all non-attributable reads as zero keeps — indistinguishable from an optimizer that
    # produced nothing. That is the same conflation the canonical-stream fix set out to remove, one
    # layer further in. Count the keep from the attempts, keep the gain coming from the entries.
    for attempt in non_attributable:
        if str(attempt.get("agent") or "") != "kernel_agent":
            continue
        by_backend = summary_by_source.setdefault("kernel_agent", {"keeps": 0, "total_gain_pct": 0.0}).setdefault(
            "by_backend", {}
        )
        backend = str(attempt.get("backend") or "unattributed")
        if backend not in by_backend:
            backend = "unattributed"
        backend_bucket = by_backend.setdefault(
            backend, {"keeps": 0, "total_gain_pct": 0.0, "non_attributable_keeps": 0}
        )
        backend_bucket["keeps"] += 1
        backend_bucket["non_attributable_keeps"] = int(backend_bucket.get("non_attributable_keeps", 0)) + 1
    # An adopted step that contributes nothing to the total is a hole in the
    # accounting, not a zero. Counting it keeps the sum honest about what it
    # could not see.
    unmeasured = [str(entry["adopted_attempt_id"]) for entry in entries if entry.get("gain_method") == "missing"]
    if unmeasured:
        warnings.append(
            f"optimizations: {len(unmeasured)} adopted step(s) recorded neither "
            "a throughput nor a gain, so they contribute nothing to the "
            f"session total: {sorted(unmeasured)[:5]}"
        )
    stale_evidence = [
        str(attempt["attempt_id"])
        for attempt in attempts
        if attempt.get("measurement_source") == "adoption_pinned_stale"
    ]
    if stale_evidence:
        # The frozen numbers still stand; what no longer stands is the trail
        # back to the readings they came from.
        warnings.append(
            f"optimizations: {len(stale_evidence)} adoption(s) cite measurements "
            "that were later written over, so their evidence cannot be "
            f"re-checked: {sorted(stale_evidence)[:5]}"
        )

    unscored_keeps = sum(
        1
        for attempt in attempts
        if attempt.get("adopted") and attempt.get("validation_basis") == "keep_verdict_unscored"
    )

    session_validation = _recorded_session_validation(operations)
    recorded_total = _to_float(session_validation.get("validated_gain_pct")) if session_validation else None
    if recorded_total is not None and abs(recorded_total - cumulative) > 0.01:
        # The one disagreement this section could never previously surface:
        # the ledger and the figure the run promoted are now two independent
        # numbers, so they can be seen to part company.
        warnings.append(
            "optimizations: the ledger totals "
            f"{cumulative:+.6f}pp but the run promoted {recorded_total:+.6f}pp as "
            "validated; adopted steps are missing from the ledger or their "
            "recorded throughputs disagree with the end-to-end measurement"
        )
    return {
        "schema_version": OPTIMIZATIONS_SCHEMA_VERSION,
        "source_of_truth": "recorder",
        # Stated on both paths. Telling a session that recorded nothing apart
        # from one that optimized nothing is the point of this section, and a
        # consumer cannot make that call against a key that is only present
        # when the answer is no.
        "available": True,
        "attempts": attempts,
        "entries": entries,
        "backend_attempts": backend_attempts,
        "summary_by_agent": summary_by_agent,
        "summary_by_source": summary_by_source,
        "summary_by_kind": summary_by_kind,
        "validation": {
            "method": ("recorded_session_validation" if recorded_total is not None else "ledger_sum"),
            "validated_at_stack_len": (
                int(_to_float(session_validation.get("validated_at_stack_len")) or 0)
                if session_validation
                else len(entries)
            ),
            # What the session moved end to end, and how much of that any
            # attempt is willing to claim. The difference is the part no
            # adopted step accounts for, and it is stated rather than absorbed.
            "validated_total_gain_pct": round(
                recorded_total if recorded_total is not None else cumulative,
                6,
            ),
            # The same figure the ledger arrives at on its own. Kept beside the
            # measured one so the two can be seen to differ; when the measured
            # one is absent they are the same number by construction, and
            # nothing here can be checked.
            "ledger_total_gain_pct": round(cumulative, 6),
            "validation_basis": (
                str(session_validation.get("measurement_basis") or "") if session_validation else "ledger_sum"
            ),
            "validation_source": (str(session_validation.get("source") or "") if session_validation else ""),
            "reconciliation_gap_pct": (round(recorded_total - cumulative, 6) if recorded_total is not None else None),
            "attributed_total_gain_pct": round(attributed, 6),
            "unattributed_gain_pct": round(unattributed, 6),
            "attribution_gap_pct": round(
                (recorded_total if recorded_total is not None else cumulative) - attributed,
                6,
            ),
            "attempt_count": len(attempts),
            "keep_count": len(entries),
            "non_attributable_keep_count": len(non_attributable),
            # Adopted, counted, but with nothing measured behind them.
            "unmeasured_keep_count": len(unmeasured),
            # Adopted steps whose finishing throughput was reconstructed from
            # the executor's own percentage rather than read from a
            # measurement.
            "projected_keep_count": sum(1 for entry in entries if entry.get("gain_method") == "local_gain_projected"),
            # Adopted steps whose evidence trail no longer resolves.
            "stale_evidence_count": len(stale_evidence),
            # Changes the ledger says landed with nothing crediting them. Any
            # number here means the unattributed figure is overstated by
            # whatever these steps earned.
            "unclaimed_integration_count": len(unclaimed_integrations),
            # Adopted on the strength of a KEEP verdict alone, with no
            # accuracy gate having ruled on them.
            "unscored_keep_count": unscored_keeps,
            "notes": ["Projected from author-time recorder streams (operations/adoptions/measurements/artifacts)."],
        },
        "gemm_tuning_runs": _collect_gemm_tuning_runs(operations, adoptions),
    }
