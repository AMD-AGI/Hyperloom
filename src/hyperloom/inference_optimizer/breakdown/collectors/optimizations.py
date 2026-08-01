# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical optimization projection for ``session_breakdown.json``.

The historical breakdown contract exposes successful optimizations through
several parallel sections (``optimization_stack``, attribution, Explore,
GEAK, and Forge).  This module projects those records into one stable,
phase-aware read model while leaving the historical sections available as
audit and compatibility data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


OPTIMIZATIONS_SCHEMA_VERSION = 1

_SOURCES = (
    "warm_replay",
    "explore",
    "framework_agent",
    "kernel_agent",
    "unattributed",
)

_NON_OPTIMIZATION_ACTIONS = {
    "baseline",
    "profile",
    "roofline",
    "sweep",
    "conc_sweep",
    "validate",
    "validate_stack",
    "target_analysis",
    "trace_analyze",
}


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


def _normalized_phase(value: Any) -> str:
    phase = str(value or "").strip().upper()
    return {
        "FRAMEWORK": "framework_agent",
        "FRAMEWORK_AGENT": "framework_agent",
        "EXPLORE": "explore",
        "KERNEL": "kernel_agent",
        "KERNEL_AGENT": "kernel_agent",
    }.get(phase, "")


def _phase_timeline(state: dict[str, Any]) -> list[tuple[float, str]]:
    timeline: list[tuple[float, str]] = []
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return timeline
    for row in history:
        if not isinstance(row, dict):
            continue
        phase = _normalized_phase(row.get("to_phase"))
        ts = _parse_ts(row.get("ts_unix"))
        if ts is None:
            ts = _parse_ts(row.get("ts"))
        if phase and ts is not None:
            timeline.append((ts, phase))
    timeline.sort(key=lambda item: item[0])
    return timeline


def _phase_at(ts: float | None, timeline: list[tuple[float, str]]) -> str:
    if ts is None:
        return ""
    current = ""
    for boundary, phase in timeline:
        if boundary > ts:
            break
        current = phase
    return current


def _kernel_backend(
    raw: dict[str, Any],
    *,
    kernel_id: str,
    geak_invocations: list[dict[str, Any]],
    forge_invocations: list[dict[str, Any]],
) -> str | None:
    values = (
        raw.get("backend"),
        raw.get("engine"),
        raw.get("source"),
        raw.get("action"),
    )
    joined = " ".join(str(value or "").lower() for value in values)
    if "forge" in joined:
        return "forge"
    if "geak" in joined:
        return "geak"

    if kernel_id:
        for backend, invocations in (
            ("forge", forge_invocations),
            ("geak", geak_invocations),
        ):
            if any(
                str(inv.get("kernel_id") or "") == kernel_id
                and str(inv.get("decision") or "").upper() == "KEEP"
                for inv in invocations
                if isinstance(inv, dict)
            ):
                return backend
    return None


def _resolve_source(
    raw: dict[str, Any],
    gain: dict[str, Any],
    *,
    timeline: list[tuple[float, str]],
) -> tuple[str, str]:
    action = str(raw.get("action") or gain.get("action") or "").strip().lower()
    if action in {"replay_warm_recipe", "warm_replay"} or "warm_replay" in action:
        return "warm_replay", "action_family"

    kernel_markers = (
        raw.get("kernel_id"),
        raw.get("backend"),
        raw.get("engine"),
        raw.get("tuned_file"),
        raw.get("final_overlay"),
    )
    if (
        any(value not in (None, "", [], {}) for value in kernel_markers)
        or action in {"geak_e2e", "gemm_tuning", "fusion", "integrate"}
        or action.startswith("kernel_opt")
    ):
        return "kernel_agent", "action_family"

    explicit = _normalized_phase(
        raw.get("source_phase")
        or gain.get("source_phase")
        or raw.get("phase")
        or gain.get("phase")
    )
    if explicit:
        return explicit, "recorded"

    ts = _parse_ts(gain.get("ts_unix"))
    if ts is None:
        ts = _parse_ts(gain.get("ts") or raw.get("ts"))
    inferred = _phase_at(ts, timeline)
    if inferred:
        return inferred, "phase_history_ts"

    if action in {"explore", "backends", "params"}:
        return "explore", "action_family"
    if action == "framework_agent":
        return "framework_agent", "action_family"
    return "unattributed", "unknown"


def _optimization_kind(
    raw: dict[str, Any],
    *,
    source: str,
) -> str:
    action = str(raw.get("action") or "").strip().lower()
    operation_kind = str(raw.get("operation_kind") or "").strip().lower()
    if source == "warm_replay":
        return "serving_config"
    if source == "kernel_agent":
        if action == "gemm_tuning":
            return "gemm_tuning"
        if action == "fusion":
            return "kernel_fusion"
        return "kernel_patch"
    if source == "framework_agent":
        patch_evidence = (
            raw.get("patch_path"),
            raw.get("target_file"),
            raw.get("source_snapshot"),
            raw.get("framework_root"),
            raw.get("base_sha"),
        )
        return (
            "framework_patch"
            if any(value not in (None, "") for value in patch_evidence)
            else "serving_config"
        )
    if source == "explore":
        return operation_kind if operation_kind in {"backend", "param", "env"} else "serving_config"
    return operation_kind or "unknown"


def _artifacts(raw: dict[str, Any]) -> list[dict[str, str]]:
    fields = (
        ("patch", "patch_path"),
        ("target_file", "target_file"),
        ("tuned_file", "tuned_file"),
        ("report", "final_report_path"),
        ("report", "report_path"),
        ("overlay", "final_overlay"),
    )
    out = [
        {"kind": str(item.get("kind") or ""), "path": str(item.get("path") or "")}
        for item in raw.get("artifacts") or []
        if isinstance(item, dict) and item.get("path")
    ]
    seen: set[tuple[str, str]] = set()
    seen.update((item["kind"], item["path"]) for item in out)
    for kind, field in fields:
        path = str(raw.get(field) or "").strip()
        key = (kind, path)
        if not path or key in seen:
            continue
        seen.add(key)
        out.append({"kind": kind, "path": path})
    return out


def _execution_mode(raw: dict[str, Any], backend: str | None) -> str | None:
    explicit = str(raw.get("execution_mode") or "").strip()
    if explicit in {"whole_pipeline", "per_kernel"}:
        return explicit
    action = str(raw.get("action") or "").strip().lower()
    if action == "geak_e2e":
        return "whole_pipeline"
    if action in {"gemm_tuning", "fusion", "integrate"} or action.startswith("kernel_opt"):
        return "per_kernel"
    if backend == "forge":
        return "per_kernel"
    if backend == "geak":
        return "whole_pipeline"
    return None


def _empty_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        source: {"keeps": 0, "total_gain_pct": 0.0}
        for source in _SOURCES
    }
    summary["kernel_agent"]["by_backend"] = {
        "geak": {"keeps": 0, "total_gain_pct": 0.0},
        "forge": {"keeps": 0, "total_gain_pct": 0.0},
        "unattributed": {"keeps": 0, "total_gain_pct": 0.0},
    }
    return summary


def collect_optimizations(
    state: dict[str, Any],
    attribution: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    forge_invocations: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the single canonical optimization read model.

    Only records in ``state.optimization_stack`` are projected: attempts,
    REVERTs, and ``no_promote`` events remain in the action timeline.  The
    legacy sections are intentionally not mutated.
    """

    raw_stack = state.get("optimization_stack") or []
    if not isinstance(raw_stack, list):
        warnings.append("optimizations: state.optimization_stack is not a list")
        raw_stack = []
    gain_rows = attribution.get("gain_per_stack_entry") or []
    if not isinstance(gain_rows, list):
        gain_rows = []
    if gain_rows and len(gain_rows) != len(raw_stack):
        warnings.append(
            "optimizations: gain_per_stack_entry length does not match "
            "optimization_stack; missing rows use throughput fallback"
        )
    try:
        validated_len = int(state.get("cumulative_gain_validated_stack_len") or 0)
    except (TypeError, ValueError):
        validated_len = 0

    baseline_tput = _to_float(state.get("baseline_tput"))
    session_id = str(state.get("session_id") or "session")
    timeline = _phase_timeline(state)
    entries: list[dict[str, Any]] = []
    previous_tput = baseline_tput

    for stack_index, raw_value in enumerate(raw_stack):
        if not isinstance(raw_value, dict):
            warnings.append(f"optimizations: stack entry {stack_index} is not an object")
            continue
        raw = dict(raw_value)
        if str(raw.get("action") or "").strip().lower() in _NON_OPTIMIZATION_ACTIONS:
            anchor_tput = _to_float(raw.get("tput"))
            if anchor_tput is not None:
                previous_tput = anchor_tput
            continue
        gain = (
            dict(gain_rows[stack_index])
            if stack_index < len(gain_rows) and isinstance(gain_rows[stack_index], dict)
            else {}
        )
        source, source_method = _resolve_source(raw, gain, timeline=timeline)
        kernel_id = str(raw.get("kernel_id") or raw.get("action_kernel_id") or "").strip()
        backend = (
            _kernel_backend(
                raw,
                kernel_id=kernel_id,
                geak_invocations=geak_invocations,
                forge_invocations=forge_invocations,
            )
            if source == "kernel_agent"
            else None
        )
        throughput_after = _to_float(raw.get("tput"))
        throughput_before = previous_tput
        delta = _to_float(gain.get("delta_pct"))
        cumulative = _to_float(gain.get("cum_gain_after"))
        if delta is None and throughput_before and throughput_after is not None:
            delta = (throughput_after / throughput_before - 1.0) * 100.0
        if cumulative is None and baseline_tput and throughput_after is not None:
            cumulative = (throughput_after / baseline_tput - 1.0) * 100.0

        name = str(
            raw.get("name")
            or raw.get("variant_name")
            or kernel_id
            or _optimization_kind(raw, source=source)
        )
        entry = {
            "id": f"{session_id}:optimization:{stack_index}",
            "stack_index": stack_index,
            "source": source,
            "source_method": source_method,
            "optimization_kind": _optimization_kind(raw, source=source),
            "name": name,
            "backend": backend,
            "execution_mode": _execution_mode(raw, backend),
            "kernel_id": kernel_id or None,
            "gain_pct": round(delta, 6) if delta is not None else None,
            "cumulative_gain_pct": (
                round(cumulative, 6) if cumulative is not None else None
            ),
            "throughput_before": throughput_before,
            "throughput_after": throughput_after,
            "validated": stack_index < validated_len,
            "task_id": str(raw.get("task_id") or gain.get("task_id") or ""),
            "ts": str(gain.get("ts") or raw.get("ts") or ""),
            "provenance": str(raw.get("provenance") or ""),
            "configuration": {
                "extra_server_args": str(
                    raw.get("extra_server_args")
                    or raw.get("candidate_extra_server_args")
                    or gain.get("extra_server_args")
                    or ""
                ),
                "extra_envs": dict(raw.get("extra_envs") or {}),
            },
            "artifacts": _artifacts(raw),
        }
        entries.append(entry)
        if throughput_after is not None:
            previous_tput = throughput_after

    summary = _empty_summary()
    for entry in entries:
        if not entry["validated"]:
            continue
        source = str(entry["source"])
        bucket = summary[source]
        bucket["keeps"] += 1
        gain = _to_float(entry.get("gain_pct")) or 0.0
        bucket["total_gain_pct"] += gain
        if source == "kernel_agent":
            backend = str(entry.get("backend") or "unattributed")
            if backend not in bucket["by_backend"]:
                backend = "unattributed"
            bucket["by_backend"][backend]["keeps"] += 1
            bucket["by_backend"][backend]["total_gain_pct"] += gain

    for bucket in summary.values():
        bucket["total_gain_pct"] = round(float(bucket["total_gain_pct"]), 6)
        for backend_bucket in bucket.get("by_backend", {}).values():
            backend_bucket["total_gain_pct"] = round(
                float(backend_bucket["total_gain_pct"]),
                6,
            )

    return {
        "schema_version": OPTIMIZATIONS_SCHEMA_VERSION,
        "entries": entries,
        "summary_by_source": summary,
    }


def collect_v4_optimizations(
    run: dict[str, Any],
    operations: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    adoptions: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Project canonical v4 adoption streams through the same read model.

    V4 never reads business-state files, so a small synthetic stack is built
    exclusively from recorder operations, validated adoptions, measurements,
    and artifact references before delegating to :func:`collect_optimizations`.
    """

    operation_by_id = {
        str(operation.get("operation_id") or ""): operation
        for operation in operations
        if isinstance(operation, dict) and operation.get("operation_id")
    }
    measurement_by_id = {
        str(measurement.get("measurement_id") or ""): measurement
        for measurement in measurements
        if isinstance(measurement, dict) and measurement.get("measurement_id")
    }
    artifact_by_id = {
        str(artifact.get("artifact_id") or ""): artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("artifact_id")
    }

    stack: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    cumulative = 0.0
    for adoption in adoptions:
        if not isinstance(adoption, dict):
            continue
        decision = str(adoption.get("decision") or "").upper()
        if decision not in {"KEEP", "ADOPT", "ADOPTED"}:
            continue
        if adoption.get("validated") is not True:
            continue
        operation_id = str(adoption.get("operation_id") or "")
        operation = operation_by_id.get(operation_id, {})
        strategy = str(operation.get("strategy") or "")
        subject = adoption.get("subject") if isinstance(adoption.get("subject"), dict) else {}
        if not subject:
            subject = operation.get("subject") if isinstance(operation.get("subject"), dict) else {}
        kernel_id = str(subject.get("name") or "") if "kernel" in str(subject.get("kind") or "").lower() else ""
        artifact_rows = [
            artifact_by_id[artifact_id]
            for artifact_id in adoption.get("artifact_ids") or []
            if artifact_id in artifact_by_id
        ]
        raw: dict[str, Any] = {
            "action": str(operation.get("kind") or adoption.get("kind") or ""),
            "variant_name": str(
                operation.get("name")
                or subject.get("name")
                or adoption.get("adoption_id")
                or ""
            ),
            "name": str(
                operation.get("name")
                or subject.get("name")
                or adoption.get("kind")
                or ""
            ),
            "task_id": operation_id,
            "source_phase": str(operation.get("phase") or ""),
            "backend": strategy,
            "execution_mode": (
                "whole_pipeline"
                if strategy == "geak"
                else "per_kernel"
                if "forge" in strategy
                else None
            ),
            "kernel_id": kernel_id,
            "operation_kind": str(operation.get("kind") or adoption.get("kind") or ""),
            "extra_server_args": str((adoption.get("configuration") or {}).get("extra_server_args") or ""),
            "extra_envs": dict((adoption.get("configuration") or {}).get("extra_envs") or {}),
            "ts": str(adoption.get("adopted_at") or operation.get("ended_at") or ""),
            "provenance": str(adoption.get("producer") or operation.get("producer") or ""),
        }
        for artifact in artifact_rows:
            kind = str(artifact.get("kind") or "").lower()
            path = str(artifact.get("path") or artifact.get("uri") or "")
            if not path:
                continue
            if "patch" in kind:
                raw.setdefault("patch_path", path)
            elif "tuned" in kind:
                raw.setdefault("tuned_file", path)
            elif "target" in kind or "source" in kind:
                raw.setdefault("target_file", path)
            elif "overlay" in kind:
                raw.setdefault("final_overlay", path)
            else:
                raw.setdefault("report_path", path)
        raw["artifacts"] = [
            {
                "kind": str(artifact.get("kind") or ""),
                "path": str(artifact.get("path") or artifact.get("uri") or ""),
            }
            for artifact in artifact_rows
            if artifact.get("path") or artifact.get("uri")
        ]

        throughput_after = None
        throughput_before = None
        for measurement_id in adoption.get("measurement_ids") or []:
            measurement = measurement_by_id.get(str(measurement_id), {})
            name = str(measurement.get("name") or "").lower()
            value = _to_float(measurement.get("value"))
            if value is None:
                continue
            if name in {"throughput_after", "final_throughput", "output_throughput"}:
                throughput_after = value
            elif name in {"throughput_before", "baseline_throughput"}:
                throughput_before = value
        if throughput_after is not None:
            raw["tput"] = throughput_after

        delta = _to_float(adoption.get("gain_pct"))
        cumulative += delta or 0.0
        gain_row: dict[str, Any] = {
            "action": raw["action"],
            "variant_name": raw["variant_name"],
            "delta_pct": delta,
            "cum_gain_after": cumulative,
            "ts": raw["ts"],
            "task_id": operation_id,
            "source_phase": raw["source_phase"],
        }
        if throughput_before is not None:
            gain_row["throughput_before"] = throughput_before
        stack.append(raw)
        gains.append(gain_row)

    baseline_tput = next(
        (
            _to_float(gain.get("throughput_before"))
            for gain in gains
            if _to_float(gain.get("throughput_before")) is not None
        ),
        None,
    )
    synthetic_state = {
        "session_id": str(run.get("session_id") or run.get("claw_session_id") or "session"),
        "baseline_tput": baseline_tput,
        "optimization_stack": stack,
        "gain_per_stack_entry": gains,
        "cumulative_gain_validated_stack_len": len(stack),
        "phase_history": [],
    }
    return collect_optimizations(
        synthetic_state,
        {"gain_per_stack_entry": gains},
        [],
        [],
        warnings,
    )

