# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Field projections shared by the SBD V6 event recorders."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from typing import Any

from hyperloom.common.timeutil import now_iso

__all__ = [
    "MAX_EXT_BLOCK_BYTES",
    "MAX_HOT_KERNELS",
    "MAX_WARNING_MESSAGE_CHARS",
    "STATUS_ORDER",
    "analysis_artifacts",
    "analysis_detail",
    "as_dict",
    "as_list",
    "bounded_block",
    "clip",
    "failure_row",
    "float_or_none",
    "graded_axes",
    "int_or_none",
    "now_iso_micros",
    "now_iso_seconds",
    "summarize_hot_kernels",
    "summarize_warnings",
    "text_or_none",
    "worst_status",
]

now_iso_seconds = functools.partial(now_iso, "seconds")

#: For rows whose order carries meaning and that land faster than one a second.
#: Ordering does not rest on this alone -- the wall clock is not monotonic across
#: an NTP step or a resume, so rows that must hold an order carry an explicit
#: ordinal and use the stamp only to read them by.
now_iso_micros = functools.partial(now_iso, "microseconds")

# The full candidate list already lives in the ``kernel_candidates`` artifact,
# so the event carries only the ranking head. 15 matches the
# ``hot_kernels_top15`` slice that the pipeline itself routes on.
MAX_HOT_KERNELS = 15

# Warning payloads carry long remediation prose.
MAX_WARNING_MESSAGE_CHARS = 600

# Ceiling for the open-ended blocks a tool fills freely (``route_ext``, per-step ``detail``).
MAX_EXT_BLOCK_BYTES = 8192


def clip(value: Any, limit: int = MAX_WARNING_MESSAGE_CHARS) -> str:
    """Coerce to str and clip to ``limit`` characters with an elision marker."""
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


def as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """Return ``value`` when it is a list/tuple, else an empty list."""
    return list(value) if isinstance(value, (list, tuple)) else []


def int_or_none(value: Any) -> int | None:
    """Best-effort int coercion that reports ``None`` instead of raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    """Best-effort float coercion that reports ``None`` instead of raising."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def text_or_none(value: Any) -> str | None:
    """Distinguish "not recorded" from "recorded empty".

    V6 reserves ``None`` for a field nothing produced; ``""`` means the producer
    ran and had nothing to say, so a caller that does not know passes ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def graded_axes(source: Any) -> dict[str, Any]:
    """The four graded axes a measurement carries, as explicit nulls where it carries none.

    A synthetic run measures none of them and an AgentX round can be missing any one. Absent keys would leave a
    reader unable to tell an unmeasured axis from one the framework failed to report, and zero reads as "measured,
    and it was zero", so all four are always present.

    Recorded beside a round's output-axis figures rather than instead of them: an AgentX session is ranked on the
    slow-tail interactivity percentile with total throughput held as a guard, and none of that is recoverable from
    the output axis -- on the canonical corpus the two throughputs differ by roughly two orders of magnitude.
    """
    from hyperloom.common.perf_metric import GRADED_AXIS_KEYS, graded_axes_of

    axes = graded_axes_of(source)
    return {key: float_or_none(axes.get(key)) for key in GRADED_AXIS_KEYS}


def summarize_hot_kernels(rows: Any) -> dict[str, Any]:
    """Project the hot-kernel ranking head into the event."""
    candidates = [row for row in as_list(rows) if isinstance(row, dict)]
    top: list[dict[str, Any]] = []
    for row in candidates[:MAX_HOT_KERNELS]:
        top.append(
            {
                "name": clip(row.get("name"), 200),
                "op_name": clip(row.get("op_name"), 200),
                "category": str(row.get("category") or ""),
                "gpu_time_us": float_or_none(row.get("gpu_time_us")),
                "gpu_pct": float_or_none(row.get("gpu_pct")),
                "count": int_or_none(row.get("count")),
            }
        )
    return {"count": len(candidates), "top": top}


def summarize_warnings(rows: Any) -> list[dict[str, Any]]:
    """Normalize trace-health warnings into queryable rows.

    ``code`` already carries its own namespace (``bypass_*`` for the TraceLens-free
    reader, bare names for TraceLens), so one flat list serves every route; the
    remaining keys are parked under ``detail`` instead of widening the row.
    """
    out: list[dict[str, Any]] = []
    for row in as_list(rows):
        if not isinstance(row, dict):
            continue
        detail = {key: value for key, value in row.items() if key not in {"code", "severity", "message"}}
        out.append(
            {
                "code": str(row.get("code") or ""),
                "severity": str(row.get("severity") or "warning"),
                "message": clip(row.get("message")),
                "detail": detail,
            }
        )
    return out


def bounded_block(value: Any, *, label: str, limit_bytes: int = MAX_EXT_BLOCK_BYTES) -> Any:
    """Drop an open-ended sub-block that would blow up the event payload."""
    if not isinstance(value, (dict, list)):
        return value
    try:
        size = len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return {"omitted": True, "reason": f"{label} is not JSON-serializable"}
    if size <= limit_bytes:
        return value
    shape: dict[str, Any] = {
        "omitted": True,
        "reason": f"{label} exceeded {limit_bytes} bytes",
        "bytes": size,
    }
    if isinstance(value, dict):
        shape["keys"] = sorted(str(key) for key in value)[:40]
    else:
        shape["length"] = len(value)
    return shape


#: Action statuses from worst to best, for the event types whose event holds an
#: array of actions. A failure ranks above everything so a later action that
#: recovered cannot hide it, and a success ranks above ``skipped`` because an
#: action refused before it ran does not unmake a sibling's anchor.
STATUS_ORDER: tuple[str, ...] = ("failed", "degraded", "running", "succeeded", "skipped")


def worst_status(statuses: Iterable[Any]) -> str:
    """Reduce the statuses of an event's actions to the one the event reports.

    The worst of them per :data:`STATUS_ORDER`, an unranked status as given
    when that is all there is, or ``"skipped"`` when there are none -- an event
    holding no action recorded nothing to judge.
    """
    present = [str(status) for status in statuses if str(status or "")]
    for status in STATUS_ORDER:
        if status in present:
            return status
    return present[0] if present else "skipped"


def failure_row(*, phase: str, error_class: str = "", message: Any = "") -> dict[str, Any]:
    """Build the canonical failure row used on runs and on the event."""
    return {
        "phase": str(phase or ""),
        "error_class": str(error_class or ""),
        "message": clip(message, 2000),
    }


def analysis_artifacts(result: dict[str, Any]) -> dict[str, Any]:
    """Project the artifact paths a ``trace_analyze`` result surfaces, absent ones as ``""``."""
    return {
        "trace_report_path": str(result.get("trace_report_path") or ""),
        "analysis_report_path": str(result.get("analysis_report_path") or ""),
        "candidates_path": str(result.get("candidates_path") or ""),
        "kernel_roofline_path": str(result.get("kernel_roofline_path") or ""),
        "tracelens_summary_path": str(result.get("tracelens_summary_path") or ""),
        "cli_log_path": str(result.get("cli_log_path") or ""),
    }


def analysis_detail(result: Any) -> dict[str, Any]:
    """Project one ``trace_analyze`` result into the shared detail block.

    ``route`` and ``tool`` both come from ``_build_analysis_meta`` -- the agent
    route reports ``agent`` / ``tracelens``, the TraceLens-free reader reports
    ``bypass`` / ``bypass`` -- so keeping both preserves routing policy and tool
    provenance. Tool-specific output stays in ``route_ext``.
    """
    payload = as_dict(result)
    meta = as_dict(payload.get("analysis_meta"))
    return {
        "route": str(meta.get("route") or ""),
        "tool": str(meta.get("tool") or ""),
        "tool_run_id": str(payload.get("run_id") or ""),
        "steady_state": bounded_block(as_dict(meta.get("steady_state")), label="steady_state"),
        "preflight": bounded_block(as_dict(meta.get("preflight")), label="preflight"),
        "split": bounded_block(as_dict(meta.get("split")), label="split"),
        "selection": bounded_block(as_dict(meta.get("selection")), label="selection"),
        "steps": bounded_block(
            [as_dict(row) for row in as_list(meta.get("steps")) if isinstance(row, dict)],
            label="steps",
        ),
        "route_ext": bounded_block(as_dict(meta.get("route_ext")), label="route_ext"),
        "hot_kernels": summarize_hot_kernels(payload.get("hot_kernels_top15") or payload.get("hot_kernels")),
        "warnings": summarize_warnings(payload.get("trace_health_warnings")),
        "artifacts": analysis_artifacts(payload),
    }
