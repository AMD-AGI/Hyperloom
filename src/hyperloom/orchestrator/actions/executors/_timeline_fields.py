# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Field projections shared by the real-time SBD V6 timeline recorders.

The ``roofline`` and ``kernel`` recorders both project the result of
``trace_analyze_handler`` -- roofline as its own ``analysis`` sub-step, kernel as
the analysis the phase requested before dispatching a rewrite. The same tool
result must land in the same shape in both, so the projection lives here rather
than once per recorder: the bounds below carry measured justifications, and a
second copy of them would drift the moment one recorder's limit is retuned.

Everything here is a pure function over a tool result, plus the one best-effort
event writer both recorders share.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import now_iso

log = logging.getLogger(__name__)

now_iso_seconds = functools.partial(now_iso, "seconds")

# Kept small on purpose. The full candidate list already lives in the
# ``kernel_candidates`` artifact, so the event carries the ranking head for
# "what did this analysis actually hand to dispatch", not the payload. p95 is
# 25 hot kernels and the max observed is 114; 15 matches the ``hot_kernels_top15``
# slice that the pipeline itself routes on.
MAX_HOT_KERNELS = 15

# Warning payloads carry long remediation prose. The code / severity pair is the
# queryable part, so the message is clipped rather than dropped.
MAX_WARNING_MESSAGE_CHARS = 600

# Ceiling for the open-ended blocks a tool fills freely (``route_ext``, per-step
# ``detail``). Generous enough for the summary dicts both tools produce today,
# small enough that a verbose one cannot dominate the SBD payload.
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

    V6 reserves ``None`` for a field nothing produced. An empty string means the
    producer ran and had nothing to say, which is a different fact, so callers
    that genuinely do not know must pass ``None`` rather than ``""``.

    Args:
        value: The raw value.

    Returns:
        The stripped text, or ``None`` when there is none.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def summarize_hot_kernels(rows: Any) -> dict[str, Any]:
    """Project the hot-kernel ranking head into the event.

    Args:
        rows: The tool's ``hot_kernels`` / ``hot_kernels_top15`` list.

    Returns:
        A dict with the full count and a bounded, trimmed ranking head.
    """
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

    Args:
        rows: The tool's ``trace_health_warnings`` list.

    Returns:
        The normalized warning rows.
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
    """Drop an open-ended sub-block that would blow up the event payload.

    Every other field here is bounded by construction, but ``route_ext`` and the
    per-step ``detail`` dicts are deliberately open: a tool can put anything in
    them, and the TraceLens-free reader in particular parks whole ``attribution``
    / ``timeline`` / ``graph_coverage`` objects there. Rather than enumerate
    tool-specific keys -- which would defeat the point of an open block -- this
    keeps the block when it is small and replaces it with its shape when it is
    not, so one verbose tool cannot silently multiply the SBD payload.

    Args:
        value: The block to bound.
        label: Block name, reported when the block is dropped.
        limit_bytes: Serialized-size ceiling for the block.

    Returns:
        The block unchanged, or a descriptor naming what was dropped.
    """
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


def failure_row(*, phase: str, error_class: str = "", message: Any = "") -> dict[str, Any]:
    """Build the canonical failure row used on runs and on the event."""
    return {
        "phase": str(phase or ""),
        "error_class": str(error_class or ""),
        "message": clip(message, 2000),
    }


def analysis_artifacts(result: dict[str, Any]) -> dict[str, Any]:
    """Project the artifact paths a ``trace_analyze`` result surfaces.

    Args:
        result: The analysis tool's result dict.

    Returns:
        The artifact path block, with absent paths as empty strings.
    """
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

    ``route`` and ``tool`` both come from ``_build_analysis_meta`` and are both
    kept: the no-LLM TraceLens route reports ``deterministic`` / ``tracelens``
    while the TraceLens-free reader reports ``bypass`` / ``bypass``, so
    collapsing them would erase the difference between "TraceLens ran without an
    LLM" and "TraceLens never ran". Tool-specific output stays in ``route_ext``.

    Args:
        result: The analysis tool's result dict.

    Returns:
        The shared per-run detail block.
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


def flush_event(session_dir: Path, event: dict[str, Any], *, component: str) -> None:
    """Persist a timeline event, parking any writer failure for the next export.

    ``write_timeline_event`` stamps its storage sequence onto ``event``, so
    re-flushing the same object updates the same file rather than appending.

    Args:
        session_dir: Session root the timeline lives under.
        event: The event dict, mutated in place with its storage sequence.
        component: Dotted component name for the write-warning sidecar.
    """
    from hyperloom.inference_optimizer.session.sbd_v6 import (
        record_write_warning,
        write_timeline_event_at,
    )

    try:
        write_timeline_event_at(session_dir, event)
    except Exception as exc:  # noqa: BLE001 — observability cannot change phase behavior
        log.debug("timeline: %s flush failed", component, exc_info=True)
        try:
            record_write_warning(session_dir, component=component, exc=exc)
        except Exception:  # noqa: BLE001 — the warning sidecar is itself best-effort
            log.debug("timeline: write-warning sidecar failed", exc_info=True)
