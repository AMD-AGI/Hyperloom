# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Projection of trajectory ledger rows onto Langfuse span specs (pure; no SDK).

Only terminal and point rows project; an open row supplies the start time and parent of the terminal row that closes
its span. Every projection is a span, never a generation, so the per-request usage carried in trajectory attributes
does not add to Langfuse's generation-usage totals, which ``llm_calls.jsonl`` already feeds.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from . import langfuse_mapping as lfmap
from .trajectory_trace import (
    OPEN_STATUSES,
    STATUS_FAILED,
    STATUS_POINT,
    STATUS_QUEUED,
    STATUS_STARTED,
    TERMINAL_STATUSES,
)

_DEFAULT_AGENT = "trajectory"


@dataclass(frozen=True)
class TrajectorySpanSpec:
    """Everything the emitter needs to create one closed Langfuse span."""

    name: str
    phase: str
    agent: str
    start: datetime | None
    end: datetime | None
    level: str
    status_message: str | None
    metadata: dict[str, Any]


def span_openings(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Map ``span_id`` to its earliest open row per open status (``queued`` / ``started``)."""
    openings: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        span_id = row.get("span_id")
        status = row.get("status")
        if not span_id or status not in OPEN_STATUSES:
            continue
        by_status = openings.setdefault(str(span_id), {})
        current = by_status.get(status)
        if current is None or str(row.get("ts") or "") < str(current.get("ts") or ""):
            by_status[status] = row
    return openings


def span_name(row: dict[str, Any]) -> str:
    """``<event_type>`` or ``<event_type>:<attributes.name>`` when the event names itself."""
    event_type = str(row.get("event_type") or "event")
    label = str(_attributes_of(row).get("name") or "").strip()
    return f"{event_type}:{label}" if label else event_type


def _first(*values: Any) -> Any:
    return next((v for v in values if v not in (None, "")), None)


def _attributes_of(row: dict[str, Any]) -> dict[str, Any]:
    attributes = row.get("attributes")
    return attributes if isinstance(attributes, dict) else {}


def project_row(
    row: dict[str, Any],
    openings: dict[str, dict[str, dict[str, Any]]],
) -> TrajectorySpanSpec | None:
    """Return the span spec for a terminal or point row; ``None`` for open or unknown rows."""
    status = row.get("status")
    if status not in TERMINAL_STATUSES and status != STATUS_POINT:
        return None
    by_status = openings.get(str(row.get("span_id") or "")) or {}
    queued = by_status.get(STATUS_QUEUED) or {}
    opening = by_status.get(STATUS_STARTED) or queued
    start_ts = _first(opening.get("ts"), row.get("start_ts"), row.get("ts"))
    attributes = {**_attributes_of(queued), **_attributes_of(opening), **_attributes_of(row)}
    error_message = attributes.get("error_message")
    return TrajectorySpanSpec(
        name=span_name({**row, "attributes": attributes}),
        phase=str(_first(opening.get("phase"), row.get("phase")) or lfmap.UNPHASED),
        agent=str(_first(row.get("agent"), opening.get("agent"), row.get("component")) or _DEFAULT_AGENT),
        start=lfmap.parse_ts(start_ts),
        end=lfmap.parse_ts(row.get("ts")),
        level=lfmap.LEVEL_ERROR if status == STATUS_FAILED else lfmap.LEVEL_DEFAULT,
        status_message=str(error_message) if status == STATUS_FAILED and error_message else None,
        metadata={
            "kind": "trajectory",
            "event_type": row.get("event_type"),
            "status": status,
            "span_id": row.get("span_id"),
            "parent_span_id": _first(
                queued.get("parent_span_id"),
                opening.get("parent_span_id"),
                row.get("parent_span_id"),
            ),
            "component": _first(row.get("component"), opening.get("component")),
            "task_id": _first(row.get("task_id"), opening.get("task_id")),
            "call_id": _first(row.get("call_id"), opening.get("call_id")),
            "tick": _first(opening.get("tick"), row.get("tick")),
            "queued_ts": queued.get("ts"),
            "attributes": attributes,
        },
    )


__all__ = ["TrajectorySpanSpec", "project_row", "span_name", "span_openings"]
