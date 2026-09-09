# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Closing the events a killed session left open."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from . import baseline_event, kernel_event, roofline_event
from .assembler import BASELINE_EVENT_SECTIONS, EVENT_SECTIONS, ROOFLINE_EVENT_SECTIONS, event_parts
from .event_timeline import EVENT_STATUS_INTERRUPTED, finish_event, residual_events

__all__ = ["finalize_events"]

log = logging.getLogger(__name__)


class _EventType(NamedTuple):
    """One timeline event type, described well enough to recover it."""

    event_type: str
    kind: str
    event_section: str
    sections: tuple[str, ...]
    assemble: Callable[..., tuple[dict[str, Any], str]]


_EVENT_TYPES: tuple[_EventType, ...] = (
    _EventType(
        event_type=kernel_event.EVENT_TYPE,
        kind=kernel_event.EVENT_KIND,
        event_section=kernel_event.SECTION_EVENT,
        # Both families, matching what the phase's own close reads: a roofline dispatched inline records into the
        # kernel event, and the re-profile block is assembled from those rows.
        sections=EVENT_SECTIONS,
        assemble=kernel_event.assemble_kernel_ext,
    ),
    _EventType(
        event_type=roofline_event.EVENT_TYPE,
        kind=roofline_event.EVENT_KIND,
        event_section=roofline_event.SECTION_EVENT,
        sections=ROOFLINE_EVENT_SECTIONS,
        assemble=roofline_event.assemble_roofline_ext,
    ),
    _EventType(
        event_type=baseline_event.EVENT_TYPE,
        kind=baseline_event.EVENT_KIND,
        event_section=baseline_event.SECTION_EVENT,
        sections=BASELINE_EVENT_SECTIONS,
        assemble=baseline_event.assemble_baseline_ext,
    ),
)


def finalize_events(session_dir: Path) -> list[str]:
    """Close every event whose fragments outlived the phase that recorded them."""
    from ...session.session_binding import session_scope

    closed: list[str] = []
    with session_scope(session_dir):
        for spec in _EVENT_TYPES:
            closed.extend(_finalize_type(spec))
    return closed


def _finalize_type(spec: _EventType) -> list[str]:
    """Close the open events of one type."""
    try:
        parts = event_parts(spec.sections)
    except Exception:  # noqa: BLE001 — a spool we cannot read costs the export nothing else
        log.warning("timeline: cannot read %s fragments to recover events", spec.event_type, exc_info=True)
        return []

    closed: list[str] = []
    for residual in residual_events(parts.get(spec.event_section) or [], event_type=spec.event_type):
        try:
            ext, _derived = spec.assemble(parts, event=residual.event_id)
        except Exception:  # noqa: BLE001 — one unrecoverable event must not cost the others
            log.warning("timeline: cannot assemble interrupted %s event %s", spec.event_type, residual.event_id)
            continue
        finish_event(
            event_type=spec.event_type,
            event=residual.event_id,
            sequence=residual.sequence,
            status=EVENT_STATUS_INTERRUPTED,
            ext=ext,
            kind=spec.kind,
            start_time=_start_time(parts.get(spec.event_section) or [], residual.event_id),
        )
        closed.append(residual.event_id)
        log.warning(
            "timeline: closed %s event %s as %s (%s)",
            spec.event_type,
            residual.event_id,
            EVENT_STATUS_INTERRUPTED,
            residual.state,
        )
    return closed


def _start_time(event_rows: list[dict[str, Any]], event: str) -> str:
    """Return the start time recorded when the event was opened."""
    for row in event_rows:
        if isinstance(row, Mapping) and str(row.get("event_id") or "") == str(event):
            return str(row.get("start_time") or "")
    return ""
