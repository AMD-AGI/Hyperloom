# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Closing the events a killed session left open.

Every event closes itself when the phase or action that opened it ends. A
session that was killed mid-phase never reaches that call, so its fragments
survive with no closing write behind them. Export runs this first, so those
fragments become events instead of being dropped for having no entry.

What a recovered event says about itself is deliberately thin. Its rows are
assembled in full -- everything that was recorded before the kill is on the
timeline -- but its status is :data:`EVENT_STATUS_INTERRUPTED` and never the
status assembly would derive. Nothing judged the run: the verdict is the
phase's to give, and it was killed before giving it. Calling such an event
``succeeded`` because its rows look complete is the inference this design
exists to remove.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from . import baseline_event, kernel_event, roofline_event
from .assembler import BASELINE_EVENT_SECTIONS, KERNEL_EVENT_SECTIONS, ROOFLINE_EVENT_SECTIONS, event_parts
from .event_timeline import EVENT_STATUS_INTERRUPTED, finish_event, residual_events

__all__ = ["finalize_events"]

log = logging.getLogger(__name__)


class _EventType(NamedTuple):
    """One timeline event type, described well enough to recover it.

    Attributes:
        event_type (str): The timeline event type, e.g. ``kernel``.
        kind (str): The sub-kind its envelope carries.
        event_section (str): The event-level section holding one fragment per
            event, which is what says an event exists at all.
        sections (tuple[str, ...]): Every section assembly reads.
        assemble (Callable[..., tuple[dict[str, Any], str]]): The assembler,
            called as ``assemble(parts, event=...)``. Its derived status is
            discarded on this path; see the module docstring.
    """

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
        sections=KERNEL_EVENT_SECTIONS,
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
    """Close every event whose fragments outlived the phase that recorded them.

    Binds the session itself rather than taking a bound one, because export
    runs from processes that never bound anything -- a re-export of a finished
    session, a CLI reading a directory handed to it.

    Args:
        session_dir (Path): The session whose spool to recover.

    Returns:
        list[str]: The event ids closed, in the order they were closed. Empty
            when nothing was left open, which is the normal case.
    """
    from ...session.session_binding import session_scope

    closed: list[str] = []
    with session_scope(session_dir):
        for spec in _EVENT_TYPES:
            closed.extend(_finalize_type(spec))
    return closed


def _finalize_type(spec: _EventType) -> list[str]:
    """Close the open events of one type.

    Args:
        spec (_EventType): The type to recover.

    Returns:
        list[str]: The event ids closed.
    """
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
    """Return the start time recorded when the event was opened.

    Args:
        event_rows (list[dict[str, Any]]): The event-level fragments.
        event (str): The event id wanted.

    Returns:
        str: The ISO timestamp, or ``""`` when the event has none.
    """
    for row in event_rows:
        if isinstance(row, Mapping) and str(row.get("event_id") or "") == str(event):
            return str(row.get("start_time") or "")
    return ""
