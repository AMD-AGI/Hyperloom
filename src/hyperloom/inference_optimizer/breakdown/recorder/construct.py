# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared construction helpers for timeline event recorders.

Each event type still owns how it is opened and assembled. What they share is
the degradation: a missing session or a spool failure must not take out the
phase, and a fragment that lands after close is republished onto the same
timeline sequence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from .event_rows import rows_for_event
from .event_timeline import finish_event
from .recorder_warnings import RECORDING_ERRORS, note_failure

log = logging.getLogger(__name__)

T = TypeVar("T")


def try_make_recorder(
    build: Callable[[], T],
    *,
    label: str,
    require_bound: bool = False,
    begin: bool = False,
    note_section: str = "",
    note_detail: str = "",
) -> T | None:
    """Build a recorder, or ``None`` when recording cannot start.

    Phase behavior must not depend on the recorder existing, so construction
    failures degrade to "no event" rather than propagating.
    """
    from ...session.session_binding import session_is_bound

    try:
        if require_bound and not session_is_bound():
            log.warning(
                "%s timeline: no session bound; this event will be missing from the "
                "breakdown. The coordinator binds at startup, so this means either "
                "that never happened or the work ran outside the session's context",
                label,
            )
            return None
        recorder = build()
        if begin:
            begin_fn = getattr(recorder, "begin", None)
            if callable(begin_fn):
                begin_fn()
        return recorder
    except RECORDING_ERRORS as exc:
        log.warning(
            "%s timeline: recorder construction failed; facts for this event will "
            "be missing from the breakdown",
            label,
            extra={"error": exc},
            exc_info=True,
        )
        if note_section:
            note_failure(
                section=note_section,
                error=exc,
                detail=note_detail or f"open {label} event failed",
            )
        return None


def event_header(parts: Mapping[str, list[dict[str, Any]]], section: str, event: str) -> dict[str, Any]:
    """The first event-level fragment for ``event``, or ``{}``."""
    rows = rows_for_event(parts.get(section) or [], event)
    return rows[0] if rows else {}


def republish_closed_event(
    event: str,
    *,
    section: str,
    event_type: str,
    kind: str,
    load_parts: Callable[[], Mapping[str, Any]],
    assemble: Callable[..., tuple[dict[str, Any], str]],
    end_time: Callable[[Mapping[str, Any], dict[str, Any]], str],
) -> None:
    """Re-assemble a closed event so a fragment written after it is published.

    The export reads the durable timeline rather than re-assembling it, so a
    row landing after the close is in the spool but not in the event. Updating
    the same storage sequence puts it there. An event that is still running is
    left alone.

    Never raises: the row this re-publishes is already in the spool.
    """
    from ...session.sbd_v6 import timeline_sequence

    try:
        parts = load_parts()
        header = event_header(parts, section, event)
        closed_end = end_time(parts, header)
        if not closed_end:
            return
        ext, derived = assemble(parts, event=event)
        finish_event(
            event_type=event_type,
            event=event,
            sequence=timeline_sequence(header),
            status=derived,
            ext=ext,
            kind=kind,
            start_time=str(header.get("start_time") or ""),
            end_time=closed_end,
        )
    except RECORDING_ERRORS as exc:
        note_failure(section=section, error=exc, detail=f"re-publishing closed event {event}")
