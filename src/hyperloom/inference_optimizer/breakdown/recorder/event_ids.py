# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two ids the recording layer uses: the event id, and the fragment key."""

from __future__ import annotations

import re
from typing import NamedTuple

__all__ = [
    "EVENT_ID_SEGMENTS",
    "EVENT_ID_SEPARATOR",
    "EventId",
    "event_id",
    "fragment_key",
    "parse_event_id",
]

#: Separates the segments of both id forms.
EVENT_ID_SEPARATOR = ":"

#: How many segments an event id has, for callers validating one they parsed.
EVENT_ID_SEGMENTS = 3

# Author-time tokens: phase names, component names, row-type names.
_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class EventId(NamedTuple):
    """The three segments of an event id."""

    phase: str
    macro_cycle: int
    component: str


def _token(value: str, *, label: str) -> str:
    """Normalize and validate one author-time id segment.

    Raises:
        ValueError: If the segment is empty or holds anything outside
            ``[a-z0-9_]`` once lowercased -- which includes the separator, so a
            segment can never split an id it is placed into.
    """
    token = str(value or "").strip().lower()
    if not _TOKEN.fullmatch(token):
        raise ValueError(f"{label} must match [a-z0-9][a-z0-9_]*, got {value!r}")
    return token


def event_id(phase: str, macro_cycle: int, component: str) -> str:
    """Build the event id for one timeline event."""
    try:
        cycle = int(macro_cycle)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"macro_cycle must be an integer, got {macro_cycle!r}") from exc
    if cycle < 0:
        raise ValueError(f"macro_cycle must not be negative, got {cycle}")
    return EVENT_ID_SEPARATOR.join(
        (
            _token(phase, label="phase"),
            str(cycle),
            _token(component, label="component"),
        )
    )


def parse_event_id(value: str) -> EventId:
    """Split an event id back into its segments.

    Args:
        value (str): An event id built by :func:`event_id`.

    Raises:
        ValueError: If ``value`` is not three separator-joined segments, or a
            segment does not survive the same validation :func:`event_id`
            applies.
    """
    parts = str(value or "").split(EVENT_ID_SEPARATOR)
    if len(parts) != EVENT_ID_SEGMENTS:
        raise ValueError(f"event id must have {EVENT_ID_SEGMENTS} segments, got {value!r}")
    phase, cycle, component = parts
    try:
        macro_cycle = int(cycle)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"event id macro_cycle must be an integer, got {value!r}") from exc
    return EventId(
        phase=_token(phase, label="phase"),
        macro_cycle=macro_cycle,
        component=_token(component, label="component"),
    )


def fragment_key(event: str, row_type: str, *natural_ids: str) -> str:
    """Build the fragment key for one row of an event."""
    parse_event_id(event)
    if not row_type and not natural_ids:
        return event
    segments = [event, _token(row_type, label="row_type")]
    for index, natural_id in enumerate(natural_ids):
        token = str(natural_id if natural_id is not None else "").strip()
        if not token:
            raise ValueError(f"natural id at position {index} must be non-empty for row_type {row_type!r}")
        if EVENT_ID_SEPARATOR in token:
            raise ValueError(
                f"natural id at position {index} must not contain {EVENT_ID_SEPARATOR!r} "
                f"for row_type {row_type!r}, got {token!r}"
            )
        segments.append(token)
    return EVENT_ID_SEPARATOR.join(segments)
