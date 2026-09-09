# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two ids the recording layer uses: the event id, and the fragment key.

An **event id** names one timeline event: ``{phase}:{macro_cycle}:{component}``,
for instance ``kernel_agent:3:kernel``. Three segments are enough because a
``(phase, macro_cycle, component)`` triple *is* one object on the timeline. A
component that runs several times inside one phase run contributes several rows
to that one event rather than several events, so there is no fourth segment for
a task id.

A **fragment key** names one row of fact: ``{event id}:{row type}:{natural
id}``, for instance ``kernel_agent:3:kernel:lane:att-7``. The event id goes in
front for two reasons. Every row of one event then shares a filename prefix, so
the spool directory can be read by eye. And the prefix is load-bearing: the
spool is per session, not per event, so two events holding a row with the same
natural id would otherwise upsert into one file and the first event's row would
vanish. The payload has to repeat the event id as a field as well, since
assembly selects rows by it -- omitting either half breaks something, but
different things.

Every segment must be derivable from persisted state and author-time constants
alone. No wall clock, no random, no in-process counter: a resumed process
recomputes the ids of the run it is continuing, and an id that changed would
split one run's facts into two incomplete halves. That rules out
``transition_id`` (which embeds a Unix timestamp) and ``state.session_id``
(``utc_now_compact()`` plus ``uuid4()``) as inputs. It also means no session
segment is needed: the spool already lives under the session directory.
"""

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

# Author-time tokens: phase names, component names, row-type names. Kept to a
# narrow alphabet so an id stays readable in a filename and can never contain
# the separator, which is what makes the prefix relationship between the two id
# forms parseable.
_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class EventId(NamedTuple):
    """The three segments of an event id.

    Attributes:
        phase (str): The phase the event belongs to.
        macro_cycle (int): The macro cycle the phase ran in.
        component (str): The component the event is about.
    """

    phase: str
    macro_cycle: int
    component: str


def _token(value: str, *, label: str) -> str:
    """Normalize and validate one author-time id segment.

    Args:
        value (str): The raw segment, in any case.
        label (str): The segment's name, for the error message.

    Returns:
        str: The lowercased segment.

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
    """Build the event id for one timeline event.

    Args:
        phase (str): The phase name, e.g. ``KERNEL_AGENT`` or ``kernel_agent``;
            case is normalized away.
        macro_cycle (int): The macro cycle from persisted state.
        component (str): The component this event is about, an author-time
            constant such as ``kernel`` or ``roofline``.

    Returns:
        str: ``{phase}:{macro_cycle}:{component}``.

    Raises:
        ValueError: If ``phase`` or ``component`` is empty or holds characters
            outside ``[a-z0-9_]``, or if ``macro_cycle`` is not a non-negative
            integer.
    """
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

    Returns:
        EventId: The parsed segments.

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
    """Build the fragment key for one row of an event.

    Args:
        event (str): The event id the row belongs to; goes in front.
        row_type (str): The kind of row, an author-time constant such as
            ``lane`` or ``rebench``.
        *natural_ids (str): The row's own identity, in narrowing order -- an
            attempt id, a task id plus a run index, whatever makes the row
            unique within its section for this event. Values come from the
            data, so anything non-empty is accepted except the separator these
            segments are joined on. Pass none at all for the event-level
            fragment, whose key is the bare event id.

    Returns:
        str: ``{event id}:{row type}:{natural ids...}``, or ``event`` when no
            row type and no natural id was given.

    Raises:
        ValueError: If ``event`` is not a valid event id, if ``row_type`` is
            empty or outside ``[a-z0-9_]``, or if any natural id is empty or
            contains the separator. An empty natural id is rejected because it
            makes two distinct rows share a key, and the second one silently
            merges into the first. A separator inside a value does the same
            thing less visibly: ``("a:b", "c")`` and ``("a", "b:c")`` join to
            one string, and the fragment filename cannot tell them apart
            either, since its digest is taken over this key.
    """
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
