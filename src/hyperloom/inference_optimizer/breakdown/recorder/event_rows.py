# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assembly primitives shared by every event type: filter, sort, group, strip.

Assembling one timeline event out of its row fragments is four steps and never
five: read the section's payloads, keep the rows belonging to this event, order
them by fields the rows carry themselves, and hand the groups to whichever wire
position they belong in. There is no de-duplication step, because one row per
fragment means duplicates cannot arise, and no inference step, because a field
that would have to be guessed is recorded where it happens instead. That second
absence is the whole line between this and the projection it replaces.

The ordering rule is the subtle one. Row order must come from explicit fields
*inside* the row, never from the fragment envelope's ``seq`` or ``ts``, because
neither means what it looks like: every upsert redraws ``seq`` and refreshes
``ts``, so both name a row's last update rather than its first write, and a row
created early but completed late sorts last. Worse, ``seq`` is a per-recorder
in-memory counter that restarts at 1 in a resumed process, so rows written
after a resume would sort ahead of rows written before it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "EVENT_ID_FIELD",
    "SCOPE_FIELDS",
    "group_rows",
    "rows_for_event",
    "sort_rows",
    "wire_row",
    "wire_rows",
]

#: The field every row payload repeats so assembly can select by it.
EVENT_ID_FIELD = "event_id"

#: Recording-side bookkeeping that never reaches the wire: the event id has
#: done its job once the rows are filtered, and ``ordinal`` is superseded by
#: the row's position once the array is sorted.
SCOPE_FIELDS: tuple[str, ...] = (EVENT_ID_FIELD, "ordinal")


def rows_for_event(rows: Iterable[Any], event: str) -> list[dict[str, Any]]:
    """Keep the rows belonging to one event.

    Args:
        rows (Iterable[Any]): Payloads read out of one section, as assembled
            from the spool; non-mapping entries are dropped.
        event (str): The event id to select.

    Returns:
        list[dict[str, Any]]: The rows whose :data:`EVENT_ID_FIELD` equals
            ``event``, in the order given.
    """
    wanted = str(event or "")
    if not wanted:
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping) and str(row.get(EVENT_ID_FIELD) or "") == wanted]


def _sort_token(value: Any) -> tuple[int, float, str]:
    """Render one field value as a totally-ordered, type-safe sort token.

    Empty values sort last: a row whose primary key was never recorded has no
    claim to a position among the rows that did record one, and putting it
    first would read as "this happened before everything else".

    Args:
        value (Any): The field value to render.

    Returns:
        tuple[int, float, str]: ``(is_empty, numeric, text)``. Numbers and
            strings both compare without raising, which matters because a
            section's rows come from disk and one malformed fragment must not
            take the whole assembly down.
    """
    if value is None or value == "":
        return (1, 0.0, "")
    if isinstance(value, bool):
        return (0, float(value), "")
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    return (0, 0.0, str(value))


def sort_rows(rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    """Order rows by explicit fields they carry, deterministically.

    Args:
        rows (Sequence[Mapping[str, Any]]): The rows of one section for one
            event.
        keys (Sequence[str]): Field names in precedence order: the primary sort
            key first, then the tiebreakers that make the order total. A
            section with no natural timestamp supplies an ``ordinal`` captured
            when its rows were replayed.

    Returns:
        list[dict[str, Any]]: The rows, ordered. The row's own JSON form is
            appended as a last-resort tiebreaker, so two rows that agree on
            every declared key still land in a fixed order rather than in
            whatever order the spool happened to be read in -- assembling the
            same fragments twice has to produce the same array.
    """
    field_names = [str(key) for key in keys if str(key or "")]

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        tokens = tuple(_sort_token(row.get(name)) for name in field_names)
        return (*tokens, json.dumps(row, sort_keys=True, default=str, ensure_ascii=False))

    return [dict(row) for row in sorted(rows, key=sort_key)]


def group_rows(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    """Split rows by a discriminating field, preserving their order.

    The dispatch step of assembly: one section holds every lane run and the
    ``lane`` field says which array each belongs in, one section holds every
    rebench attempt and ``source_kind`` says which route asked for it.

    Args:
        rows (Iterable[Mapping[str, Any]]): The ordered rows to split.
        field (str): The field to group by; rows missing it group under ``""``.

    Returns:
        dict[str, list[dict[str, Any]]]: ``{value: rows}``, each list in the
            order the rows were given.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        grouped.setdefault(str(row.get(field) or ""), []).append(dict(row))
    return grouped


def wire_row(row: Mapping[str, Any], *, drop: Sequence[str] = SCOPE_FIELDS) -> dict[str, Any]:
    """Strip recording-side bookkeeping off one row before it reaches the wire.

    Args:
        row (Mapping[str, Any]): The assembled row.
        drop (Sequence[str]): Fields to remove; defaults to
            :data:`SCOPE_FIELDS`.

    Returns:
        dict[str, Any]: A copy of ``row`` without those fields.
    """
    stripped = dict(row)
    for field in drop:
        stripped.pop(str(field), None)
    return stripped


def wire_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    drop: Sequence[str] = SCOPE_FIELDS,
) -> list[dict[str, Any]]:
    """Strip recording-side bookkeeping off every row.

    Args:
        rows (Iterable[Mapping[str, Any]]): The assembled rows.
        drop (Sequence[str]): Fields to remove; defaults to
            :data:`SCOPE_FIELDS`.

    Returns:
        list[dict[str, Any]]: The rows without those fields, in order.
    """
    return [wire_row(row, drop=drop) for row in rows if isinstance(row, Mapping)]
