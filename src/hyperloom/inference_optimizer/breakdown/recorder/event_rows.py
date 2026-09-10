# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assembly primitives shared by every event type: filter, sort, group, strip."""

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

#: Recording-side bookkeeping that never reaches the wire: the event id has done
#: its job once the rows are filtered, and ``ordinal`` once the array is sorted.
SCOPE_FIELDS: tuple[str, ...] = (EVENT_ID_FIELD, "ordinal")


def rows_for_event(rows: Iterable[Any], event: str) -> list[dict[str, Any]]:
    """Keep the rows belonging to one event."""
    wanted = str(event or "")
    if not wanted:
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping) and str(row.get(EVENT_ID_FIELD) or "") == wanted]


def _sort_token(value: Any) -> tuple[int, float, str]:
    """Render one field value as ``(is_empty, numeric, text)``.

    Empty sorts last, and numbers and strings both compare without raising, so
    one malformed fragment off disk cannot take the whole assembly down.
    """
    if value is None or value == "":
        return (1, 0.0, "")
    if isinstance(value, bool):
        return (0, float(value), "")
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    return (0, 0.0, str(value))


def sort_rows(rows: Sequence[Mapping[str, Any]], *, keys: Sequence[str]) -> list[dict[str, Any]]:
    """Order rows by explicit fields they carry, deterministically."""
    field_names = [str(key) for key in keys if str(key or "")]

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        tokens = tuple(_sort_token(row.get(name)) for name in field_names)
        return (*tokens, json.dumps(row, sort_keys=True, default=str, ensure_ascii=False))

    return [dict(row) for row in sorted(rows, key=sort_key)]


def group_rows(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    """Split rows by a discriminating field, preserving their order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        grouped.setdefault(str(row.get(field) or ""), []).append(dict(row))
    return grouped


def wire_row(row: Mapping[str, Any], *, drop: Sequence[str] = SCOPE_FIELDS) -> dict[str, Any]:
    """Strip recording-side bookkeeping off one row before it reaches the wire."""
    stripped = dict(row)
    for field in drop:
        stripped.pop(str(field), None)
    return stripped


def wire_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    drop: Sequence[str] = SCOPE_FIELDS,
) -> list[dict[str, Any]]:
    """Strip recording-side bookkeeping off every row."""
    return [wire_row(row, drop=drop) for row in rows if isinstance(row, Mapping)]
