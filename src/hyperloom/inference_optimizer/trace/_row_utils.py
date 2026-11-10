# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared row-coercion + closed-schema validation for the trace ledgers."""

from __future__ import annotations

from typing import Any

from hyperloom_common.coerce import text_or_none, to_int
from hyperloom_common.timeutil import now_iso


def call_key_fields(record: Any) -> dict[str, Any]:
    """The ``ts``-stamped join keys both halves of one LLM call (token row, conversation row) carry."""
    return {
        "session_id": str(record.session_id),
        "ts": now_iso(),
        "component": str(record.component),
        "call_id": text_or_none(record.call_id),
        "role": text_or_none(record.role),
        "task_id": text_or_none(record.task_id),
        "tick": to_int(record.tick),
        "phase": text_or_none(record.phase),
        "turn": to_int(record.turn),
        "model": text_or_none(record.model),
    }


def validate_closed_row(
    row: dict[str, Any],
    *,
    fields: frozenset[str],
    valid_components: frozenset[str],
    error_cls: type[Exception],
    label: str,
) -> None:
    """Fail fast (raising *error_cls*) if *row* deviates from the closed schema."""
    keys = set(row.keys())
    extra = sorted(keys - fields)
    missing = sorted(fields - keys)
    if extra or missing:
        raise error_cls(f"{label} row violates closed schema: extra={extra!r} missing={missing!r}")
    session_id = row.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise error_cls(f"{label} row requires a non-empty 'session_id'; got {session_id!r}")
    component = row.get("component")
    if component not in valid_components:
        raise error_cls(f"{label} row 'component'={component!r} is not one of {sorted(valid_components)!r}")


__all__ = ["call_key_fields", "validate_closed_row"]
