# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared row-coercion + closed-schema validation for the trace ledgers."""

from __future__ import annotations

from typing import Any

from hyperloom.common.timeutil import now_iso


def coerce_optional_str(value: Any) -> str | None:
    """Coerce a value to a non-empty stripped string, or ``None``."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def coerce_optional_int(value: Any) -> int | None:
    """Coerce a value to ``int``, or ``None`` on a miss / bad type."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def call_key_fields(record: Any) -> dict[str, Any]:
    """The ``ts``-stamped join keys both halves of one LLM call (token row, conversation row) carry."""
    return {
        "session_id": str(record.session_id),
        "ts": now_iso(),
        "component": str(record.component),
        "call_id": coerce_optional_str(record.call_id),
        "role": coerce_optional_str(record.role),
        "task_id": coerce_optional_str(record.task_id),
        "dyn_id": coerce_optional_str(record.dyn_id),
        "tick": coerce_optional_int(record.tick),
        "phase": coerce_optional_str(record.phase),
        "turn": coerce_optional_int(record.turn),
        "model": coerce_optional_str(record.model),
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


__all__ = ["call_key_fields", "coerce_optional_str", "coerce_optional_int", "validate_closed_row"]
