# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared row-coercion + closed-schema validation for the trace ledgers.

``conversation_trace`` and ``llm_trace`` write sibling JSONL ledgers with the
same coercion rules and the same closed-schema guard (only the field set,
exception class, and ledger label differ). These helpers are the single source
of that logic; ``parse_usage`` shares the int coercion.
"""

from __future__ import annotations

from typing import Any


def coerce_optional_str(value: Any) -> str | None:
    """Coerce a value to a non-empty stripped string, or ``None``.

    Args:
        value: Arbitrary value to normalize.

    Returns:
        The stripped string, or ``None`` when it is empty or ``None``.
    """
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def coerce_optional_int(value: Any) -> int | None:
    """Coerce a value to ``int``, or ``None`` on a miss / bad type.

    Keeps ``None`` distinct from ``0``.

    Args:
        value: Arbitrary value to convert.

    Returns:
        The integer value, or ``None`` on failure.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_closed_row(
    row: dict[str, Any],
    *,
    fields: frozenset[str],
    valid_components: frozenset[str],
    error_cls: type[Exception],
    label: str,
) -> None:
    """Fail fast (raising *error_cls*) if *row* deviates from the closed schema.

    Checks the exact field set, a non-empty string ``session_id``, and a
    ``component`` drawn from *valid_components*. *label* names the ledger in
    error messages (e.g. ``conversations`` / ``llm_calls``).

    Args:
        row: A serialized ledger row dict.
        fields: The exact set of keys the row must carry.
        valid_components: Allowed values for the ``component`` field.
        error_cls: Exception type raised on any violation.
        label: Human-readable ledger name used in error messages.
    """
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


__all__ = ["coerce_optional_str", "coerce_optional_int", "validate_closed_row"]
