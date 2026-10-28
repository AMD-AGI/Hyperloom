# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Value coercion primitives (canonical ``to_float`` / ``to_int`` / ``to_str_list`` / ...)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, TypeVar

_T = TypeVar("_T")


def to_float(value: Any, default: _T | None = None) -> float | _T | None:
    """Coerce *value* to a finite ``float``, rejecting ``bool``, ``None``, and non-finite values (``nan`` / ``inf``)."""
    if value is None or isinstance(value, bool):
        return default
    try:
        result = float(str(value).strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def recorded_float(value: Any, default: float = 0.0) -> float:
    """Coerce a recorded value to ``float``, keeping ``inf``, ``nan`` and ``bool`` as they were recorded.

    :func:`to_float` returns *default* for those three instead, which would
    render a degenerate figure as one nobody can tell from a real measurement.
    """
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: _T | None = None) -> int | _T | None:
    """Coerce *value* to ``int``, rejecting ``bool``, ``None``, and non-finite floats."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    try:
        return int(str(value).strip() if isinstance(value, str) else value)
    except (TypeError, ValueError, OverflowError):
        return default


def first_float(*values: Any, default: _T | None = None) -> float | _T | None:
    """Return the first value that :func:`to_float`-parses, else *default*."""
    for value in values:
        parsed = to_float(value)
        if parsed is not None:
            return parsed
    return default


def first_int(*values: Any, default: _T | None = None) -> int | _T | None:
    """Return the first value that :func:`to_int`-parses, else *default*."""
    for value in values:
        parsed = to_int(value)
        if parsed is not None:
            return parsed
    return default


def first_text(*values: Any) -> str:
    return str(next((value for value in values if value), "")).strip()


def optional_positive_int(value: Any, default: _T | None = None) -> int | _T | None:
    """Coerce *value* to a strictly positive ``int``, else *default*."""
    parsed = to_int(value)
    if parsed is None or parsed <= 0:
        return default
    return parsed


def optional_positive_float(value: Any, default: _T | None = None) -> float | _T | None:
    """Coerce *value* to a strictly positive finite ``float``, else *default*."""
    parsed = to_float(value)
    if parsed is None or parsed <= 0:
        return default
    return parsed


def to_unix(value: Any, default: _T | None = None) -> float | _T | None:
    """Coerce a timestamp *value* to unix seconds, rejecting ``bool``."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return default
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return default


def to_str_list(value: Any) -> list[str]:
    """Coerce *value* into a list of non-empty, stripped strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def text_or_none(value: Any) -> str | None:
    """``str(value)`` stripped, or ``None`` when *value* is ``None`` or blank."""
    if value is None:
        return None
    return str(value).strip() or None


def first_str(*values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def to_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def to_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def dict_rows(value: Any) -> list[dict[str, Any]]:
    """The dict entries of a list, in order; ``[]`` when *value* is not a list."""
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


__all__ = [
    "dict_rows",
    "to_float",
    "recorded_float",
    "to_int",
    "first_float",
    "first_int",
    "first_str",
    "optional_positive_int",
    "text_or_none",
    "to_unix",
    "to_list",
    "to_mapping",
    "to_str_list",
]
