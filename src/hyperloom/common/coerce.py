# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Value coercion primitives (canonical ``to_float`` / ``to_int`` / ``to_str_list`` / ...)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, TypeVar

_T = TypeVar("_T")


def to_float(value: Any, default: _T | None = None) -> float | _T | None:
    """Coerce *value* to a finite ``float``, rejecting ``bool``, ``None``, and"""
    if value is None or isinstance(value, bool):
        return default
    try:
        result = float(str(value).strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


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


def optional_positive_int(value: Any, default: _T | None = None) -> int | _T | None:
    """Coerce *value* to a strictly positive ``int``, else *default*."""
    parsed = to_int(value)
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


__all__ = [
    "to_float",
    "to_int",
    "first_float",
    "first_int",
    "optional_positive_int",
    "to_unix",
    "to_str_list",
]
