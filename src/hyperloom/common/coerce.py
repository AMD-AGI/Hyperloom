# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Value coercion primitives (canonical ``to_float`` / ``to_int`` / ``to_str_list`` / ...).

Single home for the "best-effort coerce a value to a number, tolerate dirty
input, reject ``bool``, fall back to a default" idiom.

Standardised semantics (one clean contract, no per-call flags):

* ``bool`` is always treated as dirty input and coerced to *default* -- never
  ``float(True) == 1.0``, since a stray boolean becoming ``1.0`` / ``0`` is
  almost always a latent bug.
* ``None`` and any non-convertible value coerce to *default* (default ``None``).
* String inputs are ``str(...).strip()``-normalised before parsing.

Zero first-party imports (stdlib only) so any package may depend on it without
creating an import cycle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypeVar

_T = TypeVar("_T")


def to_float(value: Any, default: _T | None = None) -> float | _T | None:
    """Coerce *value* to ``float``, rejecting ``bool`` and ``None``.

    Args:
        value: The value to coerce.
        default: Returned when *value* is a ``bool``, ``None``, or not
            convertible to ``float`` (default ``None``).

    Returns:
        The parsed ``float``, or *default*.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(str(value).strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: _T | None = None) -> int | _T | None:
    """Coerce *value* to ``int``, rejecting ``bool`` and ``None``.

    String inputs are stripped before parsing; a numeric string is parsed with
    ``int(str(value).strip())`` (base-10, no float truncation).

    Args:
        value: The value to coerce.
        default: Returned when *value* is a ``bool``, ``None``, or not
            convertible to ``int`` (default ``None``).

    Returns:
        The parsed ``int``, or *default*.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(str(value).strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return default


def first_float(*values: Any, default: _T | None = None) -> float | _T | None:
    """Return the first value that :func:`to_float`-parses, else *default*.

    Args:
        *values: Candidate values, tried in order.
        default: Returned when no candidate parses (default ``None``).

    Returns:
        The first successfully parsed ``float``, or *default*.
    """
    for value in values:
        parsed = to_float(value)
        if parsed is not None:
            return parsed
    return default


def first_int(*values: Any, default: _T | None = None) -> int | _T | None:
    """Return the first value that :func:`to_int`-parses, else *default*.

    Args:
        *values: Candidate values, tried in order.
        default: Returned when no candidate parses (default ``None``).

    Returns:
        The first successfully parsed ``int``, or *default*.
    """
    for value in values:
        parsed = to_int(value)
        if parsed is not None:
            return parsed
    return default


def optional_positive_int(value: Any, default: _T | None = None) -> int | _T | None:
    """Coerce *value* to a strictly positive ``int``, else *default*.

    Combines :func:`to_int` with a ``> 0`` guard: non-integer, ``bool``,
    ``None``, or non-positive inputs all collapse to *default*.

    Args:
        value: Candidate value (int, numeric string, or absent).
        default: Returned when *value* is unset, non-integer, or ``<= 0``
            (default ``None``).

    Returns:
        The positive ``int``, or *default*.
    """
    parsed = to_int(value)
    if parsed is None or parsed <= 0:
        return default
    return parsed


def to_unix(value: Any, default: _T | None = None) -> float | _T | None:
    """Coerce a timestamp *value* to unix seconds, rejecting ``bool``.

    Accepts numeric epoch seconds or an ISO-8601 string (``Z`` suffix
    tolerated). String parsing is ISO-first: an ISO-8601 timestamp is parsed to
    its epoch, falling back to a bare ``float`` cast for numeric strings.

    Args:
        value: The raw timestamp value.
        default: Returned when *value* cannot be interpreted as a timestamp
            (default ``None``).

    Returns:
        Unix seconds as ``float``, or *default*.
    """
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return default
    return default


def to_str_list(value: Any) -> list[str]:
    """Coerce *value* into a list of non-empty, stripped strings.

    Args:
        value: A string, list/tuple/set, other scalar, or ``None``.

    Returns:
        ``None`` -> ``[]``; a string -> ``[stripped]`` (dropped when blank); a
        list/tuple/set -> each element ``str``-ified, stripped, kept only when
        non-empty; any other scalar -> ``[stripped]`` when non-empty.
    """
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
