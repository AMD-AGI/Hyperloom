# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The checks the measurement contract types validate their fields with, and the error they raise.

``measurement`` and ``consistency`` share them; every type validates on construction and reads its document strictly.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import PurePosixPath
from typing import TypeVar

_SHA256 = re.compile(r"[0-9a-f]{64}")
_E = TypeVar("_E", bound=Enum)


class MeasurementError(ValueError):
    """A test request, a test result or one of their documents breaks the contract."""


def _enum(kind: type[_E], value: object, what: str) -> _E:
    try:
        return kind(value)
    except ValueError as exc:
        raise MeasurementError(f"{what} must be one of {sorted({m.value for m in kind})}, got {value!r}") from exc


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MeasurementError(f"{what} must be a non-empty string without surrounding whitespace, got {value!r}")
    return value


def _optional_text(value: object, what: str) -> str | None:
    return None if value is None else _text(value, what)


def _sha256(value: object, what: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise MeasurementError(f"{what} must be a lowercase hex sha256, got {value!r}")
    return value


def _optional_sha256(value: object, what: str) -> str | None:
    return None if value is None else _sha256(value, what)


def _int(value: object, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MeasurementError(f"{what} must be an integer of at least {minimum}, got {value!r}")
    return value


def _optional_int(value: object, what: str, *, minimum: int = 0) -> int | None:
    return None if value is None else _int(value, what, minimum=minimum)


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MeasurementError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _optional_number(value: object, what: str) -> float | None:
    return None if value is None else _number(value, what)


def _mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MeasurementError(f"{what} must be an object, got {value!r}")
    return value


def _items(value: object, what: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MeasurementError(f"{what} must be a list, got {value!r}")
    return tuple(value)


def _strings(value: object, what: str) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise MeasurementError(f"{what} must be a tuple of strings, got {value!r}")


def _relative_dir(value: object, what: str) -> str:
    text = _text(value, what) if value != "" else ""
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise MeasurementError(f"{what} must be a relative path inside the session directory, got {value!r}")
    return text


def _bool(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise MeasurementError(f"{what} must be true or false, got {value!r}")
    return value


def _optional_bool(value: object, what: str) -> bool | None:
    return None if value is None else _bool(value, what)
