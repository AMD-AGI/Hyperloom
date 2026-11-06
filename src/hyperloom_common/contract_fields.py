# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The field checks contract types validate their documents with, raising the error their own contract names.

A contract module binds one :class:`ContractFields` to its error type, so a caller catching that module's error
catches every field it rejects.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from typing import TypeVar

__all__ = ["ContractFields", "field_names"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_E = TypeVar("_E", bound=Enum)


def field_names(kind: type) -> tuple[str, ...]:
    """The dataclass ``kind``'s field names in declaration order: the keys of a document that mirrors its fields."""
    return tuple(item.name for item in dataclass_fields(kind))


@dataclass(frozen=True)
class ContractFields:
    """Strict field checks that raise ``error``; each returns the value it accepted."""

    error: type[ValueError]

    def enum(self, kind: type[_E], value: object, what: str) -> _E:
        try:
            return kind(value)
        except ValueError as exc:
            raise self.error(f"{what} must be one of {sorted({m.value for m in kind})}, got {value!r}") from exc

    def text(self, value: object, what: str) -> str:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise self.error(f"{what} must be a non-empty string without surrounding whitespace, got {value!r}")
        return value

    def optional_text(self, value: object, what: str) -> str | None:
        return None if value is None else self.text(value, what)

    def matching(self, pattern: re.Pattern[str], value: object, what: str) -> str:
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise self.error(f"{what} must match {pattern.pattern}, got {value!r}")
        return value

    def sha256(self, value: object, what: str) -> str:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise self.error(f"{what} must be a lowercase hex sha256, got {value!r}")
        return value

    def optional_sha256(self, value: object, what: str) -> str | None:
        return None if value is None else self.sha256(value, what)

    def integer(self, value: object, what: str, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise self.error(f"{what} must be an integer of at least {minimum}, got {value!r}")
        return value

    def optional_integer(self, value: object, what: str, *, minimum: int = 0) -> int | None:
        return None if value is None else self.integer(value, what, minimum=minimum)

    def number(self, value: object, what: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise self.error(f"{what} must be a finite number, got {value!r}")
        return float(value)

    def optional_number(self, value: object, what: str) -> float | None:
        return None if value is None else self.number(value, what)

    def boolean(self, value: object, what: str) -> bool:
        if not isinstance(value, bool):
            raise self.error(f"{what} must be true or false, got {value!r}")
        return value

    def optional_boolean(self, value: object, what: str) -> bool | None:
        return None if value is None else self.boolean(value, what)

    def mapping(self, value: object, what: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise self.error(f"{what} must be an object, got {value!r}")
        return value

    def known(self, value: object, keys: tuple[str, ...], what: str) -> Mapping[str, object]:
        """``value`` as a mapping of ``keys`` only: a key a writer misspelt would otherwise drop its field unseen."""

        fields = self.mapping(value, what)
        unknown = sorted(str(key) for key in fields if key not in keys)
        if unknown:
            plural = "s" if len(unknown) > 1 else ""
            named = ", ".join(repr(key) for key in unknown)
            raise self.error(f"unknown key{plural} {named} in {what}; known: {', '.join(keys)}")
        return fields

    def string_map(self, value: object, what: str) -> Mapping[str, str]:
        if not isinstance(value, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        ):
            raise self.error(f"{what} must map strings to strings, got {value!r}")
        return value

    def items(self, value: object, what: str) -> tuple[object, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise self.error(f"{what} must be a list, got {value!r}")
        return tuple(value)

    def strings(self, value: object, what: str) -> tuple[str, ...]:
        """A document's list of strings, as a tuple."""

        if (
            isinstance(value, str)
            or not isinstance(value, Sequence)
            or not all(isinstance(item, str) for item in value)
        ):
            raise self.error(f"{what} must be a list of strings, got {value!r}")
        return tuple(value)

    def string_tuple(self, value: object, what: str) -> tuple[str, ...]:
        """A constructed field that must already be a tuple of strings, so the frozen type stays hashable."""

        if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
            raise self.error(f"{what} must be a tuple of strings, got {value!r}")
        return value

    def unique(self, values: Sequence[object], what: str) -> None:
        duplicates = sorted({str(value) for value in values if values.count(value) > 1})
        if duplicates:
            raise self.error(f"duplicate {what}: {', '.join(duplicates)}")
