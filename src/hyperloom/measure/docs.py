# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Strict readers and writers shared by the instruction sheet and the result manifest.

Both documents cross a trust boundary (the application writes one for the runner, the runner writes the other for the
application), so reading is strict: unknown keys, wrong types, duplicate JSON keys and non-finite numbers are all
refusals, and a bool is never accepted where a number is expected.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, TypeVar

T = TypeVar("T")

SHA256_RE = re.compile(r"[0-9a-f]{64}")
PREFIXED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class DocumentError(ValueError):
    """A document breaks its contract; the message names the offending path."""


def canonical_json(document: Any) -> str:
    """The one JSON spelling every hash is computed over: sorted keys, no whitespace, no NaN, UTF-8 text kept."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def parse_json_object(raw: bytes, *, what: str) -> dict[str, Any]:
    """Parse ``raw`` as one JSON object, refusing duplicate keys and NaN/Infinity."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise DocumentError(f"{what}: duplicate JSON key")
        return dict(pairs)

    def no_constants(name: str) -> Any:
        raise DocumentError(f"{what}: {name} is not a JSON number")

    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=no_constants)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentError(f"{what}: not valid UTF-8 JSON ({type(exc).__name__})") from None
    if not isinstance(document, dict):
        raise DocumentError(f"{what}: the top level must be a JSON object")
    return document


class Reader:
    """A mapping being read: every key must be consumed, so an unknown key is refused when :meth:`done` runs."""

    def __init__(self, document: Any, path: str) -> None:
        if not isinstance(document, Mapping):
            raise DocumentError(f"{path}: expected an object")
        self._document = document
        self._path = path
        self._seen: set[str] = set()

    def at(self, key: str) -> str:
        return f"{self._path}.{key}"

    def raw(self, key: str) -> Any:
        if key not in self._document:
            raise DocumentError(f"{self.at(key)}: missing")
        self._seen.add(key)
        return self._document[key]

    def optional(self, key: str) -> Any:
        self._seen.add(key)
        return self._document.get(key)

    def text(self, key: str, *, pattern: re.Pattern[str] | None = None, optional: bool = False) -> str | None:
        value = self.optional(key) if optional else self.raw(key)
        if value is None and optional:
            return None
        if not isinstance(value, str) or not value:
            raise DocumentError(f"{self.at(key)}: expected a non-empty string")
        if pattern is not None and not pattern.fullmatch(value):
            raise DocumentError(f"{self.at(key)}: does not match {pattern.pattern}")
        return value

    def string(self, key: str) -> str:
        """A string that may be empty (a log tail, a YAML body)."""
        value = self.raw(key)
        if not isinstance(value, str):
            raise DocumentError(f"{self.at(key)}: expected a string")
        return value

    def integer(self, key: str, *, minimum: int | None = None, optional: bool = False) -> int | None:
        value = self.optional(key) if optional else self.raw(key)
        if value is None and optional:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise DocumentError(f"{self.at(key)}: expected an integer")
        if minimum is not None and value < minimum:
            raise DocumentError(f"{self.at(key)}: must be at least {minimum}")
        return value

    def number(self, key: str, *, minimum: float | None = None, optional: bool = False) -> float | None:
        value = self.optional(key) if optional else self.raw(key)
        if value is None and optional:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DocumentError(f"{self.at(key)}: expected a number")
        if not math.isfinite(value):
            raise DocumentError(f"{self.at(key)}: must be finite")
        if minimum is not None and value < minimum:
            raise DocumentError(f"{self.at(key)}: must be at least {minimum}")
        return float(value)

    def boolean(self, key: str) -> bool:
        value = self.raw(key)
        if not isinstance(value, bool):
            raise DocumentError(f"{self.at(key)}: expected a boolean")
        return value

    def choice(self, key: str, allowed: Sequence[str], *, optional: bool = False) -> str | None:
        value = self.text(key, optional=optional)
        if value is not None and value not in allowed:
            raise DocumentError(f"{self.at(key)}: must be one of {', '.join(allowed)}")
        return value

    def sequence(self, key: str, parse: Callable[[Any, str], T]) -> tuple[T, ...]:
        value = self.raw(key)
        if not isinstance(value, list):
            raise DocumentError(f"{self.at(key)}: expected an array")
        return tuple(parse(item, f"{self.at(key)}[{index}]") for index, item in enumerate(value))

    def section(self, key: str, parse: Callable[[Any, str], T]) -> T:
        return parse(self.raw(key), self.at(key))

    def done(self) -> None:
        extra = sorted(set(self._document) - self._seen)
        if extra:
            raise DocumentError(f"{self._path}: unknown key {extra[0]!r}")


def relative_path(value: Any, path: str) -> str:
    """A path relative to a base directory: no absolute form, no ``..``, no empty or ``.`` segment, no backslash."""
    if not isinstance(value, str) or not value:
        raise DocumentError(f"{path}: expected a non-empty relative path")
    if "\\" in value or "\x00" in value:
        raise DocumentError(f"{path}: contains a forbidden character")
    parts = value.split("/")
    if value.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise DocumentError(f"{path}: must be a relative path without '..', '.' or empty segments")
    if str(PurePosixPath(value)) != value:
        raise DocumentError(f"{path}: not in normal form")
    return value


def env_name(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _ENV_NAME_RE.fullmatch(value):
        raise DocumentError(f"{path}: not an environment variable name")
    return value


def string_map(value: Any, path: str, *, key_check: Callable[[Any, str], str] = env_name) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise DocumentError(f"{path}: expected an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        key_check(key, f"{path}.{key}")
        if not isinstance(item, str):
            raise DocumentError(f"{path}.{key}: expected a string")
        result[key] = item
    return result
