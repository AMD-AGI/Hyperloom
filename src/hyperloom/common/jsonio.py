# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared safe-JSON helpers (canonical ``hyperloom.common.jsonio``). Stdlib-only."""

from __future__ import annotations

import json
import re
from collections import deque
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

# Matches a ```json or ``` fenced block, capturing its content as group 1.
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_EMPTY_UNSET = object()


def _iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yield every top-level JSON object in *text* in document order."""
    spans: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    in_string = False
    escaped = False
    matching = {"}": "{", "]": "["}
    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append((char, idx))
        elif char in "}]":
            if not stack or stack[-1][0] != matching[char]:
                continue
            opener, start = stack.pop()
            if opener == "{":
                spans.append((start, idx + 1))
    for start, end in sorted(spans, key=lambda s: s[0]):
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            yield data


def _not_object(path: Path, data: Any) -> ValueError:
    return ValueError(f"expected a JSON object at {path}, got {type(data).__name__}")


def read_json_strict(
    path: Path,
    *,
    require_dict: bool = False,
    empty_value: Any = _EMPTY_UNSET,
    errors: str = "strict",
) -> Any:
    """Parse JSON from *path*, raising ``OSError`` / ``ValueError`` on any failure.

    ``empty_value``, when given, is returned for a blank file instead of a decode error.
    """
    text = path.read_text(encoding="utf-8", errors=errors)
    if empty_value is not _EMPTY_UNSET and not text.strip():
        data = empty_value
    else:
        data = json.loads(text)
    if require_dict and not isinstance(data, dict):
        raise _not_object(path, data)
    return data


def read_json(
    path: Path,
    default: Any = None,
    *,
    require_dict: bool = False,
    on_error: Callable[[BaseException], None] | None = None,
    errors: str = "strict",
) -> Any:
    """Parse JSON from *path*; on failure report to ``on_error`` and return ``default``."""
    try:
        data = read_json_strict(path, errors=errors)
    except (OSError, json.JSONDecodeError) as exc:
        if on_error is not None:
            on_error(exc)
        return default
    if require_dict and not isinstance(data, dict):
        if on_error is not None:
            on_error(_not_object(path, data))
        return default
    return data


def read_jsonl(
    path: Path,
    *,
    require_dict: bool = False,
    skip_malformed: bool = False,
    skip_non_dict: bool = False,
    on_error: Callable[[BaseException], None] | None = None,
) -> list[Any]:
    """Parse a JSONL file; an unreadable file is reported to ``on_error`` and yields ``[]``."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        if on_error is not None:
            on_error(exc)
        return []

    rows: list[Any] = []
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
            if require_dict and not isinstance(data, dict):
                if skip_non_dict:
                    continue
                raise ValueError(f"expected JSON object at {path}:{line_no}, got {type(data).__name__}")
        except (json.JSONDecodeError, ValueError) as exc:
            if not skip_malformed:
                raise
            if on_error is not None:
                on_error(exc)
            continue
        rows.append(data)
    return rows


def _qualifying(chunks: Iterable[str], required_key: str | None) -> Iterator[dict[str, Any]]:
    """Every JSON object in ``chunks``, in order, that carries ``required_key``."""
    for chunk in chunks:
        for data in _iter_json_objects(chunk):
            if required_key is None or required_key in data:
                yield data


def _last(objects: Iterator[dict[str, Any]]) -> dict[str, Any] | None:
    tail = deque(objects, maxlen=1)
    return tail[0] if tail else None


def extract_first_json_with_key(
    text: str,
    required_key: str | None = None,
    bare_re: re.Pattern[str] | None = None,
    *,
    last: bool = False,
) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply."""
    if not text:
        return None
    fenced = (m.group(1) for m in _FENCED_BLOCK_RE.finditer(text))
    bare = (m.group(1) for m in bare_re.finditer(text)) if bare_re is not None else ()
    candidates = _qualifying(chain(fenced, bare), required_key)
    return _last(candidates) if last else next(candidates, None)


def extract_last_json_with_key(
    text: str,
    required_key: str | None = None,
) -> dict[str, Any] | None:
    """Return the last JSON object in *text* (by start offset) that qualifies."""
    if not text:
        return None
    fenced = (m.group(1) for m in _FENCED_BLOCK_RE.finditer(text))
    return _last(_qualifying(chain(fenced, (text,)), required_key))


__all__ = [
    "extract_first_json_with_key",
    "extract_last_json_with_key",
    "read_json",
    "read_json_strict",
    "read_jsonl",
]
