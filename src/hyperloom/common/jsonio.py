# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared safe-JSON helpers (canonical ``hyperloom.common.jsonio``). Stdlib-only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator

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


def read_json(
    path: Path,
    default: Any = None,
    *,
    require_dict: bool = False,
    strict: bool = False,
    on_error: Callable[[BaseException], None] | None = None,
    empty_value: Any = _EMPTY_UNSET,
) -> Any:
    """Parse JSON from *path*."""
    try:
        text = path.read_text(encoding="utf-8")
        if empty_value is not _EMPTY_UNSET and not text.strip():
            data = empty_value
        else:
            data = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        if strict:
            raise
        if on_error is not None:
            on_error(exc)
        return default
    if require_dict and not isinstance(data, dict):
        exc = ValueError(f"expected a JSON object at {path}, got {type(data).__name__}")
        if strict:
            raise exc
        if on_error is not None:
            on_error(exc)
        return default
    return data


def read_jsonl(
    path: Path,
    default: Any = None,
    *,
    require_dict: bool = False,
    skip_malformed: bool = False,
    skip_non_dict: bool = False,
    on_error: Callable[[BaseException], None] | None = None,
) -> list[Any]:
    """Parse a JSONL file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        if on_error is not None:
            on_error(exc)
        return [] if default is None else default

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


def coerce_dict(value: dict[str, Any] | Path | str | None, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a dict value or load one from a JSON path."""
    fallback = {} if default is None else default
    if value is None:
        return fallback
    if isinstance(value, dict):
        return value
    path = Path(value) if isinstance(value, (str, Path)) else None
    if path is None:
        return fallback
    try:
        is_file = path.is_file()
    except OSError:
        return fallback
    if not is_file:
        return fallback
    return read_json(path, default=fallback, require_dict=True)


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

    def _qualifies(data: Any) -> bool:
        return isinstance(data, dict) and (required_key is None or required_key in data)

    found: dict[str, Any] | None = None
    for m in _FENCED_BLOCK_RE.finditer(text):
        for data in _iter_json_objects(m.group(1)):
            if _qualifies(data):
                if not last:
                    return data
                found = data
    if bare_re is not None:
        for m in bare_re.finditer(text):
            for data in _iter_json_objects(m.group(1)):
                if _qualifies(data):
                    if not last:
                        return data
                    found = data
    return found


def extract_last_json_with_key(
    text: str,
    required_key: str | None = None,
) -> dict[str, Any] | None:
    """Return the last JSON object in *text* (by start offset) that qualifies."""
    if not text:
        return None

    def _qualifies(data: Any) -> bool:
        return isinstance(data, dict) and (required_key is None or required_key in data)

    found: dict[str, Any] | None = None
    for m in _FENCED_BLOCK_RE.finditer(text):
        for data in _iter_json_objects(m.group(1)):
            if _qualifies(data):
                found = data
    for data in _iter_json_objects(text):
        if _qualifies(data):
            found = data
    return found


def iter_sse_objects(raw: str) -> Iterator[Any]:
    """Yield JSON objects decoded from an MCP HTTP response body."""
    text = raw.lstrip()
    if text.startswith("{") or text.startswith("["):
        try:
            yield json.loads(text)
        except json.JSONDecodeError:
            return
        return
    for block in re.split(r"\r?\n\r?\n", raw):
        parts: list[str] = []
        for line in block.splitlines():
            if line.startswith("data:"):
                seg = line[5:]
                parts.append(seg[1:] if seg.startswith(" ") else seg)
        payload = "\n".join(parts).strip()
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


__all__ = [
    "coerce_dict",
    "extract_first_json_with_key",
    "extract_last_json_with_key",
    "iter_sse_objects",
    "read_json",
    "read_jsonl",
]
