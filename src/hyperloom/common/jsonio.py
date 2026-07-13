# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared safe-JSON helpers (canonical ``_json_io``).

Relocated from ``hyperloom.orchestrator._json_io`` (P2.1); that re-export
shim was removed in P2.7 once all callers were updated to import directly
from here. One precise swallowed-exception set; stdlib-only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Fenced ```json``` block; shared by every model-reply extractor.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def read_json(
    path: Path,
    default: Any = None,
    *,
    require_dict: bool = False,
    strict: bool = False,
) -> Any:
    """Parse JSON from *path*.

    Tolerant by default: returns *default* on ``OSError`` / ``JSONDecodeError``
    (or when *require_dict* is set and the payload is not a dict). When *strict*
    is ``True`` the underlying ``OSError`` / ``JSONDecodeError`` propagate (and a
    *require_dict* violation raises ``ValueError``), letting callers wrap them in
    a domain-specific error.

    Args:
        path: JSON file to read.
        default: Value returned on failure in tolerant mode.
        require_dict: Require the top-level payload to be a dict.
        strict: Raise instead of returning *default* on any failure.

    Returns:
        The decoded JSON value, or *default* in tolerant mode.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if strict:
            raise
        return default
    if require_dict and not isinstance(data, dict):
        if strict:
            raise ValueError(f"expected a JSON object at {path}, got {type(data).__name__}")
        return default
    return data


def extract_first_json_with_key(
    text: str,
    required_key: str | None = None,
    bare_re: re.Pattern[str] | None = None,
    *,
    last: bool = False,
) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply.

    Prefers a fenced ```json block, then falls back to the bare top-level
    object matched by *bare_re*, trimming trailing prose from the right until
    ``json.loads`` accepts a candidate.

    Args:
        text: Raw model reply that may contain a fenced or bare JSON object.
        required_key: Top-level key the returned dict must contain. When
            ``None``, any parsed JSON object qualifies.
        bare_re: Compiled regex whose ``group(1)`` captures a bare JSON
            candidate. When ``None``, only fenced blocks are considered.
        last: When ``True``, return the *last* qualifying object instead of the
            first (useful when a reply ends with its final answer).

    Returns:
        The first (or last) qualifying dict, or ``None`` when none parses.
    """
    if not text:
        return None

    def _qualifies(data: Any) -> bool:
        return isinstance(data, dict) and (required_key is None or required_key in data)

    found: dict[str, Any] | None = None
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if _qualifies(data):
            if not last:
                return data
            found = data
    if bare_re is not None:
        for m in bare_re.finditer(text):
            candidate = m.group(1)
            for end in range(len(candidate), 0, -1):
                try:
                    data = json.loads(candidate[:end])
                except json.JSONDecodeError:
                    continue
                if _qualifies(data):
                    if not last:
                        return data
                    found = data
                break  # parsed but wrong shape; don't keep shrinking
    return found


__all__ = ["read_json", "extract_first_json_with_key"]
