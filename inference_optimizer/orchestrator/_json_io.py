"""Shared safe-JSON read helper with one precise swallowed-exception set."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Fenced ```json block matcher; group(1) captures the enclosed object.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def read_json(path: Path, default: Any = None, *, require_dict: bool = False) -> Any:
    """Parse JSON from *path*; return *default* on OSError/JSONDecodeError (or non-dict when *require_dict*)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if require_dict and not isinstance(data, dict):
        return default
    return data


def extract_first_json_with_key(
    text: str,
    required_key: str,
    bare_re: re.Pattern[str],
) -> dict[str, Any] | None:
    """Pull the first JSON object carrying *required_key* out of a model reply.

    Prefers a fenced ```json block, then falls back to the bare top-level
    object matched by *bare_re*, trimming trailing prose from the right until
    ``json.loads`` accepts a candidate.

    Args:
        text: Raw model reply that may contain a fenced or bare JSON object.
        required_key: Top-level key the returned dict must contain.
        bare_re: Compiled regex whose ``group(1)`` captures a bare JSON
            candidate (each backend keys it on its own envelope shape).

    Returns:
        The first dict containing *required_key*, or ``None`` when none parses.
    """
    if not text:
        return None
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and required_key in data:
            return data
    for m in bare_re.finditer(text):
        candidate = m.group(1)
        for end in range(len(candidate), 0, -1):
            try:
                data = json.loads(candidate[:end])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and required_key in data:
                return data
            break  # parsed but wrong shape; don't keep shrinking
    return None


__all__ = ["read_json", "extract_first_json_with_key"]
