"""Shared safe-JSON helpers with one precise swallowed-exception set."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Fenced ```json``` block; shared by every model-reply extractor.
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


def extract_first_json_with_key(text: str, key: str, bare_re: re.Pattern[str]) -> dict | None:
    """Pull the first JSON object containing *key* out of a model reply.

    Prefers a fenced ```json block (least ambiguous), then falls back to
    *bare_re* matches, progressively trimming trailing prose until
    ``json.loads`` accepts a candidate.

    Args:
        text: The raw model reply that may contain a JSON object.
        key: Required top-level key the returned dict must contain.
        bare_re: Caller-supplied bare-object fallback pattern (group 1 is the
            candidate ``{...}``); the required key differs per caller.

    Returns:
        The first dict containing *key*, or ``None`` when none is found.
    """
    if not text:
        return None
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and key in data:
            return data
    for m in bare_re.finditer(text):
        candidate = m.group(1)
        for end in range(len(candidate), 0, -1):
            try:
                data = json.loads(candidate[:end])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and key in data:
                return data
            break  # parsed but wrong shape; don't keep shrinking
    return None


__all__ = ["read_json", "extract_first_json_with_key"]
