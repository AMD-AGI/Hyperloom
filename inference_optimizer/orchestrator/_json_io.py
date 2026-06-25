"""Shared safe-JSON read helper with one precise swallowed-exception set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None, *, require_dict: bool = False) -> Any:
    """Parse JSON from *path*; return *default* on OSError/JSONDecodeError (or non-dict when *require_dict*)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if require_dict and not isinstance(data, dict):
        return default
    return data


__all__ = ["read_json"]
