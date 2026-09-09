# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Subprocess JSON bridge primitives (canonical ``subprocess_bridge``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class RuntimeAdapterError(RuntimeError):
    """Base class for subprocess-bridge runtime adapter errors."""


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file, returning ``None`` for a blank file."""
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else None


def emit_json(obj: Any, out: str | None, *, make_parents: bool = False) -> None:
    """Serialise ``obj`` to JSON, writing to stdout and optionally a file."""
    serialised = json.dumps(obj, ensure_ascii=False, indent=2)
    if out and out != "-":
        path = Path(out)
        if make_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialised + "\n", encoding="utf-8")
    sys.stdout.write(serialised + "\n")
    sys.stdout.flush()


__all__ = ["RuntimeAdapterError", "read_json", "emit_json"]
