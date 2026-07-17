# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Subprocess JSON bridge primitives (canonical ``subprocess_bridge``).

Single home for the "read a request JSON file / emit a response JSON to
stdout and optionally a file / raise a typed adapter error" idiom shared by
the sibling-agent runtime CLIs (critic, robustness, framework) that talk to
their host (the Coordinator or the SKILL harness) over a subprocess JSON bridge.

Zero first-party imports (stdlib only) so any package may depend on it
without creating an import cycle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class RuntimeAdapterError(RuntimeError):
    """Base class for subprocess-bridge runtime adapter errors.

    Raised on contract violations (malformed request, missing
    configuration, etc.) that the subprocess host should surface as a
    non-zero exit code rather than an uncaught traceback.
    """


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file, returning ``None`` for a blank file.

    Args:
        path: Path to the JSON file to read.

    Returns:
        The decoded JSON value, or ``None`` if the file is empty or
        contains only whitespace.
    """
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else None


def emit_json(obj: Any, out: str | None, *, make_parents: bool = False) -> None:
    """Serialise ``obj`` to JSON, writing to stdout and optionally a file.

    Always writes to stdout; additionally writes to ``out`` when it is a
    path other than ``"-"``/``None``.

    Args:
        obj: A JSON-serialisable value.
        out: Output path, or ``"-"``/``None`` for stdout only.
        make_parents: When ``True``, create ``Path(out).parent`` (``parents=True,
            exist_ok=True``) before writing the ``out`` file. Defaults to
            ``False``, so a missing parent raises ``FileNotFoundError``.
    """
    serialised = json.dumps(obj, ensure_ascii=False, indent=2)
    if out and out != "-":
        path = Path(out)
        if make_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialised + "\n", encoding="utf-8")
    sys.stdout.write(serialised + "\n")
    sys.stdout.flush()


__all__ = ["RuntimeAdapterError", "read_json", "emit_json"]
