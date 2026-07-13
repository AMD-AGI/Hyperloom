# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Subprocess JSON bridge primitives (canonical ``subprocess_bridge``).

Single home for the "read a request JSON file / emit a response JSON to
stdout and optionally a file / raise a typed adapter error" idiom shared by
the three sibling-agent runtime CLIs that talk to their host (the
Coordinator or the SKILL harness) over a subprocess JSON bridge:

* ``hyperloom.agents.critic.runtime.cli`` (``_read_json`` / ``_emit_json``;
  ``RuntimeAdapterError`` defined in the sibling ``runtime/errors.py`` and
  subclassed ~12 times there — the subclasses are untouched by this move).
* ``hyperloom.agents.robustness.runtime.cli`` (``_read_json`` / ``_emit_json`` /
  ``RuntimeAdapterError``, all three in one file).
* ``hyperloom.agents.framework.runtime.cli`` (``RuntimeAdapterError`` only —
  see "Sites intentionally NOT delegated here" below for why its
  ``_emit_json`` and ``_read_json``-shaped helpers stay local).

Zero first-party imports (stdlib only) so any package may depend on it
without creating an import cycle (anti-cycle rule: no first-party imports).

Sites intentionally NOT delegated here (kept local by design):

* ``agents/framework/runtime/cli._emit_json`` — byte-for-byte identical to
  :func:`emit_json` below *except* it additionally does
  ``Path(out).parent.mkdir(parents=True, exist_ok=True)`` before writing the
  ``--out`` file. Critic's and Robustness's ``_emit_json`` never create the
  parent directory (a missing parent raises ``FileNotFoundError``); folding
  the ``mkdir`` into the shared implementation would silently change their
  behaviour, and dropping it from framework's copy would change *its*
  behaviour (its CLI is invoked with ``--out`` paths under a fresh work dir
  that may not exist yet). This divergence is real and is left in place
  rather than force-merged.
* ``agents/framework/runtime/cli._load_request`` / ``_read_json_request`` —
  these parse a JSON request file like :func:`read_json` but additionally
  enforce "file must exist" / "must decode" / "top-level must be an object"
  and raise :class:`RuntimeAdapterError` with framework-specific messages on
  violation; they are validating parsers, not a plain "read JSON or None"
  primitive, so they are a different function shape and not merged.
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
    non-zero exit code rather than an uncaught traceback. Critic's
    ``runtime/errors.py`` subclasses this into ~12 granular error types;
    Robustness and Framework raise it directly.
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
            ``False`` to preserve the Critic/Robustness contract where a missing
            parent raises ``FileNotFoundError``; the Framework CLI passes
            ``True`` (see the divergence note in the module docstring).
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
