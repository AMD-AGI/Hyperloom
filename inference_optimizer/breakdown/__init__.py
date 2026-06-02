"""Session breakdown exporter.

Produces ``session_breakdown.json`` — a single self-contained snapshot of
every fact a dashboard needs about one hyperloom session.

Public surface:

* :func:`build` — pure builder (read-only, returns a dict)
* :func:`write_breakdown_json` — build + atomic write to disk
* :const:`BREAKDOWN_FILENAME` — canonical filename under ``session_dir``
* :const:`SCHEMA_VERSION` — the wire-shape version string
* :const:`EXPORTER_VERSION` — this exporter implementation version

See ``SKILL.md`` for usage guidance for both LLM orchestrators and
human operators.
"""

from __future__ import annotations

from .exporter import (
    BREAKDOWN_FILENAME,
    EXPORTER_VERSION,
    build,
    write_breakdown_json,
    write_minimal_final_report,
)
from .schema import SCHEMA_VERSION

__all__ = [
    "BREAKDOWN_FILENAME",
    "EXPORTER_VERSION",
    "SCHEMA_VERSION",
    "build",
    "write_breakdown_json",
    "write_minimal_final_report",
]
