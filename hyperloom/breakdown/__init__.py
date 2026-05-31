"""Session breakdown exporter.

Produces ``session_breakdown.json`` — a single self-contained snapshot of
every fact a dashboard, KB, or post-mortem reviewer needs about one
hyperloom session.

Public surface:

* :func:`build` — pure builder (read-only, returns a dict)
* :func:`write_breakdown_json` — build + atomic write to disk
* :func:`render_session_report` — build markdown report from breakdown JSON
* :const:`BREAKDOWN_FILENAME` — canonical filename under ``session_dir``
* :const:`SCHEMA_VERSION` — the wire-shape version string
"""

from __future__ import annotations

from .exporter import (
    BREAKDOWN_FILENAME,
    EXPORTER_VERSION,
    build,
    write_breakdown_json,
)
from .reporters import render_session_report
from .schema import SCHEMA_VERSION

__all__ = [
    "BREAKDOWN_FILENAME",
    "EXPORTER_VERSION",
    "SCHEMA_VERSION",
    "build",
    "render_session_report",
    "write_breakdown_json",
]
