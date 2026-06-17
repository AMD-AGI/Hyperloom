# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Session breakdown exporter.

Produces ``session_breakdown.json`` — a single self-contained snapshot of
every fact a dashboard needs about one hyperloom session.

Public surface:

* :func:`build` — pure builder (read-only, returns a dict)
* :func:`write_breakdown_json` — build + atomic write to disk
* :const:`BREAKDOWN_FILENAME` — canonical filename under ``session_dir``
* :const:`SCHEMA_VERSION` — the wire-shape version string
* :const:`EXPORTER_VERSION` — this exporter implementation version
"""

from __future__ import annotations

from .agent_timeline import (
    AGENT_TIMELINE_SCHEMA,
    build_agent_timeline,
    enrich_breakdown_with_langfuse_timeline,
)
from .exporter import (
    BREAKDOWN_FILENAME,
    EXPORTER_VERSION,
    build,
    patch_breakdown_langfuse,
    write_breakdown_json,
    write_minimal_final_report,
)
from .schema import SCHEMA_VERSION
from .session_package import package_session_artifacts

__all__ = [
    "AGENT_TIMELINE_SCHEMA",
    "BREAKDOWN_FILENAME",
    "EXPORTER_VERSION",
    "SCHEMA_VERSION",
    "build",
    "build_agent_timeline",
    "enrich_breakdown_with_langfuse_timeline",
    "package_session_artifacts",
    "patch_breakdown_langfuse",
    "write_breakdown_json",
    "write_minimal_final_report",
]
