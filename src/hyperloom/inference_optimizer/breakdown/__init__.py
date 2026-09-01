# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session breakdown exporter.

Produces ``session_breakdown.json`` — a self-contained snapshot of every
fact a dashboard needs about one hyperloom session.

Public surface:

* :func:`build` — pure builder (read-only, returns a dict)
* :func:`write_breakdown_json` — build + atomic write to disk
* :const:`BREAKDOWN_FILENAME` — canonical filename under ``session_dir``
* :const:`SCHEMA_VERSION` — the wire-shape version string
* :const:`EXPORTER_VERSION` — this exporter implementation version
"""

from __future__ import annotations

from .exporter import (
    BREAKDOWN_FILENAME,
    EXPORTER_VERSION,
    build,
    patch_breakdown_close,
    patch_breakdown_langfuse,
    write_breakdown_json,
    write_minimal_final_json,
    write_minimal_final_report,
)
from .schema import SCHEMA_VERSION
from .session_package import package_session_artifacts

__all__ = [
    "BREAKDOWN_FILENAME",
    "EXPORTER_VERSION",
    "SCHEMA_VERSION",
    "build",
    "package_session_artifacts",
    "patch_breakdown_close",
    "patch_breakdown_langfuse",
    "write_breakdown_json",
    "write_minimal_final_json",
    "write_minimal_final_report",
]
