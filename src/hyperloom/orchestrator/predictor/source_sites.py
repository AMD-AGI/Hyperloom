# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source locations TraceLens resolved for the hot kernels.

``hot_kernels_top15`` names a kernel's file but not its line or function; both
live in ``kernel_source_resolution.json`` in the analysis run directory. The
predictor shows a hot kernel as ``file:line function`` when it has all three,
which is what lets a source-change answer name a location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json
from hyperloom.common.kernel_source_contract import (
    METHOD_UNRESOLVED,
    SOURCE_RESOLUTION_FILENAME,
    SOURCE_RESOLUTION_SCHEMA_VERSION,
)

_EXPECTED_MAJOR = SOURCE_RESOLUTION_SCHEMA_VERSION.split(".")[0]


def _artifact(analysis_md_path: str) -> Path | None:
    # The deterministic route writes the artifact into the run directory and the
    # report into tracelens/ below it; the bypass route puts the two side by side.
    report = Path(analysis_md_path)
    for directory in (report.parent, report.parent.parent):
        candidate = directory / SOURCE_RESOLUTION_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _site(entry: dict[str, Any]) -> dict[str, Any] | None:
    source_file = str(entry.get("source_file") or "").strip()
    if not source_file or str(entry.get("method") or METHOD_UNRESOLVED) == METHOD_UNRESOLVED:
        return None
    line = entry.get("source_line")
    return {
        "source_file": source_file,
        "source_line": line if isinstance(line, int) and not isinstance(line, bool) else None,
        "source_function": str(entry.get("source_function") or "").strip() or None,
    }


def load_source_sites(analysis_md_path: Any) -> dict[str, dict[str, Any]]:
    """Resolved locations keyed by kernel id and by kernel name.

    Empty when the artifact is missing, unreadable, or of another schema major:
    a location enriches the request, it never gates it.
    """
    raw = str(analysis_md_path or "").strip()
    path = _artifact(raw) if raw else None
    doc = read_json(path, require_dict=True) if path else None
    if not isinstance(doc, dict) or str(doc.get("schema_version") or "").split(".")[0] != _EXPECTED_MAJOR:
        return {}
    by_name: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for entry in doc.get("entries") or []:
        site = _site(entry) if isinstance(entry, dict) else None
        if site is None:
            continue
        name = str(entry.get("name") or "").strip()
        if name:
            by_name.setdefault(name, site)
        kernel_id = str(entry.get("kernel_id") or "").strip()
        if kernel_id:
            by_id[kernel_id] = site
    # An id beats a different row that happens to be named like that id.
    return {**by_name, **by_id}
