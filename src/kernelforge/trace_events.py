# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading a Chrome/kineto trace, for every lane that asks one a question."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

#: Kernel events share this category; both annotation flavours carry a ``cat`` of their own.
KERNEL_CAT: str = "kernel"

#: Annotations as the host recorded them: the submitting thread's view, so their duration is queueing, not work.
CPU_ANNOTATION_CAT: str = "user_annotation"

#: The same annotations projected onto the device timeline, which is the one kernel events are on.
GPU_ANNOTATION_CAT: str = "gpu_user_annotation"


def load_trace_events(trace_path: str | Path) -> list[dict[str, Any]]:
    """Every event in a trace, gzipped or not; an unreadable or malformed trace reads as empty."""
    path = Path(trace_path)
    try:
        opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".json.gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def resolve_capture(trace: Any) -> str:
    """The capture file to read, from a path that may name the directory a round's ranks were written to.

    A profile records one file per rank, and what a session stores is sometimes the file and sometimes the
    directory; the newest capture in a directory is that round's own.
    """
    path = Path(str(trace or "").strip() or ".")
    if path.is_file():
        return str(path)
    if not path.is_dir():
        return ""
    captures = sorted(path.glob("*.trace.json*"), key=lambda p: p.stat().st_mtime)
    return str(captures[-1]) if captures else ""


__all__ = [
    "CPU_ANNOTATION_CAT",
    "GPU_ANNOTATION_CAT",
    "KERNEL_CAT",
    "load_trace_events",
    "resolve_capture",
]
