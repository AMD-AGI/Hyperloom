# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The failure evidence packets of a session: one JSON file per failure under ``reports/failures/``."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from hyperloom_common.io import atomic_write_json
from hyperloom.inference_optimizer.session.session_paths import failure_evidence_path, session_failures_dir


def write_failure(session_dir: Path, packet: Mapping[str, Any]) -> None:
    """Write ``packet`` under its ``failure_id``; a later packet of the same id replaces it."""
    atomic_write_json(failure_evidence_path(session_dir, str(packet["failure_id"])), dict(packet))


def read_failure(session_dir: Path, failure_id: str) -> dict[str, Any] | None:
    """The packet written as ``failure_id``, or ``None`` when the session holds no file of that name."""
    path = failure_evidence_path(session_dir, failure_id)
    # An id is a file name: one that names a path would read outside the store.
    if path.parent != session_failures_dir(session_dir) or not path.is_file():
        return None
    packet: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return packet


def recent_failures(session_dir: Path, *, task_id: str = "") -> Iterator[dict[str, Any]]:
    """Every packet of the session, or of one task, most recently written first."""
    store = session_failures_dir(session_dir)
    if not store.is_dir():
        return
    for path in sorted(store.glob("*.json"), key=lambda found: (found.stat().st_mtime_ns, found.name), reverse=True):
        packet = json.loads(path.read_text(encoding="utf-8"))
        if not task_id or packet.get("task_id") == task_id:
            yield packet


__all__ = ["read_failure", "recent_failures", "write_failure"]
