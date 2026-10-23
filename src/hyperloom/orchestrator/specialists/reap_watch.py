# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the specialist reaper polls to tell a working subprocess from a finished or stalled one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReapWatch:
    """What the reaper polls while a specialist subprocess runs.

    Attributes:
        workspace: Task workspace; supplies the ``process.log`` whose mtime is
            the second liveness signal.
        done_files: Candidate done-file paths to poll.
        heartbeat_file: Heartbeat file whose mtime is one of the activity
            signals ORed together for the liveness check.
        activity_paths: Further activity signals the launch names; a directory
            counts by the newest file below it.
        partial_files: Candidate partial-checkpoint paths polled for live
            progress; never an exit signal.
        progress_cb: Optional async callback for each new checkpoint.
        task_id: Task identifier used to pick up live ``extend_lease``
            wall-budget extensions each poll.
    """

    workspace: Path
    done_files: tuple[Path, ...]
    heartbeat_file: Path
    activity_paths: tuple[Path, ...] = ()
    partial_files: tuple[Path, ...] = ()
    progress_cb: Any = None
    task_id: str = ""


def last_activity(path: Path) -> float | None:
    """Newest mtime at ``path``, a directory counting by the newest file below it; ``None`` when there is none."""
    try:
        if not path.is_dir():
            return path.stat().st_mtime
        return max((entry.stat().st_mtime for entry in path.rglob("*") if entry.is_file()), default=None)
    except OSError:
        return None
