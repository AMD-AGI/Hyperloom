# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the specialist reaper polls to tell a working subprocess from a finished or stalled one."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dsh_launch import dsh_home

PROCESS_LOG_NAME = "process.log"


@dataclass(frozen=True)
class ReapWatch:
    """What the reaper polls while a specialist subprocess runs.

    Attributes:
        workspace: Task workspace; :func:`specialist_activity_paths` of it are
            the liveness signals.
        done_files: Candidate done-file paths to poll.
        partial_files: Candidate partial-checkpoint paths polled for live
            progress; never an exit signal.
        progress_cb: Optional async callback for each new checkpoint.
        task_id: Task identifier used to pick up live ``extend_lease``
            wall-budget extensions each poll.
    """

    workspace: Path
    done_files: tuple[Path, ...]
    partial_files: tuple[Path, ...] = ()
    progress_cb: Any = None
    task_id: str = ""


def specialist_activity_paths(workspace: Path) -> tuple[Path, ...]:
    """What shows the specialist in ``workspace`` is still working, for the reaper and the planner alike.

    Its heartbeat, its process log, and the session log ``dsh`` writes every
    step to while it prints only its final message.
    """
    return (workspace / "heartbeat.json", workspace / PROCESS_LOG_NAME, dsh_home(workspace) / "sessions")


def last_activity(path: Path) -> float | None:
    """Newest mtime at ``path``, a directory counting by the newest file below it; ``None`` when there is none."""
    try:
        if not path.is_dir():
            return path.stat().st_mtime
        return max((entry.stat().st_mtime for entry in path.rglob("*") if entry.is_file()), default=None)
    except OSError:
        return None


def newest_activity(paths: Iterable[Path]) -> float | None:
    """Newest :func:`last_activity` across ``paths``; ``None`` when none of them shows any."""
    return max((mtime for path in paths if (mtime := last_activity(path)) is not None), default=None)
