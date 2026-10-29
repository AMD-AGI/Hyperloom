# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the specialist reaper polls to tell a working subprocess from a finished or stalled one."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.proctree import boot_unix, descendants, running

from .dsh_launch import dsh_home

PROCESS_LOG_NAME = "process.log"

#: How long before its last output a child may have started and still count as the tool call that output announced.
_SPAWN_SLACK_SEC: float = 0.5


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

    Its process log, and the session log ``dsh`` writes every step to while it
    prints only its final message. Both are written by the agent CLI itself;
    ``heartbeat.json`` is not one, because the agent can keep it fresh while
    the CLI is stuck.
    """
    return (workspace / PROCESS_LOG_NAME, dsh_home(workspace) / "sessions")


def tool_in_flight(cli_pid: int, last_output_unix: float) -> bool:
    """Whether the CLI has a live process it started after its last output: a tool call it is still waiting on.

    A child older than that output (an idle helper, an earlier call's background job) does not count, and a loop
    detached into another parent is not below the CLI at all.
    """
    try:
        boot, ticks_per_sec = boot_unix(), os.sysconf("SC_CLK_TCK")
        return any(
            boot + ticks / ticks_per_sec > last_output_unix - _SPAWN_SLACK_SEC and running(pid)
            for pid, ticks in descendants(cli_pid)
        )
    except (OSError, ValueError):
        return False


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
