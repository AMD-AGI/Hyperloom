# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two files the coordinator and its supervisor share.

The supervisor never opens the session's SQLite database, whose shared-memory
journal mode is unsafe with a second writer on a network filesystem. The
coordinator stamps ``coordinator_tick.json`` at the top of every tick and the
supervisor reads it; the supervisor rewrites ``status.json`` after every
reading. Both writes are atomic and fsynced, because the reader is a different
process and, on a network filesystem, often a different host.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.io import atomic_write_json
from hyperloom.common.jsonio import read_json
from hyperloom.inference_optimizer.session.session_paths import (
    coordinator_tick_path,
    supervisor_status_path,
)

__all__ = [
    "TickStamp",
    "read_tick",
    "stamp_tick",
    "write_status",
]


@dataclass(frozen=True)
class TickStamp:
    """When the coordinator last started a tick.

    Attributes:
        pid: The process that stamped it.
        hostname: The host it stamped from; a pid means nothing off it.
        tick: The tick counter at the stamp.
        stamped_unix: When it was stamped.
    """

    pid: int
    hostname: str
    tick: int
    stamped_unix: float


def stamp_tick(session_dir: Path | str, *, tick: int, now_unix: float) -> None:
    """Record that a tick has just started.

    Args:
        session_dir: The session root directory.
        tick: The tick counter this stamp belongs to.
        now_unix: Current wall time.

    Raises:
        OSError: If the stamp cannot be written.
    """
    _write(
        coordinator_tick_path(Path(session_dir)),
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "tick": int(tick),
            "stamped_unix": float(now_unix),
        },
    )


def read_tick(session_dir: Path | str) -> TickStamp | None:
    """Read the coordinator's tick stamp.

    Args:
        session_dir: The session root directory.

    Returns:
        TickStamp | None: The stamp, ``None`` when none has been written.

    Raises:
        json.JSONDecodeError: The writer is atomic, so unparseable is corrupt.
        ValueError: The document parsed but is not an object.
    """
    try:
        raw = read_json(coordinator_tick_path(Path(session_dir)), strict=True, require_dict=True)
    except FileNotFoundError:
        return None
    return TickStamp(
        pid=int(raw["pid"]),
        hostname=str(raw["hostname"]),
        tick=int(raw["tick"]),
        stamped_unix=float(raw["stamped_unix"]),
    )


def write_status(session_dir: Path | str, payload: dict) -> None:
    """Rewrite the supervisor's status snapshot.

    Args:
        session_dir: The session root directory.
        payload: What the supervisor last observed and last did.

    Raises:
        OSError: If the snapshot cannot be written.
    """
    _write(supervisor_status_path(Path(session_dir)), payload)


def _write(path: Path, payload: dict) -> None:
    """Replace a JSON file in one step, durably enough for another host to read."""
    atomic_write_json(path, payload, trailing_newline=True, fsync=True, fsync_dir=True)
