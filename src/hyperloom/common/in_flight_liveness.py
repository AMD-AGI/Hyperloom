# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a ``state=running`` status file still describes a living process."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

#: A running subprocess refreshes its status file far more often than this. Set
#: well above the slowest observed heartbeat so a busy machine is never mistaken
#: for a dead one.
DEFAULT_MAX_SILENCE_SEC = 1800.0

STALE_NONE = ""
STALE_PID_GONE = "pid_gone"
STALE_SILENT = "silent_too_long"


@dataclass(frozen=True)
class LivenessVerdict:
    """Whether a marker counts as in flight, and why not when it does not."""

    in_flight: bool
    stale_reason: str = STALE_NONE


def evaluate_marker(
    *,
    state: object,
    pid: object = None,
    mtime: float | None = None,
    now: float | None = None,
    max_silence_sec: float = DEFAULT_MAX_SILENCE_SEC,
    pid_alive: object = None,
) -> LivenessVerdict:
    """Decide whether one status marker still represents a running subprocess."""
    if str(state or "").strip().lower() != "running":
        return LivenessVerdict(False)
    probe = pid_alive if callable(pid_alive) else _pid_is_alive
    resolved_pid = _as_pid(pid)
    if resolved_pid is not None and not probe(resolved_pid):
        return LivenessVerdict(False, STALE_PID_GONE)
    if mtime is not None and max_silence_sec > 0:
        current = time.time() if now is None else now
        if current - float(mtime) > max_silence_sec:
            return LivenessVerdict(False, STALE_SILENT)
    return LivenessVerdict(True)


def _as_pid(value: object) -> int | None:
    """Coerce a recorded pid; anything unusable means \"cannot check by pid\"."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        pid = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_is_alive(pid: int) -> bool:
    """Signal-0 probe. A pid we may not signal is still a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Cannot tell; assume alive so a probe failure never frees a kernel that is genuinely still being worked on.
        return True
    return True
