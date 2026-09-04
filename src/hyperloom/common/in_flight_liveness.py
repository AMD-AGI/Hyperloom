# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a ``state=running`` status file still describes a living process.

In-flight kernels are read off the filesystem rather than from state, so a
subprocess killed by a signal never gets to clear its own marker. Without a
liveness check that stale marker makes its kernel look permanently busy, and the
kernel is skipped for the rest of the session -- and across a resume, since the
markers outlive the process that wrote them.

Two independent signals, either of which is enough to call a marker stale: the
recorded pid is gone, or the file has not been touched for longer than a
subprocess could plausibly go without a heartbeat.
"""

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
    """Decide whether one status marker still represents a running subprocess.

    Args:
        state: The marker's ``state`` field; only ``running`` can be in flight.
        pid: Recorded process id, when the writer supplied one.
        mtime: Marker's last-modified time, in epoch seconds.
        now: Current time, injected for testability.
        max_silence_sec: How long a marker may go untouched before it is stale.
        pid_alive: Predicate used to test the pid; defaults to a signal-0 probe.

    Returns:
        The verdict. A marker that is not ``running`` is simply not in flight and
        carries no stale reason -- it completed normally.
    """
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
    """Coerce a recorded pid; anything unusable means "cannot check by pid".

    A non-integral number is refused rather than truncated: rounding 1.5 to 1
    would probe an unrelated process and answer confidently about the wrong one.
    """
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
        # Cannot tell; assume alive so a probe failure never frees a kernel that
        # is genuinely still being worked on.
        return True
    return True
