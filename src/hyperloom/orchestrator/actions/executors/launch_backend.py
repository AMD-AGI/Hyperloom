# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The boundary every server/benchmark launch crosses.

:func:`launch` forwards to :data:`PRODUCTION_LAUNCH_BACKEND`, which is read on
every call, so substituting it redirects every launch in the process --
including those on worker threads and inside Ray workers.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Callable, Protocol, runtime_checkable

__all__ = [
    "PRODUCTION_LAUNCH_BACKEND",
    "LaunchBackend",
    "SessionKillLaunchBackend",
    "launch",
]


@runtime_checkable
class LaunchBackend(Protocol):
    """Runs one launch attempt and reports what it did.

    The signature is :func:`._subprocess_kill.run_with_session_kill`'s, so a
    backend cannot silently drop the watchdog inputs executors rely on.
    """

    def run(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = ...,
        cwd: str | None = ...,
        timeout: int | float | None = ...,
        text: bool = ...,
        soft_deadline_sec: float | None = ...,
        server_log_path: str | None = ...,
        server_dead_grace_sec: float | None = ...,
        detok_stall_grace_sec: float | None = ...,
        server_already_ready: bool = ...,
        on_output: Callable[[], None] | None = ...,
        session_deadline_sec: float | None = ...,
    ) -> subprocess.CompletedProcess:
        """Run ``cmd`` to completion and return its ``CompletedProcess``.

        Raises:
            subprocess.TimeoutExpired: When the hard ``timeout`` is hit. Every
                other cause that stops a round is reported as a sentinel
                ``returncode`` instead.
        """
        ...


class SessionKillLaunchBackend:
    """The production backend: a real subprocess tree, reaped on every exit."""

    def run(self, cmd: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
        """Spawn ``cmd`` through :func:`._subprocess_kill.run_with_session_kill`."""
        from ._subprocess_kill import run_with_session_kill

        return run_with_session_kill(list(cmd), **kwargs)  # type: ignore[arg-type]


#: The backend every launch goes to, read on each call.
PRODUCTION_LAUNCH_BACKEND: LaunchBackend = SessionKillLaunchBackend()


def launch(cmd: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
    """Run one launch attempt through the installed backend.

    Args:
        cmd: The command to execute.
        **kwargs: The launch's watchdog and deadline inputs, as accepted by
            :meth:`LaunchBackend.run`.

    Returns:
        subprocess.CompletedProcess: What the attempt did.
    """
    return PRODUCTION_LAUNCH_BACKEND.run(cmd, **kwargs)  # type: ignore[arg-type]
