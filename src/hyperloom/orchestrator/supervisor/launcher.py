# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start and stop the supervisor alongside an optimizer run.

The supervisor is spawned by the process it watches, into its own session, so
that the coordinator's teardown -- which reaps process groups -- cannot take it
down as collateral.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from hyperloom.common.env_safety import scrub_benchmark_process_env
from hyperloom.common.proctree import running
from hyperloom.inference_optimizer.session.session_paths import supervisor_log_path
from hyperloom.orchestrator.supervisor.watch import DEFAULT_TICK_STALL_SEC

log = logging.getLogger(__name__)

#: Set to ``0``/``false`` to run without a supervisor at all.
SUPERVISOR_ENABLE_ENV = "HYPERLOOM_SUPERVISOR"

#: Overrides the stall window, in seconds.
SUPERVISOR_STALL_ENV = "HYPERLOOM_SUPERVISOR_TICK_STALL_SEC"

#: How long the supervisor is given to exit after being asked to.
_STOP_GRACE_SEC: float = 5.0

#: The shortest stall window worth arming. One tick can legitimately spend two
#: role turns at their five-minute cap, plus a retry each.
_TICK_STALL_FLOOR_SEC: float = 1800.0

__all__ = [
    "SUPERVISOR_ENABLE_ENV",
    "SUPERVISOR_STALL_ENV",
    "spawn_supervisor",
    "stop_supervisor",
    "tick_stall_sec",
]


def _truthy(value: str) -> bool:
    """Return whether an environment value reads as on."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def tick_stall_sec(session_sec: float) -> float:
    """Return the stall window a session of ``session_sec`` should be watched with.

    Never more than half the budget, so a wedged coordinator is caught while
    the session it is wedging still has time to run; never below the floor, so
    a slow tick is not mistaken for a stopped one.

    Args:
        session_sec: The session's wall-clock budget; ``0`` when unknown.

    Returns:
        float: Seconds a tick may go without advancing.
    """
    if session_sec <= 0.0:
        return DEFAULT_TICK_STALL_SEC
    return max(_TICK_STALL_FLOOR_SEC, min(DEFAULT_TICK_STALL_SEC, session_sec / 2.0))


def spawn_supervisor(
    session_dir: Path | str,
    *,
    session_sec: float = 0.0,
    env: dict[str, str] | None = None,
) -> subprocess.Popen | None:
    """Start the supervisor for a session.

    Args:
        session_dir: The session root directory.
        session_sec: The session's wall-clock budget, which bounds the stall
            window; ``0`` uses :data:`~.watch.DEFAULT_TICK_STALL_SEC`.
        env: Environment to read the switches from; defaults to ``os.environ``.

    Returns:
        subprocess.Popen | None: The supervisor process, or ``None`` when
        :data:`SUPERVISOR_ENABLE_ENV` switched it off.

    Raises:
        OSError: If the supervisor could not be started.
    """
    # The supervisor reads a stamp file and signals one pid; it calls no model.
    # Its environment is scrubbed of control-plane credentials and start-up
    # hooks, which would otherwise be readable from ``/proc/<pid>/environ``.
    environ = scrub_benchmark_process_env(dict(os.environ if env is None else env))
    if not _truthy(environ.get(SUPERVISOR_ENABLE_ENV, "1")):
        log.info("supervisor: disabled by %s", SUPERVISOR_ENABLE_ENV)
        return None
    argv = [
        sys.executable,
        "-m",
        "hyperloom.orchestrator.supervisor",
        "--session-dir",
        str(session_dir),
    ]
    override = environ.get(SUPERVISOR_STALL_ENV, "").strip()
    argv += ["--tick-stall-sec", override or f"{tick_stall_sec(session_sec):.0f}"]
    log_path = supervisor_log_path(Path(session_dir))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The child dups this descriptor at spawn, so the parent's copy is done the
    # moment Popen returns.
    with open(log_path, "a", encoding="utf-8") as handle:
        proc = subprocess.Popen(  # noqa: S603 — argv is built here, never from input
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environ,
            start_new_session=True,
        )
    log.info("supervisor: watching %s as pid %d (log %s)", session_dir, proc.pid, log_path)
    return proc


def stop_supervisor(proc: subprocess.Popen | None) -> None:
    """Stop a supervisor started by :func:`spawn_supervisor`.

    Terminates it, then kills its group after :data:`_STOP_GRACE_SEC`.

    Args:
        proc: The supervisor process; ``None`` is a no-op.
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_STOP_GRACE_SEC)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        # Its own session, so the group is the supervisor and nothing else.
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        # It exited between the grace period expiring and this signal.
        return
    if running(proc.pid):
        log.warning("supervisor: pid %d did not exit", proc.pid)
