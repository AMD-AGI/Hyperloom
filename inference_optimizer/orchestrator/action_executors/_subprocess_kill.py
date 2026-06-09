# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Reliable subprocess-tree teardown for Magpie-launched servers
(Hyperloom ``bugs.md`` §B).

Magpie's shell wrappers ``trap``/``setsid``/``nohup`` the vLLM/SGLang server,
so on a Hyperloom-side timeout or error the server child survives, holds GPU
memory, and keeps a ``bash`` alive that later re-sources a script mid-copy
(bugs.md §C #1). Fix: launch Magpie in its own POSIX session
(``start_new_session=True``) so ``os.killpg`` reaps the whole tree, and call
:func:`kill_my_spawned_server` from every exit path (idempotent, never raises).
Scoped strictly to the tree we created, so it's safe on every benchmark call.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

log = logging.getLogger(__name__)


# Grace window between SIGTERM and SIGKILL; 5s leaves headroom for the slow
# graphs-capture teardown.
_TERM_GRACE_SECONDS = 5.0


def new_session_kwargs() -> dict:
    """``Popen`` kwargs so the child gets its own POSIX session (killable via
    ``os.killpg``). Returns ``{}`` on non-POSIX.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _process_group_alive(pgid: int) -> bool:
    """Return True iff at least one process is still in ``pgid``.

    ``killpg(pgid, 0)`` raises ``ProcessLookupError`` once the group is empty;
    any other ``OSError`` (e.g. sandbox ``EPERM``) is treated as "still alive".
    """
    if os.name != "posix":
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _signal_group(pgid: int, sig: int) -> None:
    """Send ``sig`` to every member of ``pgid``; swallow ``ESRCH``."""
    if os.name != "posix":
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except OSError as exc:
        log.warning(
            "_subprocess_kill: killpg(%d, %d) failed: %s",
            pgid, sig, exc,
        )


def kill_my_spawned_server(
    proc: subprocess.Popen | None,
    *,
    grace_seconds: float = _TERM_GRACE_SECONDS,
) -> None:
    """Tear down the entire process tree rooted at ``proc``.

    No-op when ``proc`` is ``None`` or already exited. SIGTERM the group, wait
    up to ``grace_seconds``, then SIGKILL survivors. Never raises (logs a
    warning on unexpected signalling failure). Requires the child to have been
    launched with :func:`new_session_kwargs` so its pgid is distinct from
    Hyperloom's own; asserted defensively below.
    """
    if proc is None:
        return
    if proc.poll() is not None:
        return
    if os.name != "posix":
        try:
            proc.terminate()
            proc.wait(timeout=grace_seconds)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    except OSError as exc:
        log.warning(
            "_subprocess_kill: getpgid(%d) failed: %s; falling back to "
            "single-pid terminate", proc.pid, exc,
        )
        try:
            proc.terminate()
        except OSError:
            pass
        return

    own_pgid = os.getpgid(0)
    if pgid == own_pgid:
        log.error(
            "_subprocess_kill: refusing to killpg own session (pgid=%d, "
            "child pid=%d). The child was almost certainly launched "
            "without start_new_session=True — that is a Hyperloom bug, "
            "fix the launch site instead of widening this helper's "
            "scope.",
            pgid, proc.pid,
        )
        return

    _signal_group(pgid, signal.SIGTERM)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_group_alive(pgid):
            break
        time.sleep(0.05)

    if _process_group_alive(pgid):
        _signal_group(pgid, signal.SIGKILL)

    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        log.warning(
            "_subprocess_kill: proc.wait() did not return within 1s "
            "after SIGKILL'ing pgid=%d (pid=%d). The reaper may be "
            "wedged; leaving the zombie for init to collect.",
            pgid, proc.pid,
        )
    except OSError:
        pass


# Sentinel ``returncode`` when ``run_with_session_kill`` reaps a child for an
# elapsed ``soft_deadline_sec`` (vs the ``timeout=`` hard cap, which still
# raises ``TimeoutExpired``). Chosen not to collide with a real signal-based
# ``-N`` returncode; the ExploreExecutor maps it to ``KILLED_OVERTIME``.
OVERTIME_KILL_RETURNCODE: int = -909

# Sentinel ``returncode`` when the server-liveness watchdog reaps a child
# because the spawned inference server's engine/worker bootstrap died but the
# parent ``vllm serve`` / ``sglang.launch_server`` process hung in
# multiprocessing cleanup instead of exiting (observed on 671B MoE restarts on
# ROCm). Without this watchdog the benchmark harness keeps polling a dead
# ``/health`` until the variant hard-``timeout`` (~7800s ≈ 2h), burning the run
# budget on a server that will never come up. Distinct from
# ``OVERTIME_KILL_RETURNCODE`` so callers can label it precisely.
SERVER_DEAD_RETURNCODE: int = -910

# Fatal server-init markers. Once any appears in ``server.log`` the engine is
# unrecoverable within the same Magpie subprocess. Kept deliberately specific
# (terminal bootstrap failures, never transient per-shape warnings) so the
# watchdog cannot false-positive on a server that is merely slow to load.
_SERVER_DEAD_MARKERS: tuple[str, ...] = (
    "WorkerProc initialization failed",
    "EngineCore failed to start",
    "Engine core initialization failed",
    "Engine process failed to start",
    "AsyncEngineDeadError",
    "raise EngineDeadError",
)

# Default grace after the first fatal marker before forcing a reap. Gives the
# harness a chance to exit on its own (the clean-exit case returns normally and
# never trips the watchdog); far below the ~2h hard timeout. Overridable via
# ``INFERENCE_OPTIMIZER_SERVER_DEAD_GRACE_SEC``.
_SERVER_DEAD_GRACE_SEC_DEFAULT: float = 120.0

# Bytes read from the tail of ``server.log`` per scan (markers always land near
# the end of the bootstrap traceback).
_SERVER_LOG_TAIL_BYTES: int = 65536


def _server_log_shows_death(path: str) -> bool:
    """Return True iff ``server.log`` tail contains a terminal-init marker.

    Best-effort and never raises: a missing / unreadable log (server hasn't
    written yet) reads as "not dead" so a slow cold start is never misjudged.
    """
    try:
        with open(path, "rb") as fh:
            try:
                fh.seek(-_SERVER_LOG_TAIL_BYTES, os.SEEK_END)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", "ignore")
    except (OSError, ValueError):
        return False
    return any(marker in tail for marker in _SERVER_DEAD_MARKERS)


def run_with_session_kill(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: int | float | None = None,
    text: bool = True,
    soft_deadline_sec: float | None = None,
    server_log_path: str | None = None,
    server_dead_grace_sec: float | None = None,
) -> subprocess.CompletedProcess:
    """``subprocess.run``-compatible call that ALSO tears down the entire
    descendant tree on every exit path.

    Unlike ``subprocess.run`` (which discards the Popen handle and can't reach
    leaked grandchildren — the bugs.md §B server leak), this launches via
    ``Popen(start_new_session=True)`` and reaps the tree in a ``finally:``.
    Returns a ``CompletedProcess`` and re-raises ``TimeoutExpired`` like
    ``subprocess.run``.

    ``soft_deadline_sec`` (Fix E): an optional deadline firing before the
    ``timeout=`` hard cap; on elapse the tree is reaped and a
    ``CompletedProcess`` with ``returncode = OVERTIME_KILL_RETURNCODE`` is
    returned (does NOT raise). ``None`` / ≤ 0 keeps legacy behaviour. Tests
    should patch this function instead of ``subprocess.run``.

    ``server_log_path`` (server-liveness watchdog): when set, the spawned
    server's ``server.log`` is scanned each poll slice for terminal engine /
    worker init failures; once a marker persists past ``server_dead_grace_sec``
    (default ``INFERENCE_OPTIMIZER_SERVER_DEAD_GRACE_SEC`` or 120s) the tree is
    reaped and a ``CompletedProcess`` with
    ``returncode = SERVER_DEAD_RETURNCODE`` is returned (does NOT raise). This
    turns a crashed-but-hung server (parent never exits, ``/health`` polled
    forever) into a fast fail instead of a ~2h hard-timeout stall.
    """
    if server_dead_grace_sec is None:
        try:
            server_dead_grace_sec = float(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_SERVER_DEAD_GRACE_SEC",
                    _SERVER_DEAD_GRACE_SEC_DEFAULT,
                )
            )
        except (TypeError, ValueError):
            server_dead_grace_sec = _SERVER_DEAD_GRACE_SEC_DEFAULT
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(  # noqa: S603 — cmd is caller's responsibility
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            env=env,
            cwd=cwd,
            **new_session_kwargs(),
        )
        try:
            stdout, stderr = _communicate_with_soft_deadline(
                proc,
                hard_timeout=timeout,
                soft_deadline_sec=soft_deadline_sec,
                server_log_path=server_log_path,
                server_dead_grace_sec=server_dead_grace_sec,
            )
        except subprocess.TimeoutExpired:
            # Reap before re-raising so the caller doesn't see a running tree.
            kill_my_spawned_server(proc)
            try:
                # Drain the pipes so the exception carries partial output.
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            raise
        except _ServerDeadDetected as exc:
            kill_my_spawned_server(proc)
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            log.warning(
                "_subprocess_kill: server-liveness watchdog reaped tree — "
                "engine/worker init died but parent hung (marker=%r, "
                "grace=%.1fs, elapsed=%.1fs); returncode=%d.",
                exc.marker, exc.grace_sec, exc.elapsed_sec,
                SERVER_DEAD_RETURNCODE,
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=SERVER_DEAD_RETURNCODE,
                stdout=stdout if stdout is not None else ("" if text else b""),
                stderr=stderr if stderr is not None else ("" if text else b""),
            )
        except _SoftDeadlineExceeded as exc:
            kill_my_spawned_server(proc)
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            log.info(
                "_subprocess_kill: soft_deadline_sec=%.1fs exceeded "
                "(elapsed=%.1fs); reaped tree with sentinel returncode=%d.",
                exc.deadline_sec, exc.elapsed_sec, OVERTIME_KILL_RETURNCODE,
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=OVERTIME_KILL_RETURNCODE,
                stdout=stdout if stdout is not None else ("" if text else b""),
                stderr=stderr if stderr is not None else ("" if text else b""),
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout if stdout is not None else ("" if text else b""),
            stderr=stderr if stderr is not None else ("" if text else b""),
        )
    finally:
        kill_my_spawned_server(proc)


class _SoftDeadlineExceeded(Exception):
    """Internal sentinel for an elapsed soft (Fix-E) deadline. Never bubbles
    past :func:`run_with_session_kill` (converted to a ``CompletedProcess``).
    """

    def __init__(self, *, deadline_sec: float, elapsed_sec: float) -> None:
        super().__init__(
            f"soft deadline {deadline_sec:.1f}s elapsed "
            f"(actual={elapsed_sec:.1f}s)"
        )
        self.deadline_sec = float(deadline_sec)
        self.elapsed_sec = float(elapsed_sec)


class _ServerDeadDetected(Exception):
    """Internal sentinel: the server-liveness watchdog saw a terminal engine /
    worker init marker that persisted past the grace window. Never bubbles past
    :func:`run_with_session_kill` (converted to a ``CompletedProcess`` carrying
    ``SERVER_DEAD_RETURNCODE``).
    """

    def __init__(
        self, *, marker: str, grace_sec: float, elapsed_sec: float,
    ) -> None:
        super().__init__(
            f"server init died (marker={marker!r}) and parent hung past "
            f"grace {grace_sec:.1f}s (elapsed={elapsed_sec:.1f}s)"
        )
        self.marker = marker
        self.grace_sec = float(grace_sec)
        self.elapsed_sec = float(elapsed_sec)


def _communicate_with_soft_deadline(
    proc: subprocess.Popen,
    *,
    hard_timeout: int | float | None,
    soft_deadline_sec: float | None,
    server_log_path: str | None = None,
    server_dead_grace_sec: float | None = None,
) -> tuple[str | bytes, str | bytes]:
    """``proc.communicate`` shim enforcing the soft deadline + server watchdog.

    With no soft deadline and no server watchdog this delegates straight to
    ``proc.communicate(timeout=hard_timeout)`` (legacy fast path). Otherwise it
    polls in 0.5s slices, enforcing:

    * ``soft_deadline_sec`` — raise :class:`_SoftDeadlineExceeded` once passed;
    * ``server_log_path`` watchdog — once a terminal init marker
      (:data:`_SERVER_DEAD_MARKERS`) is observed AND it persists for
      ``server_dead_grace_sec`` without the child exiting on its own, raise
      :class:`_ServerDeadDetected`.

    The ``hard_timeout`` is always honoured so a stuck child can't dodge the
    gates.
    """
    watchdog_active = bool(server_log_path) and (
        server_dead_grace_sec is not None and float(server_dead_grace_sec) > 0.0
    )
    soft_active = (
        soft_deadline_sec is not None and float(soft_deadline_sec) > 0.0
    )
    if not soft_active and not watchdog_active:
        return proc.communicate(timeout=hard_timeout)

    deadline_sec = float(soft_deadline_sec) if soft_active else None
    grace_sec = float(server_dead_grace_sec) if watchdog_active else None
    poll_interval = 0.5
    start = time.monotonic()
    dead_marker_since: float | None = None
    while True:
        elapsed = time.monotonic() - start
        if soft_active and deadline_sec is not None:
            if deadline_sec - elapsed <= 0.0:
                raise _SoftDeadlineExceeded(
                    deadline_sec=deadline_sec, elapsed_sec=elapsed,
                )
        if watchdog_active and grace_sec is not None:
            if _server_log_shows_death(server_log_path):  # type: ignore[arg-type]
                if dead_marker_since is None:
                    dead_marker_since = time.monotonic()
                elif time.monotonic() - dead_marker_since >= grace_sec:
                    raise _ServerDeadDetected(
                        marker="server_init_failed",
                        grace_sec=grace_sec,
                        elapsed_sec=elapsed,
                    )
            else:
                dead_marker_since = None
        # Slice bounded by every active remaining window so the right gate
        # fires first; the child can still finish inside any slice.
        slice_sec = poll_interval
        if soft_active and deadline_sec is not None:
            slice_sec = min(slice_sec, deadline_sec - elapsed)
        if hard_timeout is not None:
            hard_remaining = float(hard_timeout) - elapsed
            if hard_remaining <= 0.0:
                # Let proc.communicate's own TimeoutExpired path fire.
                return proc.communicate(timeout=0.0)
            slice_sec = min(slice_sec, hard_remaining)
        slice_sec = max(slice_sec, 0.0)
        try:
            return proc.communicate(timeout=slice_sec)
        except subprocess.TimeoutExpired:
            # Not yet done; loop to re-evaluate all gates.
            continue


__all__ = [
    "OVERTIME_KILL_RETURNCODE",
    "SERVER_DEAD_RETURNCODE",
    "kill_my_spawned_server",
    "new_session_kwargs",
    "run_with_session_kill",
]
