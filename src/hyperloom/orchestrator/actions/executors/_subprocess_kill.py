# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Reliable subprocess-tree teardown for Magpie-launched servers.

Magpie's shell wrappers ``trap``/``setsid``/``nohup`` the vLLM/SGLang server,
so on a Hyperloom-side timeout or error the server child can survive, holding
GPU memory and keeping a ``bash`` alive that may later re-source a script
mid-copy. To prevent that, Magpie is launched in its own POSIX session
(``start_new_session=True``) so ``os.killpg`` reaps the whole tree, and
:func:`kill_my_spawned_server` is called from every exit path (idempotent,
never raises). Scoped strictly to the tree we created, so it's safe on every
benchmark call.
"""

from __future__ import annotations

import glob
import logging
import os
import signal
import subprocess
import sys
import threading
import time

log = logging.getLogger(__name__)


# Grace window between SIGTERM and SIGKILL.
_TERM_GRACE_SECONDS = 5.0


def new_session_kwargs() -> dict:
    """``Popen`` kwargs so the child gets its own POSIX session (killable via
    ``os.killpg``). Returns ``{}`` on non-POSIX.

    Returns:
        A kwargs dict to splat into ``Popen``; ``{"start_new_session": True}``
        on POSIX, otherwise an empty dict.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _process_group_alive(pgid: int) -> bool:
    """Return True iff at least one process is still in ``pgid``.

    ``killpg(pgid, 0)`` raises ``ProcessLookupError`` once the group is empty;
    any other ``OSError`` (e.g. sandbox ``EPERM``) is treated as "still alive".

    Args:
        pgid: The POSIX process-group id to probe.

    Returns:
        True if the group still has at least one member (or liveness is
        indeterminate), False once the group is empty or on non-POSIX.
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
    """Send ``sig`` to every member of ``pgid``; swallow ``ESRCH``.

    Args:
        pgid (int): The POSIX process-group id to signal.
        sig (int): The signal number to send.
    """
    if os.name != "posix":
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except OSError as exc:
        log.warning(
            "_subprocess_kill: killpg(%d, %d) failed: %s",
            pgid,
            sig,
            exc,
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

    Args:
        proc: The spawned process whose tree should be reaped; ``None`` is a
            no-op.
        grace_seconds: Seconds to wait after SIGTERM before SIGKILLing
            survivors.
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
            "_subprocess_kill: getpgid(%d) failed: %s; falling back to single-pid terminate",
            proc.pid,
            exc,
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
            pgid,
            proc.pid,
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
            pgid,
            proc.pid,
        )
    except OSError:
        pass


# Sentinel ``returncode`` when ``run_with_session_kill`` reaps a child for an
# elapsed ``soft_deadline_sec`` (vs the ``timeout=`` hard cap, which raises
# ``TimeoutExpired``). Chosen not to collide with a real signal-based returncode.
OVERTIME_KILL_RETURNCODE: int = -909

# Sentinel ``returncode`` when the server-liveness watchdog reaps a child whose
# engine/worker bootstrap died but whose parent ``vllm serve`` /
# ``sglang.launch_server`` process hung instead of exiting.
SERVER_DEAD_RETURNCODE: int = -910

# Fatal server-init markers: once any appears in ``server.log`` the engine is
# unrecoverable within the same Magpie subprocess. Kept specific (terminal
# bootstrap failures, never transient per-shape warnings) so the watchdog cannot
# false-positive on a server that is merely slow to load.
_SERVER_DEAD_MARKERS: tuple[str, ...] = (
    "WorkerProc initialization failed",
    "EngineCore failed to start",
    "Engine core initialization failed",
    "Engine process failed to start",
    "AsyncEngineDeadError",
    "raise EngineDeadError",
    "Failed core proc(s)",
)

# Default grace after the first fatal marker before forcing a reap. Overridable
# via ``INFERENCE_OPTIMIZER_SERVER_DEAD_GRACE_SEC``.
_SERVER_DEAD_GRACE_SEC_DEFAULT: float = 120.0


# Sentinel ``returncode`` when the detokenizer-stall watchdog reaps a child that
# came up healthy but then produced no generation progress (hung engine /
# detokenizer wedge). Distinct so callers can label it
# ``error_class="detokenizer_stall"``.
DETOKENIZER_STALL_RETURNCODE: int = -911

# Server-ready markers: their appearance in ``server.log`` means the server has
# finished startup and is accepting traffic. Only after one is observed does the
# detokenizer-stall clock start. Covers the uvicorn frontend (vLLM + sglang) and
# sglang's own ready banner.
_SERVER_READY_MARKERS: tuple[str, ...] = (
    "Application startup complete",
    "Uvicorn running on",
    "The server is fired up and ready to roll",
)

# Generation-progress markers: the periodic decode-throughput lines. Reported by
# the scanner for diagnostics; the stall gate itself keys on raw log activity
# (any new bytes) — see :func:`_communicate_with_soft_deadline`.
_SERVER_PROGRESS_MARKERS: tuple[str, ...] = (
    "gen throughput (token/s):",   # sglang
    "Avg generation throughput:",  # vLLM
)

# Default grace: how long after the server reports ready it may emit no log
# output before the watchdog declares a hang / detokenizer stall. Measured from
# the ready marker so the cold start is never counted. ``<= 0`` disables the
# gate. Overridable via ``INFERENCE_OPTIMIZER_DETOK_STALL_GRACE_SEC``.
_DETOK_STALL_GRACE_SEC_DEFAULT: float = 1800.0


class _StreamCapture:
    """Capture child output while mirroring each line to the parent stream."""

    def __init__(self, proc: subprocess.Popen, *, text: bool) -> None:
        """Set up capture/mirror threads for a child's stdout and stderr.

        Args:
            proc: The child process whose pipes should be captured.
            text: Whether the pipes are in text (``str``) or bytes mode.
        """
        self._text = text
        self._stdout_chunks: list[str | bytes] = []
        self._stderr_chunks: list[str | bytes] = []
        self._threads: list[threading.Thread] = []
        if proc.stdout is not None:
            self._threads.append(
                threading.Thread(
                    target=self._pump,
                    args=(proc.stdout, self._stdout_chunks, sys.stdout),
                    daemon=True,
                )
            )
        if proc.stderr is not None:
            self._threads.append(
                threading.Thread(
                    target=self._pump,
                    args=(proc.stderr, self._stderr_chunks, sys.stderr),
                    daemon=True,
                )
            )

    def start(self) -> None:
        """Start the capture threads."""
        for thread in self._threads:
            thread.start()

    def finish(self, timeout: float = 2.0) -> tuple[str | bytes, str | bytes]:
        """Join the capture threads and return the captured output.

        Args:
            timeout: Per-thread join timeout in seconds.

        Returns:
            A ``(stdout, stderr)`` tuple of the captured streams (``str`` or
            ``bytes`` depending on the capture mode).
        """
        for thread in self._threads:
            thread.join(timeout=timeout)
        empty: str | bytes = "" if self._text else b""
        return (
            self._join(self._stdout_chunks) if self._stdout_chunks else empty,
            self._join(self._stderr_chunks) if self._stderr_chunks else empty,
        )

    def _join(self, chunks: list[str | bytes]) -> str | bytes:
        """Concatenate captured chunks using the appropriate empty separator.

        Args:
            chunks: Captured stdout or stderr chunks.

        Returns:
            The joined ``str`` or ``bytes`` output.
        """
        return "".join(chunks) if self._text else b"".join(chunks)  # type: ignore[arg-type,return-value]

    def _pump(self, pipe, chunks: list[str | bytes], mirror) -> None:
        """Read a pipe line-by-line, capturing and mirroring each line.

        Args:
            pipe: The child pipe to read from.
            chunks: List that read chunks are appended to.
            mirror: Parent stream the chunks are echoed to.
        """
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    break
                chunks.append(chunk)
                self._mirror(chunk, mirror)
        finally:
            try:
                pipe.close()
            except Exception:  # noqa: BLE001 - best-effort close
                pass

    def _mirror(self, chunk: str | bytes, mirror) -> None:
        """Echo a captured chunk to the parent stream, ignoring errors.

        Args:
            chunk: The captured ``str``/``bytes`` chunk.
            mirror: Parent stream to write to.
        """
        try:
            if isinstance(chunk, bytes):
                stream = getattr(mirror, "buffer", mirror)
                stream.write(chunk)
            else:
                mirror.write(chunk)
            mirror.flush()
        except Exception:  # noqa: BLE001 - logging must not break subprocess
            pass


# Bytes read from the tail of ``server.log`` per scan.
_SERVER_LOG_TAIL_BYTES: int = 65536

# Glob (relative to the watched path's directory) for nested per-run server logs
# Magpie writes when its wrapper ignores ``$SERVER_LOG`` and emits to a
# ``benchmark_<framework>_<timestamp>/server.log`` subdir instead.
_NESTED_SERVER_LOG_GLOB: str = "benchmark_*/server.log"

def _server_log_tail_has_marker(path: str) -> str | None:
    """Return the death marker present in the tail of the single file ``path``,
    else None.

    Best-effort and never raises: a missing / unreadable log reads as "not
    dead" (returns None).

    Args:
        path: Filesystem path to the server's ``server.log``.

    Returns:
        The first matched terminal engine/worker-init marker string if the log
        tail contains one, otherwise None (including when the log is missing or
        unreadable). Returning the marker (rather than a bare bool) lets the
        watchdog report which fatal line tripped it instead of a placeholder.
    """
    try:
        with open(path, "rb") as fh:
            try:
                fh.seek(-_SERVER_LOG_TAIL_BYTES, os.SEEK_END)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", "ignore")
    except (OSError, ValueError):
        return None
    for marker in _SERVER_DEAD_MARKERS:
        if marker in tail:
            return marker
    return None


def _server_log_shows_death(path: str) -> str | None:
    """Return the terminal engine/worker-init marker present in a server log,
    else None.

    Scans the watched ``path`` AND any nested ``benchmark_*/server.log`` files in
    the same directory. The nested fallback covers Magpie wrappers that ignore
    ``$SERVER_LOG`` and write the real server log to a per-run subdir, which
    otherwise leaves the watchdog reading an empty/absent file forever.

    Best-effort and never raises: a missing / unreadable log (server hasn't
    written yet) reads as "not dead" so a slow cold start is never misjudged.
    Returning the marker (rather than a bare bool) lets the watchdog report
    which fatal line tripped it instead of a placeholder.

    Args:
        path: Filesystem path to the server's ``server.log``.

    Returns:
        The first matched terminal engine/worker-init marker string from any
        candidate log tail, or None when none is present (including when no log
        is present or readable).
    """
    marker = _server_log_tail_has_marker(path)
    if marker is not None:
        return marker
    try:
        base_dir = os.path.dirname(path) or "."
        for nested in glob.glob(os.path.join(base_dir, _NESTED_SERVER_LOG_GLOB)):
            if nested != path:
                nested_marker = _server_log_tail_has_marker(nested)
                if nested_marker is not None:
                    return nested_marker
    except OSError:
        return None
    return None


def server_log_death_excerpt(path: str, *, max_chars: int = 1200) -> str | None:
    """Return a short ``server.log`` excerpt around the first terminal
    engine/worker-init marker, or ``None`` when no fatal marker is present.

    Baseline / profile failure classification calls this to surface the real
    server-side root cause — e.g. vLLM's ``RuntimeError: Engine core
    initialization failed`` — instead of the Magpie wrapper's generic
    stdout/stderr tail. The excerpt keeps a couple of lines of context around
    the marker. Best-effort: a missing / unreadable log returns ``None``.

    Args:
        path: Filesystem path to the server's ``server.log``.
        max_chars: Maximum length of the returned excerpt; the tail is kept
            when the excerpt is longer.

    Returns:
        A short multi-line excerpt around the first terminal init marker, or
        ``None`` when no fatal marker is present.
    """
    candidates = [path]
    try:
        base_dir = os.path.dirname(path) or "."
        candidates.extend(
            p
            for p in glob.glob(os.path.join(base_dir, _NESTED_SERVER_LOG_GLOB))
            if p != path
        )
    except OSError:
        pass
    for candidate in candidates:
        try:
            with open(candidate, "rb") as fh:
                try:
                    fh.seek(-_SERVER_LOG_TAIL_BYTES, os.SEEK_END)
                except OSError:
                    fh.seek(0)
                tail = fh.read().decode("utf-8", "ignore")
        except (OSError, ValueError):
            continue
        lines = tail.splitlines()
        for idx, line in enumerate(lines):
            if any(marker in line for marker in _SERVER_DEAD_MARKERS):
                start = max(0, idx - 2)
                excerpt = "\n".join(lines[start : idx + 3]).strip()
                if not excerpt:
                    continue
                return excerpt[-max_chars:]
    return None


def _scan_server_log_increment(
    path: str, from_offset: int
) -> tuple[int, bool, bool]:
    """Incrementally scan the bytes appended to ``server.log`` since
    ``from_offset`` for ready / generation-progress markers.

    Reading only the NEW tail (not the whole file) keeps the per-poll cost flat
    and — crucially — makes "progress" mean *fresh* progress: a throughput line
    written ten minutes ago that still sits in the file is read exactly once,
    so a wedged server that stops appending lines correctly reads as "no new
    progress" on subsequent polls. Handles truncation/rotation (size shrank
    below the offset) by rescanning from the start. Best-effort: a missing /
    unreadable log reads as "no new markers" and leaves the offset unchanged so
    a slow cold start is never misjudged.

    Args:
        path: Filesystem path to the server's ``server.log``.
        from_offset: Byte offset already consumed by a prior scan.

    Returns:
        ``(new_offset, saw_ready, saw_progress)`` — the advanced offset plus
        whether a ready / progress marker appeared in the newly read bytes.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return from_offset, False, False
    start = from_offset
    if size < start:  # truncated / rotated — rescan from the top.
        start = 0
    if size <= start:  # nothing new appended
        return start, False, False
    try:
        with open(path, "rb") as fh:
            fh.seek(start)
            chunk = fh.read().decode("utf-8", "ignore")
    except (OSError, ValueError):
        return from_offset, False, False
    saw_ready = any(marker in chunk for marker in _SERVER_READY_MARKERS)
    saw_progress = any(marker in chunk for marker in _SERVER_PROGRESS_MARKERS)
    return size, saw_ready, saw_progress


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
    detok_stall_grace_sec: float | None = None,
    server_already_ready: bool = False,
) -> subprocess.CompletedProcess:
    """Run a subprocess in its own session and reap descendants on every exit path."""
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
    if detok_stall_grace_sec is None:
        try:
            detok_stall_grace_sec = float(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_DETOK_STALL_GRACE_SEC",
                    _DETOK_STALL_GRACE_SEC_DEFAULT,
                )
            )
        except (TypeError, ValueError):
            detok_stall_grace_sec = _DETOK_STALL_GRACE_SEC_DEFAULT
    proc: subprocess.Popen | None = None
    capture: _StreamCapture | None = None
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
        capture = _StreamCapture(proc, text=text)
        capture.start()
        try:
            stdout, stderr = _communicate_with_soft_deadline(
                proc,
                hard_timeout=timeout,
                soft_deadline_sec=soft_deadline_sec,
                server_log_path=server_log_path,
                server_dead_grace_sec=server_dead_grace_sec,
                detok_stall_grace_sec=detok_stall_grace_sec,
                capture=capture,
                server_already_ready=server_already_ready,
            )
        except subprocess.TimeoutExpired:
            kill_my_spawned_server(proc)
            if capture is not None:
                capture.finish(timeout=2.0)
            raise
        except _ServerDeadDetected as exc:
            kill_my_spawned_server(proc)
            stdout, stderr = (
                capture.finish(timeout=2.0) if capture is not None else ("" if text else b"", "" if text else b"")
            )
            log.warning(
                "_subprocess_kill: server-liveness watchdog reaped tree — "
                "engine/worker init died but parent hung (marker=%r, "
                "grace=%.1fs, elapsed=%.1fs); returncode=%d.",
                exc.marker,
                exc.grace_sec,
                exc.elapsed_sec,
                SERVER_DEAD_RETURNCODE,
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=SERVER_DEAD_RETURNCODE,
                stdout=stdout if stdout is not None else ("" if text else b""),
                stderr=stderr if stderr is not None else ("" if text else b""),
            )
        except _ServerStalledDetected as exc:
            kill_my_spawned_server(proc)
            stdout, stderr = (
                capture.finish(timeout=2.0) if capture is not None else ("" if text else b"", "" if text else b"")
            )
            log.warning(
                "_subprocess_kill: detokenizer-stall watchdog reaped tree — "
                "server reported ready but emitted no log output (grace=%.1fs, "
                "elapsed=%.1fs); returncode=%d.",
                exc.grace_sec,
                exc.elapsed_sec,
                DETOKENIZER_STALL_RETURNCODE,
            )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=DETOKENIZER_STALL_RETURNCODE,
                stdout=stdout if stdout is not None else ("" if text else b""),
                stderr=stderr if stderr is not None else ("" if text else b""),
            )
        except _SoftDeadlineExceeded as exc:
            kill_my_spawned_server(proc)
            stdout, stderr = (
                capture.finish(timeout=2.0) if capture is not None else ("" if text else b"", "" if text else b"")
            )
            log.info(
                "_subprocess_kill: soft_deadline_sec=%.1fs exceeded "
                "(elapsed=%.1fs); reaped tree with sentinel returncode=%d.",
                exc.deadline_sec,
                exc.elapsed_sec,
                OVERTIME_KILL_RETURNCODE,
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
    """Internal sentinel for an elapsed soft deadline. Never bubbles
    past :func:`run_with_session_kill` (converted to a ``CompletedProcess``).
    """

    def __init__(self, *, deadline_sec: float, elapsed_sec: float) -> None:
        """Record the deadline and actual elapsed time on the sentinel.

        Args:
            deadline_sec (float): The soft deadline that was exceeded.
            elapsed_sec (float): The actual wall-clock elapsed at trip time.
        """
        super().__init__(f"soft deadline {deadline_sec:.1f}s elapsed (actual={elapsed_sec:.1f}s)")
        self.deadline_sec = float(deadline_sec)
        self.elapsed_sec = float(elapsed_sec)


class _ServerDeadDetected(Exception):
    """Internal sentinel: the server-liveness watchdog saw a terminal engine /
    worker init marker that persisted past the grace window. Never bubbles past
    :func:`run_with_session_kill` (converted to a ``CompletedProcess`` carrying
    ``SERVER_DEAD_RETURNCODE``).
    """

    def __init__(
        self,
        *,
        marker: str,
        grace_sec: float,
        elapsed_sec: float,
    ) -> None:
        """Build the error message describing the hung-after-death condition.

        Args:
            marker: Log marker that indicated the server init died.
            grace_sec: Grace period the parent was allowed after the marker.
            elapsed_sec: Actual wall-clock elapsed at trip time.
        """
        super().__init__(
            f"server init died (marker={marker!r}) and parent hung past "
            f"grace {grace_sec:.1f}s (elapsed={elapsed_sec:.1f}s)"
        )
        self.marker = marker
        self.grace_sec = float(grace_sec)
        self.elapsed_sec = float(elapsed_sec)


class _ServerStalledDetected(Exception):
    """Internal sentinel: the detokenizer-stall watchdog saw the server report
    ready and then produce no generation progress for the grace window. Never
    bubbles past :func:`run_with_session_kill` (converted to a
    ``CompletedProcess`` carrying ``DETOKENIZER_STALL_RETURNCODE``).
    """

    def __init__(
        self,
        *,
        grace_sec: float,
        elapsed_sec: float,
    ) -> None:
        """Build the error message describing the ready-but-no-progress stall.

        Args:
            grace_sec: How long the ready server was allowed to produce no
                generation progress before the watchdog tripped.
            elapsed_sec: Actual wall-clock elapsed at trip time.
        """
        super().__init__(
            f"server reported ready but emitted no log output for "
            f"{grace_sec:.1f}s (hung engine / detokenizer stall; "
            f"elapsed={elapsed_sec:.1f}s)"
        )
        self.grace_sec = float(grace_sec)
        self.elapsed_sec = float(elapsed_sec)


def _communicate_with_soft_deadline(
    proc: subprocess.Popen,
    *,
    hard_timeout: int | float | None,
    soft_deadline_sec: float | None,
    server_log_path: str | None = None,
    server_dead_grace_sec: float | None = None,
    detok_stall_grace_sec: float | None = None,
    capture: _StreamCapture | None = None,
    server_already_ready: bool = False,
) -> tuple[str | bytes, str | bytes]:
    """Communicate with a child while enforcing soft and server-log watchdogs."""
    watchdog_active = bool(server_log_path) and (
        server_dead_grace_sec is not None and float(server_dead_grace_sec) > 0.0
    )
    stall_active = bool(server_log_path) and (
        detok_stall_grace_sec is not None and float(detok_stall_grace_sec) > 0.0
    )
    soft_active = soft_deadline_sec is not None and float(soft_deadline_sec) > 0.0
    if capture is None and not soft_active and not watchdog_active and not stall_active:
        return proc.communicate(timeout=hard_timeout)
    if capture is not None and not soft_active and not watchdog_active and not stall_active:
        proc.wait(timeout=hard_timeout)
        return capture.finish()

    deadline_sec = float(soft_deadline_sec) if soft_active else None
    grace_sec = float(server_dead_grace_sec) if watchdog_active else None
    stall_grace_sec = float(detok_stall_grace_sec) if stall_active else None
    # When a ``server.log`` is available the soft deadline measures only the
    # post-ready phase (clock starts at the server-ready marker, excluding boot /
    # weight load / first-request JIT). Without a ``server.log`` it falls back to
    # the from-spawn clock. Opt out with
    # ``INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY=0``.
    # ``server_already_ready`` (warm reuse round) forces the from-spawn path
    # since no ready marker is written, so the from-ready clock would never arm.
    soft_from_ready = (
        soft_active
        and bool(server_log_path)
        and not server_already_ready
        and os.environ.get(
            "INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY", "1"
        ).strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # The server.log increment scan feeds both the stall watchdog and the
    # from-ready soft-deadline anchor; run it once per slice when either needs it.
    scan_active = stall_active or soft_from_ready
    poll_interval = 0.5
    start = time.monotonic()
    dead_marker_since: float | None = None
    # Detokenizer-stall watchdog state: byte offset consumed from server.log,
    # whether a ready marker has been seen, and the last time the log showed any
    # new output (seeded to the ready time). The gate only arms once
    # ``server_ready_since`` is set.
    stall_log_offset = 0
    server_ready_since: float | None = None
    last_activity_at: float | None = None
    while True:
        now = time.monotonic()
        elapsed = now - start
        # Advance the server.log scan, latching the server-ready and
        # last-activity times.
        if scan_active:
            prev_offset = stall_log_offset
            stall_log_offset, saw_ready, _saw_progress = _scan_server_log_increment(
                server_log_path,  # type: ignore[arg-type]
                stall_log_offset,
            )
            if saw_ready and server_ready_since is None:
                server_ready_since = now
                last_activity_at = now  # start the silence clock at ready
            # Any new bytes in server.log count as liveness; only total silence
            # trips the stall gate.
            if stall_log_offset > prev_offset:
                last_activity_at = now
        # Soft deadline. With ``soft_from_ready`` the overtime clock is measured
        # from the server-ready marker and stays dormant until ready; otherwise
        # it is the from-spawn elapsed.
        if soft_active and deadline_sec is not None:
            if soft_from_ready:
                if server_ready_since is not None:
                    soft_elapsed = now - server_ready_since
                    if deadline_sec - soft_elapsed <= 0.0:
                        raise _SoftDeadlineExceeded(
                            deadline_sec=deadline_sec,
                            elapsed_sec=soft_elapsed,
                        )
            elif deadline_sec - elapsed <= 0.0:
                raise _SoftDeadlineExceeded(
                    deadline_sec=deadline_sec,
                    elapsed_sec=elapsed,
                )
        if watchdog_active and grace_sec is not None:
            death_marker = _server_log_shows_death(server_log_path)  # type: ignore[arg-type]
            if death_marker is not None:
                if dead_marker_since is None:
                    dead_marker_since = now
                elif now - dead_marker_since >= grace_sec:
                    raise _ServerDeadDetected(
                        marker=death_marker,
                        grace_sec=grace_sec,
                        elapsed_sec=elapsed,
                    )
            else:
                dead_marker_since = None
        # Detokenizer-stall watchdog — armed only once the server is ready.
        if stall_active and stall_grace_sec is not None:
            if server_ready_since is not None and last_activity_at is not None:
                if now - last_activity_at >= stall_grace_sec:
                    raise _ServerStalledDetected(
                        grace_sec=stall_grace_sec,
                        elapsed_sec=elapsed,
                    )
        # Slice bounded by every active remaining window so the right gate
        # fires first; the child can still finish inside any slice.
        slice_sec = poll_interval
        if soft_active and deadline_sec is not None:
            if soft_from_ready:
                if server_ready_since is not None:
                    slice_sec = min(
                        slice_sec, deadline_sec - (now - server_ready_since)
                    )
            else:
                slice_sec = min(slice_sec, deadline_sec - elapsed)
        if hard_timeout is not None:
            hard_remaining = float(hard_timeout) - elapsed
            if hard_remaining <= 0.0:
                raise subprocess.TimeoutExpired(proc.args, hard_timeout)
            slice_sec = min(slice_sec, hard_remaining)
        slice_sec = max(slice_sec, 0.0)
        try:
            if capture is None:
                return proc.communicate(timeout=slice_sec)
            proc.wait(timeout=slice_sec)
            return capture.finish()
        except subprocess.TimeoutExpired:
            continue


__all__ = [
    "DETOKENIZER_STALL_RETURNCODE",
    "OVERTIME_KILL_RETURNCODE",
    "SERVER_DEAD_RETURNCODE",
    "kill_my_spawned_server",
    "new_session_kwargs",
    "run_with_session_kill",
]
