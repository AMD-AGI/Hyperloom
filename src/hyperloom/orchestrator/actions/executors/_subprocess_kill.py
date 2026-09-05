# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reliable subprocess-tree teardown for Magpie-launched servers."""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, NamedTuple

# ``TERM_GRACE_SECONDS`` is this module's name for the shared SIGTERM-to-SIGKILL
# grace: a driver-side teardown of a server a round left behind waits for the
# same thing on the same signal, so it waits exactly as long.
from hyperloom.common.proctree import TERM_GRACE_SEC as TERM_GRACE_SECONDS
from hyperloom.common.proctree import collect_tree, group_alive, kill_tree, signal_group

from .bypass_analysis import parse_server_log_throughput

from ..cancel_channel import CancelScope, cancel_scope_listener

log = logging.getLogger(__name__)


# How long the reaper waits to collect the SIGKILL'd child before giving up on it.
_REAP_COLLECT_SECONDS: float = 1.0

# How long draining a reaped child's capture threads is given.
_CAPTURE_DRAIN_SECONDS: float = 2.0

# How often the blocking side looks up from the child to check its stop gates -- the session deadline and the cancel
# scope among them.
STOP_GATE_POLL_SECONDS: float = 0.5

# What stopping a running round costs, end to end, from the moment something asks it to: noticing at the poll,
# SIGTERM'ing the tree, waiting out the grace before SIGKILL, collecting the child, and draining its pipes.
COOPERATIVE_REAP_BUDGET_SEC: float = (
    STOP_GATE_POLL_SECONDS + TERM_GRACE_SECONDS + _REAP_COLLECT_SECONDS + _CAPTURE_DRAIN_SECONDS
)


def new_session_kwargs() -> dict:
    """``Popen`` kwargs so the child gets its own POSIX session (killable via ``os.killpg``)."""
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _process_group_alive(pgid: int) -> bool:
    """Return True iff at least one process is still in ``pgid``.

    Args:
        pgid: The POSIX process-group id to probe.

    Returns:
        True if the group still has at least one member (or liveness is
        indeterminate), False once the group is empty or on non-POSIX.
    """
    return group_alive(pgid)


def _signal_group(pgid: int, sig: int) -> None:
    """Send ``sig`` to every member of ``pgid``; swallow ``ESRCH``.

    Args:
        pgid (int): The POSIX process-group id to signal.
        sig (int): The signal number to send.
    """
    signal_group(pgid, sig, what="_subprocess_kill")


def kill_my_spawned_server(
    proc: subprocess.Popen | None,
    *,
    grace_seconds: float = TERM_GRACE_SECONDS,
) -> None:
    """Tear down the entire process tree rooted at ``proc``."""
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

    try:
        tree = collect_tree([proc.pid])
    except OSError as exc:
        # Every caller reaps from a ``finally:``, so an unreadable procfs has to
        # be reported rather than raised on top of whatever sent us here.
        log.error("_subprocess_kill: cannot enumerate the tree under pid=%d: %s", proc.pid, exc)
        return
    kill_tree(tree, grace_sec=grace_seconds, confirm_sec=grace_seconds)

    try:
        proc.wait(timeout=_REAP_COLLECT_SECONDS)
    except subprocess.TimeoutExpired:
        log.warning(
            "_subprocess_kill: proc.wait() did not return within %.0fs "
            "after SIGKILL'ing pgid=%d (pid=%d). The reaper may be "
            "wedged; leaving the zombie for init to collect.",
            _REAP_COLLECT_SECONDS,
            pgid,
            proc.pid,
        )
    except OSError:
        pass


# Sentinel ``returncode`` allocation.

# Sentinel ``returncode`` when ``run_with_session_kill`` reaps a child for an elapsed ``soft_deadline_sec`` (vs the
# ``timeout=`` hard cap, which raises ``TimeoutExpired``).
OVERTIME_KILL_RETURNCODE: int = -909

# Sentinel ``returncode`` when the server-liveness watchdog reaps a child whose engine/worker bootstrap died but whose
# parent ``vllm serve`` / ``sglang.launch_server`` process hung instead of exiting.
SERVER_DEAD_RETURNCODE: int = -910

# Fatal server-init markers: once any appears in ``server.log`` the engine is unrecoverable within the same Magpie
# subprocess.
_SERVER_DEAD_MARKERS: tuple[str, ...] = (
    # (1) runtime engine/worker bootstrap crashes
    "WorkerProc initialization failed",
    "EngineCore failed to start",
    "Engine core initialization failed",
    "Engine process failed to start",
    "AsyncEngineDeadError",
    "raise EngineDeadError",
    "Failed core proc(s)",
    # (2) config-validation-stage terminal failures (pre-engine).
    "does not recognize this architecture",
    "Transformers does not recognize",
    "ValidationError for ModelConfig",
    "are not supported for now",
)

# Default grace after the first fatal marker before forcing a reap.
_SERVER_DEAD_GRACE_SEC_DEFAULT: float = 120.0


# Sentinel ``returncode`` when the detokenizer-stall watchdog reaps a child that came up healthy but then produced no
# generation progress (hung engine / detokenizer wedge).
DETOKENIZER_STALL_RETURNCODE: int = -911

# Sentinel ``returncode`` returned by the _run_magpie AgentX hook when the execution-boundary preflight fails (aiperf
# missing or not weka-trace capable) before any benchmark launches.
AGENTX_PREFLIGHT_RETURNCODE: int = -912

#: The ``error_class`` that sentinel must carry, wherever it is classified. Named
#: here beside the return code because three call sites decide on it -- the grid
#: runner, the baseline executor and the writeback stop-reason gate -- and a
#: string literal repeated across them can drift into a class nobody handles.
#:
#: It marks an ENVIRONMENT failure, not a framework one: the AgentX client is
#: missing or is not the pinned build, and the runtime repair
#: (``agentx.repair``) could not supply it. Nothing downstream can author its way
#: out of that, so the writeback gate stops the run and names the fix for an
#: operator rather than opening an enablement round.
AGENTX_PREFLIGHT_ERROR_CLASS: str = "agentx_preflight"

# -913 is ``_ray_serving._RAY_ACTOR_DIED_RC``.

# Sentinel ``returncode`` returned by the _run_magpie eval hook when the generation bounds / pathology probe cannot be
# installed even though the target file is present and this variant runs eval.
EVAL_PROBE_UNPATCHABLE_RETURNCODE: int = -914

# Sentinel ``returncode`` when the session wall-clock budget ran out mid-round and the tree was reaped.
SESSION_TIME_EXHAUSTED_RETURNCODE: int = -915

# -916 is ``_ray_serving._ACTOR_TIMEOUT_RC``.

# Sentinel ``returncode`` when the orchestrator cancelled the action this child was launched for -- a shutdown, or a
# budget that is spent.
ORCHESTRATOR_CANCELLED_RETURNCODE: int = -917

# Server-ready markers: their appearance in ``server.log`` means the server has
# finished startup and is accepting traffic. Only after one is observed does the
# detokenizer-stall clock start. Covers the uvicorn frontend (vLLM + sglang) and
# sglang's own ready banner.
_SERVER_READY_MARKERS: tuple[str, ...] = (
    "Application startup complete",
    "Uvicorn running on",
    "The server is fired up and ready to roll",
)

# Accuracy-eval start markers: their appearance means the benchmark phase of the run is over and the accuracy eval has
# begun.
_EVAL_START_MARKERS: tuple[str, ...] = (
    "HYPERLOOM_EVAL_START",
    "[magpie_bench_remote_compat] lm_eval cmd:",
)

# The benchmark body's own stderr: Magpie redirects it here rather than into ``server.log`` or the parent's pipe, so
# it is both where the eval-start marker lands and the one resolved log whose growth is output of the very child this
# module is waiting on.
_EVAL_LOG_NAME: str = "benchmark_stderr.log"

# Default grace: how long after the server reports ready it may emit no log output before the watchdog declares a hang
# / detokenizer stall.
_DETOK_STALL_GRACE_SEC_DEFAULT: float = 1800.0


class _StreamCapture:
    """Capture child output while mirroring each line to the parent stream."""

    def __init__(
        self,
        proc: subprocess.Popen,
        *,
        text: bool,
        on_output: Callable[[], None] | None = None,
    ) -> None:
        """Set up capture/mirror threads for a child's stdout and stderr."""
        self._text = text
        self._on_output = on_output
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
        """Join the capture threads and return the captured output."""
        for thread in self._threads:
            thread.join(timeout=timeout)
        empty: str | bytes = "" if self._text else b""
        return (
            self._join(self._stdout_chunks) if self._stdout_chunks else empty,
            self._join(self._stderr_chunks) if self._stderr_chunks else empty,
        )

    def note_output(self) -> None:
        """Report one unit of child output to the caller's liveness callback."""
        if self._on_output is None:
            return
        try:
            self._on_output()
        except Exception:  # noqa: BLE001 - liveness reporting never breaks capture
            pass

    def _join(self, chunks: list[str | bytes]) -> str | bytes:
        """Concatenate captured chunks using the appropriate empty separator."""
        return "".join(chunks) if self._text else b"".join(chunks)  # type: ignore[arg-type,return-value]

    def _pump(self, pipe, chunks: list[str | bytes], mirror) -> None:
        """Read a pipe line-by-line, capturing and mirroring each line."""
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    break
                chunks.append(chunk)
                self._mirror(chunk, mirror)
                self.note_output()
        finally:
            try:
                pipe.close()
            except Exception:  # noqa: BLE001 - best-effort close
                pass

    def _mirror(self, chunk: str | bytes, mirror) -> None:
        """Echo a captured chunk to the parent stream, ignoring errors."""
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

# Glob (relative to the watched path's directory) for nested per-run server logs Magpie writes when its wrapper
# ignores ``$SERVER_LOG`` and emits to a ``benchmark_<framework>_<timestamp>/server.log`` subdir instead.
_NESTED_SERVER_LOG_GLOB: str = "benchmark_*/server.log"


def _server_log_tail_has_marker(path: str) -> str | None:
    """Return the death marker present in the tail of the single file ``path``, else None."""
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
    """Return the terminal engine/worker-init marker present in a server log, else None."""
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
    """Return a short ``server.log`` excerpt around the first terminal engine/worker-init marker, or ``None`` when no fatal marker is present."""
    candidates = [path]
    try:
        base_dir = os.path.dirname(path) or "."
        candidates.extend(p for p in glob.glob(os.path.join(base_dir, _NESTED_SERVER_LOG_GLOB)) if p != path)
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


# Name of the stamp written beside the caller's ``server.log`` the moment the server first reports ready.
_READY_STAMP_NAME = "server_ready_at"


def _ready_stamp_path(server_log_path: str) -> Path:
    """Return where a round's ready stamp lives, given its ``server.log`` path."""
    return Path(server_log_path).parent / _READY_STAMP_NAME


def stamp_server_ready(server_log_path: str, boot_sec: float) -> None:
    """Record, beside ``server_log_path``, that the server just reported ready.

    Two numbers, because they answer two questions and one clock cannot answer
    both. ``boot_sec`` is how long the round took to come up, measured from spawn
    to this moment on one ``time.monotonic()`` reading in the process that
    spawned the child. The wall-clock instant beside it only ever says *which
    round* the stamp belongs to.

    Keeping the boot a duration is what makes it safe to read across a process
    boundary. On the Ray path the round runs inside an actor, possibly on another
    host; subtracting the actor's wall-clock from the driver's would charge the
    boot for whatever the two clocks disagree by, and a positive disagreement
    inflates the boot and makes the budget gates refuse rounds that fit. A
    duration crosses the boundary meaning the same thing on both sides -- the
    same reason ``session_remaining_sec`` is passed to the actor as a duration
    rather than as a deadline.

    A file is used because it crosses that boundary without widening the round's
    return value, and the round's output directory is already how post-mortem
    evidence gets back (the caller reads the same directory's ``server.log`` to
    classify server deaths).

    Best effort: a round whose stamp cannot be written loses a measurement, which
    callers already have to handle, and must not lose the round.

    Args:
        server_log_path: The ``<output_dir>/server.log`` path from the caller.
        boot_sec: Seconds from spawn to this moment, on the spawning process's
            monotonic clock. Required rather than defaulted: a caller that
            omitted it would write a well-formed stamp claiming the round booted
            instantly, which reads as a whole round of benchmark and is the one
            wrong answer the two-field format exists to make impossible.
    """
    try:
        _ready_stamp_path(server_log_path).write_text(
            f"{time.time():.3f} {max(0.0, float(boot_sec)):.3f}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("_subprocess_kill: could not stamp server-ready time (%s)", exc)


def clear_server_ready_stamp(server_log_path: str) -> None:
    """Drop any ready stamp an earlier round left in this output directory."""
    try:
        _ready_stamp_path(server_log_path).unlink(missing_ok=True)
    except OSError as exc:
        log.warning("_subprocess_kill: could not clear stale server-ready stamp (%s)", exc)


def _read_ready_stamp(server_log_path: str) -> tuple[float, float] | None:
    """Return a round's ``(ready_unix, boot_sec)``, or ``None`` when unrecorded."""
    try:
        fields = _ready_stamp_path(server_log_path).read_text(encoding="utf-8").split()
        ready_unix = float(fields[0])
        boot_sec = float(fields[1])
    except (OSError, ValueError, IndexError):
        return None
    return (ready_unix, max(0.0, boot_sec)) if ready_unix > 0.0 else None


def server_ready_unix(server_log_path: str) -> float | None:
    """Return when the server reported ready, or ``None`` when nothing recorded it."""
    stamp = _read_ready_stamp(server_log_path)
    return None if stamp is None else stamp[0]


def post_ready_runtime_sec(
    server_log_path: str,
    *,
    started_unix: float,
    runtime_sec: float,
) -> float | None:
    """Return how long a round ran *after* its server was ready."""
    stamp = _read_ready_stamp(server_log_path)
    if stamp is None or stamp[0] < started_unix:
        return None
    return max(0.0, min(float(runtime_sec), float(runtime_sec) - stamp[1]))


def _resolve_scan_logs(server_log_path: str) -> list[str]:
    """Return the log files to scan for markers, newest-nesting first."""
    primary = Path(server_log_path)
    out: list[str] = []
    candidates = [primary]
    try:
        candidates.extend(sorted(primary.parent.glob("benchmark_*/server.log")))
    except OSError:
        pass
    for log in candidates:
        for name in (log, log.with_name(_EVAL_LOG_NAME)):
            text = str(name)
            if text not in out and name.exists():
                out.append(text)
    return out


class _LogScan(NamedTuple):
    """What one pass over the resolved logs found in the bytes appended since the last."""

    saw_ready: bool
    saw_progress: bool
    saw_eval_start: bool
    grew: bool
    child_spoke: bool


def _stale_scan_log_sizes(server_log_path: str) -> dict[str, int]:
    """Current byte length of each nested log that already exists at spawn."""
    owned_dir = Path(server_log_path).parent
    sizes: dict[str, int] = {}
    for path in _resolve_scan_logs(server_log_path):
        candidate = Path(path)
        if candidate.parent == owned_dir:
            continue
        try:
            sizes[path] = candidate.stat().st_size
        except OSError:
            continue
    return sizes


def _scan_logs_increment(server_log_path: str, offsets: dict[str, int]) -> _LogScan:
    """Scan every resolved log for markers, advancing ``offsets`` in place."""
    saw_ready = saw_progress = saw_eval_start = grew = child_spoke = False
    for path in _resolve_scan_logs(server_log_path):
        prev = offsets.get(path, 0)
        new_offset, ready, progress, eval_start = _scan_server_log_increment(path, prev)
        offsets[path] = new_offset
        saw_ready = saw_ready or ready
        saw_progress = saw_progress or progress
        saw_eval_start = saw_eval_start or eval_start
        if new_offset > prev:
            grew = True
            child_spoke = child_spoke or Path(path).name == _EVAL_LOG_NAME
    return _LogScan(saw_ready, saw_progress, saw_eval_start, grew, child_spoke)


def _scan_server_log_increment(path: str, from_offset: int) -> tuple[int, bool, bool, bool]:
    """Incrementally scan the bytes appended to ``server.log`` since ``from_offset`` for ready / generation-progress / eval-start markers."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return from_offset, False, False, False
    start = from_offset
    if size < start:  # truncated / rotated — rescan from the top.
        start = 0
    if size <= start:  # nothing new appended
        return start, False, False, False
    try:
        with open(path, "rb") as fh:
            fh.seek(start)
            chunk = fh.read().decode("utf-8", "ignore")
    except (OSError, ValueError):
        return from_offset, False, False, False
    saw_ready = any(marker in chunk for marker in _SERVER_READY_MARKERS)
    # Progress is the rate on the periodic decode-throughput line, not the line's presence: some vLLM builds log ``Avg
    # generation throughput: 0.0 tokens/s`` on an idle engine, and an engine goes idle precisely when the client
    # driving it wedges, so the marker alone lets the server vouch for the client that stopped asking it for tokens.
    saw_progress = bool(parse_server_log_throughput(chunk))
    saw_eval_start = any(marker in chunk for marker in _EVAL_START_MARKERS)
    return size, saw_ready, saw_progress, saw_eval_start


def session_deadline_to_remaining_sec(session_deadline_sec: float | None) -> float | None:
    """Convert an in-process session deadline into seconds still left on it."""
    if session_deadline_sec is None:
        return None
    return float(session_deadline_sec) - time.monotonic()


def session_remaining_to_deadline_sec(session_remaining_sec: float | None) -> float | None:
    """Re-anchor a remaining session budget onto this process's monotonic clock."""
    if session_remaining_sec is None:
        return None
    return time.monotonic() + float(session_remaining_sec)


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
    on_output: Callable[[], None] | None = None,
    session_deadline_sec: float | None = None,
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
    empty: str | bytes = "" if text else b""
    try:
        with cancel_scope_listener() as cancel_scope:
            proc = subprocess.Popen(  # noqa: S603 — cmd is caller's responsibility
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=text,
                env=env,
                cwd=cwd,
                **new_session_kwargs(),
            )
            capture = _StreamCapture(proc, text=text, on_output=on_output)
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
                    session_deadline_sec=session_deadline_sec,
                    cancel_scope=cancel_scope,
                )
            except subprocess.TimeoutExpired:
                kill_my_spawned_server(proc)
                if capture is not None:
                    capture.finish(timeout=_CAPTURE_DRAIN_SECONDS)
                raise
            except _ReapedByWatchdog as exc:
                kill_my_spawned_server(proc)
                stdout, stderr = _finish_capture(capture, text=text)
                log.log(
                    exc.log_level,
                    "_subprocess_kill: %s; reaped the tree with sentinel returncode=%d.",
                    exc,
                    exc.returncode,
                )
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=exc.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout if stdout is not None else empty,
                stderr=stderr if stderr is not None else empty,
            )
    finally:
        kill_my_spawned_server(proc)


def _finish_capture(capture: _StreamCapture | None, *, text: bool) -> tuple[str | bytes, str | bytes]:
    """Drain the capture threads of a reaped child, never returning ``None``."""
    empty: str | bytes = "" if text else b""
    if capture is None:
        return empty, empty
    stdout, stderr = capture.finish(timeout=_CAPTURE_DRAIN_SECONDS)
    return (
        stdout if stdout is not None else empty,
        stderr if stderr is not None else empty,
    )


class _ReapedByWatchdog(Exception):
    """Internal base for a cause that reaps the tree and names itself."""

    returncode: int = -1
    log_level: int = logging.WARNING


class _SessionDeadlineExceeded(_ReapedByWatchdog):
    """Internal sentinel: the session wall-clock budget ran out mid-round."""

    returncode = SESSION_TIME_EXHAUSTED_RETURNCODE

    def __init__(self, *, overrun_sec: float, elapsed_sec: float) -> None:
        """Record how far past the session deadline the round got."""
        super().__init__(
            f"the session wall-clock budget was exhausted {overrun_sec:.1f}s ago (round elapsed={elapsed_sec:.1f}s)"
        )
        self.overrun_sec = float(overrun_sec)
        self.elapsed_sec = float(elapsed_sec)


class _OrchestratorCancelled(_ReapedByWatchdog):
    """Internal sentinel: the orchestrator cancelled the action this child serves."""

    returncode = ORCHESTRATOR_CANCELLED_RETURNCODE

    def __init__(self, *, reason: str, elapsed_sec: float) -> None:
        """Record who asked for the stop and how far the round had got."""
        super().__init__(
            f"the orchestrator cancelled this action ({reason or 'no reason given'}; round elapsed={elapsed_sec:.1f}s)"
        )
        self.reason = str(reason)
        self.elapsed_sec = float(elapsed_sec)


class _SoftDeadlineExceeded(_ReapedByWatchdog):
    """Internal sentinel for an elapsed soft deadline."""

    returncode = OVERTIME_KILL_RETURNCODE
    log_level = logging.INFO

    def __init__(self, *, deadline_sec: float, elapsed_sec: float) -> None:
        """Record the deadline and actual elapsed time on the sentinel."""
        super().__init__(f"soft_deadline_sec={deadline_sec:.1f}s elapsed (actual={elapsed_sec:.1f}s)")
        self.deadline_sec = float(deadline_sec)
        self.elapsed_sec = float(elapsed_sec)


class _ServerDeadDetected(_ReapedByWatchdog):
    """Internal sentinel: the server-liveness watchdog saw a terminal engine / worker init marker that persisted past the grace window."""

    returncode = SERVER_DEAD_RETURNCODE

    def __init__(
        self,
        *,
        marker: str,
        grace_sec: float,
        elapsed_sec: float,
    ) -> None:
        """Build the error message describing the hung-after-death condition."""
        super().__init__(
            f"server init died (marker={marker!r}) and the parent hung past "
            f"grace {grace_sec:.1f}s (elapsed={elapsed_sec:.1f}s)"
        )
        self.marker = marker
        self.grace_sec = float(grace_sec)
        self.elapsed_sec = float(elapsed_sec)


class _ServerStalledDetected(_ReapedByWatchdog):
    """Internal sentinel: the detokenizer-stall watchdog saw the server report ready and then produce no generation progress for the grace window."""

    returncode = DETOKENIZER_STALL_RETURNCODE

    def __init__(
        self,
        *,
        grace_sec: float,
        elapsed_sec: float,
    ) -> None:
        """Build the error message describing the ready-but-no-progress stall."""
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
    session_deadline_sec: float | None = None,
    cancel_scope: CancelScope | None = None,
) -> tuple[str | bytes, str | bytes]:
    """Communicate with a child while enforcing soft and server-log watchdogs."""
    watchdog_active = bool(server_log_path) and (
        server_dead_grace_sec is not None and float(server_dead_grace_sec) > 0.0
    )
    stall_active = bool(server_log_path) and (detok_stall_grace_sec is not None and float(detok_stall_grace_sec) > 0.0)
    soft_active = soft_deadline_sec is not None and float(soft_deadline_sec) > 0.0
    session_active = session_deadline_sec is not None
    # A cancel scope is polled like any other gate, so its presence rules out the single-wait fast paths below: a call
    # that blocks until the child exits cannot notice a cancel that arrives while it is blocked.
    gated = soft_active or watchdog_active or stall_active or session_active or cancel_scope is not None
    if capture is None and not gated:
        return proc.communicate(timeout=hard_timeout)
    if capture is not None and not gated:
        proc.wait(timeout=hard_timeout)
        return capture.finish()

    deadline_sec = float(soft_deadline_sec) if soft_active else None
    grace_sec = float(server_dead_grace_sec) if watchdog_active else None
    stall_grace_sec = float(detok_stall_grace_sec) if stall_active else None
    # When a ``server.log`` is available the soft deadline measures only the post-ready phase (clock starts at the
    # server-ready marker, excluding boot / weight load / first-request JIT).
    soft_from_ready = (
        soft_active
        and bool(server_log_path)
        and not server_already_ready
        and os.environ.get("INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # The log increment scan feeds the stall watchdog, the from-ready soft-deadline anchor, the eval-start boundary
    # and the ready timestamp the caller prices later work off; run it once per slice whenever a log is present.
    scan_active = bool(server_log_path) and gated
    poll_interval = STOP_GATE_POLL_SECONDS
    start = time.monotonic()
    dead_marker_since: float | None = None
    # Detokenizer-stall watchdog state: per-log byte offsets consumed so far, whether a ready marker has been seen,
    # and the last time a log showed any new output (seeded to the ready time).
    scan_offsets: dict[str, int] = {}
    if scan_active:
        # A reused output_dir can still hold a prior attempt's nested workspace, whose markers are not this round's.
        scan_offsets.update(_stale_scan_log_sizes(server_log_path))  # type: ignore[arg-type]
    server_ready_since: float | None = None
    last_activity_at: float | None = None
    # Latched once the accuracy eval starts: the soft deadline bounds the throughput phase only, so it is retired for
    # the rest of the process.
    soft_deadline_suspended = False
    while True:
        now = time.monotonic()
        elapsed = now - start
        # Session budget.
        if session_active and session_deadline_sec is not None and now >= session_deadline_sec:
            raise _SessionDeadlineExceeded(
                overrun_sec=now - float(session_deadline_sec),
                elapsed_sec=elapsed,
            )
        # Orchestrator cancellation.
        if cancel_scope is not None and cancel_scope.cancelled:
            raise _OrchestratorCancelled(
                reason=cancel_scope.reason,
                elapsed_sec=elapsed,
            )
        # Advance the log scan, latching the server-ready, last-activity and eval-start signals.
        if scan_active:
            scan = _scan_logs_increment(
                server_log_path,  # type: ignore[arg-type]
                scan_offsets,
            )
            if scan.saw_ready and server_ready_since is None:
                server_ready_since = now
                last_activity_at = now  # start the silence clock at ready
                # Recorded for the caller, which prices later work off the
                # post-ready segment rather than the whole round: a pass that
                # re-attaches to this server pays none of the boot. Taken as
                # ``now - start`` so the boot is measured end to end on this
                # process's own clock, whatever host the caller reads it on.
                stamp_server_ready(server_log_path, now - start)  # type: ignore[arg-type]
            if scan.saw_eval_start and not soft_deadline_suspended:
                soft_deadline_suspended = True
                log.info(
                    "_subprocess_kill: accuracy eval started; soft_deadline_sec=%.1fs no longer enforced "
                    "(it bounds the throughput phase only)",
                    float(deadline_sec or 0.0),
                )
            # Any new bytes count as liveness; only total silence trips the stall gate.
            if scan.grew:
                last_activity_at = now
            # The liveness callback makes a narrower claim than the stall gate — that this child is working, not that
            # something on the box is — so it takes narrower evidence: tokens flowing, or the child's own redirected
            # stderr growing.
            if capture is not None and (scan.saw_progress or scan.child_spoke):
                capture.note_output()
        # Soft deadline.
        if soft_active and deadline_sec is not None and not soft_deadline_suspended:
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
        # Slice bounded by every active remaining window so the right gate fires first; the child can still finish
        # inside any slice.
        slice_sec = poll_interval
        if session_active and session_deadline_sec is not None:
            slice_sec = min(slice_sec, float(session_deadline_sec) - now)
        if soft_active and deadline_sec is not None and not soft_deadline_suspended:
            if soft_from_ready:
                if server_ready_since is not None:
                    slice_sec = min(slice_sec, deadline_sec - (now - server_ready_since))
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
    "AGENTX_PREFLIGHT_ERROR_CLASS",
    "AGENTX_PREFLIGHT_RETURNCODE",
    "COOPERATIVE_REAP_BUDGET_SEC",
    "DETOKENIZER_STALL_RETURNCODE",
    "EVAL_PROBE_UNPATCHABLE_RETURNCODE",
    "ORCHESTRATOR_CANCELLED_RETURNCODE",
    "OVERTIME_KILL_RETURNCODE",
    "SERVER_DEAD_RETURNCODE",
    "SESSION_TIME_EXHAUSTED_RETURNCODE",
    "STOP_GATE_POLL_SECONDS",
    "TERM_GRACE_SECONDS",
    "clear_server_ready_stamp",
    "kill_my_spawned_server",
    "new_session_kwargs",
    "post_ready_runtime_sec",
    "run_with_session_kill",
    "server_log_death_excerpt",
    "server_ready_unix",
    "session_deadline_to_remaining_sec",
    "session_remaining_to_deadline_sec",
    "stamp_server_ready",
]
