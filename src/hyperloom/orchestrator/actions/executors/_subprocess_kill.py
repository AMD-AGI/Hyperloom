# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reliable subprocess-tree teardown for Magpie-launched servers."""

from __future__ import annotations

import codecs
import glob
import io
import logging
import math
import os
import re
from collections.abc import Mapping
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, NamedTuple

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

#: Engine deaths that happen *after* the server reported ready and began
#: serving. Kept out of :data:`_SERVER_DEAD_MARKERS` on purpose: that tuple also
#: drives the live readiness waiter, and widening it would change when a running
#: server is torn down. These are read only when building a diagnostic excerpt.
_FATAL_ENGINE_MARKERS: tuple[str, ...] = (
    "EngineCore encountered a fatal error",
    "EngineCore encountered an issue",
    "EngineDeadError",
    "Engine core proc died",
    "died with exit code",
)

#: A framework line's exception, named anywhere in it: these logs prefix every
#: line with the worker pid, the level and the source location, so the exception
#: never starts the line.
_EXCEPTION_LINE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)\s*:")

#: Lines after a fatal marker that may carry its terminating exception.
_MARKER_WINDOW_LINES: int = 60

#: Lines of preceding context a legacy bootstrap marker keeps.
_MARKER_CONTEXT_LINES: int = 2


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

# AgentX phase boundaries. The agentic client runs a long cold-cache warmup -- close to 18 minutes on the runs measured
# so far -- before the window under measurement opens, and mixing the two makes every KV statistic meaningless. aiperf
# already announces its phase transitions, so these match its own text; anchored on the parenthesised phase id rather
# than the whole line so a reword of the human-readable half does not silently stop the split.
_AGENTX_WARMUP_BEGIN_MARKERS: tuple[str, ...] = ("(warmup) started",)
_AGENTX_MEASURED_BEGIN_MARKERS: tuple[str, ...] = ("(profiling) started",)

# aiperf writes only this file: an AgentX benchmark directory has no ``benchmark_stdout.log`` / ``benchmark_stderr.log``
# the way a synthetic one does, so the phase lines are unreachable unless it is scanned explicitly.
_AGENTX_LOG_RELPATH: str = "aiperf_artifacts/logs/aiperf.log"

# How long a round may go without any resolvable server log before it is treated as re-attached to a server an earlier
# round booted. Such a round owns no log and no ready marker will ever appear in its scan set, so KV collection has to
# start on some other signal or it never starts at all.
_WARM_REUSE_PROBE_AFTER_SEC: float = 30.0


def resolve_benchmark_timeouts(env: Mapping[str, str] | None = None) -> tuple[float, float]:
    """Resolve the invocation's silence and hard caps, rejecting invalid overrides."""
    source = os.environ if env is None else env

    def positive(name: str, default: float) -> float:
        raw = source.get(name, str(default))
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and positive, got {raw!r}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {raw!r}")
        return value

    return (
        positive("INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC", 600.0),
        positive("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", 7800.0),
    )


class _StreamCapture:
    """Capture pipe bytes promptly, decoding text incrementally for capture and mirroring."""

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
        self.last_activity_at: float | None = None
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
        """Read available bytes without waiting for a newline or a complete codepoint."""
        decoder = io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8")(errors="replace"), True)
        raw = getattr(pipe, "buffer", pipe)
        try:
            while True:
                data = raw.read1(65536)
                if not data:
                    break
                self.last_activity_at = time.monotonic()
                self.note_output()
                chunk = decoder.decode(data) if self._text else data
                if chunk:
                    chunks.append(chunk)
                    self._mirror(chunk, mirror)
            if self._text:
                final = decoder.decode(b"", final=True)
                if final:
                    chunks.append(final)
                    self._mirror(final, mirror)
        finally:
            try:
                pipe.close()
            except OSError:
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
        excerpt = _first_marker_excerpt(candidate, max_chars=max_chars)
        if excerpt is not None:
            return excerpt
    return None


def _first_marker_excerpt(path: str, *, max_chars: int) -> str | None:
    """Return the excerpt around the first terminal marker in ``path``.

    The whole file is streamed a line at a time rather than sampled from its
    head or tail. An engine that dies while serving logs one downstream error
    per rejected request afterwards, so the cause sits at the head of that
    cascade: a tail window never reaches it, and a bounded head read leaves the
    middle of a long log unsearched. Memory stays flat either way.

    Args:
        path: Log to read.
        max_chars: Cap on the returned excerpt.

    Returns:
        The excerpt, or ``None`` when the file cannot be read or names no
        terminal marker.
    """
    before: deque[str] = deque(maxlen=_MARKER_CONTEXT_LINES)
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                # Legacy first: every ``_FATAL_ENGINE_MARKERS`` entry is a
                # substring of some line these also match -- ``EngineDeadError``
                # of ``AsyncEngineDeadError`` -- and a legacy line must keep the
                # legacy extraction, whose cause sits above the marker.
                if any(marker in line for marker in _SERVER_DEAD_MARKERS):
                    after = _read_lines(fh, _MARKER_CONTEXT_LINES)
                    excerpt = "\n".join([*before, line, *after]).strip()
                    return excerpt[-max_chars:] or None
                if any(marker in line for marker in _FATAL_ENGINE_MARKERS):
                    return _cause_first_excerpt(
                        line.strip(), _read_lines(fh, _MARKER_WINDOW_LINES - 1), max_chars=max_chars
                    )
                before.append(line)
    except OSError:
        return None
    return None


def _read_lines(fh: Any, count: int) -> list[str]:
    """Return the next ``count`` lines of ``fh``, fewer at end of file."""
    out: list[str] = []
    for _ in range(max(0, count)):
        nxt = fh.readline()
        if not nxt:
            break
        out.append(nxt.rstrip("\n"))
    return out


def _cause_first_excerpt(head: str, window: list[str], *, max_chars: int) -> str | None:
    """Return the marker line followed by the exceptions it names, nearest first.

    A post-startup engine death names its cause *below* the marker, two dozen
    frames down. The frames carry no rule the classifier can use, and the window
    runs past this traceback into the errors it caused downstream -- so the last
    exception in it names a consequence (``EngineDeadError``) while the cause
    (``HIP out of memory``) sits above that. No leading context: the line above
    such a marker is routinely a scheduler-state dump tens of KB wide.
    """
    room = max_chars - len(head) - 1
    if room <= 0:
        return head[:max_chars] or None
    causes: list[str] = []
    used = 0
    for line in window:
        if not _EXCEPTION_LINE.search(line):
            continue
        text = line.strip()
        if used + len(text) + 1 > room:
            break
        causes.append(text)
        used += len(text) + 1
    if causes:
        return head + "\n" + "\n".join(causes)
    rest = "\n".join(window).strip()
    return f"{head}\n{rest[-room:]}" if rest else head[:max_chars] or None


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
        for name in (log, log.with_name(_EVAL_LOG_NAME), log.parent / _AGENTX_LOG_RELPATH):
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
    saw_warmup_begin: bool = False
    saw_measured_begin: bool = False


class _IncrementScan(NamedTuple):
    """One log's scan result: the advanced offset, the markers seen, and the partial line to carry forward."""

    offset: int
    saw_ready: bool
    saw_progress: bool
    saw_eval_start: bool
    saw_warmup_begin: bool = False
    saw_measured_begin: bool = False
    residual: str = ""


# Cap on the partial line carried between scans. A log that appends a very long line without a newline -- or none at
# all -- must not let the buffer grow without bound; a marker is far shorter than this, so nothing real is lost.
_RESIDUAL_MAX_CHARS = 8192


def _stale_scan_log_sizes(server_log_path: str) -> dict[str, int]:
    """Snapshot existing bytes before spawn so previous rounds cannot signal activity."""
    sizes: dict[str, int] = {}
    for path in _resolve_scan_logs(server_log_path):
        candidate = Path(path)
        try:
            size = candidate.stat().st_size
            sizes[path] = -1 if candidate.parent != Path(server_log_path).parent else size
        except OSError:
            continue
    return sizes


def _scan_logs_increment(
    server_log_path: str,
    offsets: dict[str, int],
    residuals: dict[str, str] | None = None,
    identities: dict[str, tuple[int, int]] | None = None,
) -> _LogScan:
    """Scan appended bytes; replacing or truncating a watched file is not activity."""
    saw_ready = saw_progress = saw_eval_start = grew = child_spoke = False
    saw_warmup_begin = saw_measured_begin = False
    for path in _resolve_scan_logs(server_log_path):
        prev = offsets.get(path, 0)
        if prev < 0:
            continue
        if identities is not None:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            replaced = path in identities and identities[path] != identity
            identities[path] = identity
            if replaced or stat.st_size < prev:
                offsets[path] = stat.st_size
                if residuals is not None:
                    residuals.pop(path, None)
                continue
        scan = _scan_server_log_increment(path, prev, "" if residuals is None else residuals.get(path, ""))
        offsets[path] = scan.offset
        if residuals is not None:
            residuals[path] = scan.residual
        saw_ready = saw_ready or scan.saw_ready
        saw_progress = saw_progress or scan.saw_progress
        saw_eval_start = saw_eval_start or scan.saw_eval_start
        saw_warmup_begin = saw_warmup_begin or scan.saw_warmup_begin
        saw_measured_begin = saw_measured_begin or scan.saw_measured_begin
        if scan.offset > prev:
            grew = True
            child_spoke = child_spoke or Path(path).name == _EVAL_LOG_NAME
    return _LogScan(saw_ready, saw_progress, saw_eval_start, grew, child_spoke, saw_warmup_begin, saw_measured_begin)


def _scan_server_log_increment(path: str, from_offset: int, residual: str = "") -> _IncrementScan:
    """Incrementally scan the bytes appended to ``server.log`` since ``from_offset`` for ready / generation-progress / eval-start markers."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return _IncrementScan(from_offset, False, False, False, residual=residual)
    start = from_offset
    carried = residual
    if size < start:  # truncated / rotated — rescan from the top.
        start = 0
        carried = ""  # the bytes it belonged to are gone
    if size <= start:  # nothing new appended
        return _IncrementScan(start, False, False, False, residual=carried)
    try:
        with open(path, "rb") as fh:
            fh.seek(start)
            chunk = fh.read().decode("utf-8", "ignore")
    except (OSError, ValueError):
        return _IncrementScan(from_offset, False, False, False, residual=carried)
    # A poll lands wherever the writer happens to be, so a marker is routinely split across two reads. Prepending the
    # previous tail and carrying the new one forward is what makes a marker findable exactly once, whole.
    chunk = carried + chunk
    tail_at = chunk.rfind("\n")
    next_residual = chunk if tail_at < 0 else chunk[tail_at + 1 :]
    next_residual = next_residual[-_RESIDUAL_MAX_CHARS:]
    saw_ready = any(marker in chunk for marker in _SERVER_READY_MARKERS)
    # Progress is the rate on the periodic decode-throughput line, not the line's presence: some vLLM builds log ``Avg
    # generation throughput: 0.0 tokens/s`` on an idle engine, and an engine goes idle precisely when the client
    # driving it wedges, so the marker alone lets the server vouch for the client that stopped asking it for tokens.
    saw_progress = bool(parse_server_log_throughput(chunk))
    saw_eval_start = any(marker in chunk for marker in _EVAL_START_MARKERS)
    saw_warmup_begin = any(marker in chunk for marker in _AGENTX_WARMUP_BEGIN_MARKERS)
    saw_measured_begin = any(marker in chunk for marker in _AGENTX_MEASURED_BEGIN_MARKERS)
    return _IncrementScan(
        size,
        saw_ready,
        saw_progress,
        saw_eval_start,
        saw_warmup_begin,
        saw_measured_begin,
        residual=next_residual,
    )


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


# Escape hatch for a round that must not pay even the cost of one HTTP GET every couple of seconds.
_KV_METRICS_ENV = "INFERENCE_OPTIMIZER_KV_METRICS"


def _build_kv_recorder(server_log_path: str | None, env: dict[str, str] | None) -> Any:
    """Build the KV-metrics recorder for this round, or ``None`` when it is off or cannot be built.

    Collection is strictly observational, so every failure here is swallowed: a round that produces a good benchmark
    number and no KV metrics is a far better outcome than one that dies because a telemetry import went wrong.
    """
    if not server_log_path:
        return None
    if os.environ.get(_KV_METRICS_ENV, "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    try:
        from ._kv_metrics import KV_ARTIFACT_NAME, AiperfProgressPoller, KvMetricsPoller, KvMetricsRecorder

        workspace = Path(server_log_path).parent
        return KvMetricsRecorder(
            # The workspace is where the port actually lives: it is pinned in the round's materialized
            # ``benchmark.envs.PORT``, which is never exported into the subprocess environment.
            poller=KvMetricsPoller(config_envs=dict(env or {}), workspace=workspace),
            # Authoritative phase boundaries when this round is an AgentX one; a no-op otherwise, since no other client
            # publishes a progress address and the poller then never resolves a URL.
            progress=AiperfProgressPoller(workspace),
            output_path=str(workspace / KV_ARTIFACT_NAME),
            scope={
                "server_log_path": str(server_log_path),
                "workspace": workspace.name,
                "run_path": _run_relative_path(workspace),
                "session": os.path.basename(
                    os.environ.get("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "").rstrip("/\\")
                ),
            },
        )
    except Exception:
        log.warning("kv_metrics: recorder could not be built; this round collects nothing", exc_info=True)
        return None


def _run_relative_path(workspace: Path) -> str:
    """Path of this round's workspace below ``runs/``, which is what the session breakdown keys rows by."""
    parts = workspace.parts
    if "runs" in parts:
        return "/".join(parts[parts.index("runs") + 1 :])
    return workspace.name


def run_with_session_kill(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: int | float | None = None,
    text: bool = True,
    server_log_path: str | None = None,
    silence_timeout_sec: float | None = None,
    server_already_ready: bool = False,
    on_output: Callable[[], None] | None = None,
    session_deadline_sec: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess in its own session and reap descendants on every exit path."""
    child_env = dict(os.environ if env is None else env)
    if silence_timeout_sec is not None:
        child_env["PYTHONUNBUFFERED"] = "1"
    scan_offsets = _stale_scan_log_sizes(server_log_path) if server_log_path else {}
    proc: subprocess.Popen | None = None
    capture: _StreamCapture | None = None
    empty: str | bytes = "" if text else b""
    try:
        with cancel_scope_listener() as cancel_scope:
            started_at = time.monotonic()
            proc = subprocess.Popen(  # noqa: S603 — cmd is caller's responsibility
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=child_env,
                cwd=cwd,
                **new_session_kwargs(),
            )
            capture = _StreamCapture(proc, text=text, on_output=on_output)
            capture.start()
            try:
                stdout, stderr = _communicate_with_watchdog(
                    proc,
                    hard_timeout=timeout,
                    server_log_path=server_log_path,
                    silence_timeout_sec=silence_timeout_sec,
                    started_at=started_at,
                    scan_offsets=scan_offsets,
                    capture=capture,
                    server_already_ready=server_already_ready,
                    session_deadline_sec=session_deadline_sec,
                    cancel_scope=cancel_scope,
                    kv_recorder=_build_kv_recorder(server_log_path, env),
                )
            except subprocess.TimeoutExpired as exc:
                kill_my_spawned_server(proc)
                exc.output, exc.stderr = _finish_capture(capture, text=text)
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
            f"benchmark reported ready but emitted no pipe or log bytes for "
            f"{grace_sec:.1f}s (silence timeout; "
            f"elapsed={elapsed_sec:.1f}s)"
        )
        self.grace_sec = float(grace_sec)
        self.elapsed_sec = float(elapsed_sec)


def _communicate_with_watchdog(
    proc: subprocess.Popen,
    *,
    hard_timeout: int | float | None,
    server_log_path: str | None = None,
    silence_timeout_sec: float | None = None,
    capture: _StreamCapture | None = None,
    server_already_ready: bool = False,
    session_deadline_sec: float | None = None,
    cancel_scope: CancelScope | None = None,
    kv_recorder: Any = None,
    started_at: float | None = None,
    scan_offsets: dict[str, int] | None = None,
) -> tuple[str | bytes, str | bytes]:
    """Wait for exit, enforcing cancellation, session budget and explicit round caps."""
    start = time.monotonic() if started_at is None else started_at
    offsets = {} if scan_offsets is None else scan_offsets
    residuals: dict[str, str] = {}
    identities: dict[str, tuple[int, int]] = {}
    for path in offsets:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        identities[path] = (stat.st_dev, stat.st_ino)
    ready_at: float | None = None
    last_activity_at: float | None = None
    completed = False
    if kv_recorder is not None and server_already_ready:
        kv_recorder.note_phase("measured", start)
    if server_log_path and server_already_ready:
        stamp_server_ready(server_log_path, 0.0)
    try:
        while True:
            # An exit already observed belongs to the child, not to an expired gate.
            if proc.poll() is not None:
                result = proc.communicate() if capture is None else capture.finish()
                completed = True
                return result
            now = time.monotonic()
            elapsed = now - start
            if session_deadline_sec is not None and now >= session_deadline_sec:
                raise _SessionDeadlineExceeded(overrun_sec=now - session_deadline_sec, elapsed_sec=elapsed)
            if cancel_scope is not None and cancel_scope.cancelled:
                raise _OrchestratorCancelled(reason=cancel_scope.reason, elapsed_sec=elapsed)
            if hard_timeout is not None and elapsed >= hard_timeout:
                raise subprocess.TimeoutExpired(proc.args, hard_timeout)
            if server_log_path:
                scan = _scan_logs_increment(server_log_path, offsets, residuals, identities)
                if scan.saw_ready and ready_at is None:
                    ready_at = last_activity_at = now
                    stamp_server_ready(server_log_path, elapsed)
                    if kv_recorder is not None:
                        kv_recorder.note_phase("measured", now)
                if scan.grew:
                    last_activity_at = now
                if kv_recorder is not None:
                    if scan.saw_warmup_begin:
                        kv_recorder.note_phase("warmup", now)
                    if scan.saw_measured_begin:
                        kv_recorder.note_phase("measured", now)
                    if scan.saw_eval_start:
                        kv_recorder.note_phase("eval", now)
                if capture is not None and (scan.saw_progress or scan.child_spoke):
                    capture.note_output()
            if capture is not None and capture.last_activity_at is not None:
                last_activity_at = max(last_activity_at or start, capture.last_activity_at)
            # This telemetry-only guess never arms the silence gate.
            warm_reuse_probe = server_log_path and not offsets and elapsed >= _WARM_REUSE_PROBE_AFTER_SEC
            if kv_recorder is not None and (ready_at is not None or server_already_ready or warm_reuse_probe):
                kv_recorder.tick(now)
            silence_remaining = None
            if ready_at is not None and silence_timeout_sec is not None:
                silence_remaining = silence_timeout_sec - (now - last_activity_at)
                if silence_remaining <= 0:
                    raise _ServerStalledDetected(grace_sec=silence_timeout_sec, elapsed_sec=elapsed)
            slice_sec = STOP_GATE_POLL_SECONDS
            if hard_timeout is not None:
                slice_sec = min(slice_sec, hard_timeout - elapsed)
            if session_deadline_sec is not None:
                slice_sec = min(slice_sec, session_deadline_sec - now)
            if silence_remaining is not None:
                slice_sec = min(slice_sec, silence_remaining)
            try:
                if capture is None:
                    result = proc.communicate(timeout=max(0.0, slice_sec))
                else:
                    proc.wait(timeout=max(0.0, slice_sec))
                    result = capture.finish()
                completed = True
                return result
            except subprocess.TimeoutExpired:
                continue
    finally:
        if kv_recorder is not None:
            kv_recorder.close(aborted=not completed)


__all__ = [
    "AGENTX_PREFLIGHT_ERROR_CLASS",
    "AGENTX_PREFLIGHT_RETURNCODE",
    "COOPERATIVE_REAP_BUDGET_SEC",
    "DETOKENIZER_STALL_RETURNCODE",
    "EVAL_PROBE_UNPATCHABLE_RETURNCODE",
    "ORCHESTRATOR_CANCELLED_RETURNCODE",
    "SERVER_DEAD_RETURNCODE",
    "SESSION_TIME_EXHAUSTED_RETURNCODE",
    "STOP_GATE_POLL_SECONDS",
    "TERM_GRACE_SECONDS",
    "clear_server_ready_stamp",
    "kill_my_spawned_server",
    "new_session_kwargs",
    "post_ready_runtime_sec",
    "resolve_benchmark_timeouts",
    "run_with_session_kill",
    "server_log_death_excerpt",
    "server_ready_unix",
    "session_deadline_to_remaining_sec",
    "session_remaining_to_deadline_sec",
    "stamp_server_ready",
]
