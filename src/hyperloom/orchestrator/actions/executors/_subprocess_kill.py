# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
from pathlib import Path
from typing import Callable, NamedTuple

from .bypass_analysis import parse_server_log_throughput

from ..cancel_channel import CancelScope, cancel_scope_listener

log = logging.getLogger(__name__)


# Grace window between SIGTERM and SIGKILL, here and for a driver-side teardown
# of a server a round left behind: the same signal, the same thing being waited
# for.
TERM_GRACE_SECONDS: float = 5.0

# How long the reaper waits to collect the SIGKILL'd child before giving up on it.
_REAP_COLLECT_SECONDS: float = 1.0

# How long draining a reaped child's capture threads is given.
_CAPTURE_DRAIN_SECONDS: float = 2.0

# How often the blocking side looks up from the child to check its stop gates --
# the session deadline and the cancel scope among them. Short enough that the
# check costs a stop almost nothing on top of the reap, long enough not to spin.
STOP_GATE_POLL_SECONDS: float = 0.5

# What stopping a running round costs, end to end, from the moment something asks
# it to: noticing at the poll, SIGTERM'ing the tree, waiting out the grace before
# SIGKILL, collecting the child, and draining its pipes.
#
# Every window that waits for a round to stop itself is derived from this rather
# than picked next to it -- the Ray submitter's grace and the dispatcher's
# cooperative window both are. A window shorter than what stopping costs looks
# generous in isolation and still expires every time, and what it discards is the
# honest sentinel the round was about to return, replacing it with a hard kill
# and an unattributed failure.
COOPERATIVE_REAP_BUDGET_SEC: float = (
    STOP_GATE_POLL_SECONDS + TERM_GRACE_SECONDS + _REAP_COLLECT_SECONDS + _CAPTURE_DRAIN_SECONDS
)


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
    grace_seconds: float = TERM_GRACE_SECONDS,
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


# Sentinel ``returncode`` allocation. This module is not the only owner of the
# space -- ``_ray_serving`` hands out Ray-actor codes from it too, and both
# arrive at their consumer as a bare ``returncode`` carrying no other tag. A
# number claimed by a second cause therefore makes attribution a coin flip, so
# allocate an unused one; ``test_every_sentinel_returncode_names_exactly_one_cause``
# enumerates both modules and fails on reuse.

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
#
# Two families:
#  1. Runtime engine/worker bootstrap crashes (engine core / worker proc).
#  2. Config-validation-stage failures that die BEFORE the engine starts — most
#     importantly a brand-new checkpoint ``model_type`` that the installed
#     transformers / vLLM does not recognise (``pydantic`` ``ModelConfig``
#     ``ValidationError``). These are just as terminal, and surfacing their
#     excerpt is what lets the enablement failure classifier see
#     ``missing_model_arch`` (and seed the ``pip install -U transformers/vllm``
#     bridge) instead of the Magpie wrapper's uninformative ``subprocess_nonzero``
#     stdout tail, which classifies as ``unknown`` and starves every enablement
#     round of the real root cause.
_SERVER_DEAD_MARKERS: tuple[str, ...] = (
    # (1) runtime engine/worker bootstrap crashes
    "WorkerProc initialization failed",
    "EngineCore failed to start",
    "Engine core initialization failed",
    "Engine process failed to start",
    "AsyncEngineDeadError",
    "raise EngineDeadError",
    "Failed core proc(s)",
    # (2) config-validation-stage terminal failures (pre-engine). Kept to
    #     specific failure phrases (never a bare "Model architectures" INFO
    #     banner) so the liveness watchdog cannot false-positive on a healthy
    #     slow-loading server.
    "does not recognize this architecture",
    "Transformers does not recognize",
    "ValidationError for ModelConfig",
    "are not supported for now",
)

# Default grace after the first fatal marker before forcing a reap. Overridable
# via ``INFERENCE_OPTIMIZER_SERVER_DEAD_GRACE_SEC``.
_SERVER_DEAD_GRACE_SEC_DEFAULT: float = 120.0


# Sentinel ``returncode`` when the detokenizer-stall watchdog reaps a child that
# came up healthy but then produced no generation progress (hung engine /
# detokenizer wedge). Distinct so callers can label it
# ``error_class="detokenizer_stall"``.
DETOKENIZER_STALL_RETURNCODE: int = -911

# Sentinel ``returncode`` returned by the _run_magpie AgentX hook when the
# execution-boundary preflight fails (aiperf missing or not weka-trace capable)
# before any benchmark launches. Distinct so callers label it
# ``error_class="agentx_preflight"`` and surface the guidance instead of crashing.
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

# Sentinel ``returncode`` returned by the _run_magpie eval hook when the
# generation bounds / pathology probe cannot be installed even though the target
# file is present and this variant runs eval. Distinct so callers label it
# ``error_class="eval_probe_unpatchable"`` -- the same class the baseline arm
# already fails with, so a bounds gap reads identically on both arms.
EVAL_PROBE_UNPATCHABLE_RETURNCODE: int = -914

# Sentinel ``returncode`` when the session wall-clock budget ran out mid-round and
# the tree was reaped. Deliberately distinct from ``OVERTIME_KILL_RETURNCODE``:
# that one says "this variant ran far longer than the baseline", which is a
# judgement about the variant, while this one says "the run was out of time",
# which says nothing about the variant at all. Sharing a code would teach the KB
# that a variant is slow whenever a session happened to end during it.
SESSION_TIME_EXHAUSTED_RETURNCODE: int = -915

# -916 is ``_ray_serving._ACTOR_TIMEOUT_RC``.

# Sentinel ``returncode`` when the orchestrator cancelled the action this child
# was launched for -- a shutdown, or a budget that is spent. Distinct from
# ``SESSION_TIME_EXHAUSTED_RETURNCODE`` because the two causes are told apart at
# the source: that one is the round's own deadline elapsing where it runs, this
# one is a decision taken outside it, and only one of them is still true when the
# same session resumes. Distinct from ``OVERTIME_KILL_RETURNCODE`` for the reason
# that one already carries: neither says anything about the variant.
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

# Accuracy-eval start markers: their appearance means the benchmark phase of the
# run is over and the accuracy eval has begun. The soft deadline measures
# throughput only, so it stops being enforced from this point on — eval duration
# is not a throughput signal and the anchor it is compared against excludes it.
# The hard timeout and the dead/stall watchdogs stay armed.
_EVAL_START_MARKERS: tuple[str, ...] = (
    "HYPERLOOM_EVAL_START",
    "[magpie_bench_remote_compat] lm_eval cmd:",
)

# The benchmark body's own stderr: Magpie redirects it here rather than into
# ``server.log`` or the parent's pipe, so it is both where the eval-start marker
# lands and the one resolved log whose growth is output of the very child this
# module is waiting on.
_EVAL_LOG_NAME: str = "benchmark_stderr.log"

# Default grace: how long after the server reports ready it may emit no log
# output before the watchdog declares a hang / detokenizer stall. Measured from
# the ready marker so the cold start is never counted. ``<= 0`` disables the
# gate. Overridable via ``INFERENCE_OPTIMIZER_DETOK_STALL_GRACE_SEC``.
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
        """Set up capture/mirror threads for a child's stdout and stderr.

        Args:
            proc: The child process whose pipes should be captured.
            text: Whether the pipes are in text (``str``) or bytes mode.
            on_output: Called once per observed unit of child output, from the
                pump thread. Used as liveness evidence by callers that report
                progress for a long step.
        """
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

    def note_output(self) -> None:
        """Report one unit of child output to the caller's liveness callback."""
        if self._on_output is None:
            return
        try:
            self._on_output()
        except Exception:  # noqa: BLE001 - liveness reporting never breaks capture
            pass

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
                self.note_output()
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

    Baseline / profile / explore failure classification calls this to surface
    the real server-side root cause — e.g. vLLM's ``RuntimeError: Engine core
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


# Name of the stamp written beside the caller's ``server.log`` the moment the
# server first reports ready.
_READY_STAMP_NAME = "server_ready_at"


def _ready_stamp_path(server_log_path: str) -> Path:
    """Return where a round's ready stamp lives, given its ``server.log`` path."""
    return Path(server_log_path).parent / _READY_STAMP_NAME


def _stamp_server_ready(server_log_path: str, boot_sec: float) -> None:
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
    """Drop any ready stamp an earlier round left in this output directory.

    Args:
        server_log_path: The ``<output_dir>/server.log`` path from the caller.
    """
    try:
        _ready_stamp_path(server_log_path).unlink(missing_ok=True)
    except OSError as exc:
        log.warning("_subprocess_kill: could not clear stale server-ready stamp (%s)", exc)


def _read_ready_stamp(server_log_path: str) -> tuple[float, float] | None:
    """Return a round's ``(ready_unix, boot_sec)``, or ``None`` when unrecorded.

    Both fields are required. A stamp missing its boot is reported as no stamp at
    all rather than as a boot of zero: zero is a legitimate boot, so it cannot
    also stand for "not recorded", and reading it that way would hand the whole
    round to the benchmark -- the figure every later variant is then admitted on.

    Args:
        server_log_path: The ``<output_dir>/server.log`` path from the caller.

    Returns:
        tuple[float, float] | None: The wall-clock instant the stamp was written
        and the boot it measured, or ``None`` when no readable stamp exists.
    """
    try:
        fields = _ready_stamp_path(server_log_path).read_text(encoding="utf-8").split()
        ready_unix = float(fields[0])
        boot_sec = float(fields[1])
    except (OSError, ValueError, IndexError):
        return None
    return (ready_unix, max(0.0, boot_sec)) if ready_unix > 0.0 else None


def server_ready_unix(server_log_path: str) -> float | None:
    """Return when the server reported ready, or ``None`` when nothing recorded it.

    Args:
        server_log_path: The ``<output_dir>/server.log`` path from the caller.

    Returns:
        float | None: The wall-clock instant, or ``None`` when no stamp exists or
        it is unreadable.
    """
    stamp = _read_ready_stamp(server_log_path)
    return None if stamp is None else stamp[0]


def post_ready_runtime_sec(
    server_log_path: str,
    *,
    started_unix: float,
    runtime_sec: float,
) -> float | None:
    """Return how long a round ran *after* its server was ready.

    This is the part of a round's wall-clock that is the benchmark itself, with
    boot, weight load, compile and graph capture excluded. It is what makes two
    rounds comparable when one paid for a cold start and the other re-attached to
    a server already up.

    The round's wall-clock less the boot the stamp measured. The boot is a
    duration taken on one clock, so this holds however far apart the writer and
    the reader are; ``started_unix`` is only compared against the stamp's own
    instant, to tell this round's stamp from one an earlier round left behind.

    Args:
        server_log_path: The ``<output_dir>/server.log`` path from the caller.
        started_unix: When the round was spawned.
        runtime_sec: The round's full wall-clock.

    Returns:
        float | None: Seconds after ready, bounded by the round's own runtime, or
        ``None`` when the round never reported ready or the only stamp present
        predates it (an earlier round's, left behind by a failed clear -- reading
        it would report a cold round as though it had never booted).
    """
    stamp = _read_ready_stamp(server_log_path)
    if stamp is None or stamp[0] < started_unix:
        return None
    return max(0.0, min(float(runtime_sec), float(runtime_sec) - stamp[1]))


def _resolve_scan_logs(server_log_path: str) -> list[str]:
    """Return the log files to scan for markers, newest-nesting first.

    The caller passes ``<output_dir>/server.log``, but the Magpie benchmark
    scripts write their logs into a ``benchmark_<fw>_<ts>/`` subdir of
    ``$RESULT_DIR``, so that exact path usually does not exist. Resolve the
    nested location and pair each ``server.log`` with the sibling
    ``benchmark_stderr.log`` that carries the eval-start marker (the patcher
    echoes it to stderr, which never reaches ``server.log``).

    Args:
        server_log_path: The ``<output_dir>/server.log`` path from the caller.

    Returns:
        Existing log paths to scan; empty when none are present yet.
    """
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
    """What one pass over the resolved logs found in the bytes appended since the last.

    Attributes:
        saw_ready: A server-ready marker appeared.
        saw_progress: A periodic decode-throughput line reported a non-zero
            rate, so tokens were being produced during the interval whoever
            logged them.
        saw_eval_start: The accuracy eval announced itself.
        grew: Some resolved log got longer, from any writer at all.
        child_spoke: The benchmark body's own redirected stderr got longer,
            which is output of the child being waited on that never reaches its
            pipe.
    """

    saw_ready: bool
    saw_progress: bool
    saw_eval_start: bool
    grew: bool
    child_spoke: bool


def _stale_scan_log_sizes(server_log_path: str) -> dict[str, int]:
    """Current byte length of each nested log that already exists at spawn.

    Seeded as starting offsets so such a log contributes only what it grows by,
    never what a previous attempt left in it.

    Only the nested ``benchmark_*/`` workspaces, not the path the caller named.
    That one the caller owns: it clears it when it means this round to start
    clean, and a caller that means to attach to a server already up says so with
    ``server_already_ready``. The nested ones are found by globbing a directory
    the caller reuses, so nothing in them was asserted by anyone -- and a stale
    ready line there would otherwise latch on the first poll and report a boot
    that took minutes as one that took none.

    Args:
        server_log_path: The ``<output_dir>/server.log`` path from the caller.

    Returns:
        dict[str, int]: Per-path byte lengths; a path whose size cannot be read
        is left out, so it is scanned from the start as an absent one would be.
    """
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
    """Scan every resolved log for markers, advancing ``offsets`` in place.

    Args:
        server_log_path: The ``<output_dir>/server.log`` path from the caller.
        offsets: Per-path byte offsets already consumed; mutated in place.

    Returns:
        _LogScan: The markers seen in the newly appended bytes, plus who — if
        anyone — did the appending.
    """
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
    """Incrementally scan the bytes appended to ``server.log`` since
    ``from_offset`` for ready / generation-progress / eval-start markers.

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
        ``(new_offset, saw_ready, saw_progress, saw_eval_start)`` — the advanced
        offset plus whether the newly read bytes carried a ready marker, a
        non-zero generation-throughput rate, or the eval-start marker.
    """
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
    # Progress is the rate on the periodic decode-throughput line, not the
    # line's presence: some vLLM builds log ``Avg generation throughput: 0.0
    # tokens/s`` on an idle engine, and an engine goes idle precisely when the
    # client driving it wedges, so the marker alone lets the server vouch for
    # the client that stopped asking it for tokens. Reusing the post-mortem
    # estimator's parse keeps the frameworks the two recognise from drifting
    # apart. A line whose rate it cannot read counts as no progress: this
    # evidence only ever suppresses a stall accusation, and one suppressed by
    # mistake is invisible, where a missing one is visible and answerable from
    # the child's own redirected stderr. The stall gate is deliberately broader
    # and keys on raw log activity (any new bytes); see
    # :func:`_communicate_with_soft_deadline`.
    saw_progress = bool(parse_server_log_throughput(chunk))
    saw_eval_start = any(marker in chunk for marker in _EVAL_START_MARKERS)
    return size, saw_ready, saw_progress, saw_eval_start


def session_deadline_to_remaining_sec(session_deadline_sec: float | None) -> float | None:
    """Convert an in-process session deadline into seconds still left on it.

    The pair of this and :func:`session_remaining_to_deadline_sec` is how a
    session deadline crosses a process boundary. ``time.monotonic()`` has an
    unspecified, per-process origin, so the absolute instant means nothing to a
    reader in another process; a duration means the same thing everywhere.

    Args:
        session_deadline_sec: Absolute ``time.monotonic()`` instant at which the
            session budget expires, or ``None`` when the budget is unbounded.

    Returns:
        Seconds left on the budget, or ``None`` when unbounded. Non-positive
        when the deadline has already passed, which the receiving side is meant
        to act on immediately rather than treat as "no budget given".
    """
    if session_deadline_sec is None:
        return None
    return float(session_deadline_sec) - time.monotonic()


def session_remaining_to_deadline_sec(session_remaining_sec: float | None) -> float | None:
    """Re-anchor a remaining session budget onto this process's monotonic clock.

    The inverse of :func:`session_deadline_to_remaining_sec`. Whatever the trip
    itself cost is forgiven: no clock is shared across the boundary to measure it
    with, so the receiver starts a fresh window of the full duration. The
    receiver can therefore run marginally past the sender's deadline, which is
    the safe direction -- the alternative is guessing at the transit and charging
    a round for time it never had.

    Args:
        session_remaining_sec: Seconds left on the session budget as measured by
            the sender, or ``None`` when the budget is unbounded.

    Returns:
        An absolute ``time.monotonic()`` deadline usable in this process, or
        ``None`` when unbounded.
    """
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
    """Run a subprocess in its own session and reap descendants on every exit path.

    ``session_deadline_sec`` is an absolute ``time.monotonic()`` instant, unlike
    ``soft_deadline_sec`` which is a relative duration. It is the session's own
    wall-clock budget and is enforced in every phase, including the accuracy
    eval -- the phase that retires ``soft_deadline_sec``. That distinction is the
    reason it is a separate channel rather than a reuse of the soft deadline: the
    eval-start boundary is meaningful for "is this variant abnormally slow" and
    meaningless for "is the run out of time".

    The cancel scope published by the dispatcher (:mod:`..cancel_channel`) is
    watched for as long as the child lives, so an orchestrator that cancels the
    action reaches the tree rather than just the coroutine awaiting this call.
    Every cause that reaps the tree is reported as its own sentinel
    ``returncode``, which is all a caller of a subprocess gets to tell them apart
    by.

    Args:
        on_output: Called from a reader thread each time the child produces
            output, so a caller can report a long step alive on the child's own
            activity rather than on a timer.
        session_deadline_sec: Absolute ``time.monotonic()`` instant at which the
            session budget expires. Reaps the tree and returns
            ``SESSION_TIME_EXHAUSTED_RETURNCODE``.
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
            empty: str | bytes = "" if text else b""
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout if stdout is not None else empty,
                stderr=stderr if stderr is not None else empty,
            )
    finally:
        kill_my_spawned_server(proc)


def _finish_capture(capture: _StreamCapture | None, *, text: bool) -> tuple[str | bytes, str | bytes]:
    """Drain the capture threads of a reaped child, never returning ``None``.

    Args:
        capture: The stream capture to finish, or ``None`` when the child's
            output was not captured.
        text: Whether the streams are ``str`` (``True``) or ``bytes``, which
            decides what an absent stream reads as.

    Returns:
        tuple[str | bytes, str | bytes]: The captured ``(stdout, stderr)``.
    """
    empty: str | bytes = "" if text else b""
    if capture is None:
        return empty, empty
    stdout, stderr = capture.finish(timeout=_CAPTURE_DRAIN_SECONDS)
    return (
        stdout if stdout is not None else empty,
        stderr if stderr is not None else empty,
    )


class _ReapedByWatchdog(Exception):
    """Internal base for a cause that reaps the tree and names itself.

    None of these bubble past :func:`run_with_session_kill`: each is converted
    to a ``CompletedProcess`` carrying the subclass's own ``returncode``, since
    a returncode is the only thing that survives the trip back to a caller of a
    subprocess. Subclasses build their own message and declare how much the
    event is worth logging.
    """

    returncode: int = -1
    log_level: int = logging.WARNING


class _SessionDeadlineExceeded(_ReapedByWatchdog):
    """Internal sentinel: the session wall-clock budget ran out mid-round."""

    returncode = SESSION_TIME_EXHAUSTED_RETURNCODE

    def __init__(self, *, overrun_sec: float, elapsed_sec: float) -> None:
        """Record how far past the session deadline the round got.

        Args:
            overrun_sec (float): Seconds past the session deadline at trip time.
            elapsed_sec (float): Wall-clock elapsed for this round at trip time.
        """
        super().__init__(
            f"the session wall-clock budget was exhausted {overrun_sec:.1f}s ago (round elapsed={elapsed_sec:.1f}s)"
        )
        self.overrun_sec = float(overrun_sec)
        self.elapsed_sec = float(elapsed_sec)


class _OrchestratorCancelled(_ReapedByWatchdog):
    """Internal sentinel: the orchestrator cancelled the action this child serves.

    The cause is a decision taken outside the round -- a shutdown, or a budget
    the dispatcher found spent -- which is why it carries the caller's reason
    rather than a measurement of its own.
    """

    returncode = ORCHESTRATOR_CANCELLED_RETURNCODE

    def __init__(self, *, reason: str, elapsed_sec: float) -> None:
        """Record who asked for the stop and how far the round had got.

        Args:
            reason (str): Short cause from the canceller, e.g. ``shutdown_requested``.
            elapsed_sec (float): Wall-clock elapsed for this round at trip time.
        """
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
        """Record the deadline and actual elapsed time on the sentinel.

        Args:
            deadline_sec (float): The soft deadline that was exceeded.
            elapsed_sec (float): The actual wall-clock elapsed at trip time.
        """
        super().__init__(f"soft_deadline_sec={deadline_sec:.1f}s elapsed (actual={elapsed_sec:.1f}s)")
        self.deadline_sec = float(deadline_sec)
        self.elapsed_sec = float(elapsed_sec)


class _ServerDeadDetected(_ReapedByWatchdog):
    """Internal sentinel: the server-liveness watchdog saw a terminal engine /
    worker init marker that persisted past the grace window.
    """

    returncode = SERVER_DEAD_RETURNCODE

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
            f"server init died (marker={marker!r}) and the parent hung past "
            f"grace {grace_sec:.1f}s (elapsed={elapsed_sec:.1f}s)"
        )
        self.marker = marker
        self.grace_sec = float(grace_sec)
        self.elapsed_sec = float(elapsed_sec)


class _ServerStalledDetected(_ReapedByWatchdog):
    """Internal sentinel: the detokenizer-stall watchdog saw the server report
    ready and then produce no generation progress for the grace window.
    """

    returncode = DETOKENIZER_STALL_RETURNCODE

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
    # A cancel scope is polled like any other gate, so its presence rules out the
    # single-wait fast paths below: a call that blocks until the child exits
    # cannot notice a cancel that arrives while it is blocked.
    gated = soft_active or watchdog_active or stall_active or session_active or cancel_scope is not None
    if capture is None and not gated:
        return proc.communicate(timeout=hard_timeout)
    if capture is not None and not gated:
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
        and os.environ.get("INFERENCE_OPTIMIZER_SOFT_DEADLINE_FROM_READY", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # The log increment scan feeds the stall watchdog, the from-ready
    # soft-deadline anchor, the eval-start boundary and the ready timestamp the
    # caller prices later work off; run it once per slice whenever a log is
    # present. Not narrowed to the watchdogs that consume it: tying it to
    # ``stall_active`` let an unrelated knob
    # (``INFERENCE_OPTIMIZER_DETOK_STALL_GRACE_SEC=0``) silently withdraw the
    # boot/benchmark split for a whole session. It costs nothing where it did not
    # already run, since without a gate this call never enters the loop at all.
    scan_active = bool(server_log_path) and gated
    poll_interval = STOP_GATE_POLL_SECONDS
    start = time.monotonic()
    dead_marker_since: float | None = None
    # Detokenizer-stall watchdog state: per-log byte offsets consumed so far,
    # whether a ready marker has been seen, and the last time a log showed any
    # new output (seeded to the ready time). The gate only arms once
    # ``server_ready_since`` is set.
    scan_offsets: dict[str, int] = {}
    if scan_active:
        # A reused output_dir can still hold a prior attempt's nested workspace,
        # whose markers are not this round's. The workspace this round creates
        # does not exist yet, so it is discovered later at offset zero, which is
        # correct: all of its bytes are this round's.
        scan_offsets.update(_stale_scan_log_sizes(server_log_path))  # type: ignore[arg-type]
    server_ready_since: float | None = None
    last_activity_at: float | None = None
    # Latched once the accuracy eval starts: the soft deadline bounds the
    # throughput phase only, so it is retired for the rest of the process.
    soft_deadline_suspended = False
    while True:
        now = time.monotonic()
        elapsed = now - start
        # Session budget. Checked before every other gate and never suspended:
        # unlike the soft deadline it makes no claim about the variant, so the
        # eval-start boundary that retires the soft deadline does not apply. An
        # accuracy eval that starts one minute before the run is out of time still
        # has to stop.
        if session_active and session_deadline_sec is not None and now >= session_deadline_sec:
            raise _SessionDeadlineExceeded(
                overrun_sec=now - float(session_deadline_sec),
                elapsed_sec=elapsed,
            )
        # Orchestrator cancellation. Checked after the session deadline so a
        # round that was already out of time keeps that reason: the budget is a
        # fact about the run, while the cancel is only the dispatcher acting on
        # it, and the more specific of the two is the one worth recording.
        if cancel_scope is not None and cancel_scope.cancelled:
            raise _OrchestratorCancelled(
                reason=cancel_scope.reason,
                elapsed_sec=elapsed,
            )
        # Advance the log scan, latching the server-ready, last-activity
        # and eval-start signals.
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
                _stamp_server_ready(server_log_path, now - start)  # type: ignore[arg-type]
            if scan.saw_eval_start and not soft_deadline_suspended:
                soft_deadline_suspended = True
                log.info(
                    "_subprocess_kill: accuracy eval started; soft_deadline_sec=%.1fs no longer enforced "
                    "(it bounds the throughput phase only)",
                    float(deadline_sec or 0.0),
                )
            # Any new bytes count as liveness; only total silence trips the
            # stall gate.
            if scan.grew:
                last_activity_at = now
            # The liveness callback makes a narrower claim than the stall gate —
            # that this child is working, not that something on the box is — so
            # it takes narrower evidence: tokens flowing, or the child's own
            # redirected stderr growing. A ``server.log`` that grew is neither.
            # The server keeps logging while its benchmark client is wedged, and
            # both vLLM and sglang write an access line per request, including
            # the health probe the robustness agent issues on its own tick —
            # which would let the monitor manufacture the evidence that
            # suppresses its own stall accusation.
            if capture is not None and (scan.saw_progress or scan.child_spoke):
                capture.note_output()
        # Soft deadline. With ``soft_from_ready`` the overtime clock is measured
        # from the server-ready marker and stays dormant until ready; otherwise
        # it is the from-spawn elapsed.
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
        # Slice bounded by every active remaining window so the right gate
        # fires first; the child can still finish inside any slice.
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
]
