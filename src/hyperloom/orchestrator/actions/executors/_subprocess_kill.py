# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a subprocess in its own session under stop gates, and reap its whole tree on every exit path."""

from __future__ import annotations

import codecs
import io
import logging
import math
import os
from collections.abc import Mapping
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from hyperloom_common.proctree import REAP_MARK_ENV, TERM_GRACE_SEC, collect_tree, kill_tree, marked_pids


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
    STOP_GATE_POLL_SECONDS + TERM_GRACE_SEC + _REAP_COLLECT_SECONDS + _CAPTURE_DRAIN_SECONDS
)


def _terminate_single_process(proc: subprocess.Popen, grace_seconds: float) -> None:
    """Terminate ``proc`` alone, escalating to kill after ``grace_seconds``, where process groups are unavailable."""
    try:
        proc.terminate()
        proc.wait(timeout=grace_seconds)
    except (subprocess.TimeoutExpired, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def kill_my_spawned_server(
    proc: subprocess.Popen | None,
    *,
    grace_seconds: float = TERM_GRACE_SEC,
    mark: str = "",
) -> None:
    """Tear down the entire process tree rooted at ``proc``, and every process carrying ``mark`` in its environment."""
    if proc is None:
        return
    marked = marked_pids(REAP_MARK_ENV, mark) if mark else []
    if proc.poll() is not None:
        if marked:
            kill_tree(collect_tree(marked), grace_sec=grace_seconds, confirm_sec=grace_seconds)
        return
    if os.name != "posix":
        _terminate_single_process(proc, grace_seconds)
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
        tree = collect_tree([proc.pid, *marked])
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


# Sentinel returned when the delegated MLPerf client fails its execution preflight.
AGENTX_PREFLIGHT_RETURNCODE: int = -912

#: The ``error_class`` that sentinel must carry, wherever it is classified. Named
#: here beside the return code because three call sites decide on it -- the grid
#: runner, the baseline executor and the writeback stop-reason gate -- and a
#: string literal repeated across them can drift into a class nobody handles.
#:
#: A missing MLPerf dependency stops the run instead of opening a framework enablement round.
AGENTX_PREFLIGHT_ERROR_CLASS: str = "agentx_preflight"

# -913 is ``_ray_serving._RAY_ACTOR_DIED_RC``.

# Sentinel ``returncode`` when the session wall-clock budget ran out mid-round and the tree was reaped.
SESSION_TIME_EXHAUSTED_RETURNCODE: int = -915

# -916 is ``_ray_serving._ACTOR_TIMEOUT_RC``.

# Sentinel ``returncode`` when the orchestrator cancelled the action this child was launched for -- a shutdown, or a
# budget that is spent.
ORCHESTRATOR_CANCELLED_RETURNCODE: int = -917

# Sentinel ``returncode`` and ``error_class`` of an AgentX round its checkpoint stopped for trailing the reference.
AGENTX_EARLY_STOP_RETURNCODE: int = -919
AGENTX_EARLY_STOP_ERROR_CLASS: str = "agentx_early_stop"

# The operator's two grants. They are the operator's to set, read from the process env and
# validated there once by preflight, so nothing re-reads them from an env a variant composed.
BENCHMARK_TIMEOUT_ENV = "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC"
BENCHMARK_SILENCE_TIMEOUT_ENV = "INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC"


def default_benchmark_timeout_sec(env: Mapping[str, str] | None = None) -> float:
    """The hard cap the workload this env names needs, before any operator override.

    A grant has to bound the work the invocation was composed to do, so an
    agentic run's cap is sized to its own workload: an MLPerf run works through a
    fixed trajectory count rather than a fixed window, and a canonical run warms
    up one set of requests per lane before it opens its window, so both grow with
    a size the caller pinned. The stock default reaped a healthy 150-trajectory
    baseline at 38%, and reaped a canonical CONC 20 rung 1.3s after its profiling
    phase completed.

    Sizing reads only workload pins -- the backend, the flow, ``CONC`` -- but it
    reads them as the run will: a pin it cannot parse raises
    :class:`~hyperloom_common.env.EnvValueError` here rather than sizing the
    grant for a workload the round will refuse to start.
    """
    from hyperloom.common.agentx_workload import (
        canonical_benchmark_timeout_sec,
        is_canonical_agentx,
        is_mlperf_backend,
        mlperf_benchmark_timeout_sec,
    )

    source = os.environ if env is None else env
    stock = 7800.0
    if is_mlperf_backend(source):
        return max(stock, mlperf_benchmark_timeout_sec(source))
    if is_canonical_agentx(source):
        return max(stock, canonical_benchmark_timeout_sec(source))
    return stock


def resolve_benchmark_timeouts(env: Mapping[str, str] | None = None) -> tuple[float, float]:
    """Resolve the invocation's silence and hard caps, rejecting invalid overrides.

    The defaults are what the workload needs; an explicit
    ``INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC`` still wins. Both names are the
    operator's, so ``env`` has to be an env the operator set: preflight validates
    the process env once and this rejects what it cannot parse. A per-variant env
    is sized with ``default_benchmark_timeout_sec`` instead.
    """
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
        positive(BENCHMARK_SILENCE_TIMEOUT_ENV, 600.0),
        positive(BENCHMARK_TIMEOUT_ENV, default_benchmark_timeout_sec(source)),
    )


def sync_agentx_phase_wait(envs: dict[str, Any], timeout_sec: float) -> None:
    """Move an exported ``AGENTX_PHASE_WAIT_TIMEOUT_S`` to the cap; the client accepts only an unsigned integer."""
    if "AGENTX_PHASE_WAIT_TIMEOUT_S" in envs:
        envs["AGENTX_PHASE_WAIT_TIMEOUT_S"] = str(math.ceil(timeout_sec))


class _StreamCapture:
    """Capture pipe bytes promptly, decoding text incrementally for capture and mirroring."""

    def __init__(
        self,
        proc: subprocess.Popen,
        *,
        on_output: Callable[[], None] | None = None,
    ) -> None:
        """Set up capture/mirror threads for a child's stdout and stderr."""
        self._on_output = on_output
        self.last_activity_at: float | None = None
        self._stdout_chunks: list[str] = []
        self._stderr_chunks: list[str] = []
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

    def finish(self) -> tuple[str, str]:
        """Join the capture threads and return the captured output."""
        for thread in self._threads:
            thread.join(timeout=_CAPTURE_DRAIN_SECONDS)
        return "".join(self._stdout_chunks), "".join(self._stderr_chunks)

    def note_output(self) -> None:
        """Report one unit of child output to the caller's liveness callback."""
        if self._on_output is None:
            return
        self._on_output()

    def _pump(self, pipe, chunks: list[str], mirror) -> None:
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
                chunk = decoder.decode(data)
                if chunk:
                    chunks.append(chunk)
                    self._mirror(chunk, mirror)
            final = decoder.decode(b"", final=True)
            if final:
                chunks.append(final)
                self._mirror(final, mirror)
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    def _mirror(self, chunk: str, mirror) -> None:
        """Echo a captured chunk to the parent stream, ignoring errors."""
        try:
            mirror.write(chunk)
            mirror.flush()
        except (OSError, ValueError):
            pass  # a closed or broken mirror stream must not stop the capture


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


class RoundJudge(Protocol):
    """Rules on a running round at each tick, raising :class:`AgentxCheckpointMissed` to stop it."""

    def tick(self, mono: float) -> None: ...

    def close(self, *, aborted: bool = False) -> None: ...


class RoundCheckpoint(Protocol):
    """A ruling made while the round runs, by the judge it builds when the round starts."""

    def judge(self) -> RoundJudge:
        """The judge that rules on this round."""


@dataclass(frozen=True)
class RoundWatch:
    """The stop gates the watchdog enforces on one round."""

    timeout: int | float | None = None
    session_deadline_sec: float | None = None
    checkpoint: RoundCheckpoint | None = None


def run_with_session_kill(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    watch: RoundWatch = RoundWatch(),
    on_output: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess in its own session and reap descendants, and what carries ``REAP_MARK_ENV``, on every exit path."""
    child_env = dict(os.environ if env is None else env)
    mark = child_env.get(REAP_MARK_ENV, "")
    proc: subprocess.Popen | None = None
    try:
        with cancel_scope_listener() as cancel_scope:
            started_at = time.monotonic()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=child_env,
                cwd=cwd,
                start_new_session=True,
            )
            capture = _StreamCapture(proc, on_output=on_output)
            capture.start()
            try:
                stdout, stderr = _communicate_with_watchdog(
                    proc,
                    _WatchedRound(watch, started_at=started_at),
                    capture=capture,
                    cancel_scope=cancel_scope,
                )
            except subprocess.TimeoutExpired as exc:
                kill_my_spawned_server(proc, mark=mark)
                exc.output, exc.stderr = capture.finish()
                raise
            except _ReapedByWatchdog as exc:
                kill_my_spawned_server(proc, mark=mark)
                stdout, stderr = capture.finish()
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
            return subprocess.CompletedProcess(args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    finally:
        kill_my_spawned_server(proc, mark=mark)


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


class _OrchestratorCancelled(_ReapedByWatchdog):
    """Internal sentinel: the orchestrator cancelled the action this child serves."""

    returncode = ORCHESTRATOR_CANCELLED_RETURNCODE

    def __init__(self, *, reason: str, elapsed_sec: float) -> None:
        """Record who asked for the stop and how far the round had got."""
        super().__init__(
            f"the orchestrator cancelled this action ({reason or 'no reason given'}; round elapsed={elapsed_sec:.1f}s)"
        )


class AgentxCheckpointMissed(_ReapedByWatchdog):
    """The round's checkpoint median trailed its reference by more than the allowed drop."""

    returncode = AGENTX_EARLY_STOP_RETURNCODE
    log_level = logging.INFO


@dataclass(frozen=True)
class _WatchedRound:
    """One round's watch; ``started_at`` defaults to the moment the watch begins."""

    watch: RoundWatch
    started_at: float | None = None

    def enforce_stop_gates(self, proc: subprocess.Popen, scope: CancelScope | None, now: float, elapsed: float) -> None:
        """Raise as soon as the session budget, an orchestrator cancellation or the round cap is spent."""
        if self.watch.session_deadline_sec is not None and now >= self.watch.session_deadline_sec:
            raise _SessionDeadlineExceeded(overrun_sec=now - self.watch.session_deadline_sec, elapsed_sec=elapsed)
        if scope is not None and scope.cancelled:
            raise _OrchestratorCancelled(reason=scope.reason, elapsed_sec=elapsed)
        if self.watch.timeout is not None and elapsed >= self.watch.timeout:
            raise subprocess.TimeoutExpired(proc.args, self.watch.timeout)

    def poll_slice_sec(self, now: float, elapsed: float) -> float:
        """How long to wait on the child before the nearest armed gate has to be re-checked."""
        hard = None if self.watch.timeout is None else self.watch.timeout - elapsed
        session = None if self.watch.session_deadline_sec is None else self.watch.session_deadline_sec - now
        return min(sec for sec in (STOP_GATE_POLL_SECONDS, hard, session) if sec is not None)


def _communicate_with_watchdog(
    proc: subprocess.Popen,
    watched: _WatchedRound,
    *,
    capture: _StreamCapture,
    cancel_scope: CancelScope | None = None,
) -> tuple[str, str]:
    """Wait for exit, enforcing cancellation, session budget, explicit round caps and the round's checkpoint."""
    start = time.monotonic() if watched.started_at is None else watched.started_at
    judge = None if watched.watch.checkpoint is None else watched.watch.checkpoint.judge()
    completed = False
    try:
        while True:
            # An exit already observed belongs to the child, not to an expired gate.
            if proc.poll() is not None:
                result = capture.finish()
                completed = True
                return result
            now = time.monotonic()
            elapsed = now - start
            watched.enforce_stop_gates(proc, cancel_scope, now, elapsed)
            if judge is not None:
                judge.tick(now)
            try:
                proc.wait(timeout=max(0.0, watched.poll_slice_sec(now, elapsed)))
                result = capture.finish()
                completed = True
                return result
            except subprocess.TimeoutExpired:
                continue
    finally:
        if judge is not None:
            judge.close(aborted=not completed)


__all__ = [
    "AGENTX_EARLY_STOP_ERROR_CLASS",
    "AGENTX_EARLY_STOP_RETURNCODE",
    "AGENTX_PREFLIGHT_ERROR_CLASS",
    "AGENTX_PREFLIGHT_RETURNCODE",
    "BENCHMARK_SILENCE_TIMEOUT_ENV",
    "BENCHMARK_TIMEOUT_ENV",
    "AgentxCheckpointMissed",
    "COOPERATIVE_REAP_BUDGET_SEC",
    "ORCHESTRATOR_CANCELLED_RETURNCODE",
    "RoundCheckpoint",
    "RoundJudge",
    "RoundWatch",
    "SESSION_TIME_EXHAUSTED_RETURNCODE",
    "STOP_GATE_POLL_SECONDS",
    "default_benchmark_timeout_sec",
    "kill_my_spawned_server",
    "resolve_benchmark_timeouts",
    "run_with_session_kill",
    "session_deadline_to_remaining_sec",
    "session_remaining_to_deadline_sec",
    "sync_agentx_phase_wait",
]
