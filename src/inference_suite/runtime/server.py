# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round's server: own, boot or reuse it, and :func:`stop` one a round left running.

``own`` and ``boot`` first refuse a port something listens on and GPUs that are not visible or not free. ``own``
starts the server, logs to the run_dir's ``server.log``, keeps ``server.pid`` there while it runs (so a host
can still stop it if the round's process dies) and tears it down at release. ``boot`` starts it with its live log and
pidfile in ``pid_dir`` and leaves it running. ``reuse`` runs against the server ``pid_dir`` records, once it has shown
it owns that server. ``external`` runs against a server something else starts and stops. A boot or reuse round
copies the bytes the server logged during the round into its run_dir's ``server.log`` and records their range.

Ready means three things at once: ``/health`` answers 200, the process is alive, and the log shows a ready marker.
"""

from __future__ import annotations

import http.client
import os
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common.proctree import proc_identity

from inference_suite.contracts.config import LifecycleMode, ServerConfig, SuiteConfig
from inference_suite.contracts.layout import SERVER_LOG, SERVER_PID, ReadyStamp, RunLayout
from inference_suite.contracts.report import FailedStep, LogRange
from inference_suite.frameworks import Framework
from inference_suite.runtime.gpus import unavailable_gpus
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.process import (
    Reaped,
    ServerPidfile,
    Spawned,
    group_listens,
    listening_inodes,
    reap,
    spawn,
)
from inference_suite.runtime.watch import POLL_SEC, Cancel, RoundGate, ServerWatch, Stop

_HEALTH_PATH = "/health"
_HEALTH_TIMEOUT_SEC = 5.0
_HTTP_OK = 200
_COPY_CHUNK = 1 << 20


@dataclass(frozen=True)
class ServerLaunch:
    """The server process a round starts: its final argv and env, and the launch digest that names it."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    framework: Framework
    digest: str


@dataclass(frozen=True)
class Released:
    """What releasing the round's server left: the ``server.log`` range copied in, and a teardown that fell short."""

    log_range: LogRange | None
    teardown: Stop | None


class Server:
    """The live server a round's load and eval run against."""

    def __init__(self, base_url: str, reused: bool, watch: ServerWatch | None, alive: Callable[[], bool]) -> None:
        self.base_url = base_url
        self.reused = reused
        self._watch = watch
        self._alive = alive

    def check(self, now: float) -> Stop | None:
        """Why the round has to stop because of its server, if it has to; a server the round does not run passes."""
        if self._watch is None:
            return None
        self._watch.advance(now)
        return self._watch.stalled(now) if self._alive() else self._watch.died()

    def note_output(self, now: float) -> None:
        if self._watch is not None:
            self._watch.note_output(now)


def _healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + _HEALTH_PATH, timeout=_HEALTH_TIMEOUT_SEC) as response:  # nosec B310
            return response.status == _HTTP_OK
    except (OSError, http.client.HTTPException):
        return False


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _copy_range(source: Path, start: int, end: int, target: Path) -> int:
    """Write ``source``'s bytes ``[start, end)`` to ``target``; return how many there were."""
    if end <= start:
        target.write_bytes(b"")
        return 0
    copied = 0
    with source.open("rb") as reader, target.open("wb") as writer:
        reader.seek(start)
        while copied < end - start:
            chunk = reader.read(min(_COPY_CHUNK, end - start - copied))
            if not chunk:
                break
            writer.write(chunk)
            copied += len(chunk)
    return copied


def _settle_pidfile(path: Path, outcome: Reaped) -> Stop | None:
    """Drop ``path`` once nothing it names is left; while a member outlives SIGKILL, keep it for the next stop."""
    if outcome.survivors:
        survivors = ", ".join(map(str, outcome.survivors))
        return Stop(FailedStep.TEARDOWN_INCOMPLETE, f"pids {survivors} outlived SIGKILL; {path} is kept")
    path.unlink(missing_ok=True)
    return None


def stop(pid_dir: Path | str) -> Reaped:
    """Reap the server ``pid_dir``'s ``server.pid`` records, and every process its run mark reaches.

    The recorded pid is signalled only while it is still the incarnation that was started; once that leader is gone,
    its workers are found by the mark alone. The pidfile is removed when nothing is left, and kept while a member
    outlives SIGKILL.
    """

    path = Path(pid_dir) / SERVER_PID
    record = ServerPidfile.read(path)
    if record is None:
        return Reaped((), ())
    outcome = reap((record.pid,) if record.leader_alive else (), record.run_mark)
    _settle_pidfile(path, outcome)
    return outcome


def server_url(server: ServerConfig) -> str:
    """The URL the round's server answers at; the client argv ``launch.json`` records and the one run share it."""
    return f"http://{server.host}:{server.port}"


class ServerLifecycle:
    """One round's server, from :meth:`acquire` to :meth:`release`; the round calls both, whatever happens between."""

    def __init__(self, config: SuiteConfig, launch: ServerLaunch, clock: PhaseClock, cancel: Cancel) -> None:
        self._config = config
        self._launch = launch
        self._clock = clock
        self._layout = RunLayout(Path(config.paths.run_dir))
        pid_dir = config.lifecycle.pid_dir
        home = self._layout.root if pid_dir is None else Path(pid_dir)
        self._pidfile = home / SERVER_PID
        self._log = home / SERVER_LOG
        self._log_start = _size(self._log)
        self._gate = RoundGate(config.deadlines, cancel, time.monotonic())
        self._spawned: Spawned | None = None
        self._teardown_stop: Stop | None = None

    @property
    def mode(self) -> LifecycleMode:
        return self._config.lifecycle.mode

    @property
    def base_url(self) -> str:
        return server_url(self._config.server)

    def acquire(self) -> Server | Stop:
        self._layout.root.mkdir(parents=True, exist_ok=True)
        if self.mode is LifecycleMode.EXTERNAL:
            return Server(self.base_url, True, None, lambda: True)
        if self.mode is LifecycleMode.REUSE:
            return self._reuse()
        return self._boot()

    def release(self) -> Released:
        """Tear down an own round's server, or copy a boot or reuse round's stretch of the live log into the run_dir."""
        if self._spawned is not None and self.mode is LifecycleMode.OWN:
            self._teardown_stop = self._teardown()
        own_log = self.mode in (LifecycleMode.OWN, LifecycleMode.EXTERNAL)
        return Released(None if own_log else self._copy_log(), self._teardown_stop)

    def _reuse(self) -> Server | Stop:
        record = ServerPidfile.read(self._pidfile)
        refusal = f"{self._pidfile} records no server" if record is None else self._reuse_refusal(record)
        if record is None or refusal is not None:
            return Stop(FailedStep.REUSE_NOT_OWNED, refusal or "")
        now = time.monotonic()
        self._clock.mark(Phase.READY, now)
        self._layout.write_ready_stamp(ReadyStamp(time.time(), 0.0))
        watch = ServerWatch(self._log, self._log_start, self._config.deadlines, now)
        return Server(self.base_url, True, watch, lambda: record.leader_alive)

    def _reuse_refusal(self, record: ServerPidfile) -> str | None:
        port = self._config.server.port
        if not record.leader_alive:
            return f"pid {record.pid} is no longer the process started at {record.start_time}"
        if record.port != port or not group_listens(record.pgid, port):
            return f"process group {record.pgid} does not hold the listener on port {port}"
        if record.launch_digest != self._launch.digest:
            return f"the server runs launch {record.launch_digest}, this round's is {self._launch.digest}"
        return None

    def _boot(self) -> Server | Stop:
        port = self._config.server.port
        if listening_inodes(port):
            return Stop(FailedStep.PORT_IN_USE, f"something already listens on port {port}")
        busy = unavailable_gpus(self._config.server.gpus)
        if busy is not None:
            return Stop(FailedStep.GPUS_UNAVAILABLE, busy)
        started = time.monotonic()
        self._clock.mark(Phase.SPAWN, started)
        spawned = self._spawned = spawn(
            self._launch.argv, self._server_env(), self._log, cwd=self._layout.framework_cwd
        )
        self._record(spawned)
        watch = ServerWatch(self._log, self._log_start, self._config.deadlines, started)
        stop = self._await_ready(spawned, watch)
        if stop is not None:
            self._teardown_stop = self._teardown()
            return stop
        ready = time.monotonic()
        self._clock.mark(Phase.READY, ready)
        self._layout.write_ready_stamp(ReadyStamp(time.time(), ready - started))
        watch.mark_ready(ready)
        return Server(self.base_url, False, watch, lambda: spawned.proc.poll() is None)

    def _server_env(self) -> dict[str, str]:
        inherited = {name: os.environ[name] for name in self._config.server.env_inherit if name in os.environ}
        return {**inherited, **self._launch.env}

    def _record(self, spawned: Spawned) -> None:
        identity = proc_identity(spawned.proc.pid)
        if identity is None:
            # Already gone: nothing to record, and the readiness wait reports the exit.
            return
        ServerPidfile(
            pid=spawned.proc.pid,
            # A new session makes the child its own process group's leader.
            pgid=spawned.proc.pid,
            start_time=identity[1],
            run_mark=spawned.mark,
            port=self._config.server.port,
            framework=self._launch.framework,
            launch_digest=self._launch.digest,
        ).write(self._pidfile)

    def _await_ready(self, spawned: Spawned, watch: ServerWatch) -> Stop | None:
        while True:
            now = time.monotonic()
            stop = self._gate.check(now)
            if stop is not None:
                return stop
            watch.advance(now)
            stop = watch.before_ready(now, spawned.proc.poll())
            if stop is not None:
                return stop
            if watch.ready_logged and _healthy(self.base_url) and spawned.proc.poll() is None:
                return None
            time.sleep(POLL_SEC)

    def _teardown(self) -> Stop | None:
        spawned, self._spawned = self._spawned, None
        if spawned is None:
            return None
        self._clock.mark(Phase.TEARDOWN)
        return _settle_pidfile(self._pidfile, spawned.reap())

    def _copy_log(self) -> LogRange:
        copied = _copy_range(self._log, self._log_start, _size(self._log), self._layout.server_log)
        return LogRange(self._log_start, self._log_start + copied)


__all__ = ["Released", "Server", "ServerLaunch", "ServerLifecycle", "server_url", "stop"]
