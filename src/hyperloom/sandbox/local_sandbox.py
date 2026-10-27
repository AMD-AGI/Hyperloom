# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One local sandbox: a private directory tree plus the processes started in it."""

from __future__ import annotations

import logging
import os
import posixpath
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from hyperloom.sandbox.launcher import StatusPipe, launcher_argv
from hyperloom.sandbox.reaper import kill_launchers, reap_sandbox_processes
from hyperloom.sandbox.socket import (
    CommandResult,
    OutputSinks,
    RunSettings,
    Role,
    SandboxDenied,
    SandboxGone,
    SandboxInfo,
    SandboxUnavailable,
)
from hyperloom.sandbox.templates import RESERVED_ENV_NAMES, LocalTemplate

log = logging.getLogger(__name__)

__all__ = ["WORK_DIR", "LocalSandbox", "SandboxPlan", "SandboxRuntime"]

#: The working directory every sandbox starts in, under the root the file API shows (``/work``).
WORK_DIR = "work"

#: Parent environment names a sandbox process inherits; everything else (keys,
#: tokens, job topology) is dropped.
_INHERITED_ENV = ("PATH", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV")
_READER_JOIN_SEC = 2.0
_CHUNK = 1 << 20

_Sink = IO[bytes] | int | None


@dataclass(frozen=True)
class SandboxPlan:
    """What a sandbox is: its id, template, role, directory, metadata, environments and lifetime."""

    sandbox_id: str
    template: LocalTemplate
    role: Role
    directory: Path
    metadata: Mapping[str, str]
    create_envs: Mapping[str, str]
    platform_envs: Mapping[str, str]
    timeout_sec: float


@dataclass(frozen=True)
class SandboxRuntime:
    """How a sandbox's processes run: the end callback, the launcher's python and the account they run as."""

    on_end: Callable[[LocalSandbox, bool], None]
    launcher_python: str
    user: tuple[int, int] | None = None


class LocalSandbox:
    """A sandbox on this machine; every path it shows is mapped under its own root.

    Layout under the sandbox directory: ``fs/`` is the root callers see
    (``/in``, ``/out``, ``/work``), ``private/`` holds the signing key and process logs
    and is not reachable through the file API.
    """

    def __init__(self, plan: SandboxPlan, runtime: SandboxRuntime) -> None:
        self.sandbox_id = plan.sandbox_id
        self.template = plan.template
        self.role = plan.role
        self.directory = plan.directory
        self.fs = plan.directory / "fs"
        self.private = plan.directory / "private"
        self._metadata = dict(plan.metadata)
        self._base_env = self._build_base_env(plan.create_envs, plan.platform_envs)
        self._on_end = runtime.on_end
        self._launcher_python = runtime.launcher_python
        self._user = runtime.user
        self._lock = threading.RLock()
        self._ended = False
        self._procs: list[subprocess.Popen[bytes]] = []
        self._startup: tuple[subprocess.Popen[bytes], StatusPipe] | None = None
        self._startup_exit: int | None = None
        self._timer: threading.Timer | None = None
        self._bg_count = 0
        self.set_timeout(plan.timeout_sec)

    def _build_base_env(self, create_envs: Mapping[str, str], platform_envs: Mapping[str, str]) -> dict[str, str]:
        env = {k: os.environ[k] for k in _INHERITED_ENV if k in os.environ}
        env["HOME"] = str(self.fs / WORK_DIR)
        env.update(self.template.envs)
        env.update(create_envs)
        env.update(platform_envs)
        return env

    # -- lifecycle ---------------------------------------------------------

    @property
    def ended(self) -> bool:
        """bool: True once the sandbox was killed or timed out."""
        return self._ended

    def _require_alive(self) -> None:
        if self._ended:
            raise SandboxGone(f"sandbox {self.sandbox_id} has ended")

    def info(self) -> SandboxInfo:
        """Return the sandbox's id, template and metadata.

        Raises:
            SandboxGone: When the sandbox has ended.
        """
        self._require_alive()
        return SandboxInfo(self.sandbox_id, self.template.name, dict(self._metadata))

    def set_timeout(self, timeout_sec: float) -> None:
        """Restart the lifetime clock: the sandbox ends ``timeout_sec`` from now.

        Raises:
            SandboxGone: When the sandbox has ended.
            ValueError: When the timeout is not positive.
        """
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        with self._lock:
            self._require_alive()
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(timeout_sec, self._expire)
            self._timer.daemon = True
            self._timer.start()

    def _expire(self) -> None:
        try:
            self.kill()
        except SandboxUnavailable:
            log.exception("sandbox %s: reaping after its timeout failed", self.sandbox_id)

    def kill(self) -> None:
        """End the sandbox, reap every process it started, and free its resources.

        Safe to call again; a second call waits for the first to finish.

        Raises:
            SandboxUnavailable: When some process could not be confirmed gone.
                The GPU cards then stay leased rather than reach another test.
        """
        with self._lock:
            if self._ended:
                return
            self._ended = True
            if self._timer is not None:
                self._timer.cancel()
            if self._startup is not None:
                self._startup[1].close()
            reaped = reap_sandbox_processes(self._procs, self.sandbox_id)
            shutil.rmtree(self.directory, ignore_errors=True)
            self._on_end(self, reaped)
        if not reaped:
            raise SandboxUnavailable(f"sandbox {self.sandbox_id}: processes outlived SIGKILL; its GPUs stay leased")

    # -- files -------------------------------------------------------------

    def _virtual(self, path: str) -> str:
        virt = posixpath.normpath(path if path.startswith("/") else f"/{WORK_DIR}/{path}")
        return "/" + virt.lstrip("/")

    def _real(self, virt: str) -> Path:
        """Map a virtual absolute path under ``fs``, refusing anything that resolves outside it."""
        root = self.fs.resolve()
        real = (root / virt.lstrip("/")).resolve()
        if real != root and root not in real.parents:
            raise SandboxDenied(f"path {virt!r} resolves outside the sandbox")
        return real

    def _gate(self, path: str, *, write: bool) -> Path:
        self._require_alive()
        virt = self._virtual(path)
        if self.template.sealed:
            allowed = virt.startswith("/in/") if write else (virt == "/out" or virt.startswith("/out/"))
            if not allowed:
                verb = "write" if write else "read"
                raise SandboxDenied(f"sealed sandbox: cannot {verb} {virt!r} (write /in/, read /out/ only)")
        return self._real(virt)

    def write_file(self, path: str, data: bytes) -> None:
        """Write a file; a sealed sandbox only accepts paths under ``/in/``.

        Raises:
            SandboxDenied: By the sealed policy, or for a path that escapes the root.
            SandboxGone: When the sandbox has ended.
        """
        target = self._gate(path, write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target = self._gate(path, write=True)  # re-check after the parents exist
        target.write_bytes(data)

    def read_file(self, path: str) -> bytes:
        """Read a file; a sealed sandbox only serves ``/out/``.

        Raises:
            SandboxDenied: By the sealed policy, or for a path that escapes the root.
            SandboxGone: When the sandbox has ended.
            FileNotFoundError: When there is no such file.
        """
        return self._gate(path, write=False).read_bytes()

    def copy_out(self, path: str, dest: Path) -> int:
        """Stream a file to ``dest`` in chunks and return the byte count.

        Raises:
            SandboxDenied: By the sealed policy, or for a path that escapes the root.
            SandboxGone: When the sandbox has ended.
            FileNotFoundError: When there is no such file.
        """
        source = self._gate(path, write=False)
        total = 0
        with source.open("rb") as src, dest.open("wb") as out:
            while chunk := src.read(_CHUNK):
                out.write(chunk)
                total += len(chunk)
        return total

    def list_dir(self, path: str) -> list[str]:
        """List entry names of a directory, sorted.

        Raises:
            SandboxDenied: By the sealed policy, or for a path that escapes the root.
            SandboxGone: When the sandbox has ended.
            FileNotFoundError: When there is no such directory.
        """
        return sorted(entry.name for entry in self._gate(path, write=False).iterdir())

    # -- processes ---------------------------------------------------------

    def _start(
        self, argv: list[str], env: Mapping[str, str], cwd: Path, stdout: _Sink, stderr: _Sink
    ) -> tuple[subprocess.Popen[bytes], int]:
        """Start a process under a launcher, as the leader of its own session, and track it.

        Args:
            argv: The command line.
            env: Extra environment on top of the sandbox's.
            cwd: The working directory.
            stdout: Where its stdout goes.
            stderr: Where its stderr goes.

        Returns:
            tuple: The launcher process and the read end of its exit-code pipe.

        Raises:
            SandboxGone: When the sandbox ended before the process could start.
        """
        read_fd, write_fd = os.pipe()
        try:
            with self._lock:
                self._require_alive()
                extra: dict[str, object] = {}
                if self._user is not None:
                    extra = {"user": self._user[0], "group": self._user[1], "extra_groups": []}
                proc = subprocess.Popen(
                    launcher_argv(self._launcher_python, write_fd, argv),
                    env={**self._base_env, **env},
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    pass_fds=(write_fd,),
                    **extra,
                )
                self._procs.append(proc)
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        return proc, read_fd

    def spawn(
        self, argv: list[str], *, env: Mapping[str, str], cwd: Path, stdout: _Sink, stderr: _Sink
    ) -> subprocess.Popen[bytes]:
        """Start a process nobody waits for (its exit code is dropped); it is reaped with the sandbox."""
        proc, read_fd = self._start(argv, env, cwd, stdout, stderr)
        os.close(read_fd)
        return proc

    def spawn_watched(
        self, argv: list[str], *, env: Mapping[str, str], cwd: Path, stdout: _Sink, stderr: _Sink
    ) -> tuple[subprocess.Popen[bytes], StatusPipe]:
        """Start a process and also return the pipe that delivers its exit code."""
        proc, read_fd = self._start(argv, env, cwd, stdout, stderr)
        return proc, StatusPipe(read_fd)

    def start_template_command(self) -> None:
        """Launch the template's startup command, logging to the private directory."""
        if not self.template.command:
            return
        # The launcher would hide a missing executable behind exit code 127; fail the create instead.
        if shutil.which(self.template.command[0], path=self._base_env.get("PATH")) is None:
            raise FileNotFoundError(f"startup command {self.template.command[0]!r} not found or not executable")
        log_file = (self.private / "runner.log").open("ab")
        try:
            self._startup = self.spawn_watched(
                list(self.template.command), env={}, cwd=self.fs / WORK_DIR, stdout=log_file, stderr=subprocess.STDOUT
            )
        finally:
            log_file.close()

    def startup_exit_code(self) -> int | None:
        """The exit code of the template's startup command once it has exited; ``None`` while it runs, when the template
        has none, or when its launcher died without reporting one."""
        with self._lock:
            if self._startup is not None and self._startup_exit is None and not self._ended:
                proc, status = self._startup
                try:
                    self._startup_exit = status.wait_exit(proc, 0)
                except TimeoutError:
                    pass
            return self._startup_exit

    def _check_run_args(self, envs: Mapping[str, str] | None) -> dict[str, str]:
        self._require_alive()
        if self.template.sealed:
            raise SandboxDenied("sealed sandbox: commands are not allowed")
        extra = dict(envs or {})
        reserved = RESERVED_ENV_NAMES & set(extra)
        if reserved:
            raise SandboxDenied(f"envs name platform variables {sorted(reserved)}")
        return extra

    def run(
        self,
        cmd: str,
        *,
        background: bool = False,
        settings: RunSettings = RunSettings(),
        sinks: OutputSinks = OutputSinks(),
    ) -> CommandResult:
        """Run a shell command inside the sandbox.

        A background command returns at once with exit code 0; its output goes
        to a private log and the sandbox reaps it on kill or timeout.

        Raises:
            SandboxDenied: In a sealed sandbox, or when ``envs`` names a platform variable.
            SandboxGone: When the sandbox has ended (also if it ends during the command).
            SandboxUnavailable: When ``timeout_sec`` expires; the command's tree is killed.
        """
        extra = self._check_run_args(settings.envs)
        workdir = self._real(self._virtual(settings.cwd)) if settings.cwd else self.fs / WORK_DIR
        argv = ["/bin/sh", "-c", cmd]
        if background:
            self._bg_count += 1
            with (self.private / f"background-{self._bg_count}.log").open("ab") as log_file:
                self.spawn(argv, env=extra, cwd=workdir, stdout=log_file, stderr=subprocess.STDOUT)
            return CommandResult(0, "", "")
        proc, status = self.spawn_watched(argv, env=extra, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return self._collect(proc, status, settings.timeout_sec, sinks.on_stdout, sinks.on_stderr)

    def _collect(
        self,
        proc: subprocess.Popen[bytes],
        status: StatusPipe,
        timeout_sec: float | None,
        on_stdout: Callable[[str], None] | None,
        on_stderr: Callable[[str], None] | None,
    ) -> CommandResult:
        out: list[str] = []
        err: list[str] = []
        readers = [
            threading.Thread(target=_drain, args=(proc.stdout, out, on_stdout), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err, on_stderr), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            code = status.wait_exit(proc, timeout_sec)
        except TimeoutError:
            status.close()
            kill_launchers([proc], grace_sec=1.0)
            raise SandboxUnavailable(f"command timed out after {timeout_sec}s") from None
        finally:
            join_by = time.monotonic() + _READER_JOIN_SEC
            for reader in readers:
                reader.join(max(0.0, join_by - time.monotonic()))
        self._require_alive()
        if code is None:
            raise SandboxUnavailable("the command's launcher died before reporting an exit code")
        return CommandResult(code, "".join(out), "".join(err))


def _drain(stream: IO[bytes] | None, sink: list[str], callback: Callable[[str], None] | None) -> None:
    """Copy a pipe line by line into ``sink`` and the callback until EOF."""
    if stream is None:
        return
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", errors="replace")
        sink.append(line)
        if callback is not None:
            callback(line)
