# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs one real ``envd`` per sandbox, each in private pid and mount namespaces over a root directory of its own.

``envd`` is the process E2B puts inside a sandbox: it serves commands and files over HTTP. Here it runs as an ordinary
root process of this machine, started under ``unshare`` and ``chroot`` so that

- every sandbox sees only a tmpfs root of its own (``/work``, ``/app/work``, ``/in``, ``/out``, ``/tmp``) plus read-only
  binds of the system directories and of whatever the template names, so one sandbox cannot read another's files;
- killing the process the stand-in started ends everything inside the sandbox, a ``setsid`` child included, because the
  pid namespace's init dies with it;
- no mount ever reaches this machine's namespace, and nothing is written outside the stand-in's own directories.

Every process this module starts is recorded in :attr:`EnvdHost.pids`, and only those are ever signalled; a sandbox
that outlives its stand-in (a crashed test run) is killed by ``timeout`` after two hours. ``envd`` is run with
``-isnotfc -no-cgroups``: it must not create cgroups on a host whose cgroup root is shared. It listens on every
interface of this machine's network namespace and only its access token keeps other local users out.
"""

from __future__ import annotations

import http.client
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ENVD_ENV",
    "EnvdHost",
    "RunningSandbox",
    "SandboxPlan",
    "interpreter_binds",
    "unavailable_reason",
    "without_network",
]

#: Names the environment variable that points at an envd binary (``packages/envd`` of e2b-dev/infra built with
#: ``GOWORK=off CGO_ENABLED=0 go build``, or an official release binary).
ENVD_ENV = "HYPERLOOM_E2B_ENVD"

_SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
#: What a sandbox's own /etc and /dev hold: copies of the few files programs need to start, and the harmless devices.
#: Nothing else of this machine's /etc (keys, shadow, credentials) or /dev is shown.
_ETC_FILES = ("passwd", "group", "nsswitch.conf", "hosts", "resolv.conf", "ld.so.cache", "ld.so.conf", "ld.so.conf.d")
_DEVICES = ("null", "zero", "full", "random", "urandom", "tty")
_EMPTY_DIRS = ("proc", "dev", "dev/shm", "etc", "tmp", "run", "work", "app/work", "in", "out", "root", "var/tmp")
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_MAX_LIFETIME_SEC = 7200
_HEALTH_WAIT_SEC = 20.0
_KILL_WAIT_SEC = 10.0


def unavailable_reason(envd: str | None) -> str | None:
    """Why the stand-in cannot run here (a sentence for a skip message), or None when it can."""
    if sys.platform != "linux":
        return "the E2B stand-in runs envd under unshare and needs Linux"
    if os.geteuid() != 0:
        return "the E2B stand-in needs root: envd switches the account of every command it starts"
    if any(shutil.which(program) is None for program in ("unshare", "chroot", "timeout")):
        return "the E2B stand-in needs the unshare, chroot and timeout programs"
    if not envd:
        return f"set {ENVD_ENV} to an envd binary to run the E2B stand-in"
    if not (Path(envd).is_file() and os.access(envd, os.X_OK)):
        return f"{ENVD_ENV}={envd} is not an executable file"
    probe = subprocess.run(
        ["unshare", "--mount", "--pid", "--fork", "--kill-child", "true"], capture_output=True, timeout=30, check=False
    )
    if probe.returncode != 0:
        return f"unshare of mount and pid namespaces is refused here: {probe.stderr.decode(errors='replace').strip()}"
    return None


def interpreter_binds() -> tuple[str, ...]:
    """The directories a sandbox must show for this Python to start there: its prefixes and its symlink chain."""
    binds = [sys.prefix, sys.base_prefix, os.path.realpath(sys.base_prefix)]
    path = sys.executable
    while os.path.islink(path):
        path = os.path.normpath(os.path.join(os.path.dirname(path), os.readlink(path)))
        binds.append(os.path.dirname(path))
    return tuple(dict.fromkeys(binds))


def without_network(command: Sequence[str]) -> tuple[str, ...]:
    """``command`` run in a network namespace of its own: no interface, not even a loopback that is up, so nothing it
    starts can open a connection.

    The sandbox's ``envd`` stays in the namespace it was started in, so the data plane keeps working. A template whose
    sandboxes have no egress (the ``measurement`` role's) wraps its start command in this.
    """
    return ("unshare", "--net", "--", *command)


@dataclass(frozen=True)
class SandboxPlan:
    """What one sandbox's init does: where its root is, what it binds in, what it starts and its envd's port.

    Attributes:
        root: An empty directory on this machine that becomes the sandbox's tmpfs root (inside its namespace only).
        binds: Host directories shown read-only at the same path, on top of the system directories.
        start: The template's start command, run inside the sandbox next to envd (empty: none).
        start_env: Environment of the start command.
        log: Where the init's and envd's output goes.
        port: The port envd listens on.
    """

    root: Path
    binds: tuple[str, ...]
    start: tuple[str, ...]
    start_env: Mapping[str, str]
    log: Path
    port: int


@dataclass
class RunningSandbox:
    """A sandbox's init process (``unshare``), as this machine sees it."""

    plan: SandboxPlan
    process: subprocess.Popen[bytes]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _bind_lines(directory: str) -> list[str]:
    dest = f'"$R"{shlex.quote(directory)}'
    if os.path.islink(directory):
        parent = shlex.quote(os.path.dirname(directory))
        return [f'mkdir -p "$R"{parent}', f"ln -s {shlex.quote(os.readlink(directory))} {dest}"]
    return [f"mkdir -p {dest}", f"mount --bind {shlex.quote(directory)} {dest}", f"mount -o remount,bind,ro {dest}"]


def _init_script(plan: SandboxPlan, envd: Path) -> str:
    q = shlex.quote
    binds = [d for d in dict.fromkeys((*_SYSTEM_DIRS, *plan.binds)) if os.path.lexists(d)]
    lines = ["set -e", f"R={q(str(plan.root))}", 'mount -t tmpfs -o mode=755 tmpfs "$R"']
    for directory in binds:
        lines += _bind_lines(directory)
    lines += [f'mkdir -p "$R"/{d}' for d in _EMPTY_DIRS]
    lines += ['mount -t proc proc "$R/proc"', 'mount -t tmpfs tmpfs "$R/dev/shm"', 'chmod 1777 "$R/tmp"']
    lines += [f'cp -rL /etc/{f} "$R/etc/{f}"' for f in _ETC_FILES if os.path.exists(f"/etc/{f}")]
    for device in (d for d in _DEVICES if os.path.exists(f"/dev/{d}")):
        lines += [f': > "$R/dev/{device}"', f'mount --bind /dev/{device} "$R/dev/{device}"']
    lines += [': > "$R/envd"', f'mount --bind {q(str(envd))} "$R/envd"']
    if plan.start:
        env = " ".join(f"{q(k)}={q(v)}" for k, v in plan.start_env.items())
        lines.append(f'(chroot "$R" env {env} {" ".join(q(a) for a in plan.start)}) >"$R/run/template.log" 2>&1 &')
    lines.append(f'exec chroot "$R" /envd -isnotfc -no-cgroups -port {plan.port}')
    return "\n".join(lines)


def _health(port: int) -> bool:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        conn.request("GET", "/health")
        return conn.getresponse().status == 204
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


class EnvdHost:
    """Starts and stops sandboxes; remembers the PID of every init it starts.

    Args:
        envd: The envd binary.
        workdir: A directory the host may fill with sandbox roots and logs; it is left empty by :meth:`stop`.
    """

    def __init__(self, envd: Path, workdir: Path) -> None:
        self._envd = envd
        self._workdir = workdir
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self.version = subprocess.run(
            [str(envd), "-version"], capture_output=True, text=True, check=True
        ).stdout.strip()

    def start(
        self,
        name: str,
        *,
        binds: Sequence[str],
        start: Sequence[str],
        start_env: Mapping[str, str],
        init: Mapping[str, object],
    ) -> RunningSandbox:
        """Start a sandbox's init, wait until its envd is healthy and send it ``init`` (``POST /init``).

        Raises:
            RuntimeError: When envd does not become healthy; the init is killed first.
        """
        root = self._workdir / f"{name}-root"
        root.mkdir()
        plan = SandboxPlan(
            root, tuple(binds), tuple(start), dict(start_env), self._workdir / f"{name}.log", _free_port()
        )
        env = {"PATH": _PATH, "HOME": "/root", "LANG": "C.UTF-8"}
        script = _init_script(plan, self._envd)
        unshare = ["unshare", "--mount", "--pid", "--fork", "--kill-child", "sh", "-c", script]
        argv = ["timeout", "-s", "KILL", str(_MAX_LIFETIME_SEC), *unshare]
        with plan.log.open("wb") as log:
            process = subprocess.Popen(
                argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True
            )
        with self._lock:
            self._processes[process.pid] = process
        sandbox = RunningSandbox(plan, process)
        try:
            self._await_healthy(sandbox)
            self._init(plan.port, init)
        except BaseException:
            self.stop(sandbox)
            raise
        return sandbox

    @property
    def pids(self) -> list[int]:
        """The PID of every init this host ever started."""
        with self._lock:
            return list(self._processes)

    def alive(self) -> list[int]:
        """The recorded PIDs whose process has not ended."""
        with self._lock:
            return [pid for pid, process in self._processes.items() if process.poll() is None]

    def _await_healthy(self, sandbox: RunningSandbox) -> None:
        deadline = time.monotonic() + _HEALTH_WAIT_SEC
        while time.monotonic() < deadline:
            if sandbox.process.poll() is not None:
                break
            if _health(sandbox.plan.port):
                return
            time.sleep(0.02)
        tail = sandbox.plan.log.read_text(errors="replace")[-2000:]
        raise RuntimeError(f"envd did not become healthy (init exited with {sandbox.process.poll()}): {tail}")

    @staticmethod
    def _init(port: int, body: Mapping[str, object]) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            conn.request("POST", "/init", json.dumps(body), {"Content-Type": "application/json"})
            response = conn.getresponse()
            response.read()
            if response.status >= 300:
                raise RuntimeError(f"envd refused /init with {response.status}")
        finally:
            conn.close()

    def stop(self, sandbox: RunningSandbox) -> None:
        """Kill the sandbox's init, which ends every process inside it, and remove its directories.

        Raises:
            ValueError: When the init was not started by this host; nothing is signalled then.
        """
        process = sandbox.process
        with self._lock:
            if self._processes.get(process.pid) is not process:
                raise ValueError(f"process {process.pid} was not started by this host")
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=_KILL_WAIT_SEC)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"sandbox init {process.pid} outlived SIGKILL") from exc
        sandbox.plan.root.rmdir()
        sandbox.plan.log.unlink(missing_ok=True)
