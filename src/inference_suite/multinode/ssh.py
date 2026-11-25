# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Non-interactive SSH to the hosts of a PD cluster, with strict host-key checking.

A host's key is learned with ``ssh-keyscan`` on first contact into a known_hosts file of the session (or the one the
node list names) and pinned there; a pinned host whose key no longer verifies is refused, never rescanned. The
private key is passed by path; nothing here reads or logs its content. The hardened options, the host-key error
markers and the key scan are shared with the Infera control plane.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from inference_suite.multinode.nodes import SshAccess

#: stderr fragments that mean a pinned host key no longer verifies, lowercase.
HOST_KEY_ERRORS = (
    "host key verification failed",
    "remote host identification has changed",
    "offending ed25519 key",
    "offending rsa key",
)
_KEYSCAN_TIMEOUT_S = 30
_DEFAULT_SSH_PORT = 22


class SshError(RuntimeError):
    """A host cannot be reached, its host key cannot be learned, or its pinned host key no longer verifies."""


@dataclass(frozen=True)
class SshResult:
    """What a remote script returned."""

    returncode: int
    stdout: str
    stderr: str


#: Runs ``argv`` with ``stdin`` and a timeout; the seam tests replace.
Runner = Callable[[Sequence[str], str, int], SshResult]


def run_process(argv: Sequence[str], stdin: str, timeout: int) -> SshResult:
    """``argv`` as a subprocess; a timeout is exit code 124."""
    try:
        done = subprocess.run(list(argv), input=stdin, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return SshResult(124, "", f"timed out after {timeout}s")
    return SshResult(done.returncode, done.stdout or "", done.stderr or "")


def ssh_options(known_hosts: Path) -> list[str]:
    """The hardened, non-interactive ``-o`` options that pin host keys in ``known_hosts``."""
    options = (
        "StrictHostKeyChecking=yes",
        f"UserKnownHostsFile={known_hosts}",
        "UpdateHostKeys=no",
        "PasswordAuthentication=no",
        "BatchMode=yes",
        "ConnectTimeout=15",
        "ServerAliveInterval=30",
        "ServerAliveCountMax=3",
        "LogLevel=ERROR",
    )
    return [flag for option in options for flag in ("-o", option)]


def is_host_key_error(stderr: str) -> bool:
    """Whether ``stderr`` reports a host key that no longer verifies."""
    text = stderr.lower()
    return any(marker in text for marker in HOST_KEY_ERRORS)


def keyscan(host: str, port: int, runner: Runner = run_process) -> str:
    """The host keys ``ssh-keyscan`` returns for ``host`` on ``port``, hashed as known_hosts lines.

    Raises:
        SshError: ``ssh-keyscan`` returns no key for the host.
    """
    scanned = runner(["ssh-keyscan", "-p", str(port), "-H", host], "", _KEYSCAN_TIMEOUT_S)
    if scanned.returncode != 0 or not scanned.stdout.strip():
        raise SshError(f"ssh-keyscan returned no host key for {host}:{port}")
    return scanned.stdout if scanned.stdout.endswith("\n") else f"{scanned.stdout}\n"


@dataclass(frozen=True)
class SshClient:
    """Runs bash scripts on the hosts ``access`` reaches, host keys pinned in ``known_hosts``."""

    access: SshAccess
    known_hosts: Path
    runner: Runner | None = None

    def _exec(self, argv: Sequence[str], stdin: str, timeout: int) -> SshResult:
        """Run ``argv`` through the runner, the subprocess runner unless one is given."""
        run = self.runner if self.runner is not None else run_process
        return run(argv, stdin, timeout)

    def _argv(self, host: str) -> list[str]:
        argv = ["ssh", *ssh_options(self.known_hosts), "-o", "IdentitiesOnly=yes"]
        return [*argv, "-i", self.access.key_file, "-p", str(self.access.port_of(host)), f"{self.access.user}@{host}"]

    def _pinned(self, host: str) -> bool:
        """Whether ``known_hosts`` holds a key for ``host`` on its port, hashed entries included."""
        port = self.access.port_of(host)
        name = host if port == _DEFAULT_SSH_PORT else f"[{host}]:{port}"
        found = self._exec(["ssh-keygen", "-F", name, "-f", str(self.known_hosts)], "", _KEYSCAN_TIMEOUT_S)
        return found.returncode == 0 and bool(found.stdout.strip())

    def learn(self, hosts: Sequence[str]) -> None:
        """Append the host keys of ``hosts`` not yet pinned to ``known_hosts``; a pinned host is left as pinned.

        Raises:
            SshError: ``ssh-keyscan`` returns no key for a host.
        """
        self.known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.known_hosts.touch(mode=0o600, exist_ok=True)
        for host in hosts:
            if self._pinned(host):
                continue
            port = self.access.port_of(host)
            scanned = keyscan(host, port, self._exec)
            with self.known_hosts.open("a", encoding="utf-8") as pinned:
                pinned.write(scanned)

    def run(self, host: str, script: str, timeout: int) -> SshResult:
        """``script`` run by ``bash -s`` on ``host``, learning its host key first when none is pinned.

        Raises:
            SshError: the host's pinned key no longer verifies, or its key cannot be learned.
        """
        body = f"set -uo pipefail\n{script}\n"
        result = self._exec([*self._argv(host), "bash", "-s"], body, timeout)
        if result.returncode == 0 or not is_host_key_error(result.stderr):
            return result
        if self._pinned(host):
            raise SshError(
                f"the host key of {host} changed from the one pinned in {self.known_hosts}; remove the pinned key "
                "there to trust a new one"
            )
        self.learn([host])
        return self._exec([*self._argv(host), "bash", "-s"], body, timeout)


__all__ = [
    "HOST_KEY_ERRORS",
    "Runner",
    "SshClient",
    "SshError",
    "SshResult",
    "is_host_key_error",
    "keyscan",
    "run_process",
    "ssh_options",
]
