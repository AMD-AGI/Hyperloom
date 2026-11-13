# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Non-interactive SSH to the hosts of a PD cluster, with strict host-key checking.

Host keys are learned once per session with ``ssh-keyscan`` into a known_hosts file of the session (or the one the
node list names); a host whose key changed is rescanned once and the command retried. The private key is passed by
path; nothing here reads or logs its content.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from inference_suite.multinode.nodes import SshAccess

_HOST_KEY_ERRORS = (
    "host key verification failed",
    "remote host identification has changed",
    "offending ed25519 key",
    "offending rsa key",
)
_KEYSCAN_TIMEOUT_S = 30


class SshError(RuntimeError):
    """A host cannot be reached, or its host key cannot be learned."""


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


@dataclass(frozen=True)
class SshClient:
    """Runs bash scripts on the hosts ``access`` reaches, host keys pinned in ``known_hosts``."""

    access: SshAccess
    known_hosts: Path
    runner: Runner = run_process

    def _argv(self, host: str) -> list[str]:
        options = ("StrictHostKeyChecking=yes", f"UserKnownHostsFile={self.known_hosts}", "UpdateHostKeys=no")
        options += ("PasswordAuthentication=no", "BatchMode=yes", "ConnectTimeout=15", "ServerAliveInterval=30")
        options += ("ServerAliveCountMax=3", "LogLevel=ERROR", "IdentitiesOnly=yes")
        argv = ["ssh"]
        for option in options:
            argv += ["-o", option]
        return [*argv, "-i", self.access.key_file, "-p", str(self.access.port_of(host)), f"{self.access.user}@{host}"]

    def learn(self, hosts: Sequence[str]) -> None:
        """Append the host keys of ``hosts`` not yet pinned to ``known_hosts``.

        Raises:
            SshError: ``ssh-keyscan`` returns no key for a host.
        """
        self.known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.known_hosts.touch(mode=0o600, exist_ok=True)
        for host in hosts:
            port = self.access.port_of(host)
            scanned = self.runner(["ssh-keyscan", "-p", str(port), "-H", host], "", _KEYSCAN_TIMEOUT_S)
            if scanned.returncode != 0 or not scanned.stdout.strip():
                raise SshError(f"ssh-keyscan returned no host key for {host}:{port}")
            with self.known_hosts.open("a", encoding="utf-8") as pinned:
                pinned.write(scanned.stdout if scanned.stdout.endswith("\n") else f"{scanned.stdout}\n")

    def run(self, host: str, script: str, timeout: int) -> SshResult:
        """``script`` run by ``bash -s`` on ``host``, after one rescan when the host key changed."""
        body = f"set -uo pipefail\n{script}\n"
        result = self.runner([*self._argv(host), "bash", "-s"], body, timeout)
        if result.returncode == 0 or not any(marker in result.stderr.lower() for marker in _HOST_KEY_ERRORS):
            return result
        self.learn([host])
        return self.runner([*self._argv(host), "bash", "-s"], body, timeout)


__all__ = ["Runner", "SshClient", "SshError", "SshResult", "run_process"]
