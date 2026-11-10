"""SSH control plane for the Infera multi-node backend."""

from __future__ import annotations

import base64
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .env_safety import assert_forward_env_keys

# Default sshd port baked into the image's mn-sshd-init.sh.
DEFAULT_SSH_PORT = 2233


def _ssh_common_opts(known_hosts: Path) -> list[str]:
    """Build hardened non-interactive SSH options using a session known_hosts file."""
    return [
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "LogLevel=ERROR",
    ]


def _ssh_identity(key_path: Path | str) -> str:
    """Return the identity path to hand to ``ssh -i``."""
    return str(key_path)


@dataclass(frozen=True)
class SshTarget:
    """One pod reachable over SSH as root: its address, identity file and session known_hosts."""

    host: str
    key_path: Path | str
    known_hosts: Path
    port: int = DEFAULT_SSH_PORT

    def argv(self) -> list[str]:
        """The ``ssh`` argv up to and including ``root@host``."""
        return [
            "ssh",
            *_ssh_common_opts(self.known_hosts),
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            _ssh_identity(self.key_path),
            "-p",
            str(self.port),
            f"root@{self.host}",
        ]


@dataclass(frozen=True)
class RemoteScript:
    """A script shipped to a pod and run there as ``[env] interpreter remote_path args``."""

    text: str
    interpreter: str
    args: str
    env: dict[str, str] | None = None
    remote_path: str = str(Path(tempfile.gettempdir()) / "mn_infera_launch")


def ssh_run(target: SshTarget, command: str, *, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run ``command`` on the target over SSH and return the CompletedProcess."""
    argv = [*target.argv(), "bash", "-lc", shlex.quote(command)]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ssh_run_script(target: SshTarget, script: RemoteScript, *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Ship ``script`` to the pod (base64 over the command line) and run it."""
    enc = base64.b64encode(script.text.encode("utf-8")).decode("ascii")
    env_prefix = ""
    if script.env:
        assert_forward_env_keys(script.env)
        env_prefix = "".join(f"{k}={shlex.quote(str(v))} " for k, v in script.env.items())
    remote_path = shlex.quote(script.remote_path)
    remote_cmd = (
        f"echo {enc} | base64 -d > {remote_path} && {env_prefix}{script.interpreter} {remote_path} {script.args}"
    )
    return ssh_run(target, remote_cmd, timeout=timeout)


def ssh_run_bash_stdin(target: SshTarget, script_text: str, *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run ``script_text`` on the target by piping it to ``bash -s`` under ``set -uo pipefail``."""
    full = f"set -uo pipefail\n{script_text}\n"
    argv = [*target.argv(), "bash", "-s"]
    return subprocess.run(
        argv,
        input=full,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
