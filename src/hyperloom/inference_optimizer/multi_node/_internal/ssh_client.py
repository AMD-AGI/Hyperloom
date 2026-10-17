"""SSH control plane for the Infera multi-node backend."""

from __future__ import annotations

import base64
import shlex
import subprocess
import tempfile
from pathlib import Path

from .env_safety import assert_env_key_shapes, assert_forward_env_keys

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


def ssh_run(
    host: str,
    command: str,
    *,
    key_path: Path | str,
    known_hosts: Path,
    port: int = DEFAULT_SSH_PORT,
    user: str = "root",
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run ``command`` on ``host`` over SSH and return the CompletedProcess."""
    argv = [
        "ssh",
        *_ssh_common_opts(known_hosts),
        "-o",
        "IdentitiesOnly=yes",
        "-i",
        _ssh_identity(key_path),
        "-p",
        str(port),
        f"{user}@{host}",
        "bash",
        "-lc",
        shlex.quote(command),
    ]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ssh_run_script(
    host: str,
    script_text: str,
    interpreter: str,
    script_args: str,
    *,
    key_path: Path | str,
    known_hosts: Path,
    port: int = DEFAULT_SSH_PORT,
    user: str = "root",
    timeout: int = 600,
    remote_path: str = str(Path(tempfile.gettempdir()) / "mn_infera_launch"),
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Ship ``script_text`` to the pod (base64 over the command line) and run it."""
    enc = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
    env_prefix = ""
    if env:
        assert_forward_env_keys(env)
        env_prefix = "".join(f"{k}={shlex.quote(str(v))} " for k, v in env.items())
    remote_cmd = (
        f"echo {enc} | base64 -d > {shlex.quote(remote_path)} && "
        f"{env_prefix}{interpreter} {shlex.quote(remote_path)} {script_args}"
    )
    return ssh_run(
        host,
        remote_cmd,
        key_path=key_path,
        known_hosts=known_hosts,
        port=port,
        user=user,
        timeout=timeout,
    )


def ssh_run_bash_with_env(
    host: str,
    script_text: str,
    env: dict[str, str] | None,
    *,
    key_path: Path | str,
    known_hosts: Path,
    port: int = DEFAULT_SSH_PORT,
    user: str = "root",
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run ``script_text`` on ``host`` via ``bash -s`` with ``env`` exported."""
    if env:
        assert_env_key_shapes(env)
    prologue = "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
    full = f"set -uo pipefail\n{prologue}\n{script_text}\n"
    argv = [
        "ssh",
        *_ssh_common_opts(known_hosts),
        "-o",
        "IdentitiesOnly=yes",
        "-i",
        _ssh_identity(key_path),
        "-p",
        str(port),
        f"{user}@{host}",
        "bash",
        "-s",
    ]
    return subprocess.run(
        argv,
        input=full,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
