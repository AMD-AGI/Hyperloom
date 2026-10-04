"""SSH control plane for the Infera multi-node backend."""

from __future__ import annotations

import base64
import shlex
import subprocess
import tempfile
from pathlib import Path

from .env_safety import assert_env_key_shapes, assert_forward_env_keys
from .log import info, warn

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


def generate_session_keypair(dest_dir: Path) -> tuple[Path, str]:
    """Generate (or reuse) an ed25519 keypair under ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        dest_dir.chmod(0o700)
    except OSError as exc:
        warn(f"could not chmod SSH key dir {dest_dir} to 0700: {exc}")
    priv = dest_dir / "mn_id_ed25519"
    pub = dest_dir / "mn_id_ed25519.pub"
    if priv.is_file() and pub.is_file():
        return priv, pub.read_text(encoding="utf-8").strip()
    # Remove any half-written remnant before regenerating.
    for p in (priv, pub):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    proc = subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-q",
            "-C",
            "hyperloom-mn-infera",
            "-f",
            str(priv),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh-keygen failed rc={proc.returncode}: {proc.stderr.strip()}")
    priv.chmod(0o600)
    pub_str = pub.read_text(encoding="utf-8").strip()
    info(f"generated session SSH keypair at {priv}")
    return priv, pub_str


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
