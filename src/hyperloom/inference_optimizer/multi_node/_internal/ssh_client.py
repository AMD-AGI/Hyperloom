"""SSH control plane for the Dynamo multi-node backend.

The Dynamo backend reuses the RayJob "long-lived pod + external server
restart" pattern, but the pods are a SaFE DynamoDeployment (LeaderWorkerSet
worker) deployed IDLE (``mn-idle.sh`` starts sshd on ``$MN_SSH_PORT``), and the
control plane is SSH instead of the Ray Dashboard REST API. ``restart-server``
fans out by SSHing into each LWS worker pod and (re)launching sglang/vllm —
each pod self-determines its rank via the LWS-injected ``$LWS_WORKER_INDEX`` /
``$LWS_LEADER_ADDRESS`` env vars (read from ``/proc/1/environ`` pod-side), so a
single command is issued identically to every worker.

No paramiko: we shell out to the system ``ssh`` / ``ssh-keygen`` (present in
the sandbox) to avoid a new transitive dependency on the optimizer image.

Keys are session-scoped and ephemeral:

* :func:`generate_session_keypair` makes an ed25519 pair under the session dir.
* The public key is injected into the workload body as ``MN_SSH_AUTHORIZED_KEY``
  (``mn-sshd-init.sh`` writes it to the pod's ``authorized_keys`` at start).
* The private key stays in the sandbox and is passed to :func:`ssh_run`.
* Host keys are recorded under ``known_hosts`` beside the keypair.
"""

from __future__ import annotations

import base64
import shlex
import subprocess
from pathlib import Path

from .env_safety import assert_env_key_shapes, assert_forward_env_keys
from .log import info, warn

# Default sshd port baked into the image's mn-sshd-init.sh (not 22, to avoid a
# hostNetwork collision with the node's own sshd).
DEFAULT_SSH_PORT = 2222

def _ssh_common_opts(known_hosts: Path) -> list[str]:
    """Build hardened non-interactive SSH options using a session known_hosts file.

    Args:
        known_hosts: Path to the session-scoped known_hosts file.

    Returns:
        list[str]: OpenSSH CLI option tokens.
    """
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


def generate_session_keypair(dest_dir: Path) -> tuple[Path, str]:
    """Generate (or reuse) an ed25519 keypair under ``dest_dir``.

    Returns ``(private_key_path, public_key_str)``. Idempotent: if the key
    already exists it is reused (so retries of ``create-dynamo`` keep the same
    authorized key that the running pods already trust).

    Args:
        dest_dir: Directory to create (or reuse) the keypair under.

    Returns:
        A ``(private_key_path, public_key_str)`` tuple.

    Raises:
        RuntimeError: If ``ssh-keygen`` exits non-zero.
    """
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
            "hyperloom-mn-dynamo",
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
    """Run ``command`` on ``host`` over SSH and return the CompletedProcess.

    ``command`` is run via ``bash -lc`` on the remote so login-shell PATH and
    the framework venv resolve. Does not raise on non-zero exit — the caller
    inspects ``.returncode`` / ``.stdout`` / ``.stderr``.

    Args:
        host: The target host/IP.
        command: The shell command to run remotely.
        key_path: Path to the private SSH key.
        known_hosts: Session known_hosts file for host-key verification.
        port: The remote sshd port.
        user: The remote login user.
        timeout: Subprocess timeout in seconds.

    Returns:
        The completed SSH subprocess (not raised on non-zero exit).
    """
    argv = [
        "ssh",
        *_ssh_common_opts(known_hosts),
        "-i",
        str(key_path),
        "-p",
        str(port),
        f"{user}@{host}",
        "bash",
        "-lc",
        shlex.quote(command),
    ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


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
    remote_path: str = "/tmp/mn_dynamo_launch",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Ship ``script_text`` to the pod (base64 over the command line) and run it.

    The script body is base64-encoded, decoded into ``remote_path`` on the pod,
    then executed with ``interpreter`` (e.g. ``python3`` / ``bash``) and
    ``script_args`` appended verbatim. ``env`` (optional) is prepended as
    ``KEY=VAL`` assignments before the interpreter so tuning vars reach the
    SSH-launched framework child.

    Args:
        host: The target host/IP.
        script_text: The script body to ship and run.
        interpreter: Interpreter used to run the script (e.g. ``python3``).
        script_args: Arguments appended verbatim after the script path.
        key_path: Path to the private SSH key.
        known_hosts: Session known_hosts file for host-key verification.
        port: The remote sshd port.
        user: The remote login user.
        timeout: Subprocess timeout in seconds.
        remote_path: Destination path for the decoded script on the pod.
        env: Optional ``KEY=VAL`` assignments prepended before the interpreter.

    Returns:
        The completed SSH subprocess.
    """
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
    """Run ``script_text`` on ``host`` via ``bash -s`` with ``env`` exported.

    The env (which may include credentials) is prepended as shell-quoted
    ``export`` lines and the whole script is piped over SSH **stdin**, so
    secrets never appear in argv or on the pod's disk.

    Args:
        host: The target host/IP.
        script_text: The script body run via ``bash -s``.
        env: Environment variables exported before the script, or ``None``.
        key_path: Path to the private SSH key.
        known_hosts: Session known_hosts file for host-key verification.
        port: The remote sshd port.
        user: The remote login user.
        timeout: Subprocess timeout in seconds.

    Returns:
        The completed SSH subprocess.
    """
    if env:
        assert_env_key_shapes(env)
    prologue = "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
    full = f"set -uo pipefail\n{prologue}\n{script_text}\n"
    argv = [
        "ssh",
        *_ssh_common_opts(known_hosts),
        "-i",
        str(key_path),
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


def probe_ssh(
    host: str,
    *,
    key_path: Path | str,
    known_hosts: Path,
    port: int = DEFAULT_SSH_PORT,
    user: str = "root",
    timeout: int = 20,
) -> bool:
    """Return True iff a trivial SSH command succeeds (pod reachable + key OK).

    Args:
        host: The target host/IP.
        key_path: Path to the private SSH key.
        known_hosts: Session known_hosts file for host-key verification.
        port: The remote sshd port.
        user: The remote login user.
        timeout: Subprocess timeout in seconds.

    Returns:
        ``True`` when a trivial probe command succeeds (pod reachable and key
        accepted), ``False`` otherwise (including on timeout).
    """
    try:
        cp = ssh_run(
            host,
            "echo mn_ssh_ok",
            key_path=key_path,
            known_hosts=known_hosts,
            port=port,
            user=user,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        warn(f"ssh probe to {host}:{port} timed out")
        return False
    ok = cp.returncode == 0 and "mn_ssh_ok" in (cp.stdout or "")
    if not ok:
        warn(f"ssh probe to {host}:{port} failed rc={cp.returncode} stderr={(cp.stderr or '').strip()[:200]}")
    return ok
