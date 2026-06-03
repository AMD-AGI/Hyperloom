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
T
* :func:`generate_session_keypair` makes an ed25519 pair under the session dir.
* The public key is injected into the workload body as ``MN_SSH_AUTHORIZED_KEY``
  (``mn-sshd-init.sh`` writes it to the pod's ``authorized_keys`` at start).
* The private key stays in the sandbox and is passed to :func:`ssh_run`.
"""

from __future__ import annotations

import base64
import shlex
import subprocess
from pathlib import Path

from .log import info, warn

# Default sshd port baked into the image's mn-sshd-init.sh. Not 22: the SaFE
# Dynamo pod template runs ClusterFirstWithHostNet and may be promoted to
# hostNetwork, where :22 collides with the node's own sshd.
DEFAULT_SSH_PORT = 2222

# ssh options for ephemeral, key-only, non-interactive automation against
# short-lived pod IPs. Host keys change per pod recreate, so we deliberately
# skip known_hosts (StrictHostKeyChecking=no + UserKnownHostsFile=/dev/null).
_SSH_COMMON_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "PasswordAuthentication=no",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-o", "LogLevel=ERROR",
]


def generate_session_keypair(dest_dir: Path) -> tuple[Path, str]:
    """Generate (or reuse) an ed25519 keypair under ``dest_dir``.

    Returns ``(private_key_path, public_key_str)``. Idempotent: if the key
    already exists it is reused (so retries of ``create-dynamo`` keep the same
    authorized key that the running pods already trust).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
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
            "ssh-keygen", "-t", "ed25519", "-N", "", "-q",
            "-C", "hyperloom-mn-dynamo",
            "-f", str(priv),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ssh-keygen failed rc={proc.returncode}: {proc.stderr.strip()}"
        )
    priv.chmod(0o600)
    pub_str = pub.read_text(encoding="utf-8").strip()
    info(f"generated session SSH keypair at {priv}")
    return priv, pub_str


def ssh_run(
    host: str,
    command: str,
    *,
    key_path: Path | str,
    port: int = DEFAULT_SSH_PORT,
    user: str = "root",
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run ``command`` on ``host`` over SSH and return the CompletedProcess.

    ``command`` is run via ``bash -lc`` on the remote so login-shell PATH and
    the framework venv resolve. Does not raise on non-zero exit — the caller
    inspects ``.returncode`` / ``.stdout`` / ``.stderr``.
    """
    argv = [
        "ssh",
        *_SSH_COMMON_OPTS,
        "-i", str(key_path),
        "-p", str(port),
        f"{user}@{host}",
        "bash", "-lc", shlex.quote(command),
    ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def ssh_run_script(
    host: str,
    script_text: str,
    interpreter: str,
    script_args: str,
    *,
    key_path: Path | str,
    port: int = DEFAULT_SSH_PORT,
    user: str = "root",
    timeout: int = 600,
    remote_path: str = "/tmp/mn_dynamo_launch",
) -> subprocess.CompletedProcess:
    """Ship ``script_text`` to the pod (base64 over the command line) and run it.

    Mirrors the RayJob ``_wrap_for_dash`` heredoc pattern: the script body is
    base64-encoded so it survives the SSH command line without quoting issues,
    decoded into ``remote_path`` on the pod, then executed with ``interpreter``
    (e.g. ``python3`` / ``bash``) and ``script_args`` appended verbatim.
    """
    enc = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
    remote_cmd = (
        f"echo {enc} | base64 -d > {shlex.quote(remote_path)} && "
        f"{interpreter} {shlex.quote(remote_path)} {script_args}"
    )
    return ssh_run(
        host, remote_cmd, key_path=key_path, port=port, user=user, timeout=timeout,
    )


def ssh_run_bash_with_env(
    host: str,
    script_text: str,
    env: dict[str, str] | None,
    *,
    key_path: Path | str,
    port: int = DEFAULT_SSH_PORT,
    user: str = "root",
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run ``script_text`` on ``host`` via ``bash -s`` with ``env`` exported.

    The env (which may include credentials) is prepended as shell-quoted
    ``export`` lines and the whole script is piped over SSH **stdin** — so
    secrets never appear in argv or on the pod's disk. Used by the OOB pod
    install (needs OOB_API_KEY / OOB_BASE_URL).
    """
    prologue = "\n".join(
        f"export {k}={shlex.quote(str(v))}" for k, v in (env or {}).items()
    )
    full = f"set -uo pipefail\n{prologue}\n{script_text}\n"
    argv = [
        "ssh", *_SSH_COMMON_OPTS, "-i", str(key_path), "-p", str(port),
        f"{user}@{host}", "bash", "-s",
    ]
    return subprocess.run(
        argv, input=full, capture_output=True, text=True, timeout=timeout,
    )


def probe_ssh(
    host: str,
    *,
    key_path: Path | str,
    port: int = DEFAULT_SSH_PORT,
    user: str = "root",
    timeout: int = 20,
) -> bool:
    """Return True iff a trivial SSH command succeeds (pod reachable + key OK)."""
    try:
        cp = ssh_run(host, "echo mn_ssh_ok", key_path=key_path, port=port,
                     user=user, timeout=timeout)
    except subprocess.TimeoutExpired:
        warn(f"ssh probe to {host}:{port} timed out")
        return False
    ok = cp.returncode == 0 and "mn_ssh_ok" in (cp.stdout or "")
    if not ok:
        warn(
            f"ssh probe to {host}:{port} failed rc={cp.returncode} "
            f"stderr={(cp.stderr or '').strip()[:200]}"
        )
    return ok
