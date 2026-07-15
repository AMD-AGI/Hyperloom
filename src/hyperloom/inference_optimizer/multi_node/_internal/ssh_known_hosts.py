"""Session-scoped SSH known_hosts management for the Infera control plane."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .log import info, warn

_HOST_KEY_ERROR_MARKERS: tuple[str, ...] = (
    "host key verification failed",
    "REMOTE HOST IDENTIFICATION HAS CHANGED",
    "Offending ED25519 key",
    "Offending RSA key",
)


def default_known_hosts_path(ssh_dir: Path) -> Path:
    """Return the default known_hosts file adjacent to the SSH keypair.

    Args:
        ssh_dir: Session directory holding ``mn_id_ed25519``.

    Returns:
        Path: ``<ssh_dir>/known_hosts``.
    """
    return ssh_dir / "known_hosts"


def is_host_key_error(stderr: str | None) -> bool:
    """Return True when SSH stderr indicates a host-key mismatch.

    Args:
        stderr: SSH subprocess stderr text.

    Returns:
        bool: True for known host-key verification failures.
    """
    text = (stderr or "").lower()
    return any(marker.lower() in text for marker in _HOST_KEY_ERROR_MARKERS)


def refresh_known_hosts(
    hosts: list[tuple[str, int]],
    dest: Path,
) -> Path:
    """Append ssh-keyscan results for ``hosts`` into ``dest``.

    Idempotent per host line: re-scanning the same IP updates the file via
    ssh-keyscan append semantics (duplicates are harmless for OpenSSH).

    Args:
        hosts: ``(ip_or_hostname, port)`` pairs to scan.
        dest: Target known_hosts file.

    Returns:
        Path: The ``dest`` path (created or updated).

    Raises:
        RuntimeError: When every host scan fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.parent.chmod(0o700)
    except OSError as exc:
        warn(f"could not chmod SSH dir {dest.parent} to 0700: {exc}")
    if not dest.is_file():
        dest.touch()
    try:
        dest.chmod(0o600)
    except OSError as exc:
        warn(f"could not chmod known_hosts {dest} to 0600: {exc}")

    scanned = 0
    for host, port in hosts:
        host = (host or "").strip()
        if not host:
            continue
        proc = subprocess.run(
            ["ssh-keyscan", "-p", str(int(port)), "-H", host],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            warn(
                f"ssh-keyscan failed for {host}:{port} "
                f"rc={proc.returncode} stderr={(proc.stderr or '').strip()[:200]}"
            )
            continue
        with dest.open("a", encoding="utf-8") as fh:
            fh.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                fh.write("\n")
        scanned += 1
    if scanned == 0 and hosts:
        raise RuntimeError(f"ssh-keyscan produced no keys for hosts={hosts!r}")
    if scanned:
        info(f"refreshed SSH known_hosts ({scanned} host(s)) -> {dest}")
    return dest
