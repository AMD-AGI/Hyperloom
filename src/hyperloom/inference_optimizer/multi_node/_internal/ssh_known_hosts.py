"""Session-scoped SSH known_hosts management for the Infera control plane."""

from __future__ import annotations

from pathlib import Path

from inference_suite.multinode import SshError, keyscan

from .log import info, warn


def default_known_hosts_path(ssh_dir: Path) -> Path:
    """Return the default known_hosts file adjacent to the SSH keypair."""
    return ssh_dir / "known_hosts"


def refresh_known_hosts(
    hosts: list[tuple[str, int]],
    dest: Path,
) -> Path:
    """Append ssh-keyscan results for ``hosts`` into ``dest``."""
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
        try:
            keys = keyscan(host, int(port))
        except SshError as exc:
            warn(f"ssh-keyscan failed for {host}:{port}: {exc}")
            continue
        with dest.open("a", encoding="utf-8") as fh:
            fh.write(keys)
        scanned += 1
    if scanned == 0 and hosts:
        raise RuntimeError(f"ssh-keyscan produced no keys for hosts={hosts!r}")
    if scanned:
        info(f"refreshed SSH known_hosts ({scanned} host(s)) -> {dest}")
    return dest
