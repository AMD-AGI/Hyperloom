# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start the loopback Experience service on demand and keep it running with the settings its client launches with."""

from __future__ import annotations

import ipaddress
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from hyperloom_kb.config import PACKAGED_DECLARATION, load_declaration
from hyperloom_kb.http_service import ServiceSettings
from hyperloom_kb.remote import RemoteClient, RemoteClientError, RemoteConfig
from hyperloom_kb.schema import JsonValue

LOG_NAME = "service.log"
DEFAULT_START_TIMEOUT_SECONDS = 60.0
_POLL_SECONDS = 0.2


class LocalServiceError(RuntimeError):
    """Raised when the loopback Experience service cannot be brought to serving."""


@dataclass(frozen=True)
class LocalService:
    health: dict[str, JsonValue]
    # Only set when this call started the service; a service that was already serving is not ours to hold.
    process: subprocess.Popen[bytes] | None = None
    restarted: bool = False


def is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _address(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    if parts.scheme != "http" or not is_loopback(url):
        raise LocalServiceError(f"a local Experience service URL must be http on a loopback host, not {url}")
    return parts.hostname or "", parts.port or 80


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _spawn(host: str, port: int, home: Path, token: str, env: Mapping[str, str]) -> subprocess.Popen[bytes]:
    home.mkdir(parents=True, exist_ok=True)
    child_env = dict(env)
    child_env["HYPERLOOM_KB_TOKEN"] = token
    package_root = str(Path(__file__).resolve().parent.parent)
    child_env["PYTHONPATH"] = os.pathsep.join(path for path in (package_root, child_env.get("PYTHONPATH", "")) if path)
    # ``-m hyperloom_kb.http_service`` would re-execute a module the package ``__init__`` already imported.
    command = [
        sys.executable,
        "-c",
        "from hyperloom_kb.http_service import main; raise SystemExit(main())",
        "--home",
        str(home),
        "--host",
        host,
        "--port",
        str(port),
    ]
    with (home / LOG_NAME).open("ab") as log:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )


def _wait_until_listening(
    process: subprocess.Popen[bytes],
    host: str,
    port: int,
    log_path: Path,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not _listening(host, port):
        # A concurrent start may have won the port, which makes this process exit while the service serves.
        if process.poll() is not None and not _listening(host, port):
            raise LocalServiceError(f"Experience service exited with status {process.returncode}; see {log_path}")
        if time.monotonic() >= deadline:
            raise LocalServiceError(
                f"Experience service is not listening on {host}:{port} after {timeout_seconds:g}s; see {log_path}"
            )
        time.sleep(_POLL_SECONDS)


def _health(config: RemoteConfig) -> dict[str, JsonValue]:
    try:
        return RemoteClient(config).health()
    except RemoteClientError as exc:
        raise LocalServiceError(
            f"{config.base_url} is listening but did not answer as this Experience service: {exc}"
        ) from exc


def _stale(health: Mapping[str, JsonValue], env: Mapping[str, str]) -> str:
    expected = load_declaration(PACKAGED_DECLARATION).schema_ref
    if health.get("schema_ref") != expected:
        return f"it serves {health.get('schema_ref')!r} but this client writes {expected}"
    if health.get("config_digest") != ServiceSettings.from_env(env).digest():
        return "it was started with other settings"
    return ""


def _stop(health: Mapping[str, JsonValue], host: str, port: int, timeout_seconds: float) -> None:
    pid = health.get("pid")
    # A service in this very process (an embedded server) is never ours to signal.
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        raise LocalServiceError(f"the Experience service on {host}:{port} reports no process to restart; stop it")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise LocalServiceError(f"cannot stop the Experience service process {pid} on {host}:{port}") from exc
    deadline = time.monotonic() + timeout_seconds
    while _listening(host, port):
        if time.monotonic() >= deadline:
            raise LocalServiceError(f"the Experience service process {pid} still listens on {host}:{port}")
        time.sleep(_POLL_SECONDS)


def ensure_local_service(
    config: RemoteConfig,
    home: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_START_TIMEOUT_SECONDS,
) -> LocalService:
    """Serve ``config.base_url`` from ``home`` with ``env``'s settings, restarting a service started otherwise.

    Only a service that answers with this client's token is reused or restarted; anything else on the port is refused.
    """

    host, port = _address(config.base_url)
    home = Path(home).expanduser()
    launch_env = os.environ if env is None else env
    restarted = False
    if _listening(host, port):
        health = _health(config)
        if not _stale(health, launch_env):
            return LocalService(health)
        _stop(health, host, port, timeout_seconds)
        restarted = True
    process = _spawn(host, port, home, config.token, launch_env)
    _wait_until_listening(process, host, port, home / LOG_NAME, timeout_seconds)
    health = _health(config)
    if reason := _stale(health, launch_env):
        raise LocalServiceError(f"the Experience service on {host}:{port} is not usable: {reason}")
    return LocalService(health, process, restarted)


__all__ = [
    "DEFAULT_START_TIMEOUT_SECONDS",
    "LOG_NAME",
    "LocalService",
    "LocalServiceError",
    "ensure_local_service",
    "is_loopback",
]
