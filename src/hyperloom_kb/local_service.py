# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start the loopback Experience service on demand and keep it running with the settings its client launches with."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from hyperloom_kb.config import PACKAGED_DECLARATION, load_declaration
from hyperloom_kb.http_service import ServiceSettings, code_digest
from hyperloom_kb.remote import RemoteClient, RemoteClientError, RemoteConfig, is_loopback
from hyperloom_kb.schema import JsonValue

LOG_NAME = "service.log"
# A log past this size is kept as ``service.log.1`` when the next service starts, replacing the one kept before.
LOG_ROTATE_BYTES = 8 * 1024 * 1024
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
    # Why a service left serving runs with other settings than the caller's; empty when they match.
    stale: str = ""


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
    child_env = dict(env)
    child_env["HYPERLOOM_KB_TOKEN"] = token
    package_root = str(Path(__file__).resolve().parent.parent)
    child_env["PYTHONPATH"] = os.pathsep.join(path for path in (package_root, child_env.get("PYTHONPATH", "")) if path)
    command = [
        sys.executable,
        "-m",
        "hyperloom_kb",
        "--home",
        str(home),
        "--host",
        host,
        "--port",
        str(port),
    ]
    try:
        home.mkdir(parents=True, exist_ok=True)
        log_path = home / LOG_NAME
        if log_path.is_file() and log_path.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(log_path, log_path.with_name(f"{LOG_NAME}.1"))
        with log_path.open("ab") as log:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env,
                start_new_session=True,
            )
    except OSError as exc:
        raise LocalServiceError(f"cannot start the Experience service from {home}: {exc}") from exc


def _last_line(log_path: Path) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[-1][:300] if lines else ""


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
            raise LocalServiceError(
                f"Experience service exited with status {process.returncode}: {_last_line(log_path)}; see {log_path}"
            )
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
    if health.get("code_digest") != code_digest():
        return "it runs other Experience KB code than this client"
    return ""


def _require_home(health: Mapping[str, JsonValue], home: Path, host: str, port: int) -> None:
    """Refuse a service that holds another workspace's data, such as one a copied ``.env`` points at."""

    served = health.get("home")
    if isinstance(served, str) and Path(served).resolve() != home.resolve():
        raise LocalServiceError(
            f"the Experience service on {host}:{port} serves {served}, not {home}; "
            "give this workspace its own port in HYPERLOOM_KB_URL"
        )


def _stop(health: Mapping[str, JsonValue], host: str, port: int, timeout_seconds: float) -> None:
    pid = health.get("pid")
    # A service in this very process (an embedded server) is never ours to signal.
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        raise LocalServiceError(f"the Experience service on {host}:{port} reports no process to restart; stop it")
    try:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
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
    restart: bool = True,
) -> LocalService:
    """Serve ``config.base_url`` from ``home`` with ``env``'s settings.

    Only a service that answers with this client's token and serves ``home`` is reused; anything else on the port is
    refused. One started with other settings is restarted, keeping its data, unless ``restart`` is false: then it is
    left serving and returned with why it is stale, so a caller that only needs it answering never stops it under a run.
    """

    host, port = _address(config.base_url)
    home = Path(home).expanduser()
    launch_env = os.environ if env is None else env
    restarted = False
    if _listening(host, port):
        health = _health(config)
        _require_home(health, home, host, port)
        reason = _stale(health, launch_env)
        if not reason:
            return LocalService(health)
        if not restart:
            return LocalService(health, stale=reason)
        _stop(health, host, port, timeout_seconds)
        restarted = True
    process = _spawn(host, port, home, config.token, launch_env)
    _wait_until_listening(process, host, port, home / LOG_NAME, timeout_seconds)
    health = _health(config)
    _require_home(health, home, host, port)
    if reason := _stale(health, launch_env):
        raise LocalServiceError(f"the Experience service on {host}:{port} is not usable: {reason}")
    return LocalService(health, process, restarted)


__all__ = [
    "DEFAULT_START_TIMEOUT_SECONDS",
    "LOG_NAME",
    "LOG_ROTATE_BYTES",
    "LocalService",
    "LocalServiceError",
    "ensure_local_service",
]
