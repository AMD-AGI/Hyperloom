# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Finding the endpoint the engine serves Prometheus ``/metrics`` on.

The port is not pinned anywhere a sampler can simply read: on an AgentX round the config carries no ``PORT``
at all and vLLM falls back to its own 8000, while the synthetic path pins 8888. So the round's own evidence is
searched in order of how directly it witnessed the bind, and the default is only ever right by coincidence.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from pathlib import Path
from typing import Any, Callable

import yaml

__all__ = [
    "DEFAULT_METRICS_PORT",
    "port_from_server_command",
    "port_from_server_log",
    "port_from_workspace",
    "resolve_metrics_port",
    "resolve_metrics_url",
]

#: Port the persistent server binds when ``benchmark.envs.PORT`` is unset. Mirrors
#: ``_server_lifecycle.REUSE_PORT_DEFAULT``; duplicated rather than imported because that module imports
#: ``_subprocess_kill``, which is where this one gets wired in.
DEFAULT_METRICS_PORT = 8888

#: How much of a server log to read when hunting for the bound-port banner. It is logged during startup and these logs
#: grow to megabytes on shared storage, so reading the head is both sufficient and the difference between a cheap check
#: and a slow one.
_SERVER_LOG_HEAD_BYTES = 262144

_SERVER_BIND_HINT = re.compile(r"\b(?:server|uvicorn|listening|running)\b", re.IGNORECASE)
_SERVER_BIND_URL = re.compile(r"https?://[^\s/:]+:(\d+)")
_SERVER_COMMAND_PORT = re.compile(r"(?:^|\s)--port(?:=|\s+)(\d+)(?:\s|$)")


def _first_port_found(workspace: Any, patterns: tuple[str, ...], read: Callable[[Path], int | None]) -> int | None:
    """The first port *read* reports from this round's workspace, searching *patterns* in the order given."""
    try:
        root = Path(workspace)
        for pattern in patterns:
            for candidate in sorted(root.glob(pattern)):
                port = read(candidate)
                if port is not None:
                    return port
    except OSError:
        return None
    return None


def _port_value(raw: Any) -> int | None:
    """One PORT reading as a usable port number, or ``None``."""
    if raw in (None, ""):
        return None
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return port if port > 0 else None


def _port_in_server_log(path: Path) -> int | None:
    """The port the server announced binding, read from the head of one log."""
    # The banner is near the top; a served round's log grows to megabytes and is on shared storage.
    with path.open(encoding="utf-8", errors="ignore") as handle:
        head = handle.read(_SERVER_LOG_HEAD_BYTES)
    for line in head.splitlines():
        # Matched on what the line is about rather than on its exact wording. The observed builds say
        # "Starting vLLM server on http://0.0.0.0:8000"; older ones and SGLang say "Uvicorn running on".
        # A list of phrasings misses the next build's wording, so the test is the combination -- a
        # line announcing the server, carrying a URL with a port.
        if not _SERVER_BIND_HINT.search(line):
            continue
        match = _SERVER_BIND_URL.search(line)
        if match:
            port = _port_value(match.group(1))
            if port is not None:
                return port
    return None


def port_from_server_log(workspace: Any) -> int | None:
    """Read the port the server actually bound out of its own log.

    The last resort before the default, and the only source that is right by construction: it is what the server said
    it was listening on. Needed because the port is not always pinned anywhere else -- on an AgentX round the config
    carries no ``PORT`` at all and vLLM falls back to its own 8000, while the synthetic path pins 8888. A default can
    only ever match one of those.
    """
    return _first_port_found(workspace, ("server.log", "*/server.log"), _port_in_server_log)


def port_from_workspace(workspace: Any) -> int | None:
    """Read ``benchmark.envs.PORT`` out of the round's materialized config.

    This is the only place the port is reliably written. The subprocess environment does not carry it -- the port is
    pinned in the YAML that Magpie reads, not exported to the parent -- so resolving from the environment alone lands
    on the default and scrapes a port nothing is listening on. Observed on a real session: two of three rounds bound an
    ephemeral 34407 while collection sat on 8888 and recorded the engine as unavailable, and the one round that worked
    did so only because it happened to bind the default.
    """
    # The round's own config first, then the benchmark's copy, which only exists once the client has started.
    return _first_port_found(workspace, ("*.yaml", "*/config.yaml"), _port_in_round_config)


def _port_in_round_config(path: Path) -> int | None:
    """``benchmark.envs.PORT`` as one materialized round config records it."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    benchmark = payload.get("benchmark")
    envs = benchmark.get("envs") if isinstance(benchmark, dict) else None
    return _port_value(envs.get("PORT")) if isinstance(envs, dict) else None


def _port_in_server_command(path: Path) -> int | None:
    """The ``--port`` one materialized launch command carries."""
    match = _SERVER_COMMAND_PORT.search(path.read_text(encoding="utf-8", errors="ignore"))
    return _port_value(match.group(1)) if match else None


def port_from_server_command(workspace: Any) -> int | None:
    """Read ``--port`` from the command that launched the serving process.

    AgentX vLLM rounds do not pin ``benchmark.envs.PORT`` and the Rust frontend
    does not emit the Python API server's bind banner.  The materialized command
    is therefore the only durable authority for those rounds.
    """
    return _first_port_found(workspace, ("vllm_command.txt", "*/vllm_command.txt"), _port_in_server_command)


def resolve_metrics_port(config_envs: dict[str, Any] | None = None, workspace: Any = None) -> int:
    """Resolve the port the engine serves ``/metrics`` on.

    A ``PORT`` in the caller's env is an operator pin and wins. Then the round's own evidence: its materialized
    ``benchmark.envs.PORT`` (an ephemeral port assigned per session, never exported into the subprocess env), the port
    the server logged at bind, and the ``--port`` of the command that launched it. Only then a ``PORT`` in the ambient
    env, and the default last, which is only ever right by coincidence.
    """
    port = _port_value((config_envs or {}).get("PORT"))
    if port is not None:
        return port
    if workspace is not None:
        for probe in (port_from_workspace, port_from_server_log, port_from_server_command):
            port = probe(workspace)
            if port is not None:
                return port
    port = _port_value(os.environ.get("PORT"))
    if port is not None:
        return port
    return DEFAULT_METRICS_PORT


def resolve_metrics_url(config_envs: dict[str, Any] | None = None) -> str | None:
    """Resolve a remote serving endpoint to its Prometheus metrics URL."""
    raw = str((config_envs or {}).get("BENCHMARK_BASE_URL") or "").strip()
    if not raw:
        return None
    try:
        parsed = urllib.parse.urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))
