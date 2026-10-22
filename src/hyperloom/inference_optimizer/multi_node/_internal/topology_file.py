# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the cluster hand-off a platform writes as a TopologyHandoff v1 JSON file.

``CLAW_TOPOLOGY_FILE`` names the file (schema string ``claw.topology-handoff/v1``).
The file is translated into the legacy variable names (``HYPERLOOM_MN_EXT_*``,
``INFERENCE_OPTIMIZER_*``) so the rest of the hand-off code resolves a file and a
set of variables through one path and the two cannot drift apart.

A file that is named but unusable is an error, never a silent fallback to the
legacy variables: the caller asked for the file, and half-using it would hand the
optimizer a cluster nobody described.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TOPOLOGY_FILE_ENV = "CLAW_TOPOLOGY_FILE"

_SCHEMA_RE = re.compile(r"^claw\.topology-handoff/v(\d+)(?:\.\d+)*$")
_SUPPORTED_MAJOR = 1
_BACKENDS = ("rayjob", "infera")

# Every legacy variable the file describes. When a file is in force these are read
# from it only (empty when the file omits them), never from the process env.
COVERED_NAMES = (
    "INFERENCE_OPTIMIZER_MN_BACKEND",
    "INFERENCE_OPTIMIZER_NODES",
    "INFERENCE_OPTIMIZER_GPUS_PER_NODE",
    "HYPERLOOM_MN_EXT_SERVICE_URL",
    "HYPERLOOM_MN_EXT_HEAD_IP",
    "HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN",
    "HYPERLOOM_MN_EXT_SSH_KEY",
    "HYPERLOOM_MN_EXT_SSH_PORT",
    "HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS",
    "HYPERLOOM_MN_EXT_PREFILL_IPS",
    "HYPERLOOM_MN_EXT_DECODE_IPS",
    "HYPERLOOM_MN_EXT_WORKER_IPS",
)

# File field each legacy variable comes from, for the disagreement warning.
_FIELD_OF = {
    "INFERENCE_OPTIMIZER_MN_BACKEND": "backend",
    "INFERENCE_OPTIMIZER_NODES": "nodes",
    "INFERENCE_OPTIMIZER_GPUS_PER_NODE": "gpus_per_node",
    "HYPERLOOM_MN_EXT_SERVICE_URL": "service_url",
    "HYPERLOOM_MN_EXT_HEAD_IP": "ray.head_ip",
    "HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN": "ray.dashboard_token_file",
    "HYPERLOOM_MN_EXT_SSH_KEY": "infera.ssh_key_file",
    "HYPERLOOM_MN_EXT_SSH_PORT": "infera.ssh_port_base",
    "HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS": "infera.known_hosts_file",
    "HYPERLOOM_MN_EXT_PREFILL_IPS": "infera.prefill_ips",
    "HYPERLOOM_MN_EXT_DECODE_IPS": "infera.decode_ips",
    "HYPERLOOM_MN_EXT_WORKER_IPS": "infera.worker_ips",
}

# Names whose value is compared as an IP list, not as text.
_LIST_NAMES = frozenset(n for n in COVERED_NAMES if n.endswith("_IPS"))

Lookup = Callable[[str], str]

# Disagreements already reported, so a hand-off consulted many times warns once.
_WARNED: set[tuple[str, tuple[str, ...]]] = set()


class TopologyHandoffError(RuntimeError):
    """``CLAW_TOPOLOGY_FILE`` names a file that cannot be used."""


def _fail(path: str, problem: str) -> TopologyHandoffError:
    return TopologyHandoffError(f"{TOPOLOGY_FILE_ENV}={path}: {problem}")


def _is_int(value: Any, low: int, high: int | None = None) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= low and (high is None or value <= high)


def _text(path: str, doc: dict[str, Any], key: str, where: str) -> str:
    """An optional non-empty string field; empty when absent."""
    value = doc.get(key)
    if value is None:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, f"{where}{key} must be a non-empty string")
    return value.strip()


def _ip_list(path: str, doc: dict[str, Any], key: str) -> str:
    value = doc.get(key, [])
    if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
        raise _fail(path, f"infera.{key} must be a list of non-empty strings")
    return ",".join(x.strip() for x in value)


def _block(path: str, doc: dict[str, Any], key: str, *, required: bool) -> dict[str, Any]:
    value = doc.get(key)
    if value is None:
        if required:
            raise _fail(path, f"backend infera needs an {key!r} block")
        return {}
    if not isinstance(value, dict):
        raise _fail(path, f"{key} must be an object")
    return value


def _read_secret(path: str, secret_path: str) -> str:
    """Read the credential a file field points at; refuse when it cannot be read or is empty."""
    try:
        content = Path(secret_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise _fail(path, f"ray.dashboard_token_file {secret_path} cannot be read: {exc}") from exc
    if not content:
        raise _fail(path, f"ray.dashboard_token_file {secret_path} is empty")
    return content


def _ray_overlay(path: str, doc: dict[str, Any]) -> dict[str, str]:
    ray = _block(path, doc, "ray", required=False)
    token_file = _text(path, ray, "dashboard_token_file", "ray.")
    return {
        "HYPERLOOM_MN_EXT_HEAD_IP": _text(path, ray, "head_ip", "ray."),
        "HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN": _read_secret(path, token_file) if token_file else "",
    }


def _infera_overlay(path: str, doc: dict[str, Any]) -> dict[str, str]:
    infera = _block(path, doc, "infera", required=True)
    port = infera.get("ssh_port_base")
    if port is not None and not _is_int(port, 1, 65535):
        raise _fail(path, "infera.ssh_port_base must be an integer in [1, 65535]")
    return {
        "HYPERLOOM_MN_EXT_SSH_KEY": _text(path, infera, "ssh_key_file", "infera."),
        "HYPERLOOM_MN_EXT_SSH_PORT": "" if port is None else str(port),
        "HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS": _text(path, infera, "known_hosts_file", "infera."),
        "HYPERLOOM_MN_EXT_PREFILL_IPS": _ip_list(path, infera, "prefill_ips"),
        "HYPERLOOM_MN_EXT_DECODE_IPS": _ip_list(path, infera, "decode_ips"),
        "HYPERLOOM_MN_EXT_WORKER_IPS": _ip_list(path, infera, "worker_ips"),
    }


def _check_schema(path: str, doc: dict[str, Any]) -> None:
    schema = doc.get("schema")
    matched = _SCHEMA_RE.match(schema) if isinstance(schema, str) else None
    if matched is None:
        raise _fail(path, f"schema {schema!r} is not a claw.topology-handoff/v<major> string")
    if int(matched.group(1)) != _SUPPORTED_MAJOR:
        raise _fail(
            path,
            f"unsupported schema major in {schema!r}; this reader handles claw.topology-handoff/v{_SUPPORTED_MAJOR}",
        )


def _parse(path: str) -> dict[str, str]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _fail(path, f"cannot read the file: {exc}") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail(path, f"not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise _fail(path, "the document must be a JSON object")
    _check_schema(path, doc)

    backend = doc.get("backend")
    if backend not in _BACKENDS:
        raise _fail(path, f"backend {backend!r} must be one of {', '.join(_BACKENDS)}")
    for key in ("nodes", "gpus_per_node"):
        if not _is_int(doc.get(key), 1):
            raise _fail(path, f"{key} must be an integer >= 1")
    url = doc.get("service_url")
    if not isinstance(url, str) or not url.strip().startswith(("http://", "https://")):
        raise _fail(path, "service_url must be an http(s) URL")

    overlay = dict.fromkeys(COVERED_NAMES, "")
    overlay.update(
        {
            "INFERENCE_OPTIMIZER_MN_BACKEND": backend,
            "INFERENCE_OPTIMIZER_NODES": str(doc["nodes"]),
            "INFERENCE_OPTIMIZER_GPUS_PER_NODE": str(doc["gpus_per_node"]),
            "HYPERLOOM_MN_EXT_SERVICE_URL": url.strip(),
        }
    )
    overlay.update(_ray_overlay(path, doc) if backend == "rayjob" else _infera_overlay(path, doc))
    return overlay


def _normalise(name: str, value: str) -> str:
    if name in _LIST_NAMES:
        return ",".join(x.strip() for x in value.split(",") if x.strip())
    return value.strip()


def _warn_on_disagreement(path: str, overlay: dict[str, str]) -> None:
    """Name every field where a set legacy variable differs from the file that overrides it."""
    differing = tuple(
        f"{_FIELD_OF[name]} ({name})"
        for name in COVERED_NAMES
        if os.environ.get(name, "").strip() and _normalise(name, os.environ[name]) != overlay[name]
    )
    if not differing or (path, differing) in _WARNED:
        return
    _WARNED.add((path, differing))
    log.warning(
        "%s=%s overrides the legacy hand-off variables; they disagree with the file on: %s",
        TOPOLOGY_FILE_ENV,
        path,
        ", ".join(differing),
    )


def topology_file_overlay() -> dict[str, str] | None:
    """Legacy-name view of ``CLAW_TOPOLOGY_FILE``, or ``None`` when the variable is unset.

    Raises ``TopologyHandoffError`` when the variable is set and the file cannot be used.
    """
    path = os.environ.get(TOPOLOGY_FILE_ENV, "").strip()
    if not path:
        return None
    overlay = _parse(path)
    _warn_on_disagreement(path, overlay)
    return overlay


def handoff_lookup() -> Lookup:
    """Return ``name -> value``: the file's view of covered names when a file is in force, else the env."""
    overlay = topology_file_overlay()
    if overlay is None:
        return lambda name: os.environ.get(name, "")
    return lambda name: overlay[name] if name in overlay else os.environ.get(name, "")
