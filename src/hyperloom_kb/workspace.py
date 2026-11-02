# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workspace's one local Experience KB, as setup configures it and every run reaches it.

Setup writes a loopback ``HYPERLOOM_KB_URL`` and a generated ``HYPERLOOM_KB_TOKEN`` into the workspace's ``.env``
(``init_env``), and the service serves ``$USER_DATA_PATH/experience-kb`` unless ``HYPERLOOM_KB_HOME`` names another
home (``service_home``). An optimize run and a ``kernelforge forge-loop`` run both start it from those settings, so
the workspace has one Experience KB, with each product's schema in it.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import socket
from collections.abc import Mapping
from pathlib import Path

from hyperloom_common.llm.llm_config import DEFAULT_CLAUDE_MODEL
from hyperloom_common.workspace import data_root
from hyperloom_kb.local_service import LocalServiceError

URL_ENV, TOKEN_ENV = "HYPERLOOM_KB_URL", "HYPERLOOM_KB_TOKEN"
# Names a file holding the token, for a process that should not carry the token itself in its environment.
TOKEN_FILE_ENV = "HYPERLOOM_KB_TOKEN_FILE"
# A workspace's service port is drawn from here, below the ephemeral range a later client socket could take.
LOCAL_PORTS = range(20_000, 30_000)
SERVICE_DIR = "experience-kb"
# Names the service's data home instead of $USER_DATA_PATH/experience-kb. One home serves one service at a time, so
# runs on other hosts that share a session root each need their own.
HOME_ENV = "HYPERLOOM_KB_HOME"
PLACEHOLDER = "<PLEASE_FILL_IN>"
_PLANNER_MODEL_KEYS = ("LOCAL_KB_PLANNER_MODEL", "CLAUDE_MODEL", "ANTHROPIC_MODEL")


def service_home(environ: Mapping[str, str] = os.environ, *, root: Path | None = None) -> Path:
    """``HYPERLOOM_KB_HOME``, else ``experience-kb`` under the workspace's data root (``root``, else ``data_root``)."""
    named = (environ.get(HOME_ENV) or "").strip()
    if named:
        return Path(named).expanduser()
    return (root if root is not None else data_root(environ)) / SERVICE_DIR


def configured_token(environ: Mapping[str, str] = os.environ) -> str:
    """The KB token: the value of HYPERLOOM_KB_TOKEN, else the contents of the file HYPERLOOM_KB_TOKEN_FILE names;
    empty when neither gives one."""
    token = environ.get(TOKEN_ENV, "").strip()
    named = environ.get(TOKEN_FILE_ENV, "").strip()
    if token or not named:
        return token
    try:
        return Path(named).expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def service_env(environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    """The environment a local service is started with: ``environ``, with a planner model when it names none."""
    env = dict(environ)
    if not any(env.get(key) for key in _PLANNER_MODEL_KEYS):
        env["LOCAL_KB_PLANNER_MODEL"] = DEFAULT_CLAUDE_MODEL
    return env


def _port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def local_url(workspace: Path) -> str:
    """The loopback URL of ``workspace``'s own service: a port derived from its path, past any port in use.

    Two workspaces on one host land on different ports even while neither service runs.
    """

    start = int(hashlib.sha256(str(workspace.resolve()).encode()).hexdigest(), 16) % len(LOCAL_PORTS)
    for offset in range(len(LOCAL_PORTS)):
        port = LOCAL_PORTS[(start + offset) % len(LOCAL_PORTS)]
        if _port_is_free(port):
            return f"http://127.0.0.1:{port}"
    raise LocalServiceError(f"no free port in {LOCAL_PORTS.start}-{LOCAL_PORTS.stop - 1} for the Experience service")


def init_env(env_file: Path) -> dict[str, str]:
    """Point ``env_file`` at its workspace's local service with a generated token, keeping any value it already sets."""

    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.is_file() else []
    defaults = {
        URL_ENV: local_url(env_file.resolve().parent),
        TOKEN_ENV: secrets.token_urlsafe(32),
    }
    status: dict[str, str] = {}
    for key, default in defaults.items():
        assignment = re.compile(rf"^\s*(?:export\s+)?{key}\s*=(.*)$")
        index = next((i for i in reversed(range(len(lines))) if assignment.match(lines[i])), None)
        match = None if index is None else assignment.match(lines[index])
        value = "" if match is None else match.group(1).strip().strip("\"'")
        if value and value != PLACEHOLDER:
            status[key] = "kept"
            continue
        if index is None:
            lines.append(f"{key}={default}")
        else:
            lines[index] = f"{key}={default}"
        status[key] = "written"
    if "written" in status.values():
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


__all__ = [
    "HOME_ENV",
    "LOCAL_PORTS",
    "PLACEHOLDER",
    "SERVICE_DIR",
    "TOKEN_ENV",
    "TOKEN_FILE_ENV",
    "URL_ENV",
    "configured_token",
    "init_env",
    "local_url",
    "service_env",
    "service_home",
]
