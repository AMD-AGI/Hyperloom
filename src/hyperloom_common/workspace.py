# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator's Hyperloom workspace as every product reads it: its data root and its ``.env``.

Setup writes the workspace's ``.env``; an optimize run sources the settings it holds that the environment does not,
and a ``kernelforge forge-loop`` run reads its Experience KB settings from the same file, so the two share one
workspace.
"""

from __future__ import annotations

import io
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path

DEFAULT_SESSION_DIR = Path("/workspace/hyperloom")
ENV_USER_DATA_PATH = "USER_DATA_PATH"
POD_LOCAL_WORKSPACE = Path("/workspace")
REPO_ROOT_ENV = "REPO_ROOT"
DOTENV_NAME = ".env"

_SHELL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_workspace_root() -> Path:
    """The workspace to use when ``$USER_DATA_PATH`` is unset."""
    # The nearest *existing* ancestor decides: os.access is False for a path that does not exist yet, which would
    # divert root off a /workspace it can create.
    probe = POD_LOCAL_WORKSPACE
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if os.access(probe, os.W_OK):
        return DEFAULT_SESSION_DIR
    return Path.cwd() / "session"


def data_root(environ: Mapping[str, str] = os.environ) -> Path:
    """``$USER_DATA_PATH``, else ``default_workspace_root()``."""
    user_data = (environ.get(ENV_USER_DATA_PATH) or "").strip()
    return Path(user_data).expanduser() if user_data else default_workspace_root()


def _dotenv_roots(environ: Mapping[str, str]) -> list[Path]:
    explicit_root = (environ.get(REPO_ROOT_ENV) or "").strip()
    if explicit_root:
        return [Path(explicit_root)]
    # Development checkout: workspace.py -> hyperloom_common -> src -> repo root.
    roots = [Path(__file__).resolve().parents[2]]
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file() and (cwd / "src" / "hyperloom").is_dir():
        roots.append(cwd)
    return roots


def workspace_dotenv(environ: Mapping[str, str] = os.environ) -> Path | None:
    """The trusted workspace ``.env`` that exists, resolved without trusting an arbitrary working directory."""
    for root in _dotenv_roots(environ):
        env_file = root / DOTENV_NAME
        if env_file.is_file():
            return env_file
    return None


def workspace_dotenv_target(environ: Mapping[str, str] = os.environ) -> Path:
    """The workspace ``.env``, the one that exists or else where setup would write it."""
    return workspace_dotenv(environ) or _dotenv_roots(environ)[0] / DOTENV_NAME


def parse_env_assignments(text: str) -> dict[str, str]:
    """Read assignments as data (first wins), decoding quoted values without expansion."""
    out: dict[str, str] = {}
    stream = io.StringIO(text)
    while True:
        start = stream.tell()
        raw = stream.readline()
        if not raw:
            break
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator:
            continue
        if not _SHELL_NAME_RE.fullmatch(key):
            if not any(char.isspace() for char in key):
                out.setdefault(key, value.strip())  # The allowlist reports invalid assignment names.
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            value_start = raw.index("=") + 1
            while value_start < len(raw) and raw[value_start] in " \t":
                value_start += 1
            stream.seek(start + value_start)
            lexer = shlex.shlex(stream, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            try:
                value = lexer.get_token()
                if stream.tell() and text[stream.tell() - 1] != "\n":
                    stream.readline()
            except ValueError as exc:
                raise ValueError(f"Invalid quoted environment assignment for {key}: {exc}") from exc
        out.setdefault(key, value)
    return out


__all__ = [
    "DEFAULT_SESSION_DIR",
    "DOTENV_NAME",
    "ENV_USER_DATA_PATH",
    "POD_LOCAL_WORKSPACE",
    "REPO_ROOT_ENV",
    "data_root",
    "default_workspace_root",
    "parse_env_assignments",
    "workspace_dotenv",
    "workspace_dotenv_target",
]
