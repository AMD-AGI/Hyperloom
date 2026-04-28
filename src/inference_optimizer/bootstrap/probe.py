"""Detect ``node`` and ``claude`` binaries on the host.

Pure stdlib (``shutil.which`` + ``subprocess``). All side-effects (calls to
``node --version`` / ``claude --version``) go through ``_run`` so tests
can monkeypatch one function and stub everything.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# Minimum Node.js major version required by recent claude-agent-sdk
# releases (the SDK's package.json declares ``"engines": {"node": ">=18"}``).
NODE_MIN_VERSION = (18, 0, 0)


@dataclass(frozen=True)
class ProbeResult:
    """Snapshot of what is currently on PATH."""

    node_path: Path | None
    node_version: tuple[int, int, int] | None
    npm_path: Path | None
    claude_path: Path | None
    claude_version: str | None
    extra_path_dirs: tuple[Path, ...]  # forwarded to the subprocess env

    @property
    def has_node(self) -> bool:
        return self.node_path is not None and self.node_version is not None

    @property
    def has_claude(self) -> bool:
        return self.claude_path is not None

    @property
    def node_is_recent_enough(self) -> bool:
        return (
            self.node_version is not None
            and self.node_version >= NODE_MIN_VERSION
        )


# ---------------------------------------------------------------------------
def parse_version(text: str) -> tuple[int, int, int] | None:
    """Pull the first ``X.Y.Z`` from ``text`` (e.g. ``v20.18.0\n``)."""
    if not text:
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _which(binary: str, *, extra_dirs: Sequence[Path] = ()) -> Path | None:
    """``shutil.which`` with optional extra search dirs prepended.

    Honours platform-appropriate extensions (``.cmd`` etc.) via
    ``shutil.which``'s own logic.
    """
    path = os.environ.get("PATH", "")
    if extra_dirs:
        joiner = os.pathsep
        path = joiner.join(str(d) for d in extra_dirs) + joiner + path
    found = shutil.which(binary, path=path)
    return Path(found) if found else None


def _run(args: Sequence[str], *, timeout: float = 5.0) -> str:
    """Run ``args`` and return stdout+stderr as text. Returns ``""`` on
    failure so the caller can decide what to do."""
    try:
        completed = subprocess.run(  # noqa: S603 — args are not user-provided
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


# ---------------------------------------------------------------------------
def find_node(extra_dirs: Sequence[Path] = ()) -> tuple[Path | None, tuple[int, int, int] | None, Path | None]:
    """Locate ``node`` and ``npm``. Returns ``(node, version_tuple, npm)``.

    ``npm`` may be ``None`` even when ``node`` is found (e.g. stripped down
    container) — :func:`probe_environment` treats that as ``has_node=True``
    but not installable.
    """
    node = _which("node", extra_dirs=extra_dirs)
    if node is None:
        return None, None, None
    version = parse_version(_run([str(node), "--version"]))
    npm = _which("npm", extra_dirs=extra_dirs)
    return node, version, npm


def find_claude(extra_dirs: Sequence[Path] = ()) -> tuple[Path | None, str | None]:
    """Locate ``claude`` CLI. Returns ``(path, version_string)``."""
    claude = _which("claude", extra_dirs=extra_dirs)
    if claude is None:
        return None, None
    raw = _run([str(claude), "--version"]).strip().splitlines()
    version = raw[0] if raw else None
    return claude, version


def probe_environment(extra_dirs: Sequence[Path] = ()) -> ProbeResult:
    """One-shot probe used by ``ensure_claude_cli``."""
    node, node_ver, npm = find_node(extra_dirs)
    claude, claude_ver = find_claude(extra_dirs)
    return ProbeResult(
        node_path=node,
        node_version=node_ver,
        npm_path=npm,
        claude_path=claude,
        claude_version=claude_ver,
        extra_path_dirs=tuple(extra_dirs),
    )
