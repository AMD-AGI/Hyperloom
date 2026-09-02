# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve the operator-selected Hermes CLI consistently across transports."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

HERMES_EXTERNAL_SANDBOX_ENV = "HYPERLOOM_HERMES_EXTERNAL_SANDBOX"


def running_in_container(env: Mapping[str, str] | None = None) -> bool:
    """Return whether this process has a concrete container-runtime marker."""

    del env
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "kubepods", "containerd", "podman", "lxc"))


def hermes_external_sandbox_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether a declared and verifiable outer container is active."""

    source = os.environ if env is None else env
    declared = str(source.get(HERMES_EXTERNAL_SANDBOX_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}
    return declared and running_in_container(source)


def resolve_hermes_executable(override: str = "") -> str:
    """Return one executable Hermes path, or ``""`` when unavailable.

    ``override`` wins, followed by ``HYPERLOOM_HERMES_BIN`` and finally the
    normal ``PATH`` lookup. Explicit paths must already exist and be executable;
    an invalid override fails closed instead of silently selecting another CLI.
    """

    configured = str(override or os.environ.get("HYPERLOOM_HERMES_BIN", "")).strip()
    if configured:
        found = shutil.which(configured)
        if found:
            return str(Path(found).resolve())
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        return ""
    found = shutil.which("hermes")
    return str(Path(found).resolve()) if found else ""


__all__ = [
    "HERMES_EXTERNAL_SANDBOX_ENV",
    "hermes_external_sandbox_enabled",
    "resolve_hermes_executable",
    "running_in_container",
]
