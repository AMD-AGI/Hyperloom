# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a native build checks of its machine before it compiles: a failed check is the machine's, never the candidate's.

:func:`native_preflight` checks a recipe that builds no Python extension and so needs no torch: disk for one candidate,
the GPU architecture, and that each build tool the recipe runs answers its version, which the record keeps.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

__all__ = ["Preflight", "native_preflight"]

_VERSION_TIMEOUT_SEC = 30


@dataclass(frozen=True)
class Preflight:
    """The outcome of a recipe's machine checks: ``problem`` is an INFRA failure, ``toolchain`` the facts to record."""

    problem: str | None
    toolchain: Mapping[str, str]


def native_preflight(tools: Mapping[str, tuple[str, ...]], disk_gb: float, scratch: Path, gfx_arch: str) -> Preflight:
    """Disk for one candidate below ``scratch``, a GPU architecture, and each of ``tools`` (its name in the record, the
    command that prints its version) on this machine; the first line each prints is the toolchain's fact."""
    try:
        disk_preflight(scratch, 1, per_candidate_gb=disk_gb)
    except DiskPreflightError as exc:
        return Preflight(f"preflight_disk: {exc}", {})
    toolchain: dict[str, str] = {}
    for name, command in tools.items():
        version = _version(command)
        if version is None:
            return Preflight(f"preflight_toolchain: {name} ({command[0]}) does not run on this machine", {})
        toolchain[name] = version
    return Preflight(None, toolchain)


def _version(command: tuple[str, ...]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=_VERSION_TIMEOUT_SEC, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return lines[0] if done.returncode == 0 and lines else None
