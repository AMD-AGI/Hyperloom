# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The execution scope a process records on the ownership it takes: this boot and PID namespace."""

from __future__ import annotations

from pathlib import Path


def local_owner_scope() -> str:
    """Identify this boot and PID namespace, or leave ownership unobservable."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        namespace = Path("/proc/self/ns/pid").stat().st_ino
    except OSError:
        return ""
    return f"{boot}:{namespace}" if boot else ""
