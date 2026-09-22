# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which machine this is, so the right roofs get measured and checked.

Only identity lives here. The roofs themselves are measured by the analyst
during its session and never stored: a committed figure ages out of date
silently, and a measured one is not ours to publish.

Identity still matters for two things. The analyst is told which card it is
estimating for, so a session that cannot reach a profiler at least names the
right machine when it falls back. And the architecture selects which published
peaks :func:`~kernelforge.roofline_ceiling.contract.build_hardware` checks the
reported roofs against -- a roof above the vendor's own number cannot be a
measurement, and that check is the only one standing between a mistyped column
and the denominator of every ceiling.

Partition mode is part of the identity rather than metadata on it: splitting an
MI355X into CPX changes the bandwidth one slice can reach, so the same card
under SPX is a different machine and its roofs are different numbers.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from kernelforge.fusion.gpu_arch import canon_arch, detect_arch

_MARKETING_RE = re.compile(r"^\s*Marketing Name:\s*(AMD Instinct\s+\S+)", re.MULTILINE)
_COMPUTE_PARTITION_RE = re.compile(r"Compute Partition:\s*(\S+)")
_MEMORY_PARTITION_RE = re.compile(r"Memory Partition:\s*(\S+)")


@dataclass(frozen=True)
class DeviceIdentity:
    """What distinguishes one measurable machine configuration from another."""

    arch: str
    device_name: str = ""
    compute_partition: str = ""
    memory_partition: str = ""

    def slug(self) -> str:
        """A filename-safe key for this configuration."""
        parts = [
            self.arch or "unknown",
            (self.device_name or "").lower().replace("amd instinct", "").strip().replace(" ", "-"),
            (self.compute_partition or "").lower(),
            (self.memory_partition or "").lower(),
        ]
        return "-".join(part for part in parts if part) or "unknown"

    def describe(self) -> dict[str, Any]:
        """The identity as the analyst request carries it."""
        return {
            "arch": self.arch,
            "device_name": self.device_name,
            "compute_partition": self.compute_partition,
            "memory_partition": self.memory_partition,
        }


def _run_text(argv: list[str], timeout_sec: float = 20.0) -> str:
    """Run a probe binary and return its output, or ``""`` on any failure."""
    if not shutil.which(argv[0]):
        return ""
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def describe_device(arch: str = "") -> DeviceIdentity:
    """Identify this machine's measurable configuration.

    Every field degrades independently to an empty string. An empty field says
    "not determined" rather than "absent", and the analyst is told as much: a
    machine whose partition mode could not be read is one whose roofs deserve a
    lower confidence, not one assumed to be in the default mode.
    """
    resolved = canon_arch(arch) or detect_arch()
    marketing = _MARKETING_RE.search(_run_text(["rocminfo"]))
    partitions = _run_text(["rocm-smi", "--showcomputepartition", "--showmemorypartition"])
    compute = _COMPUTE_PARTITION_RE.search(partitions)
    memory = _MEMORY_PARTITION_RE.search(partitions)
    return DeviceIdentity(
        arch=resolved,
        device_name=(marketing.group(1).strip() if marketing else ""),
        compute_partition=(compute.group(1).strip() if compute else ""),
        memory_partition=(memory.group(1).strip() if memory else ""),
    )


__all__ = [
    "DeviceIdentity",
    "describe_device",
]
