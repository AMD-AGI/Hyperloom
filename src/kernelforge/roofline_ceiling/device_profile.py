# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which machine this is, so the analyst measures and names the right one.

Only identity lives here. The roofs themselves are measured by the analyst
during its session and never stored: a committed figure ages out of date
silently, and a measured one is not ours to publish.

Identity is what the analyst is told about the card it is estimating for, so a
session that cannot reach a profiler still names the right machine when it
falls back. It checks nothing: no code compares a reported roof against this
card's published peaks, so a misread profiler column reaches the ceiling unless
a reader of the derivation catches it.

Partition mode is part of the identity rather than metadata on it: splitting an
MI355X into CPX changes the bandwidth one slice can reach, so the same card
under SPX is a different machine and its roofs are different numbers.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from hyperloom.common import gpu_partition
from kernelforge.fusion.gpu_arch import canon_arch, detect_arch

_MARKETING_RE = re.compile(r"^\s*Marketing Name:\s*(AMD Instinct\s+\S+)", re.MULTILINE)


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


def _partition_label(reader: Callable[[int], str]) -> str:
    """Read one partition-mode field via ``gpu_partition``, degrading to "" like every other identity field here.

    ``gpu_partition`` does not filter the literal string ``"N/A"`` amd-smi reports for hardware that does not
    partition at all (an APU, or a card whose ROCm build dropped partition info from ``static``, per the plan's
    6.4.2/7.0 schema notes) -- that is this module's call to make, not the shared reader's, so it is made here.
    """
    try:
        mode = reader(0)
    except gpu_partition.PartitionError:
        return ""
    return mode if mode and mode != "N/A" else ""


def describe_device(arch: str = "") -> DeviceIdentity:
    """Identify this machine's measurable configuration.

    Every field degrades independently to an empty string. An empty field says
    "not determined" rather than "absent", and the analyst is told as much: a
    machine whose partition mode could not be read is one whose roofs deserve a
    lower confidence, not one assumed to be in the default mode.

    Compute/memory partition go through ``hyperloom.common.gpu_partition`` (``amd-smi partition``) rather than a
    parser of this module's own: ``observe_partition`` is not used here because it also sizes the partition against
    the per-board CU/VRAM table, which this identity-only probe has no use for and should not pay the cost of.
    """
    resolved = canon_arch(arch) or detect_arch()
    marketing = _MARKETING_RE.search(_run_text(["rocminfo"]))
    return DeviceIdentity(
        arch=resolved,
        device_name=(marketing.group(1).strip() if marketing else ""),
        compute_partition=_partition_label(gpu_partition.read_partition_mode),
        memory_partition=_partition_label(gpu_partition.read_memory_partition_mode),
    )


__all__ = [
    "DeviceIdentity",
    "describe_device",
]
