# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recognize, in a failed run's output, a fault of the GPU itself rather than a failure of the program that ran on it.

Two faults are told apart by what they take down with them. A hardware exception -- a hang the driver resets the GPU
for -- ends every process on the machine that holds the GPU open, so the run that reports it may only have been near
the work that caused it. A memory access fault ends only the process whose kernel touched the bad address. A device
that runs no work for a reason its output does not name is unresponsive; only a probe reports that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The driver reset the GPU: every process holding it open was ended, the one that caused it or not.
DEVICE_HANG = "hang"
#: A kernel touched an address it may not: only its own process was ended.
MEMORY_FAULT = "memory_fault"
#: The device ran no work, and nothing the run printed says why.
DEVICE_UNRESPONSIVE = "unresponsive"

_SIGNATURES = (
    (DEVICE_HANG, re.compile(r"HW Exception by GPU node-(\d+)")),
    (MEMORY_FAULT, re.compile(r"Memory access fault by GPU node-(\d+)")),
)


@dataclass(frozen=True)
class DeviceFault:
    """One GPU fault: what kind, the line that said so, and the GPU node when that line names one."""

    kind: str
    line: str
    node: int | None = None

    def describe(self) -> str:
        where = "" if self.node is None else f" on GPU node-{self.node}"
        return f"{self.kind}{where}: {self.line}"


def classify(output: str) -> DeviceFault | None:
    """The GPU fault ``output`` reports, a hang before a memory access fault, or ``None`` when it reports none."""
    found: dict[str, DeviceFault] = {}
    for line in (output or "").splitlines():
        for kind, pattern in _SIGNATURES:
            match = pattern.search(line)
            if match and kind not in found:
                found[kind] = DeviceFault(kind=kind, line=line.strip(), node=int(match.group(1)))
    return found.get(DEVICE_HANG) or found.get(MEMORY_FAULT)


__all__ = ["DEVICE_HANG", "DEVICE_UNRESPONSIVE", "MEMORY_FAULT", "DeviceFault", "classify"]
