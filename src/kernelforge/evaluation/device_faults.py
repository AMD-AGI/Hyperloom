# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recognize, in a failed run's output, a fault of the GPU itself rather than a failure of the program that ran on it.

Two faults are told apart by what they take down with them. A hardware exception -- a hang the driver resets the GPU
for -- ends every process on the machine that holds the GPU open, so the run that reports it may only have been near
the work that caused it. A memory access fault ends only the process whose kernel touched the bad address.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The driver reset the GPU: every process holding it open was ended, the one that caused it or not.
DEVICE_HANG = "hang"
#: A kernel touched an address it may not: only its own process was ended.
MEMORY_FAULT = "memory_fault"

_SIGNATURES = (
    (DEVICE_HANG, re.compile(r"HW Exception by GPU node-(\d+)")),
    (MEMORY_FAULT, re.compile(r"Memory access fault by GPU node-(\d+)")),
)


@dataclass(frozen=True)
class DeviceFault:
    """One GPU fault a run reported: what kind, on which GPU node, and the line that said so."""

    kind: str
    node: int
    line: str

    def describe(self) -> str:
        return f"{self.kind} on GPU node-{self.node}: {self.line}"


def classify(output: str) -> DeviceFault | None:
    """The GPU fault ``output`` reports, a hang before a memory access fault, or ``None`` when it reports none."""
    found: dict[str, DeviceFault] = {}
    for line in (output or "").splitlines():
        for kind, pattern in _SIGNATURES:
            match = pattern.search(line)
            if match and kind not in found:
                found[kind] = DeviceFault(kind=kind, node=int(match.group(1)), line=line.strip())
    return found.get(DEVICE_HANG) or found.get(MEMORY_FAULT)


__all__ = ["DEVICE_HANG", "MEMORY_FAULT", "DeviceFault", "classify"]
