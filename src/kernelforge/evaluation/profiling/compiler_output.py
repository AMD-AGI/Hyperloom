# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parser for GPU compiler output (hipcc/clang register info)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kernelforge.utils.gpu_arch import CDNA_OCCUPANCY, OccupancyRule


@dataclass
class RegisterInfo:
    """Register usage extracted from compiler output or ISA dump."""

    vgpr: int = 0
    agpr: int = 0
    sgpr: int = 0
    lds_bytes: int = 0
    spill_bytes: int = 0
    occupancy: int = 0
    #: The target's occupancy steps; the Instinct ones unless the caller names the target.
    occupancy_rule: OccupancyRule = CDNA_OCCUPANCY

    @property
    def has_spill(self) -> bool:
        return self.spill_bytes > 0

    @property
    def occupancy_analysis(self) -> str:
        """Where the counts sit against the occupancy steps of ``occupancy_rule``."""
        rule = self.occupancy_rule
        parts = []
        if self.vgpr > 0:
            if self.vgpr <= rule.vgpr_limit:
                parts.append(f"VGPR={self.vgpr} (≤{rule.vgpr_limit}: {rule.vgpr_within})")
            else:
                parts.append(f"VGPR={self.vgpr} (>{rule.vgpr_limit}: {rule.vgpr_beyond})")
        if self.agpr > 0:
            parts.append(f"AGPR={self.agpr}")
        if self.sgpr > 0:
            parts.append(f"SGPR={self.sgpr}")
        if self.lds_bytes > 0:
            lds_kb = self.lds_bytes / 1024
            if lds_kb <= rule.lds_limit_kib:
                parts.append(f"LDS={lds_kb:.1f}KB (≤{rule.lds_limit_kib}KB: {rule.lds_within})")
            else:
                parts.append(f"LDS={lds_kb:.1f}KB (>{rule.lds_limit_kib}KB: {rule.lds_beyond})")
        if self.has_spill:
            parts.append(f"SPILL={self.spill_bytes}B ⚠️")
        return "; ".join(parts) if parts else "unknown"

    def summary(self) -> str:
        return (
            f"VGPR={self.vgpr} AGPR={self.agpr} SGPR={self.sgpr} "
            f"LDS={self.lds_bytes}B spill={self.spill_bytes}B\n"
            f"Analysis: {self.occupancy_analysis}"
        )


def parse_register_info(text: str) -> RegisterInfo:
    """Extract register usage from hipcc -v output or ISA dump."""
    info = RegisterInfo()

    # .vgpr_count patterns
    m = re.search(r"\.vgpr_count:\s*(\d+)", text)
    if m:
        info.vgpr = int(m.group(1))
    else:
        # Alternative: NumVgprs from clang verbose
        m = re.search(r"NumVgprs:\s*(\d+)", text)
        if m:
            info.vgpr = int(m.group(1))

    # .agpr_count
    m = re.search(r"\.agpr_count:\s*(\d+)", text)
    if m:
        info.agpr = int(m.group(1))

    # .sgpr_count
    m = re.search(r"\.sgpr_count:\s*(\d+)", text)
    if m:
        info.sgpr = int(m.group(1))
    else:
        m = re.search(r"NumSgprs:\s*(\d+)", text)
        if m:
            info.sgpr = int(m.group(1))

    # LDS size
    m = re.search(r"\.lds_size:\s*(\d+)", text)
    if m:
        info.lds_bytes = int(m.group(1))
    else:
        m = re.search(r"LDSByteSize:\s*(\d+)", text)
        if m:
            info.lds_bytes = int(m.group(1))

    # Spill
    m = re.search(r"ScratchSize:\s*(\d+)", text)
    if m:
        info.spill_bytes = int(m.group(1))
    else:
        m = re.search(r"\.scratch_memory_size:\s*(\d+)", text)
        if m:
            info.spill_bytes = int(m.group(1))

    # Occupancy
    m = re.search(r"Occupancy:\s*(\d+)", text)
    if m:
        info.occupancy = int(m.group(1))

    return info
