# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parser for GPU compiler output (hipcc/clang register info, errors, warnings)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RegisterInfo:
    """Register usage extracted from compiler output or ISA dump."""

    vgpr: int = 0
    agpr: int = 0
    sgpr: int = 0
    lds_bytes: int = 0
    spill_bytes: int = 0
    occupancy: int = 0

    @property
    def has_spill(self) -> bool:
        return self.spill_bytes > 0

    @property
    def occupancy_analysis(self) -> str:
        """Occupancy heuristic for CDNA3/CDNA4 (gfx942/gfx950): 256-VGPR and
        ~80KB-LDS dual-occupancy thresholds hold for both."""
        parts = []
        if self.vgpr > 0:
            if self.vgpr <= 256:
                parts.append(f"VGPR={self.vgpr} (≤256: occupancy≥2 possible)")
            else:
                parts.append(f"VGPR={self.vgpr} (>256: occupancy=1 ONLY)")
        if self.agpr > 0:
            parts.append(f"AGPR={self.agpr}")
        if self.sgpr > 0:
            parts.append(f"SGPR={self.sgpr}")
        if self.lds_bytes > 0:
            lds_kb = self.lds_bytes / 1024
            if lds_kb <= 80:
                parts.append(f"LDS={lds_kb:.1f}KB (≤80KB: dual-occupancy OK)")
            else:
                parts.append(f"LDS={lds_kb:.1f}KB (>80KB: single-occupancy)")
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


def parse_compiler_errors(text: str) -> list[str]:
    """Extract error lines from compiler output."""
    errors = []
    for line in text.splitlines():
        if re.search(r"\berror\b", line, re.IGNORECASE):
            errors.append(line.strip())
    return errors


def parse_compiler_warnings(text: str) -> list[str]:
    """Extract warning lines from compiler output."""
    warnings = []
    for line in text.splitlines():
        if re.search(r"\bwarning\b", line, re.IGNORECASE):
            warnings.append(line.strip())
    return warnings
