# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-shape theoretical achievable latency for one kernel.

Self-contained: it collects its own evidence, runs its own analyst session, and
shares no state with the optimization loop. It answers one question -- what is
the best latency this operator could have on this box, for each scored shape --
and deliberately answers nothing else. It does not measure a baseline and does
not compute an attainment ratio: whoever holds a baseline, measured their own
way, divides by these numbers themselves. Two timing methodologies producing one
ratio is how an efficiency figure becomes meaningless.

The division of labour: the framework measures the box and fixes the case set,
the analyst estimates the ceiling and writes the derivation that defends it.
Nothing recomputes that estimate, so the published document is the audit trail
rather than a summary of one.

The ceiling is estimated, not measured, and is advisory everywhere it is
consumed. It must never gate a KEEP.
"""

from kernelforge.roofline_ceiling.contract import (
    CaseCeiling,
    CeilingContractError,
    CeilingReport,
    Hardware,
)
from kernelforge.roofline_ceiling.report import (
    read_report,
    render_for_prompt,
)

__all__ = [
    "CaseCeiling",
    "CeilingContractError",
    "CeilingReport",
    "Hardware",
    "read_report",
    "render_for_prompt",
]
