# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-shape theoretical achievable latency, and how much of it a kernel reached.

Two questions, deliberately kept apart.

**The ceiling** is the best latency this operator could have on this box, per
scored shape. The framework measures the box and fixes the case set; the analyst
derives the minimum legal work and composes it into a latency, writing the
derivation that defends it. Nothing recomputes that estimate, so the published
document is the audit trail rather than a summary of one.

**Attainment** is ``ceiling / measured``, and it is computed in
:mod:`~kernelforge.roofline_ceiling.attainment` against whatever latencies the
caller measured itself -- never against the figure this module timed under a
profiler. One ratio built from two clocks is how an efficiency number stops
meaning anything, so the divisor stays with whoever owns the measurement.

What the ceiling may and may not decide is worth stating exactly, because it is
now both. A campaign may **stop** on attainment: reaching the estimated ceiling
is a reason to stop spending budget. A campaign may not **KEEP** on it: whether
one candidate beats another is a measurement, settled the same way it always
was. The ceiling says when to stop trying, not what is better.
"""

from kernelforge.roofline_ceiling.attainment import (
    Attainment,
    CaseAttainment,
    measure_attainment,
)
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
    "Attainment",
    "CaseAttainment",
    "CaseCeiling",
    "CeilingContractError",
    "CeilingReport",
    "Hardware",
    "measure_attainment",
    "read_report",
    "render_for_prompt",
]
