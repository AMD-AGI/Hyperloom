# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-shape theoretical achievable latency, and how much of it a kernel reached.

Two questions, deliberately kept apart.

**The ceiling** is the best latency this operator could have on this box, per
scored shape. The analyst owns all of it: it measures the machine's roofs
during its session, derives the minimum legal work, composes the two into a
latency, and writes both the answer and the derivation behind it. The framework
settles what must not vary between runs -- which shapes are scored, which
machine this is, what the kernel really dispatches -- and then reads back one
file, checking only that it can be read.

**Attainment** is ``ceiling / measured``, computed in
:mod:`~kernelforge.roofline_ceiling.attainment` against whatever latencies the
caller measured itself. One ratio built from two clocks is how an efficiency
number stops meaning anything, so the divisor stays with whoever owns the
measurement.

What the ceiling may and may not decide is worth stating exactly, because it is
both. A campaign may **stop** on attainment: reaching the estimated ceiling is
a reason to stop spending budget. A campaign may not **KEEP** on it: whether
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
    "measure_attainment",
    "read_report",
    "render_for_prompt",
]
