# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a FlyDSL kernel measured against the source kernel it replaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlydslMeasurement:
    """A FlyDSL kernel's wall time beside the source kernel's, the arena speedup between them, and its accuracy.

    ``speedup`` is the arena's equal-weight mean over per-case ratios; the two wall times travel with it as raw
    evidence, not as the claim. ``snr_db`` is the accuracy measured for *this* kernel, or ``None`` when it was not
    measured, because a reading taken from another kernel is not a substitute.
    """

    source_ms: float | None
    flydsl_ms: float | None
    speedup: float | None
    snr_db: float | None = None
