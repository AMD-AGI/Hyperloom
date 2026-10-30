# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a GEAK end-to-end rebench decided about the candidate it re-measured.

A rebench either validated the candidate, found its win immaterial, measured it truthfully without beating
``current_best``, or could not conclude because the config it was asked to reproduce did not engage. The revalidation
that produced it writes the same value as the candidate's settled status, and the breakdown reads it off the rebench
rows; both spell it through these names. A revalidation that could not reach a verdict at all settles as one of the
failed statuses instead.
"""

from __future__ import annotations

REBENCH_VALIDATED = "validated"
REBENCH_NO_MATERIAL = "no_material"
REBENCH_NO_PROMOTE = "no_promote"
#: The 2b rebench could not rule, so the round replays through GEAK's own harness (2a).
REBENCH_FALLBACK = "fallback"
#: The revalidation itself failed.
REBENCH_FAILED = "failed"
#: The 2a replay a fallback sent the candidate to failed as well.
REBENCH_FALLBACK_FAILED = "fallback_failed"
#: The settled statuses that drop the candidate unjudged; ``no_material`` and ``no_promote`` are verdicts.
REBENCH_FAILED_STATUSES: frozenset[str] = frozenset({REBENCH_FAILED, REBENCH_FALLBACK_FAILED})

__all__ = [
    "REBENCH_FAILED",
    "REBENCH_FAILED_STATUSES",
    "REBENCH_FALLBACK",
    "REBENCH_FALLBACK_FAILED",
    "REBENCH_NO_MATERIAL",
    "REBENCH_NO_PROMOTE",
    "REBENCH_VALIDATED",
]
