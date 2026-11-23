# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a GEAK end-to-end rebench decided about the candidate it re-measured.

A rebench either validated the candidate, found its win immaterial, or measured it truthfully without beating
``current_best``. The revalidation that produced it writes the same value as the candidate's settled status, and the
breakdown reads it off the rebench rows; both spell it through these names. A revalidation that could not reach a
verdict settles as one of the failed statuses instead.
"""

from __future__ import annotations

REBENCH_VALIDATED = "validated"
REBENCH_NO_MATERIAL = "no_material"
REBENCH_NO_PROMOTE = "no_promote"
#: The rebench failed, or measured something other than the candidate.
REBENCH_FAILED = "failed"
#: The candidate's declared source could not be measured.
REBENCH_FALLBACK_FAILED = "fallback_failed"
#: The settled statuses that drop the candidate unjudged; ``no_material`` and ``no_promote`` are verdicts.
REBENCH_FAILED_STATUSES: frozenset[str] = frozenset({REBENCH_FAILED, REBENCH_FALLBACK_FAILED})
#: The error class of a ``failed`` revalidation that is final: the candidate's measurements ended with no verdict
#: as many times as the session measures one candidate, so it is dropped unmeasured.
REBENCH_UNMEASURED_GIVEN_UP = "unmeasured_attempts_exhausted"

__all__ = [
    "REBENCH_FAILED",
    "REBENCH_FAILED_STATUSES",
    "REBENCH_FALLBACK_FAILED",
    "REBENCH_NO_MATERIAL",
    "REBENCH_NO_PROMOTE",
    "REBENCH_UNMEASURED_GIVEN_UP",
    "REBENCH_VALIDATED",
]
