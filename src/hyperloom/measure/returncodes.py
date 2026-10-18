# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Return codes only the hosted measurement implementation produces.

They continue the negative sentinel range the local flow already uses (-910 to -917 in
``_subprocess_kill`` and ``_ray_serving``), so no process exit status can collide with them. Downstream code treats
them as ordinary non-zero return codes; :mod:`hyperloom.measure.failures` decides what each one means.
"""

from __future__ import annotations

#: No authoritative sandbox could be obtained: creation timed out, capacity was short, or the runner never came up.
HOSTED_RUNNER_UNAVAILABLE_RC: int = -920
#: The runner or the platform failed in a way that says nothing about the candidate: node fault, lost runner, an
#: unreadable result, a failed archive write.
HOSTED_INFRA_FAILURE_RC: int = -921
#: The runner refused the instruction sheet (hash mismatch, incompatible version): a program error, not a verdict.
HOSTED_SPEC_REJECTED_RC: int = -922
#: The result bundle failed verification: a program error or a tampered result, never a measurement.
HOSTED_RESULT_REJECTED_RC: int = -923

HOSTED_RETURNCODES: frozenset[int] = frozenset(
    {
        HOSTED_RUNNER_UNAVAILABLE_RC,
        HOSTED_INFRA_FAILURE_RC,
        HOSTED_SPEC_REJECTED_RC,
        HOSTED_RESULT_REJECTED_RC,
    }
)
