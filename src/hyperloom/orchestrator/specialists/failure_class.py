# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a failed specialist run is classified, and whether it is retried."""

from __future__ import annotations

import enum

__all__ = [
    "SpecialistFailureType",
    "classify_specialist_failure",
]


class SpecialistFailureType(str, enum.Enum):
    """Coarse failure taxonomy for a finished specialist run.

    Only the transient infrastructure members (``TIMEOUT`` /
    ``STALE_HEARTBEAT`` / ``CRASH``) are retry-eligible; semantic outcomes are
    left for the orchestrator to act on.
    """

    NONE = "none"  # succeeded
    TIMEOUT = "timeout"  # subprocess wall-clock kill
    STALE_HEARTBEAT = "stale_heartbeat"  # heartbeat went silent (hang)
    CRASH = "crash"  # nonzero exit / backend error
    NO_OUTPUT = "no_output"  # ran clean but emitted no done / empty
    CONFIG = "config"  # unknown domain / no workspace
    UNKNOWN = "unknown"


def classify_specialist_failure(
    runner_status: str,
    error: str,
) -> tuple[SpecialistFailureType, bool]:
    """Map a :class:`SpecialistRunResult` ``(status, error)`` to a failure
    type + retry-eligibility flag.

    ``status == 'stale'`` marks a subprocess that died with a ``backend_error``
    (timeout / stale-heartbeat / crash) and left nothing usable behind;
    ``partial`` means it died the same way but a checkpoint was salvaged, so the
    failure is reported without discarding the work; ``empty_synthesised`` means
    it exited cleanly without a usable ``specialist_done``.

    Args:
        runner_status: The :class:`SpecialistRunResult` status string.
        error: The associated error string (drives sub-classification).

    Returns:
        A ``(failure_type, retry_eligible)`` tuple.
    """
    status = (runner_status or "").strip().lower()
    err = (error or "").strip().lower()
    if status == "succeeded":
        return SpecialistFailureType.NONE, False
    if status == "partial":
        # Salvaged work: classify the failure but never retry over it.
        if "timeout" in err:
            return SpecialistFailureType.TIMEOUT, False
        if "stale_heartbeat" in err:
            return SpecialistFailureType.STALE_HEARTBEAT, False
        return SpecialistFailureType.CRASH, False
    if status == "stale":
        if "timeout" in err:
            ftype = SpecialistFailureType.TIMEOUT
        elif "stale_heartbeat" in err:
            ftype = SpecialistFailureType.STALE_HEARTBEAT
        else:
            ftype = SpecialistFailureType.CRASH
        return ftype, True
    if status == "empty_synthesised":
        if "unknown_domain" in err or "no_workspace" in err:
            return SpecialistFailureType.CONFIG, False
        return SpecialistFailureType.NO_OUTPUT, False
    return SpecialistFailureType.UNKNOWN, False
