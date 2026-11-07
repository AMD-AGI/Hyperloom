# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``error_class`` a round that inference_suite stopped at a :class:`FailedStep` is recorded under."""

from __future__ import annotations

from inference_suite.contracts import FailedStep

from ...state.failure_evidence import FAILED_STEP_ERROR_CLASSES
from ..stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, SESSION_TIME_EXHAUSTED_CLASS

# The failed steps whose class an action module owns; every other step is mapped in ``failure_evidence``. A round
# inference_suite stopped on its deadline or on cancel was stopped by this run, so it is recorded as the run's own
# stop rather than as a failure.
_ACTION_OWNED_STEP_CLASSES: dict[FailedStep, str] = {
    FailedStep.DEADLINE_EXHAUSTED: SESSION_TIME_EXHAUSTED_CLASS,
    FailedStep.CANCELLED: ORCHESTRATOR_CANCELLED_CLASS,
}


def failed_step_error_class(step: FailedStep) -> str:
    """Return the ``error_class`` a round that failed at ``step`` is recorded under."""
    return _ACTION_OWNED_STEP_CLASSES.get(step) or FAILED_STEP_ERROR_CLASSES[step]


__all__ = ["failed_step_error_class"]
