# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tests the scheduling side itself issues: all of them run on the integration branch.

PRELUDE's baseline, the integrator's merge tests and FINAL's validation differ only in what they measure, from which
commit, and at which priority; this module builds each one the same way.
"""

from __future__ import annotations

import uuid

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.peers import TestKind, TestRequest
from hyperloom_multibranch.contracts.scheduling import PriorityClass

NO_DELTA = RecipeDelta()


def integration_test_request(
    kind: TestKind,
    parent_commit: str,
    intent: str,
    purpose: str,
    priority: PriorityClass,
    delta: RecipeDelta = NO_DELTA,
) -> TestRequest:
    """A fresh request on the integration branch measuring ``delta`` from ``parent_commit``."""

    return TestRequest(
        request_id=uuid.uuid4().hex,
        kind=kind,
        branch=INTEGRATION_BRANCH,
        parent_commit=parent_commit,
        intent_fingerprint=intent,
        delta=delta,
        purpose=purpose,
        priority=priority.value,
    )


__all__ = ["NO_DELTA", "integration_test_request"]
