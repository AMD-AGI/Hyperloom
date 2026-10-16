# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule gate as the scheduler calls it, for the tests the scheduler submits itself.

PRELUDE establishes the baseline from its root tests, the integrator compares a merge test, FINAL checks a candidate.
The gate judges them as it judges a specialist's candidate, the confirming retest included, and settles nothing: what
a judgement leads to is the caller's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hyperloom_multibranch.contracts.judgement import Judgement
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord


class NotMeasured(Exception):
    """The tests gave the gate nothing to judge by: withdrawn, not measured, measured without a value the rule reads,
    or, for the baseline, root tests whose eval score is below the floor."""


class SessionGate(Protocol):
    """Each call waits for the tests it names; one that ends without a measurement raises :class:`NotMeasured`."""

    async def baseline_test_count(self) -> int:
        """How many root tests PRELUDE submits for the baseline."""

    async def establish_baseline(self, root_commit: str, request_ids: Sequence[str]) -> BaselineRecord:
        """Write the session's baseline and bar from the root tests that measured; the bar needs two of them, and
        their eval score must clear the floor."""

    async def compare(self, request_id: str) -> Judgement:
        """Judge the test against its parent's recorded value, retesting a gain on another node before it counts."""

    async def accuracy_passes(self, request_id: str) -> bool:
        """Whether the test clears the accuracy gate every commit clears."""


__all__ = ["NotMeasured", "SessionGate"]
