# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule gate as the scheduler calls it, for the tests the scheduler submits itself.

PRELUDE establishes the baseline from its root tests, the integrator compares a merge test, FINAL checks a candidate.
The gate judges them as it judges a specialist's candidate, the confirming retest included, and settles nothing: what
a judgement leads to is the caller's.

:class:`ResultVerifier` is the other direction: what the gate asks of the toolbox before a decision rests on a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hyperloom_multibranch.contracts.judgement import Judgement
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord


class NotMeasured(Exception):
    """The tests gave the gate nothing to judge by: withdrawn, not measured, measured without a value the rule reads,
    a result that does not re-verify (:class:`ResultUnverified`), or, for the baseline, root tests whose eval score is
    below the floor."""


class ResultUnverified(NotMeasured):
    """The notebook's result of ``request_id`` is not backed by its archived signed bundle, or the bundle cannot be
    read; ``reason`` says which. Final: the gate does not ask again."""

    def __init__(self, request_id: str, reason: str) -> None:
        super().__init__(f"{request_id} did not re-verify: {reason}")
        self.request_id = request_id
        self.reason = reason


class ResultVerifier(Protocol):
    """What the gate asks before a decision rests on a result (hosting design L26); the toolbox implements it."""

    async def reverify(self, request_id: str) -> None:
        """Check the notebook's result of ``request_id`` against its archived signed bundle, its values included;
        raises :class:`ResultUnverified` when it does not match or cannot be checked."""


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


__all__ = ["NotMeasured", "ResultUnverified", "ResultVerifier", "SessionGate"]
