# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule gate as the scheduler calls it, for the tests the scheduler submits itself.

PRELUDE establishes the baseline from its root tests, or for AgentX from the PRELUDE child's own run of the root
(:class:`PreludeMeasurement`) where it starts one, the integrator compares a merge test, FINAL checks a candidate. The gate judges them as it judges a specialist's candidate, the confirming retest included, and settles nothing: what
a judgement leads to is the caller's.

:class:`ResultVerifier` is the other direction: what the gate asks of the toolbox before a decision rests on a result.
:class:`RegressionCheck` is the regression set the integrator runs before a merge and FINAL on every candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from hyperloom.contracts.measurement import TestResult
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.judgement import Judgement
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord


class NotMeasured(Exception):
    """The tests gave the gate nothing to judge by: withdrawn, not measured, measured without a value the rule reads,
    a result that does not re-verify (:class:`ResultUnverified`), or, for the baseline, root tests whose eval score is
    below the floor."""


class ResultUnverified(NotMeasured):
    """The blackboard's result of ``request_id`` is not backed by its archived signed bundle, or the bundle cannot be
    read; ``reason`` says which. Final: the gate does not ask again."""

    def __init__(self, request_id: str, reason: str) -> None:
        super().__init__(f"{request_id} did not re-verify: {reason}")
        self.request_id = request_id
        self.reason = reason


class ResultVerifier(Protocol):
    """What the gate asks before a decision rests on a result (hosting design L26); the toolbox implements it."""

    async def reverify(self, request_id: str) -> None:
        """Check the blackboard's result of ``request_id`` against its archived signed bundle, its values included;
        raises :class:`ResultUnverified` when it does not match or cannot be checked."""


@dataclass(frozen=True)
class PreludeMeasurement:
    """The PRELUDE child's own measurement of the root, read from its landed run in ``run_dir``.

    ``report`` and ``export`` are the session's copies of the run's ``benchmark_report.json`` and per-request export,
    with the sha256 they had when ``result`` was read from them. The run took ``wall_seconds`` from launch to exit. No
    card guard watched it and no signed bundle backs it, so the copies are all that ties ``result`` to the run.
    """

    result: TestResult
    run_dir: str
    report: str
    report_sha256: str
    export: str
    export_sha256: str
    wall_seconds: float


class SessionGate(Protocol):
    """Each call waits for the tests it names; one that ends without a measurement raises :class:`NotMeasured`."""

    async def baseline_test_count(self) -> int:
        """How many root tests PRELUDE submits for the baseline; 0 for a rule that anchors on the PRELUDE child's own
        run, which takes one root test instead where PRELUDE starts no child."""

    async def establish_baseline(self, root_commit: str, request_ids: Sequence[str]) -> BaselineRecord:
        """Write the session's baseline and bar from the root tests that measured; the rule says how many it needs,
        and their eval score must clear the floor."""

    async def establish_prelude_baseline(self, root_commit: str, measured: PreludeMeasurement) -> BaselineRecord:
        """Write the session's baseline from the PRELUDE child's own measurement of the root, for a rule that asks for
        no root test (:meth:`baseline_test_count` is 0); raises :class:`NotMeasured` when the copies no longer match
        their sha256 or the measurement lacks what the rule reads, and ``ValueError`` for a rule that asks for root
        tests."""

    async def compare(self, request_id: str) -> Judgement:
        """Judge the test against its parent's recorded value, retesting a gain on another node before it counts."""

    async def request_confirm(self, request_id: str) -> str:
        """Submit the test's confirming retest before it has a result, returning its id; :meth:`compare` judges a gain
        with it, and a caller whose test needs no confirm withdraws it."""

    async def accuracy_passes(self, request_id: str) -> bool:
        """Whether the test clears the accuracy gate every commit clears."""


@dataclass(frozen=True)
class RegressionOutcome:
    """What the regression set found: ``passed``, or why not in ``reason``, naming each failed case and its cause."""

    passed: bool
    reason: str = ""


class RegressionCheck(Protocol):
    """Runs the reproduction case of every correctness fix ``base_commit`` or a commit in ``sources`` builds on, once
    each, as a repro test of ``base_commit`` with ``candidate_delta`` applied.

    The integrator names the commit a merge brings in as ``sources``: a fix only that commit's ancestry holds is not in
    the integration head's. FINAL names none, because the candidate's own commit carries its whole ancestry.

    It passes only when every case ran clean, and passes without a test when there is no case. A case that reproduced,
    or ended without a verdict for any reason, fails it; a test outcome is never raised. A caller that is cancelled
    takes the repro tests back.
    """

    async def check(
        self, base_commit: str, candidate_delta: RecipeDelta, *, sources: tuple[str, ...]
    ) -> RegressionOutcome: ...


__all__ = [
    "NotMeasured",
    "PreludeMeasurement",
    "RegressionCheck",
    "RegressionOutcome",
    "ResultUnverified",
    "ResultVerifier",
    "SessionGate",
]
