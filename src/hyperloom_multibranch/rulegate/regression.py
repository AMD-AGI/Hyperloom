# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Correctness fixes and their reproduction cases.

A reproduction case is the attachments of a repro test: a directory with a ``repro.sh`` at its top, whose verdict on a
commit is ``reproduced`` (the failure occurs) or ``not_reproduced`` (it ran clean). :class:`ReproRunner` is the one
place that submits repro tests; :class:`RegressionSet` runs the cases of every fix a commit builds on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from hyperloom.contracts.measurement import (
    NOT_REPRODUCED,
    REPRODUCED,
    AttachmentRef,
    Purpose,
    TestKind,
    TestRequest,
    Validity,
)
from hyperloom.contracts.recipe import RecipeDelta, apply
from hyperloom_multibranch.contracts.gate import RegressionOutcome, ResultUnverified, ResultVerifier
from hyperloom_multibranch.contracts.blackboard.interfaces import CommitQuery, GateRecords, RecordSearch
from hyperloom_multibranch.contracts.blackboard.records import RegressionCase, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import Backpressure, RequestRejected, Toolbox
from hyperloom_multibranch.rulegate.service import Polling

#: Seconds before a repro test the toolbox refused with backpressure is submitted again.
RESUBMIT_SECONDS = 30.0
NO_DELTA = RecipeDelta()

T = TypeVar("T")


@dataclass(frozen=True)
class ReproRun:
    """One run of a case: the ``request_id`` the toolbox accepted it as (empty when it refused it), and either the
    case's ``verdict`` (``reproduced`` or ``not_reproduced``) or the ``failure`` that left it without one."""

    request_id: str
    verdict: str = ""
    failure: str = ""

    @property
    def outcome(self) -> str:
        """The verdict, or what kept the case from one."""

        return self.verdict or self.failure


class ReproRunner:
    """Runs a case as a repro test through ``toolbox`` and reads its verdict back from ``blackboard``.

    A verdict counts only once ``verifier`` re-verified the result behind it. A run never raises for how the test
    ended: a refusal, a withdrawal, a failure, a success without a verdict and a result that does not re-verify are each
    a :attr:`ReproRun.failure`. A caller that stops waiting takes its test back, shielded from a second cancel.
    """

    def __init__(
        self, toolbox: Toolbox, blackboard: RecordSearch, verifier: ResultVerifier, *, polling: Polling = Polling()
    ) -> None:
        self._toolbox = toolbox
        self._blackboard = blackboard
        self._verifier = verifier
        self._polling = polling

    async def run(
        self, commit: str, delta: RecipeDelta, attachments: tuple[AttachmentRef, ...], *, branch: str
    ) -> ReproRun:
        """Run the case ``attachments`` on ``commit`` with ``delta`` applied, as a test of ``branch``."""

        request = TestRequest(
            kind=TestKind.REPRO,
            purpose=Purpose.REPRO,
            branch_id=branch,
            parent_commit=commit,
            delta=delta,
            attachments=attachments,
        )
        try:
            request_id = await self._submit(request)
        except RequestRejected as error:
            return ReproRun("", failure=f"refused by the toolbox ({error})")
        try:
            return await self._verdict(await self._ended(request_id))
        except asyncio.CancelledError:
            await asyncio.shield(self._toolbox.cancel(request_id))
            raise

    async def _submit(self, request: TestRequest) -> str:
        while True:
            try:
                return await self._toolbox.submit_test(request)
            except Backpressure:
                await self._polling.sleep(RESUBMIT_SECONDS)

    async def _ended(self, request_id: str) -> TestRecord:
        while (record := await self._blackboard.record(request_id)).in_flight:
            await self._polling.sleep(self._polling.seconds)
        return record

    async def _verdict(self, record: TestRecord) -> ReproRun:
        request_id, result = record.request_id, record.result
        if result is None:
            return ReproRun(request_id, failure=f"withdrawn ({record.withdrawal_reason})")
        if result.validity is not Validity.SUCCEEDED:
            reason = result.failure_reason or "no reason given"
            return ReproRun(request_id, failure=f"ended {result.validity.value} ({reason})")
        if not result.repro_verdict:
            return ReproRun(request_id, failure="succeeded without a verdict")
        try:
            await self._verifier.reverify(request_id)
        except ResultUnverified as error:
            return ReproRun(request_id, failure=str(error))
        return ReproRun(request_id, verdict=result.repro_verdict)


class RegressionBlackboard(RecordSearch, CommitQuery, Protocol):
    """What the regression set reads: the cases of a commit's ancestry and the repro test each case came from."""


class RegressionSet:
    """The session's :class:`~hyperloom_multibranch.contracts.gate.RegressionCheck`, running each case on the
    integration branch with ``runner``, with the attachments of the repro test the case was registered from.

    A case's verdict on a recipe is kept, keyed by the digest of the recipe ``repo`` and the delta make, so the same
    case on the same recipe is not run again; a case that ran to no verdict runs again next time.
    """

    def __init__(self, blackboard: RegressionBlackboard, runner: ReproRunner, repo: ExperimentRepo) -> None:
        self._blackboard = blackboard
        self._runner = runner
        self._repo = repo
        self._verdicts: dict[tuple[str, str], str] = {}

    async def check(
        self, base_commit: str, candidate_delta: RecipeDelta, *, sources: tuple[str, ...]
    ) -> RegressionOutcome:
        cases = await self._cases((base_commit, *sources))
        if not cases:
            return RegressionOutcome(True)
        digest = apply(await self._repo.read_recipe(base_commit), candidate_delta).digest()
        outcomes = await _gathered(self._outcome(case, base_commit, candidate_delta, digest) for case in cases)
        failed = [f"{case.case_ref}: {outcome}" for case, outcome in zip(cases, outcomes) if outcome != NOT_REPRODUCED]
        return RegressionOutcome(not failed, "; ".join(failed))

    async def _cases(self, commits: Iterable[str]) -> list[RegressionCase]:
        """Every commit's cases, each once, in the order they are first found."""

        found: dict[str, RegressionCase] = {}
        for commit in commits:
            for case in await self._blackboard.regression_cases_for(commit):
                found.setdefault(case.request_id, case)
        return list(found.values())

    async def _outcome(self, case: RegressionCase, commit: str, delta: RecipeDelta, digest: str) -> str:
        key = (case.request_id, digest)
        if key in self._verdicts:
            return self._verdicts[key]
        attachments = (await self._blackboard.record(case.request_id)).request.attachments
        run = await self._runner.run(commit, delta, attachments, branch=INTEGRATION_BRANCH)
        if run.verdict:
            self._verdicts[key] = run.verdict
        return run.outcome


class FixNotProven(Exception):
    """The case did not reproduce on the fix's parent, or did not run clean on the candidate; ``reason`` says which
    and why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class FixProof:
    """A correctness fix's case: ``reproduced`` on the ``parent``, ``not_reproduced`` on the ``candidate`` (whose
    request is the one its regression case is registered from), and what it did on the session's ``root``."""

    parent: ReproRun
    candidate: ReproRun
    root: ReproRun

    @property
    def foundation(self) -> bool:
        """The bug was in the baseline already; a root run that ended without a verdict says it was not, and keeps
        why in ``root.failure``."""

        return self.root.verdict == REPRODUCED


async def prove_fix(
    runner: ReproRunner, blackboard: CommitQuery, candidate: TestRequest, case: tuple[AttachmentRef, ...]
) -> FixProof:
    """Run ``case`` on ``candidate``'s branch, on its parent, on the parent with its delta and on the baseline's root
    commit, at once; the parent's run stands for the root's when the parent is the root.

    Raises :class:`FixNotProven` unless the parent reproduced and the candidate ran clean.
    """

    parent, root = candidate.parent_commit, (await blackboard.baseline()).root_commit
    targets = [(parent, NO_DELTA), (parent, candidate.delta)]
    if root != parent:
        targets.append((root, NO_DELTA))
    runs = await _gathered(runner.run(commit, delta, case, branch=candidate.branch) for commit, delta in targets)
    parent_run, candidate_run, *root_run = runs
    proof = FixProof(parent_run, candidate_run, root_run[0] if root_run else parent_run)
    if proof.parent.verdict != REPRODUCED:
        raise FixNotProven(f"the case did not reproduce on the parent {parent}: {proof.parent.outcome}")
    if proof.candidate.verdict != NOT_REPRODUCED:
        raise FixNotProven(f"the case did not run clean on the candidate: {proof.candidate.outcome}")
    return proof


class CaseRegistry(RecordSearch, GateRecords, Protocol):
    """Where a fix's case is registered: the rule gate's handle on the blackboard."""


async def register_case(blackboard: CaseRegistry, fix_commit: str, repro_request_id: str) -> RegressionCase:
    """Register the case of the repro test ``repro_request_id`` as the regression case of ``fix_commit``.

    The test is the fix's candidate run: raises ``ValueError`` unless it is a repro test whose case ran clean.
    """

    record = await blackboard.record(repro_request_id)
    clean = record.result is not None and record.result.repro_verdict == NOT_REPRODUCED
    if record.request.kind is not TestKind.REPRO or not clean:
        raise ValueError(f"{repro_request_id} is not a repro test whose case ran clean")
    case_ref = ", ".join(attachment.id for attachment in record.request.attachments)
    case = RegressionCase(case_ref, fix_commit, repro_request_id)
    await blackboard.add_regression_case(case)
    return case


async def _gathered(awaitables: Iterable[Awaitable[T]]) -> list[T]:
    """Await every one concurrently; once one raises, or the caller is cancelled, the others are cancelled and awaited,
    so each takes its test back before this returns."""

    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "NO_DELTA",
    "RESUBMIT_SECONDS",
    "CaseRegistry",
    "FixNotProven",
    "FixProof",
    "RegressionBlackboard",
    "RegressionSet",
    "ReproRun",
    "ReproRunner",
    "prove_fix",
    "register_case",
]
