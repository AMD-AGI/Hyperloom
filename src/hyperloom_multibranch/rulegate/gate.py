# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule gate as the one writer of specialist branches: it lands what a judgement decides in the experiment
repository and the blackboard, takes fork and repair requests to the scheduler, and lands or records the forks and
repairs the scheduler reserved a branch for."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from hyperloom.contracts.recipe import apply
from hyperloom_multibranch.contracts.fork import (
    AdmissionStatus,
    ForkAdmission,
    ForkAdmissionService,
    ForkRequest,
    RepairRequest,
)
from hyperloom_multibranch.contracts.gate import ResultUnverified, ResultVerifier
from hyperloom_multibranch.contracts.intent import Intent
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.blackboard.interfaces import JudgementSink, LineageSink
from hyperloom_multibranch.contracts.blackboard.peers import TestRequest
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion, LineageKind
from hyperloom_multibranch.contracts.repo import BranchMoved, ExperimentRepo
from hyperloom_multibranch.rulegate.decision import Action, decide
from hyperloom_multibranch.rulegate.regression import CaseRegistry, FixProof, register_case

logger = logging.getLogger(__name__)


class GateWriter(JudgementSink, LineageSink, CaseRegistry, Protocol):
    """The gate's handle on the blackboard: it writes judgements, the ``SPLIT`` of each fork it lands, and the cases of
    the fixes it commits."""


@dataclass(frozen=True)
class _Reserved:
    """A fork or repair the scheduler reserved a branch for; ``reason`` and ``fix`` are a fork's, empty for a repair."""

    request: TestRequest
    judgement: Judgement
    reason: str
    fix: FixProof | None = None


def _recorded_as(judgement: Judgement) -> Conclusion:
    """An inconclusive retest is not an accuracy failure, even when one of its measurements failed the gate."""
    if judgement.accuracy is AccuracyVerdict.FAILED and judgement.verdict is not Verdict.INCONCLUSIVE:
        return Conclusion.ACCURACY_FAILED
    return Conclusion.RECORDED_ONLY


def _require(judgement: Judgement, *actions: Action) -> None:
    action = decide(judgement)
    if action not in actions:
        raise ValueError(f"a {judgement.verdict.value} judgement calls for {action.value}")


def _require_proof(request: TestRequest, judgement: Judgement, fix: FixProof | None) -> None:
    if judgement.fixes_correctness and fix is None:
        raise ValueError(f"{request.request_id} lands a correctness fix without the proof of its case")


class RuleGate:
    """``settle`` lands a commit or a record, or holds a blocked gain as ``DIAGNOSIS_PENDING`` until its specialist
    sends it for repair or declines it; a retest, a diagnosis and a fork are its caller's next step.

    A commit, a fork and a repair first re-verify every result behind the judgement with ``verifier``; one that does
    not re-verify is recorded ``UNVERIFIED`` and its :class:`ResultUnverified` raised, with nothing landed or asked.

    As :class:`~hyperloom_multibranch.contracts.fork.ReservedBranches` it keeps, in memory, each fork and repair the
    scheduler reserved a branch for, until it lands the fork or records the void. Asking, landing and voiding take
    turns, so a landing never runs before its fork recorded where the candidate stands.
    """

    def __init__(
        self, blackboard: GateWriter, repo: ExperimentRepo, scheduler: ForkAdmissionService, verifier: ResultVerifier
    ) -> None:
        self._blackboard = blackboard
        self._repo = repo
        self._scheduler = scheduler
        self._verifier = verifier
        self._lock = asyncio.Lock()
        self._reserved: dict[str, _Reserved] = {}

    async def settle(self, request: TestRequest, judgement: Judgement, *, fix: FixProof | None = None) -> Action:
        """The judgement is written against ``request``, the first measurement of the candidate.

        A commit whose branch has moved past ``request.parent_commit`` is recorded only, and ``RECORD_ONLY`` returned:
        a gain measured against one parent must not land on another. A commit carrying ``fix`` registers its case as a
        regression case of the new commit; a correctness fix cannot commit without one (``ValueError``).
        """
        action = decide(judgement)
        if action is Action.COMMIT:
            _require_proof(request, judgement, fix)
            await self._reverify(request, judgement)
            # The scheduler may stop a specialist at any moment; a stop must not split a commit from its judgement.
            if not await asyncio.shield(self._commit(request, judgement, fix)):
                return Action.RECORD_ONLY
        elif action is Action.RECORD_ONLY:
            await self._record(request, judgement)
        elif action is Action.DIAGNOSE:
            await self._blackboard.write_judgement(request.request_id, judgement, Conclusion.DIAGNOSIS_PENDING)
        return action

    async def decline(self, request: TestRequest, judgement: Judgement) -> TestRecord:
        """Record a blocked gain its specialist found no repair for."""
        _require(judgement, Action.DIAGNOSE)
        return await self._record(request, judgement)

    async def fork(
        self, request: TestRequest, judgement: Judgement, reason: str, *, fix: FixProof | None = None
    ) -> ForkAdmission:
        """Asks for the fork and records where the candidate stands; the branch is cut when the scheduler has it
        landed (:meth:`land_split`), and ``fix``'s case joins the regression set there. A fork is a correctness fix,
        so it cannot be asked without its proof (``ValueError``)."""
        _require(judgement, Action.REQUEST_FORK)
        _require_proof(request, judgement, fix)
        await self._reverify(request, judgement)
        async with self._lock:
            admission = await self._scheduler.admit(ForkRequest(request.branch, request.parent_commit, reason))
            if admission.status is AdmissionStatus.REJECTED:
                conclusion = _recorded_as(judgement)
            elif admission.status is AdmissionStatus.PENDING:
                conclusion = Conclusion.SPLIT_PENDING
            else:
                conclusion = Conclusion.SPLIT_REQUESTED
            self._reserve(admission, _Reserved(request, judgement, reason, fix))
            await self._blackboard.write_judgement(request.request_id, judgement, conclusion)
        return admission

    async def repair(self, request: TestRequest, judgement: Judgement, intent: Intent, diagnosis: str) -> ForkAdmission:
        await self._reverify(request, judgement)
        async with self._lock:
            admission = await self._scheduler.admit_repair(RepairRequest(intent, request.delta, judgement, diagnosis))
            rejected = admission.status is AdmissionStatus.REJECTED
            conclusion = _recorded_as(judgement) if rejected else Conclusion.REPAIR_REQUESTED
            self._reserve(admission, _Reserved(request, judgement, ""))
            await self._blackboard.write_judgement(request.request_id, judgement, conclusion)
        return admission

    async def land_split(self, branch: str) -> bool:
        """Cut the fork reserved as ``branch`` at its candidate's parent and commit the fix on it; false when no fork
        waits on ``branch``. A commit whose judgement cannot be written is taken back, and the error raised."""
        # The scheduler bounds the landing with a timeout; giving up the wait must not leave a fork half landed.
        return await asyncio.shield(self._land(branch))

    async def settle_voided(self, branch: str, reason: str) -> None:
        """Record the candidate of the fork or repair the scheduler voided as ``branch`` as only recorded."""
        await asyncio.shield(self._void(branch, reason))

    def _reserve(self, admission: ForkAdmission, reserved: _Reserved) -> None:
        if admission.branch:
            self._reserved[admission.branch] = reserved

    async def _land(self, branch: str) -> bool:
        async with self._lock:
            split = self._reserved.get(branch)
            if split is None or split.fix is None:
                return False
            request, parent = split.request, split.request.parent_commit
            recipe = apply(await self._repo.read_recipe(parent), request.delta)
            await self._repo.create_branch(branch, parent)
            message = f"{request.request_id}: split from {request.branch}: {split.reason}"
            commit = await self._repo.commit(branch, parent, recipe, message)
            try:
                await register_case(self._blackboard, commit, split.fix.candidate.request_id)
                await self._blackboard.write_judgement(
                    request.request_id, split.judgement, Conclusion.SPLIT_COMMITTED, commit, recipe.digest()
                )
            except Exception:
                await self._repo.rewind(branch, commit, parent)
                raise
            await self._blackboard.record_lineage(
                LineageEvent(
                    LineageKind.SPLIT,
                    branch,
                    commit_id=commit,
                    from_commit=parent,
                    from_branch=request.branch,
                    reason=split.reason,
                )
            )
            del self._reserved[branch]
            return True

    async def _void(self, branch: str, reason: str) -> None:
        async with self._lock:
            voided = self._reserved.pop(branch, None)
            if voided is None:
                logger.warning("no fork or repair waits on %s: its void (%s) records nothing", branch, reason)
                return
            logger.info("the scheduler voided %s (%s): %s is recorded only", branch, reason, voided.request.request_id)
            await self._record(voided.request, voided.judgement)

    async def _commit(self, request: TestRequest, judgement: Judgement, fix: FixProof | None) -> bool:
        # The repository commit comes first: the judgement names the commit it made, and so does the case.
        recipe = apply(await self._repo.read_recipe(request.parent_commit), request.delta)
        message = f"{request.request_id}: {judgement.verdict.value} {judgement.gain_pct:+.2f}%"
        try:
            commit = await self._repo.commit(request.branch, request.parent_commit, recipe, message)
        except BranchMoved:
            await self._record(request, judgement)
            return False
        conclusion = Conclusion.COMMITTED_WITH_REPRO if judgement.fixes_correctness else Conclusion.COMMITTED
        try:
            if fix is not None:
                await register_case(self._blackboard, commit, fix.candidate.request_id)
            await self._blackboard.write_judgement(request.request_id, judgement, conclusion, commit, recipe.digest())
        except Exception:
            # Whatever kept the judgement out, a commit no judgement names would block the branch for good.
            await self._repo.rewind(request.branch, commit, request.parent_commit)
            raise
        return True

    async def _reverify(self, request: TestRequest, judgement: Judgement) -> None:
        try:
            for request_id in judgement.request_ids:
                await self._verifier.reverify(request_id)
        except ResultUnverified:
            await self._blackboard.write_judgement(request.request_id, judgement, Conclusion.UNVERIFIED)
            raise

    async def _record(self, request: TestRequest, judgement: Judgement) -> TestRecord:
        return await self._blackboard.write_judgement(request.request_id, judgement, _recorded_as(judgement))


__all__ = ["GateWriter", "RuleGate"]
