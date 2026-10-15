# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule gate as the one writer of specialist branches: it lands what a judgement decides in the experiment
repository and the notebook, and takes fork and repair requests to the scheduler."""

from __future__ import annotations

import asyncio

from hyperloom.contracts.recipe import apply
from hyperloom_multibranch.contracts.fork import (
    AdmissionStatus,
    ForkAdmission,
    ForkAdmissionService,
    ForkRequest,
    RepairRequest,
)
from hyperloom_multibranch.contracts.intent import Intent
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.interfaces import JudgementSink
from hyperloom_multibranch.contracts.notebook.peers import TestRequest
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion
from hyperloom_multibranch.contracts.repo import BranchMoved, ExperimentRepo
from hyperloom_multibranch.rulegate.decision import Action, decide


def _recorded_as(judgement: Judgement) -> Conclusion:
    """An inconclusive retest is not an accuracy failure, even when one of its measurements failed the gate."""
    if not judgement.accuracy_ok and judgement.verdict is not Verdict.INCONCLUSIVE:
        return Conclusion.ACCURACY_FAILED
    return Conclusion.RECORDED_ONLY


def _require(judgement: Judgement, *actions: Action) -> None:
    action = decide(judgement)
    if action not in actions:
        raise ValueError(f"a {judgement.verdict.value} judgement calls for {action.value}")


class RuleGate:
    """``settle`` lands a commit or a record; a retest, a diagnosis and a fork are its caller's next step."""

    def __init__(self, notebook: JudgementSink, repo: ExperimentRepo, scheduler: ForkAdmissionService) -> None:
        self._notebook = notebook
        self._repo = repo
        self._scheduler = scheduler

    async def settle(self, request: TestRequest, judgement: Judgement) -> Action:
        """The judgement is written against ``request``, the first measurement of the candidate.

        A commit whose branch has moved past ``request.parent_commit`` is recorded only, and ``RECORD_ONLY`` returned:
        a gain measured against one parent must not land on another.
        """
        action = decide(judgement)
        if action is Action.COMMIT:
            # The scheduler may stop a specialist at any moment; a stop must not split a commit from its judgement.
            if not await asyncio.shield(self._commit(request, judgement)):
                return Action.RECORD_ONLY
        elif action is Action.RECORD_ONLY:
            await self._record(request, judgement)
        return action

    async def decline(self, request: TestRequest, judgement: Judgement) -> TestRecord:
        """Record a blocked gain its specialist found no repair for."""
        _require(judgement, Action.DIAGNOSE)
        return await self._record(request, judgement)

    async def fork(self, request: TestRequest, judgement: Judgement, reason: str) -> ForkAdmission:
        """Asks for the fork and records where the candidate stands; cutting the branch is not part of this call."""
        _require(judgement, Action.REQUEST_FORK)
        admission = await self._scheduler.admit(ForkRequest(request.branch, request.parent_commit, reason))
        if admission.status is AdmissionStatus.REJECTED:
            conclusion = _recorded_as(judgement)
        elif admission.status is AdmissionStatus.PENDING:
            conclusion = Conclusion.SPLIT_PENDING
        else:
            conclusion = Conclusion.SPLIT_REQUESTED
        await self._notebook.write_judgement(request.request_id, judgement, conclusion)
        return admission

    async def repair(self, request: TestRequest, judgement: Judgement, intent: Intent, diagnosis: str) -> ForkAdmission:
        admission = await self._scheduler.admit_repair(RepairRequest(intent, request.delta, judgement, diagnosis))
        rejected = admission.status is AdmissionStatus.REJECTED
        conclusion = _recorded_as(judgement) if rejected else Conclusion.REPAIR_REQUESTED
        await self._notebook.write_judgement(request.request_id, judgement, conclusion)
        return admission

    async def _commit(self, request: TestRequest, judgement: Judgement) -> bool:
        # The repository commit comes first: the judgement names the commit it made.
        recipe = apply(await self._repo.read_recipe(request.parent_commit), request.delta)
        message = f"{request.request_id}: {judgement.verdict.value} {judgement.gain_pct:+.2f}%"
        try:
            commit = await self._repo.commit(request.branch, request.parent_commit, recipe, message)
        except BranchMoved:
            await self._record(request, judgement)
            return False
        conclusion = Conclusion.COMMITTED_WITH_REPRO if judgement.fixes_correctness else Conclusion.COMMITTED
        try:
            await self._notebook.write_judgement(request.request_id, judgement, conclusion, commit, recipe.digest())
        except Exception:
            # Whatever kept the judgement out, a commit no judgement names would block the branch for good.
            await self._repo.rewind(request.branch, commit, request.parent_commit)
            raise
        return True

    async def _record(self, request: TestRequest, judgement: Judgement) -> TestRecord:
        return await self._notebook.write_judgement(request.request_id, judgement, _recorded_as(judgement))


__all__ = ["RuleGate"]
