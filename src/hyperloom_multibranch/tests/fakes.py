# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-memory implementations of the multi-branch contracts, for tests that need a repository or a lifecycle without
git or processes."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from hyperloom.contracts.measurement import FailureClass, TestRequest, TestResult, Validity
from hyperloom.contracts.recipe import Recipe
from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission, ForkRequest, RepairRequest
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.notebook.interfaces import TestRecordSink
from hyperloom_multibranch.contracts.notebook.records import Withdrawal
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import BranchMoved
from hyperloom_multibranch.contracts.toolbox import (
    Backpressure,
    CancelOutcome,
    MeasurementCapacity,
    QueueState,
    RequestRejected,
)

#: How a :class:`ReproToolbox` ends a test: the case's verdict (empty for a success without one), the validity it
#: failed with, or ``None`` to withdraw it.
ReproAnswer = str | Validity | None


class InMemoryRepo:
    """Starts with the baseline recipe as the root commit on ``INTEGRATION_BRANCH``."""

    def __init__(self, baseline: Recipe) -> None:
        self._recipes: dict[str, Recipe] = {}
        self._heads: dict[str, str] = {}
        self.messages: dict[str, str] = {}
        self._heads[INTEGRATION_BRANCH] = self._store(baseline, "baseline")

    def _store(self, recipe: Recipe, message: str) -> str:
        commit = f"c{len(self._recipes)}"
        self._recipes[commit] = recipe
        self.messages[commit] = message
        return commit

    async def read_recipe(self, commit: str) -> Recipe:
        return self._recipes[commit]

    async def create_branch(self, branch: str, at_commit: str) -> None:
        if branch in self._heads:
            raise ValueError(f"branch {branch!r} already exists")
        if at_commit not in self._recipes:
            raise KeyError(at_commit)
        self._heads[branch] = at_commit

    async def head(self, branch: str) -> str:
        return self._heads[branch]

    async def commit(self, branch: str, parent: str, recipe: Recipe, message: str) -> str:
        if self._heads[branch] != parent:
            raise BranchMoved(f"{branch} is at {self._heads[branch]}, not {parent}")
        self._heads[branch] = self._store(recipe, message)
        return self._heads[branch]

    async def rewind(self, branch: str, commit: str, parent: str) -> None:
        if self._heads[branch] != commit:
            raise BranchMoved(f"{branch} is at {self._heads[branch]}, not {commit}")
        self._heads[branch] = parent


class RecordingScheduler:
    """Answers every fork and repair request with ``status``, naming branch ``fork-<n>`` on acceptance, and keeps what
    it was asked."""

    def __init__(self, status: AdmissionStatus) -> None:
        self.status = status
        self.forks: list[ForkRequest] = []
        self.repairs: list[RepairRequest] = []

    async def admit(self, request: ForkRequest) -> ForkAdmission:
        self.forks.append(request)
        return self._answer()

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission:
        self.repairs.append(request)
        return self._answer()

    def _answer(self) -> ForkAdmission:
        if self.status is AdmissionStatus.REJECTED:
            return ForkAdmission(self.status)
        return ForkAdmission(self.status, branch=f"fork-{len(self.forks) + len(self.repairs)}")


class RecordingVerifier:
    """Re-verifies every result but those of ``rejects``, and keeps the request ids it was asked about."""

    def __init__(self, rejects: Iterable[str] = ()) -> None:
        self.rejects = frozenset(rejects)
        self.asked: list[str] = []

    async def reverify(self, request_id: str) -> None:
        self.asked.append(request_id)
        if request_id in self.rejects:
            raise ResultUnverified(request_id, "the manifest hash is not the recorded one")


class ReproToolbox:
    """A toolbox that runs repro tests without running anything: it accepts each request as ``p<n>``, writes it to
    ``notebook`` and ends it as ``answer`` says.

    ``busy`` answers that many submissions with :class:`Backpressure` first, ``refuse`` rejects the requests it
    accepts, and ``hold`` keeps every accepted test in flight until :meth:`finish`.
    """

    def __init__(self, notebook: TestRecordSink, answer: Callable[[TestRequest], ReproAnswer]) -> None:
        self.notebook = notebook
        self.answer = answer
        self.busy = 0
        self.refuse: Callable[[TestRequest], bool] = lambda request: False
        self.hold = False
        self.submitted: list[TestRequest] = []
        self.cancelled: list[str] = []
        self._in_flight: list[TestRequest] = []

    async def submit_test(self, request: TestRequest) -> str:
        if self.busy:
            self.busy -= 1
            raise Backpressure(request.branch_id, 1, 1)
        if self.refuse(request):
            raise RequestRejected(f"{request.parent_commit} is not in the repository", "unknown_parent")
        accepted = request.accepted(f"p{len(self.submitted) + 1}")
        self.submitted.append(accepted)
        await self.notebook.write_request(accepted)
        self._in_flight.append(accepted)
        if not self.hold:
            await self.finish()
        return str(accepted.request_id)

    async def finish(self) -> None:
        """End every test in flight as ``answer`` says."""

        ending, self._in_flight = self._in_flight, []
        for request in ending:
            await self._end(request)

    async def _end(self, request: TestRequest) -> None:
        request_id = str(request.request_id)
        answer = self.answer(request)
        if answer is None:
            await self.notebook.write_withdrawal(Withdrawal(request_id, "the branch was stopped"))
        elif isinstance(answer, Validity):
            failure = FailureClass.CANDIDATE if answer is Validity.CANDIDATE_FAILED else FailureClass.INFRA
            await self.notebook.write_result(TestResult(request_id, answer, failure, "the server did not start"))
        else:
            await self.notebook.write_result(
                TestResult(request_id, Validity.SUCCEEDED, manifest_sha256="a" * 64, repro_verdict=answer)
            )

    async def cancel(self, request_id: str) -> CancelOutcome:
        self.cancelled.append(request_id)
        held = [request for request in self._in_flight if request.request_id == request_id]
        if not held:
            return CancelOutcome.FINISHED
        self._in_flight.remove(held[0])
        await self.notebook.write_withdrawal(Withdrawal(request_id, "cancelled by its caller"))
        return CancelOutcome.WITHDRAWN

    async def queue_state(self) -> QueueState:
        return QueueState()

    async def measurement_capacity(self) -> MeasurementCapacity:
        return MeasurementCapacity.unknown()


class RecordingLifecycle:
    """Keeps the start commit, the brief, the repair and the current K of every running branch."""

    def __init__(self) -> None:
        self.running: dict[str, str] = {}
        self.briefs: dict[str, str] = {}
        self.repairs: dict[str, RepairRequest | None] = {}
        self.k: dict[str, int] = {}

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        if branch in self.running:
            raise ValueError(f"branch {branch!r} already has a specialist")
        self.running[branch] = start_commit
        self.briefs[branch] = task.brief
        self.repairs[branch] = task.repair
        self.k[branch] = 1

    async def stop(self, branch: str) -> None:
        del self.running[branch]
        del self.briefs[branch]
        del self.repairs[branch]
        del self.k[branch]

    async def set_k(self, branch: str, k: int) -> None:
        if branch not in self.running:
            raise KeyError(branch)
        if k < 1:
            raise ValueError("stop the branch instead of giving it no room")
        self.k[branch] = k
