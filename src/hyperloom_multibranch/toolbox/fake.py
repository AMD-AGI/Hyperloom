# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A toolbox that measures nothing: the real toolbox with a fake executor and seeded, configurable results.

:class:`FakeToolbox` is a :class:`~hyperloom_multibranch.toolbox.toolbox.RealToolbox` whose executor is
:class:`FakeExecutor`. Acceptance, the queue (priorities, per-branch K, global limit, backpressure), the retry rules of
design 4.5, quarantine, the submit log and the notebook writes are the real toolbox's own code, so a test written
against the fake exercises the semantics the hosted executor will have. Only one attempt is faked: the fate drawn for
it, the node it lands on and the values it measures come from :mod:`hyperloom_multibranch.toolbox.fake_model`, seeded
by ``FakeToolboxConfig.seed`` and by the request and attempt number, so a run does not depend on the order tests ran in.

Time is virtual: each attempt takes ``attempt_seconds`` on a clock that only attempts advance, so results carry
timestamps and GPU seconds and nothing sleeps.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from hyperloom.contracts.measurement import TestKind, TestRequest, TestResult, Validity
from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.notebook.interfaces import ClaimService, RecordSearch
from hyperloom_multibranch.contracts.notebook.records import QuarantineEvent, Withdrawal
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import (
    CancelOutcome,
    MeasurementCapacity,
    RequestRejected,
    WorkloadKind,
)
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.toolbox.attempts import MeasurementRefused, TestJob
from hyperloom_multibranch.toolbox.fake_model import (
    AttemptOutcome,
    AttemptConstraints,
    FakeMeasurer,
    FakeToolboxConfig,
    Fate,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.submission_log import SubmissionLog
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, RealToolbox, ToolboxStores, ToolboxTuning


class ToolboxNotebook(Protocol):
    """What the fake writes to: the notebook's ``TestRecordSink`` and ``NodeHealth`` (a ``LocalNotebook`` is one)."""

    async def write_request(self, request: object) -> object: ...

    async def write_result(self, result: object) -> object: ...

    async def write_withdrawal(self, withdrawal: Withdrawal) -> object: ...

    async def record_quarantine(self, event: QuarantineEvent) -> None: ...

    async def quarantined(self) -> Sequence[QuarantineEvent]: ...


class MemoryNotebook:
    """The notebook the fake writes to when it is given none: it keeps what it is told, in order."""

    def __init__(self) -> None:
        self.log: list[tuple[str, object]] = []

    async def write_request(self, request: object) -> None:
        self.log.append(("request", request))

    async def write_result(self, result: object) -> None:
        self.log.append(("result", result))

    async def write_withdrawal(self, withdrawal: Withdrawal) -> None:
        self.log.append(("withdrawal", withdrawal))

    async def record_quarantine(self, event: QuarantineEvent) -> None:
        self.log.append(("quarantine", event))

    async def quarantined(self) -> list[QuarantineEvent]:
        return []


class _Repo:
    """The repository acceptance reads a parent's recipe from.

    With a ``repo`` the recipe is the one that repository holds for the commit, read as the real toolbox reads it, so a
    parent that pins a component takes a delta of patches only; a commit the repository does not know is an unknown
    parent. Without one every parent exists and its recipe is the empty one on the fake's image, which pins no
    component. ``parent_exists`` narrows either: a commit it refuses is unknown.
    """

    def __init__(
        self, image: str, parent_exists: Callable[[str], bool] | None, repo: ExperimentRepo | None = None
    ) -> None:
        self._image, self._exists, self._repo = image, parent_exists, repo

    async def read_recipe(self, commit: str) -> Recipe:
        if self._exists is not None and not self._exists(commit):
            raise KeyError(commit)
        if self._repo is not None:
            return await self._repo.read_recipe(commit)
        return Recipe(base_image_digest=self._image)


class _NoPlanner:
    """No environment to compile, no measurement points and no surface to check: every request plans to nothing."""

    def plan(self, request: TestRequest, parent: Recipe) -> object:
        return None


_MEASURED_FIELDS = (
    "validity",
    "rounds",
    "accuracy",
    "manifest_sha256",
    "signing_key_fingerprint",
    "spec_fingerprint",
    "surface_sha256",
)


class FakeExecutor:
    """One attempt, faked: draws the fate, picks the node (away from ``avoid_nodes``, best effort) and measures.

    A drawn measurement failure is a :class:`MeasurementRefused`, as the hosted executor raises it for a rejected spec
    or a result that fails verification. ``gate`` holds attempts back until it is set (``autorun=False``).
    """

    def __init__(self, config: FakeToolboxConfig, *, autorun: bool = True) -> None:
        self.config = config
        self.now = config.start_time
        self.gate = asyncio.Event()
        if autorun:
            self.gate.set()
        self._measurer = FakeMeasurer(config)
        self._scripts: list[tuple[Callable[[TestRequest], bool], list[Fate]]] = []
        self._gpu_seconds: dict[str, float] = {}
        self._produced: dict[str, TestResult] = {}

    def script(self, match: Callable[[TestRequest], bool], fates: Sequence[Fate]) -> None:
        self._scripts.append((match, list(fates)))

    def _forced(self, request: TestRequest, attempt: int) -> Fate | None:
        for match, fates in self._scripts:
            if match(request):
                return fates[attempt] if attempt < len(fates) else None
        return None

    async def run(self, job: TestJob) -> TestResult:
        await self.gate.wait()
        if self.config.run_delay_s:
            await asyncio.sleep(self.config.run_delay_s)
        request, request_id, index = job.request, job.request.request_id or "", job.attempt - 1
        constraints = AttemptConstraints(avoid=frozenset(job.avoid_nodes), forced=self._forced(request, index))
        outcome = self._measurer.run(request, request_id, index, constraints)
        started, self.now = self.now, self.now + self.config.attempt_seconds
        gpus = outcome.sandbox.gpu_count if outcome.sandbox is not None else 0
        spent = self._gpu_seconds[request_id] = self._gpu_seconds.get(request_id, 0.0) + (self.now - started) * gpus
        if outcome.fate is Fate.MEASUREMENT:
            raise MeasurementRefused(outcome.failure_reason)
        result = self._produced[request_id] = self._result(request, outcome, started, spent)
        return result

    def withdraw(self, request_id: str) -> None:
        """A faked attempt holds no card, so there is nothing to release."""

    async def reverify(self, result: TestResult) -> None:
        """What the result says of the measurement must be what this executor produced for its request; the runner's
        own bookkeeping (queue time, sandbox marks) is not part of it."""
        produced = self._produced.get(result.request_id)
        if produced is None:
            raise ResultUnverified(result.request_id, "the fake executor produced no result for this request")
        for name in _MEASURED_FIELDS:
            if getattr(produced, name) != getattr(result, name):
                raise ResultUnverified(result.request_id, f"{name}: differs from what the fake executor produced")

    def _result(self, request: TestRequest, outcome: AttemptOutcome, started: float, gpu_seconds: float) -> TestResult:
        """The attempt as a result; the runner decides the request's final validity from the failure class."""

        request_id, measured = request.request_id or "", outcome.fate is Fate.OK
        validity = Validity.SUCCEEDED if measured else self._failed_validity(outcome)
        digest = hashlib.sha256(f"{request_id}/{validity.value}".encode()).hexdigest()
        return TestResult(
            request_id=request_id,
            validity=validity,
            failure_class=outcome.failure_class,
            failure_reason=outcome.failure_reason,
            rounds=outcome.rounds,
            accuracy=outcome.accuracy,
            sandbox=outcome.sandbox,
            manifest_sha256=digest if measured else None,
            signing_key_fingerprint="fake-signing-key" if measured else None,
            spec_fingerprint=digest[:16] if measured else None,
            surface_sha256=digest if measured else None,
            output_dir=f"tests/{request_id}",
            started_at=started,
            finished_at=self.now,
            gpu_seconds=gpu_seconds,
            profile_location=f"tests/{request_id}/profile" if measured and request.kind is TestKind.PROFILE else "",
            cold_start_seconds=self.config.attempt_seconds,
        )

    @staticmethod
    def _failed_validity(outcome: AttemptOutcome) -> Validity:
        return Validity.MEASUREMENT_FAILED if outcome.fate is Fate.INFRA else Validity.CANDIDATE_FAILED


@dataclass(frozen=True)
class FakeWiring:
    """What a :class:`FakeToolbox` is connected to besides its notebook; every field is optional.

    ``repo`` is the experiment repository acceptance reads each parent's recipe from, as the real toolbox does; its
    recipes must sit on ``FakeToolboxConfig.image_digest``, the active image. Without one a parent's recipe is the
    empty one, so a delta must give a component's upstream repository and commit. ``parent_exists`` answers whether a
    parent commit is known, ``log_path`` places the submit log, ``claims`` and ``records`` are the notebook's claim and
    record ports and ``workload_kind`` is the campaign's workload.
    """

    parent_exists: Callable[[str], bool] | None = None
    log_path: Path | None = None
    claims: ClaimService | None = None
    records: RecordSearch | None = None
    workload_kind: WorkloadKind = WorkloadKind.INFERENCEX
    repo: ExperimentRepo | None = None


class FakeToolbox(RealToolbox):
    """The real toolbox over a :class:`FakeExecutor`.

    ``autorun=False`` holds every attempt until :meth:`drain`; tests that want queued tests use it with a small
    ``max_parallel_tests``. Without a ``notebook`` the writes go to a :class:`MemoryNotebook`; without
    ``wiring.log_path`` the submit log lives in a temporary directory that goes with the toolbox. The toolbox starts
    on first use.
    """

    def __init__(
        self,
        config: FakeToolboxConfig | None = None,
        *,
        notebook: ToolboxNotebook | None = None,
        autorun: bool = True,
        wiring: FakeWiring = FakeWiring(),
    ) -> None:
        log_path = wiring.log_path
        self.config = config or FakeToolboxConfig()
        self.executor = FakeExecutor(self.config, autorun=autorun)
        self.notebook = notebook if notebook is not None else MemoryNotebook()
        self._scratch = None if log_path else tempfile.TemporaryDirectory(prefix="fake-toolbox-")
        path = log_path or Path(self._scratch.name) / "submissions.jsonl"  # type: ignore[union-attr]
        self._rejections: list[Callable[[TestRequest], RequestRejected | None]] = []
        self._started = False
        self._blob_scratch = tempfile.TemporaryDirectory(prefix="fake-blobs-")
        self.blobs = LocalBlobStore(Path(self._blob_scratch.name))
        stores = ToolboxStores(
            log=SubmissionLog(path, str(self.config.campaign_id)),
            sink=self.notebook,  # type: ignore[arg-type]
            health=self.notebook,  # type: ignore[arg-type]
            claims=wiring.claims,
            records=wiring.records,
        )
        acceptance = AcceptanceSetup(
            _Repo(self.config.image_digest, wiring.parent_exists, wiring.repo),
            _NoPlanner(),
            self.config.image_digest,
            wiring.workload_kind,
        )
        tuning = ToolboxTuning(
            policy=self._policy(self.config.default_k),
            max_infra_retries=self.config.max_infra_retries,
            clock=lambda: self.executor.now,
        )
        super().__init__(stores, self.executor, acceptance, tuning=tuning)

    def _policy(self, k: int) -> QueuePolicy:
        return QueuePolicy(
            per_branch_k=k,
            global_concurrency=self.config.max_parallel_tests,
            max_queued_per_branch=self.config.branch_queue_limit,
        )

    # --- what the toolbox knows, read from its log -------------------------------------------------------------------

    @property
    def now(self) -> float:
        """The virtual clock."""

        return self.executor.now

    @property
    def requests(self) -> dict[str, TestRequest]:
        return {e.request_id: TestRequest.from_document(e.request) for e in self._log.entries() if not e.aborted}

    @property
    def results(self) -> dict[str, TestResult]:
        """Every finished test's result, a ``measurement_failed`` one included (the notebook gets that as a withdrawal)."""

        return {
            e.request_id: TestResult.from_document(e.outcome["result"])
            for e in self._log.entries()
            if e.outcome is not None and "result" in e.outcome
        }

    @property
    def withdrawn(self) -> dict[str, str]:
        """Tests taken back unmeasured, with the reason."""

        return {
            e.request_id: str(e.outcome["withdrawal"])
            for e in self._log.entries()
            if e.outcome is not None and "withdrawal" in e.outcome and "result" not in e.outcome
        }

    # --- the interface, started on first use -------------------------------------------------------------------------

    async def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            await self.start()

    async def submit_test(self, request: TestRequest) -> str:
        await self._ensure_started()
        for check in self._rejections:
            rejection = check(request)
            if rejection is not None:
                raise rejection
        return await super().submit_test(request)

    async def cancel(self, request_id: str) -> CancelOutcome:
        await self._ensure_started()
        return await super().cancel(request_id)

    @property
    def single_node(self) -> bool:
        return len(self.config.nodes) == 1

    async def measurement_capacity(self) -> MeasurementCapacity:
        """Cards of the fake fleet: ``gpu_count`` per node. The fake refuses every test once all nodes are
        quarantined, so the cards of a quarantined node are not usable here (the real toolbox keeps them usable)."""

        per_node = self.config.gpu_count
        usable = len(set(self.config.nodes) - self.quarantined)
        return MeasurementCapacity(len(self.config.nodes) * per_node, usable * per_node)

    async def drain(self) -> None:
        """Release held attempts, run everything queued and wait for every running test."""

        await self._ensure_started()
        self.executor.gate.set()
        await super().drain()

    # --- controls for tests ------------------------------------------------------------------------------------------

    async def set_k(self, k: int) -> None:
        """Tests a branch may run at once, for every branch (the real queue has one K, set through its policy)."""

        if k < 1:
            raise ValueError("a branch needs room for at least one test")
        await self.set_policy(self._policy(k))

    def script(self, match: Callable[[TestRequest], bool], fates: Sequence[Fate]) -> None:
        """Force the fate of the attempts of the requests ``match`` accepts: ``fates[n]`` for attempt ``n``, then random."""

        self.executor.script(match, fates)

    def reject_if(self, check: Callable[[TestRequest], RequestRejected | None]) -> None:
        """Refuse requests at acceptance: ``check`` returns the rejection to raise, or ``None`` to accept."""

        self._rejections.append(check)


__all__ = ["FakeExecutor", "FakeToolbox", "FakeWiring", "MemoryNotebook", "ToolboxNotebook"]
