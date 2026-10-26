# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The native builds accepted tests wait for: started from the submission, coalesced by key, charged once.

A test whose stack needs native builds enters the queue as ``pending_build``: it holds no card but takes a slot of its
branch's backlog. :class:`PendingBuilds` runs each key's build once for every test waiting on it, through a
:class:`BuildDriver` (the store's :func:`~hyperloom.measure.runner.native.service.ensure_build`, whose ``flock`` also
coalesces equal keys across toolboxes). When every build a test needs is ready the test is queued for a card; a build
that failed for the candidate ends the test with a ``compile_error`` result; a build that failed for the machine is
retried, and the test ends ``measurement_failed`` only once the retries are spent. Every waiting test is told of each
of its builds that became ready or failed for the machine (a ``build`` inbox item), never of a candidate failure: its
result says that.

``build_seconds`` are charged :attr:`~hyperloom.contracts.measurement.BuildPayment.PAID` to the first test still waiting
on the build this toolbox ran, and :attr:`~hyperloom.contracts.measurement.BuildPayment.REUSED` to everyone else.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

from hyperloom.contracts.measurement import (
    BuildCharge,
    BuildPayment,
    FailureClass,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import READY, BuildRecord
from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.jit.native.trust import BuildRefused
from hyperloom.measure.runner.native.builder import COMPILE_ERROR
from hyperloom.measure.runner.native.service import Ensured
from hyperloom_multibranch.contracts.notebook.records import BuildNotice
from hyperloom_multibranch.contracts.notebook.vocab import BuildOutcome
from hyperloom_multibranch.contracts.toolbox import BuildState, BuildStatus, RequestRejected
from hyperloom_multibranch.toolbox.submission_log import LogEntry

logger = logging.getLogger(__name__)

BUILD_AFTER_DEADLINE = "build_after_deadline"
_RESULT = "result"


class BuildDriver(Protocol):
    """What the queue asks of the native build store. ``ensure`` must not compile in this process (it holds the signing
    key): the store's driver hands the compile to a runner that has no key."""

    def needs(self, request: TestRequest, environment: object) -> tuple[BuildInputs, ...]:
        """The builds the accepted ``request``'s stack needs, given what acceptance prepared; empty for most stacks."""

    def lookup(self, inputs: BuildInputs) -> BuildRecord | None:
        """The store's verified entry of ``inputs``, without building; ``None`` when there is none or it fails a check
        (``ensure`` then raises ``BuildRefused``)."""

    def eta_seconds(self, inputs: BuildInputs) -> float:
        """How long a build of ``inputs`` is expected to take, in wall seconds."""

    async def ensure(self, inputs: BuildInputs) -> Ensured:
        """The entry of ``inputs``, built when there is none; raises ``BuildInfraError`` or ``BuildRefused``."""


@dataclass(frozen=True)
class BuildNeed:
    """One build a test needs, and the store's entry of it when the test was accepted."""

    inputs: BuildInputs
    stored: BuildRecord | None


@dataclass(frozen=True)
class BuildEnd:
    """Why a test's builds end it before it ran: its result's failure class and reason."""

    failure_class: FailureClass
    reason: str


@dataclass(frozen=True)
class BuildHooks:
    """What :class:`PendingBuilds` tells the toolbox: a test's builds are all ready, a test ends, a build notice."""

    ready: Callable[[str], Awaitable[None]]
    ended: Callable[[str, BuildEnd], Awaitable[None]]
    notify: Callable[[BuildNotice], Awaitable[None]]


@dataclass
class _Build:
    inputs: BuildInputs
    eta_seconds: float
    state: BuildState = BuildState.QUEUED
    build_seconds: float = 0.0
    infra_failures: int = 0
    waiters: list[str] = field(default_factory=list)
    task: asyncio.Task[None] | None = None

    def status(self) -> BuildStatus:
        over = self.state in (BuildState.READY, BuildState.FAILED)
        eta = 0.0 if over else self.eta_seconds
        return BuildStatus(self.inputs.key, self.inputs.component, self.state, eta, self.build_seconds)


@dataclass
class _Waiting:
    """A test waiting for builds: every key it needs in order, the ones not ready yet, and its charges so far."""

    keys: tuple[str, ...]
    pending: set[str]
    stored: dict[str, BuildStatus]


def _ready_status(record: BuildRecord) -> BuildStatus:
    return BuildStatus(record.key, record.inputs.component, BuildState.READY, 0.0, record.build_seconds)


def _excerpt(failure: str) -> str:
    prefix = f"{COMPILE_ERROR}: "
    return failure[len(prefix) :] if failure.startswith(prefix) else failure


class PendingBuilds:
    """The builds of the tests in ``pending_build``; a toolbox without a driver has none, and nothing changes."""

    def __init__(
        self,
        driver: BuildDriver | None,
        hooks: BuildHooks,
        *,
        max_infra_retries: int,
        retry_seconds: float,
    ) -> None:
        self._driver, self._hooks = driver, hooks
        self._max_infra_retries, self._retry_seconds = max_infra_retries, retry_seconds
        self._builds: dict[str, _Build] = {}
        self._waiting: dict[str, _Waiting] = {}
        self._charges: dict[str, dict[str, BuildCharge]] = {}

    # --- what the toolbox asks -----------------------------------------------------------------------------------

    def plan(self, request: TestRequest, environment: object) -> tuple[BuildNeed, ...]:
        """The builds ``request`` needs, each with the store's entry of it now."""

        if self._driver is None:
            return ()
        driver = self._driver
        return tuple(BuildNeed(inputs, driver.lookup(inputs)) for inputs in driver.needs(request, environment))

    def check_deadline(self, needs: tuple[BuildNeed, ...], deadline: float, test_seconds: float, now: float) -> None:
        """Refuse with ``build_after_deadline`` when the slowest build not ready, then one test, ends past ``deadline``
        (0: no deadline)."""

        unready = [(self._eta(need), need.inputs.component) for need in needs if self._must_build(need)]
        if not deadline or not unready:
            return
        eta, component = max(unready)
        if now + eta + test_seconds > deadline:
            raise RequestRejected(
                f"the {component} build (eta {eta:.0f} s) and one test ({test_seconds:.0f} s) would end "
                f"{now + eta + test_seconds - deadline:.0f} s after the branch's deadline {deadline:.0f}",
                BUILD_AFTER_DEADLINE,
            )

    def wait(self, request_id: str, needs: tuple[BuildNeed, ...]) -> bool:
        """Start or join the builds ``request_id`` needs; whether it must wait for any (False: all ready now)."""

        stored: dict[str, BuildStatus] = {}
        pending: set[str] = set()
        for need in needs:
            key, ready = need.inputs.key, self._ready_now(need)
            if ready is not None:
                stored[key] = ready
                self._charge(request_id, key, need.inputs.component, None)
            else:
                self._join(request_id, need.inputs)
                pending.add(key)
        if pending:
            self._waiting[request_id] = _Waiting(tuple(need.inputs.key for need in needs), pending, stored)
        return bool(pending)

    def statuses(self, request_id: str) -> tuple[BuildStatus, ...]:
        """One status per build a waiting test needs, in the order it needs them; empty for a test not waiting."""

        waiting = self._waiting.get(request_id)
        if waiting is None:
            return ()
        return tuple(waiting.stored.get(key) or self._builds[key].status() for key in waiting.keys)

    def charges(self, request_id: str) -> tuple[BuildCharge, ...]:
        return tuple(self._charges.get(request_id, {}).values())

    def forget(self, request_id: str) -> None:
        """The test ran, ended or was taken back: it waits for nothing and its charges are written or dropped."""

        self._waiting.pop(request_id, None)
        self._charges.pop(request_id, None)

    def ledger(self) -> tuple[BuildStatus, ...]:
        return tuple(build.status() for build in self._builds.values())

    @property
    def busy(self) -> bool:
        return any(build.task is not None and not build.task.done() for build in self._builds.values())

    async def settle(self) -> None:
        """Wait for the builds running now (for tests and orderly shutdown)."""

        tasks = [build.task for build in self._builds.values() if build.task is not None]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        tasks = [build.task for build in self._builds.values() if build.task is not None and not build.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # --- building ------------------------------------------------------------------------------------------------

    def _ready_now(self, need: BuildNeed) -> BuildStatus | None:
        """The status of a build that is ready already, in the store or in this session; ``None`` otherwise."""

        if need.stored is not None and need.stored.status == READY:
            return _ready_status(need.stored)
        build = self._builds.get(need.inputs.key)
        return build.status() if build is not None and build.state is BuildState.READY else None

    def _must_build(self, need: BuildNeed) -> bool:
        """Whether the test waits for a build of ``need`` (an entry in the store, ready or failed, ends the wait at once)."""

        return need.stored is None and self._ready_now(need) is None

    def _eta(self, need: BuildNeed) -> float:
        build = self._builds.get(need.inputs.key)
        if build is not None and build.state is not BuildState.FAILED:
            return build.eta_seconds
        return self._driver.eta_seconds(need.inputs) if self._driver is not None else 0.0

    def _join(self, request_id: str, inputs: BuildInputs) -> None:
        build = self._builds.get(inputs.key)
        if build is None or build.state is BuildState.FAILED:
            eta = self._driver.eta_seconds(inputs) if self._driver is not None else 0.0
            build = self._builds[inputs.key] = _Build(inputs, eta)
            build.task = asyncio.create_task(self._run(build))
        build.waiters.append(request_id)

    async def _run(self, build: _Build) -> None:
        assert self._driver is not None
        build.state = BuildState.BUILDING
        try:
            ensured = await self._driver.ensure(build.inputs)
        except BuildRefused as exc:
            await self._end(
                build, BuildEnd(FailureClass.INFRA, f"the build store refused {build.inputs.component}: {exc}")
            )
            return
        except (BuildInfraError, OSError) as exc:
            await self._infra_failed(build, f"{type(exc).__name__}: {exc}")
            return
        record = ensured.record
        build.build_seconds = record.build_seconds
        payer = None if ensured.reused else next(iter(self._still_waiting(build)), None)
        if record.status == READY:
            await self._finish_ready(build, payer)
        else:
            reason = f"{COMPILE_ERROR}: {build.inputs.component} {build.inputs.key[:12]}: {_excerpt(record.failure)}"
            await self._end(build, BuildEnd(FailureClass.CANDIDATE, reason), payer)

    def _still_waiting(self, build: _Build) -> list[str]:
        return [request_id for request_id in build.waiters if request_id in self._waiting]

    def _charge(self, request_id: str, key: str, component: str, paid: float | None) -> None:
        payment = BuildPayment.REUSED if paid is None else BuildPayment.PAID
        charge = BuildCharge(key, component, paid or 0.0, payment)
        self._charges.setdefault(request_id, {})[key] = charge

    def _charge_waiters(self, build: _Build, payer: str | None) -> list[str]:
        waiters = self._still_waiting(build)
        for request_id in waiters:
            paid = build.build_seconds if request_id == payer else None
            self._charge(request_id, build.inputs.key, build.inputs.component, paid)
        build.waiters.clear()
        return waiters

    async def _finish_ready(self, build: _Build, payer: str | None) -> None:
        build.state = BuildState.READY
        for request_id in self._charge_waiters(build, payer):
            waiting = self._waiting[request_id]
            waiting.pending.discard(build.inputs.key)
            await self._notify(request_id, build, BuildOutcome.READY)
            if not waiting.pending:
                del self._waiting[request_id]
                await self._hooks.ready(request_id)

    async def _end(self, build: _Build, end: BuildEnd, payer: str | None = None) -> None:
        build.state = BuildState.FAILED
        for request_id in self._charge_waiters(build, payer):
            self._waiting.pop(request_id, None)
            await self._hooks.ended(request_id, end)

    async def _infra_failed(self, build: _Build, detail: str) -> None:
        build.infra_failures += 1
        for request_id in self._still_waiting(build):
            await self._notify(request_id, build, BuildOutcome.FAILED, detail)
        if build.infra_failures > self._max_infra_retries:
            reason = f"the {build.inputs.component} build failed {build.infra_failures} times for the machine: {detail}"
            await self._end(build, BuildEnd(FailureClass.INFRA, reason))
            return
        build.state = BuildState.QUEUED
        build.task = asyncio.create_task(self._retry(build))

    async def _retry(self, build: _Build) -> None:
        await asyncio.sleep(self._retry_seconds)
        await self._run(build)

    async def _notify(self, request_id: str, build: _Build, outcome: BuildOutcome, detail: str = "") -> None:
        notice = BuildNotice(request_id, build.inputs.key, build.inputs.component, outcome, detail)
        try:
            await self._hooks.notify(notice)
        except Exception:  # any failure of the notebook's push: the waiting test goes on without its notice
            logger.warning(
                "the build notice of %s for %s was not delivered", build.inputs.key, request_id, exc_info=True
            )


def median_test_seconds(entries: Iterable[LogEntry], kind: TestKind) -> float:
    """The median wall time of the finished tests of ``kind`` the submit log holds; 0 when it holds none."""

    seconds = []
    for entry in entries:
        result = (entry.outcome or {}).get(_RESULT)
        if entry.request.get("kind") != kind.value or not isinstance(result, dict):
            continue
        started, finished = result.get("started_at"), result.get("finished_at")
        if isinstance(started, (int, float)) and isinstance(finished, (int, float)) and finished >= started:
            seconds.append(float(finished) - float(started))
    return float(statistics.median(seconds)) if seconds else 0.0


def build_failure_result(request_id: str, end: BuildEnd, builds: tuple[BuildCharge, ...]) -> TestResult:
    """The result of a test its builds ended before it ran: the candidate's compile error, or spent machine retries."""

    validity = Validity.CANDIDATE_FAILED if end.failure_class is FailureClass.CANDIDATE else Validity.MEASUREMENT_FAILED
    return TestResult(
        request_id=request_id,
        validity=validity,
        failure_class=end.failure_class,
        failure_reason=end.reason,
        builds=builds,
    )


__all__ = [
    "BUILD_AFTER_DEADLINE",
    "BuildDriver",
    "BuildEnd",
    "BuildHooks",
    "BuildNeed",
    "PendingBuilds",
    "build_failure_result",
    "median_test_seconds",
]
