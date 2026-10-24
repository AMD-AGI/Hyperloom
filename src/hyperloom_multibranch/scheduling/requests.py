# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tests the scheduling side itself issues, all on the integration branch, and waiting for their outcome.

PRELUDE's baseline, the integrator's merge tests and FINAL's validation differ only in what they measure, from which
commit and why; :func:`integration_test_request` builds each one the same way. Whether a test runs the eval is the
toolbox's call: it adds the accuracy rounds the session's rule reads from every session measurement. The integrator
and FINAL check a commit the same way too: :class:`SessionChecks` measures it, judges the measurement and runs the
regression set on it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest, TestResult
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.gate import RegressionCheck, SessionGate
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import Backpressure
from hyperloom_multibranch.scheduling.clock import Clock

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

NO_DELTA = RecipeDelta()
#: Clock seconds before a request the toolbox refused with backpressure is submitted again.
RESUBMIT_SECONDS = 30.0

T = TypeVar("T")


def integration_test_request(
    kind: TestKind,
    purpose: Purpose,
    parent_commit: str,
    delta: RecipeDelta = NO_DELTA,
    *,
    concurrency: tuple[int, ...] = (),
) -> TestRequest:
    """A request on the integration branch measuring ``delta`` from ``parent_commit``.

    It names no intent: the session's own tests claim nothing, and the toolbox refuses a request whose intent has no
    claim behind it.
    """

    return TestRequest(
        kind=kind,
        purpose=purpose,
        branch_id=INTEGRATION_BRANCH,
        parent_commit=parent_commit,
        delta=delta,
        concurrency=concurrency,
    )


class SessionTests:
    """Submits the session's own tests to ``toolbox`` and waits for their outcome.

    A caller that stops waiting takes its test back; the take-back is shielded, so a second cancel of the caller does
    not cut the withdrawal short.
    """

    def __init__(self, toolbox: RealToolbox, clock: Clock) -> None:
        self._toolbox = toolbox
        self._clock = clock

    async def submit(self, request: TestRequest) -> str:
        """The accepted request's id; a branch queue that is full is waited out, a refusal raises."""

        while True:
            try:
                return await self._toolbox.submit_test(request)
            except Backpressure:
                await self._clock.sleep(RESUBMIT_SECONDS)

    async def result(self, request: TestRequest) -> TestResult:
        """Submit ``request`` and return its result; raises ``TestWithdrawn`` when the toolbox withdrew it instead."""

        return await self.judged(request, self._toolbox.outcome)

    async def judged(self, request: TestRequest, judge: Callable[[str], Awaitable[T]]) -> T:
        """Submit ``request`` and return what ``judge`` makes of its request id once the test has ended."""

        request_id = await self.submit(request)
        try:
            return await judge(request_id)
        except asyncio.CancelledError:
            await asyncio.shield(self._toolbox.cancel(request_id))
            raise

    async def withdraw_integration(self, reason: str) -> tuple[str, ...]:
        """Take back every test of the integration branch still queued or running, retests included."""

        return await self._toolbox.withdraw_branch(INTEGRATION_BRANCH, reason)


async def together(*measurements: asyncio.Future[Any]) -> None:
    """Await every measurement; once one fails, the others are cancelled, which takes their tests back, and its error
    is raised. Each is cancelled once, so the take-back is never cut short by a second cancel."""

    try:
        await asyncio.wait(measurements, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        for measurement in measurements:
            measurement.cancel()
        await asyncio.gather(*measurements, return_exceptions=True)
    failures = [measurement.exception() for measurement in measurements if not measurement.cancelled()]
    failure = next((error for error in failures if error is not None), None)
    if failure is not None:
        raise failure


@dataclass(frozen=True)
class SessionChecks:
    """What the scheduling side checks a commit on the integration branch with: ``tests`` measures it, ``gate``
    judges the measurement and ``regression`` runs the regression set on it."""

    tests: SessionTests
    gate: SessionGate
    regression: RegressionCheck


__all__ = [
    "NO_DELTA",
    "RESUBMIT_SECONDS",
    "SessionChecks",
    "SessionTests",
    "integration_test_request",
    "together",
]
