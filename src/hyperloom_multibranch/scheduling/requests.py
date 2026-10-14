# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tests the scheduling side itself issues, all on the integration branch, and waiting for their outcome.

PRELUDE's baseline, the integrator's merge tests and FINAL's validation differ only in what they measure, from which
commit and why; :func:`integration_test_request` builds each one the same way. The toolbox hands no result back: it
writes the result, or the withdrawal of a test it will not measure, to the notebook, so :class:`SessionTests` reads
the outcome from there.
"""

from __future__ import annotations

import asyncio
from typing import Any

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.toolbox import Backpressure, Toolbox
from hyperloom_multibranch.scheduling.clock import Clock

NO_DELTA = RecipeDelta()
#: Clock seconds between two reads of a test's notebook record while it is in flight; a test takes minutes.
SETTLE_POLL_SECONDS = 5.0
#: Clock seconds before a request the toolbox refused with backpressure is submitted again.
RESUBMIT_SECONDS = 30.0


class TestWithdrawn(Exception):
    """The toolbox took the test back without measuring it; ``reason`` is the notebook's withdrawal reason."""

    __test__ = False

    def __init__(self, request_id: str, reason: str) -> None:
        super().__init__(f"{request_id} was withdrawn: {reason}")
        self.request_id = request_id
        self.reason = reason


def integration_test_request(
    kind: TestKind,
    purpose: Purpose,
    parent_commit: str,
    delta: RecipeDelta = NO_DELTA,
    *,
    accuracy: bool = False,
    concurrency: tuple[int, ...] = (),
) -> TestRequest:
    """A request on the integration branch measuring ``delta`` from ``parent_commit``, with accuracy rounds when
    ``accuracy`` is set.

    It names no intent: the session's own tests claim nothing, and the toolbox refuses a request whose intent has no
    claim behind it.
    """

    return TestRequest(
        kind=kind,
        purpose=purpose,
        branch_id=INTEGRATION_BRANCH,
        parent_commit=parent_commit,
        delta=delta,
        accuracy=accuracy,
        concurrency=concurrency,
    )


class SessionTests:
    """Submits the session's own tests to ``toolbox`` and waits for their outcome in ``notebook``."""

    def __init__(self, toolbox: Toolbox, notebook: RecordSearch, clock: Clock) -> None:
        self._toolbox = toolbox
        self._notebook = notebook
        self._clock = clock

    async def submit(self, request: TestRequest) -> str:
        """The accepted request's id; a branch queue that is full is waited out, a refusal raises."""

        while True:
            try:
                return await self._toolbox.submit_test(request)
            except Backpressure:
                await self._clock.sleep(RESUBMIT_SECONDS)

    async def settled(self, request_id: str) -> TestRecord:
        """The record once the test has a result; raises :class:`TestWithdrawn` when it was withdrawn instead."""

        while True:
            record = await self._notebook.record(request_id)
            if record.withdrawn_at is not None:
                raise TestWithdrawn(request_id, record.withdrawal_reason)
            if record.result is not None:
                return record
            await self._clock.sleep(SETTLE_POLL_SECONDS)

    async def run(self, request: TestRequest) -> TestRecord:
        """Submit ``request`` and wait for its settled record; a caller that stops waiting takes the test back.

        The take-back is shielded, so a second cancel of the caller does not cut the withdrawal short.
        """

        request_id = await self.submit(request)
        try:
            return await self.settled(request_id)
        except asyncio.CancelledError:
            await asyncio.shield(self._toolbox.cancel(request_id))
            raise


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


__all__ = [
    "NO_DELTA",
    "RESUBMIT_SECONDS",
    "SETTLE_POLL_SECONDS",
    "SessionTests",
    "TestWithdrawn",
    "integration_test_request",
    "together",
]
