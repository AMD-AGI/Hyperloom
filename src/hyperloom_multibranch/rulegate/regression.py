# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Correctness fixes and their reproduction cases.

A reproduction case is the attachments of a repro test: a directory with a ``repro.sh`` at its top, whose verdict on a
commit is ``reproduced`` (the failure occurs) or ``not_reproduced`` (it ran clean). :class:`ReproRunner` is the one
place that submits repro tests.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind, TestRequest, Validity
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.gate import ResultUnverified, ResultVerifier
from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.toolbox import Backpressure, RequestRejected, Toolbox
from hyperloom_multibranch.rulegate.service import Polling

#: Seconds before a repro test the toolbox refused with backpressure is submitted again.
RESUBMIT_SECONDS = 30.0


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
    """Runs a case as a repro test through ``toolbox`` and reads its verdict back from ``notebook``.

    A verdict counts only once ``verifier`` re-verified the result behind it. A run never raises for how the test
    ended: a refusal, a withdrawal, a failure, a success without a verdict and a result that does not re-verify are each
    a :attr:`ReproRun.failure`. A caller that stops waiting takes its test back, shielded from a second cancel.
    """

    def __init__(
        self, toolbox: Toolbox, notebook: RecordSearch, verifier: ResultVerifier, *, polling: Polling = Polling()
    ) -> None:
        self._toolbox = toolbox
        self._notebook = notebook
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
        while (record := await self._notebook.record(request_id)).in_flight:
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


__all__ = ["RESUBMIT_SECONDS", "ReproRun", "ReproRunner"]
