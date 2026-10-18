# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule gate the scheduler calls, over the notebook and the toolbox.

Every test it judges is read back from the notebook, so a scheduler that asks again after a restart finds the retest
it asked for before instead of paying for another.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from hyperloom.contracts.measurement import Validity
from hyperloom_multibranch.contracts.gate import NotMeasured, ResultVerifier
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.interfaces import CommitQuery, GateRecords, RecordSearch, SnapshotQuery
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord, TestRecord
from hyperloom_multibranch.contracts.toolbox import Toolbox
from hyperloom_multibranch.rulegate.accuracy import BaselineAccuracyError
from hyperloom_multibranch.rulegate.inferencex import MIN_BASELINES
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.rulegate.rules import Rule, repeats, retest_request

#: PRELUDE measures the root this many times, so one failed measurement still leaves the bar its two.
BASELINE_TESTS = 3


class GateNotebook(RecordSearch, SnapshotQuery, CommitQuery, GateRecords, Protocol):
    """The gate's handle: it writes the baseline and reads every test and recorded value."""


@dataclass(frozen=True)
class Polling:
    """How the gate waits for a test to end: ``seconds`` between two reads of its record, waited out with ``sleep``."""

    seconds: float = 5.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


class GateService:
    """The session's :class:`~hyperloom_multibranch.contracts.gate.SessionGate`; ``rule`` judges its workload.

    Every measured test it judges by is first re-verified with ``verifier``; one that does not re-verify raises
    :class:`~hyperloom_multibranch.contracts.gate.ResultUnverified`, a :class:`NotMeasured`.
    """

    def __init__(
        self,
        notebook: GateNotebook,
        toolbox: Toolbox,
        rule: Rule,
        verifier: ResultVerifier,
        *,
        polling: Polling = Polling(),
    ) -> None:
        self._notebook = notebook
        self._toolbox = toolbox
        self._rule = rule
        self._verifier = verifier
        self._polling = polling

    async def baseline_test_count(self) -> int:
        return BASELINE_TESTS

    async def establish_baseline(self, root_commit: str, request_ids: Sequence[str]) -> BaselineRecord:
        records = [await self._ended(request_id) for request_id in request_ids]
        measured = [record for record in records if _succeeded(record)]
        if len(measured) < MIN_BASELINES:
            raise NotMeasured(f"{len(measured)} of {len(records)} root tests measured; the bar needs {MIN_BASELINES}")
        for record in measured:
            await self._verifier.reverify(record.request_id)
        with _judgeable():
            return await self._rule.baseline(self._notebook, root_commit, measured)

    async def compare(self, request_id: str) -> Judgement:
        first = await self._measured(request_id)
        parent = (await self._notebook.recorded_value(first.request.parent_commit)).values
        with _judgeable():
            judgement = await self._rule.first(first, parent)
        if judgement.verdict is not Verdict.NEEDS_CONFIRM:
            return judgement
        retest = await self._measured(await self._retest_of(first))
        with _judgeable():
            return await self._rule.retest(judgement, retest, parent)

    async def accuracy_passes(self, request_id: str) -> bool:
        record = await self._measured(request_id)
        with _judgeable():
            return await self._rule.accuracy_passes(record)

    async def _retest_of(self, first: TestRecord) -> str:
        snapshot = await self._notebook.snapshot()
        asked = [
            record
            for record in snapshot.records
            if record.request.retest_of == first.request_id and repeats(record, first, self._rule)
        ]
        if asked:
            return max(asked, key=lambda record: record.requested_at).request_id
        return await self._toolbox.submit_test(retest_request(first, self._rule))

    async def _measured(self, request_id: str) -> TestRecord:
        record = await self._ended(request_id)
        if record.result is None:
            raise NotMeasured(f"{request_id} was withdrawn ({record.withdrawal_reason})")
        if not _succeeded(record):
            reason = record.result.failure_reason or "no reason given"
            raise NotMeasured(f"{request_id} ended {record.result.validity.value} ({reason})")
        await self._verifier.reverify(request_id)
        return record

    async def _ended(self, request_id: str) -> TestRecord:
        while (record := await self._notebook.record(request_id)).in_flight:
            await self._polling.sleep(self._polling.seconds)
        return record


@contextmanager
def _judgeable() -> Iterator[None]:
    """A result without what the rule reads, or root tests scoring below the floor, give the gate no more to judge by
    than a test that never measured."""
    try:
        yield
    except (UnreadableResult, BaselineAccuracyError) as error:
        raise NotMeasured(str(error)) from error


def _succeeded(record: TestRecord) -> bool:
    return record.result is not None and record.result.validity is Validity.SUCCEEDED


__all__ = ["BASELINE_TESTS", "GateNotebook", "GateService", "Polling"]
