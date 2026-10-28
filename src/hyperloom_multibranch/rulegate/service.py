# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule gate the scheduler calls, over the blackboard and the toolbox.

Every test it judges is read back from the blackboard, so a scheduler that asks again after a restart finds the retest
it asked for before instead of paying for another.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hyperloom.contracts.measurement import Validity
from hyperloom_multibranch.contracts.gate import NotMeasured, PreludeMeasurement, ResultVerifier
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.blackboard.interfaces import CommitQuery, GateRecords, RecordSearch, SnapshotQuery
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, TestRecord
from hyperloom_multibranch.contracts.toolbox import Toolbox
from hyperloom_multibranch.rulegate.accuracy import BaselineAccuracyError
from hyperloom_multibranch.rulegate.inferencex import BASELINE_TESTS
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.rulegate.rules import Rule, repeats, retest_request

#: Why a baseline read from the PRELUDE child's run is weaker than one from root tests, kept on its record.
PRELUDE_SOURCE_CAVEAT = "not card-guarded, not re-verified"
_HASH_CHUNK = 1 << 20


class GateBlackboard(RecordSearch, SnapshotQuery, CommitQuery, GateRecords, Protocol):
    """The gate's handle: it writes the baseline and reads every test and recorded value."""


@dataclass(frozen=True)
class Polling:
    """How the gate waits for a test to end: ``seconds`` between two reads of its record, waited out with ``sleep``."""

    seconds: float = 5.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


class GateService:
    """The session's :class:`~hyperloom_multibranch.contracts.gate.SessionGate`; ``rule`` judges its workload.

    Every measured test it judges by is first re-verified with ``verifier``; one that does not re-verify raises
    :class:`~hyperloom_multibranch.contracts.gate.ResultUnverified`, a :class:`NotMeasured`. The PRELUDE child's run
    has no signed bundle: its copied report and export are checked against the sha256 they were read with instead.
    """

    def __init__(
        self,
        blackboard: GateBlackboard,
        toolbox: Toolbox,
        rule: Rule,
        verifier: ResultVerifier,
        *,
        polling: Polling = Polling(),
    ) -> None:
        self._blackboard = blackboard
        self._toolbox = toolbox
        self._rule = rule
        self._verifier = verifier
        self._polling = polling

    async def baseline_test_count(self) -> int:
        return self._rule.baseline_tests

    async def establish_baseline(self, root_commit: str, request_ids: Sequence[str]) -> BaselineRecord:
        records = [await self._ended(request_id) for request_id in request_ids]
        measured = [record for record in records if _succeeded(record)]
        if len(measured) < self._rule.min_baselines:
            needed = self._rule.min_baselines
            raise NotMeasured(f"{len(measured)} of {len(records)} root tests measured; the baseline needs {needed}")
        for record in measured:
            await self._verifier.reverify(record.request_id)
        with _judgeable():
            return await self._rule.baseline(self._blackboard, root_commit, measured)

    async def establish_prelude_baseline(self, root_commit: str, measured: PreludeMeasurement) -> BaselineRecord:
        await asyncio.to_thread(_check_copies, measured)
        if measured.result.validity is not Validity.SUCCEEDED:
            raise NotMeasured(f"the PRELUDE child's run in {measured.run_dir} did not measure")
        source = (
            f"prelude-child:{measured.run_dir} sha256:{measured.report_sha256}/{measured.export_sha256}; "
            f"{PRELUDE_SOURCE_CAVEAT}"
        )
        with _judgeable():
            return await self._rule.prelude_baseline(
                self._blackboard, root_commit, measured.result, source=source, wall_seconds=measured.wall_seconds
            )

    async def compare(self, request_id: str) -> Judgement:
        first = await self._measured(request_id)
        parent = (await self._blackboard.recorded_value(first.request.parent_commit)).values
        with _judgeable():
            judgement = await self._rule.first(first, parent)
        if judgement.verdict is not Verdict.NEEDS_CONFIRM:
            return judgement
        retest = await self._measured(await self._retest_of(first))
        with _judgeable():
            return await self._rule.retest(judgement, retest, parent)

    async def request_confirm(self, request_id: str) -> str:
        record = await self._blackboard.record(request_id)
        return await self._toolbox.submit_test(retest_request(record, self._rule))

    async def accuracy_passes(self, request_id: str) -> bool:
        record = await self._measured(request_id)
        with _judgeable():
            return await self._rule.accuracy_passes(record)

    async def _retest_of(self, first: TestRecord) -> str:
        snapshot = await self._blackboard.snapshot()
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
        while (record := await self._blackboard.record(request_id)).in_flight:
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


def _check_copies(measured: PreludeMeasurement) -> None:
    """The copies of the child's report and export still hold what the result was read from. Blocks."""
    for path, expected in ((measured.report, measured.report_sha256), (measured.export, measured.export_sha256)):
        try:
            actual = _sha256(Path(path))
        except OSError as error:
            raise NotMeasured(f"cannot read {path}: {error}") from error
        if actual != expected:
            raise NotMeasured(f"{path} has sha256 {actual}, not {expected} the baseline was read from")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _succeeded(record: TestRecord) -> bool:
    return record.result is not None and record.result.validity is Validity.SUCCEEDED


__all__ = ["BASELINE_TESTS", "PRELUDE_SOURCE_CAVEAT", "GateBlackboard", "GateService", "Polling"]
