# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule gate the scheduler calls, over the blackboard and the toolbox.

Every test it judges is read back from the blackboard, so a scheduler that asks again after a restart finds the retest
it asked for before instead of paying for another.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Protocol

from hyperloom.contracts.measurement import Validity
from hyperloom.measure.jit_kinds import HIT
from hyperloom_multibranch.contracts.gate import NotMeasured, ResultVerifier
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.blackboard.interfaces import (
    CommitQuery,
    GateRecords,
    NotRecorded,
    RecordSearch,
    SnapshotQuery,
)
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, TestRecord
from hyperloom_multibranch.contracts.toolbox import Toolbox
from hyperloom_multibranch.rulegate.accuracy import BaselineAccuracyError
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.rulegate.regression import ended
from hyperloom_multibranch.rulegate.rules import Rule, repeats, retest_request


class GateBlackboard(RecordSearch, SnapshotQuery, CommitQuery, GateRecords, Protocol):
    """The gate's handle: it writes the baseline and reads every test and recorded value."""


class GateService:
    """The session's :class:`~hyperloom_multibranch.contracts.gate.SessionGate`; ``rule`` judges its workload.

    Every measured test it judges by is first re-verified with ``verifier``; one that does not re-verify raises
    :class:`~hyperloom_multibranch.contracts.gate.ResultUnverified`, a :class:`NotMeasured`.
    """

    def __init__(self, blackboard: GateBlackboard, toolbox: Toolbox, rule: Rule, verifier: ResultVerifier) -> None:
        self._blackboard = blackboard
        self._toolbox = toolbox
        self._rule = rule
        self._verifier = verifier

    async def baseline_test_count(self) -> int:
        return self._rule.baseline_tests

    async def establish_baseline(self, root_commit: str, request_ids: Sequence[str]) -> BaselineRecord:
        records = [await ended(self._blackboard, request_id) for request_id in request_ids]
        measured = [record for record in records if _succeeded(record)]
        if len(measured) < self._rule.min_baselines:
            needed = self._rule.min_baselines
            raise NotMeasured(f"{len(measured)} of {len(records)} root tests measured; the baseline needs {needed}")
        for record in measured:
            await self._verifier.reverify(record.request_id)
        with _judgeable():
            return await self._rule.baseline(self._blackboard, root_commit, measured)

    async def compare(self, request_id: str) -> Judgement:
        first = await self._measured(request_id)
        parent = await self._blackboard.recorded_value(first.request.parent_commit)
        with _judgeable():
            judgement = await self._rule.first(first, parent.values)
        if judgement.verdict is Verdict.NEEDS_CONFIRM:
            retest = await self._measured(await self._retest_of(first))
            with _judgeable():
                judgement = await self._rule.retest(judgement, retest, parent.values)
        differing = cache_mismatch(
            await self._recorded(judgement.request_ids), await self._recorded(parent.request_ids)
        )
        return replace(judgement, cache_mismatch=differing) if differing else judgement

    async def request_confirm(self, request_id: str) -> str:
        if not self._rule.confirms_ahead:
            return ""
        record = await self._blackboard.record(request_id)
        return await self._toolbox.submit_test(retest_request(record, self._rule))

    async def accuracy_passes(self, request_id: str) -> bool:
        record = await self._measured(request_id)
        with _judgeable():
            return await self._rule.accuracy_passes(record)

    async def speed_gates(self) -> frozenset[str]:
        return self._rule.speed_gates

    async def _recorded(self, request_ids: Sequence[str]) -> list[TestRecord]:
        """The records of ``request_ids`` this blackboard holds; a parent's tests run in another session are not here."""
        records = []
        for request_id in request_ids:
            try:
                records.append(await self._blackboard.record(request_id))
            except NotRecorded:
                continue
        return records

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
        record = await ended(self._blackboard, request_id)
        if record.result is None:
            raise NotMeasured(f"{request_id} was withdrawn ({record.withdrawal_reason})")
        if not _succeeded(record):
            reason = record.result.failure_reason or "no reason given"
            raise NotMeasured(f"{request_id} ended {record.result.validity.value} ({reason})")
        await self._verifier.reverify(request_id)
        return record


@contextmanager
def _judgeable() -> Iterator[None]:
    """A result without what the rule reads, or root tests scoring below the floor, give the gate no more to judge by
    than a test that never measured."""
    try:
        yield
    except (UnreadableResult, BaselineAccuracyError) as error:
        raise NotMeasured(str(error)) from error


def cache_mismatch(candidate: Sequence[TestRecord], parent: Sequence[TestRecord]) -> tuple[str, ...]:
    """The kinds of shared cache that were warm in some test on one side and not alike on the other; none when a side has
    no test to say. A test whose result states no cache counts as cold for every kind."""
    if not candidate or not parent:
        return ()
    kinds = {kind for record in (*candidate, *parent) for kind in _states(record)}
    return tuple(sorted(kind for kind in kinds if _warmth(candidate, kind) != _warmth(parent, kind)))


def _states(record: TestRecord) -> Mapping[str, str]:
    return record.result.cache_states if record.result is not None else {}


def _warmth(records: Sequence[TestRecord], kind: str) -> frozenset[bool]:
    """Whether each of ``records`` found the cache of ``kind`` warm."""
    return frozenset(_states(record).get(kind) == HIT for record in records)


def _succeeded(record: TestRecord) -> bool:
    return record.result is not None and record.result.validity is Validity.SUCCEEDED


__all__ = ["GateBlackboard", "GateService", "cache_mismatch"]
