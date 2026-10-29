# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The PRELUDE baseline the rule gate writes, and what judging a candidate reads back from it.

The root's value, its eval score and the bar are set once at PRELUDE and hold for the whole session, so every candidate
is held to the same bar; judging reads them back from the record, so nothing it judges by lives only in memory.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace

from hyperloom.common.perf_metric import (
    AGENTX_KEEP_P50_THRESHOLD_PCT,
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
)
from hyperloom.contracts.measurement import Purpose, TestResult
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE
from hyperloom_multibranch.contracts.blackboard.interfaces import CommitQuery, GateRecords
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, TestRecord
from hyperloom_multibranch.rulegate.accuracy import baseline_accuracy
from hyperloom_multibranch.rulegate.inferencex import DispersionThreshold, InferenceXBaseline
from hyperloom_multibranch.rulegate.readings import UnreadableResult, agentx_reading, inferencex_reading

# Without these an AgentX comparison is graded on nothing, or found incomparable for every candidate.
_AGENTX_REQUIRED = (GRADED_INTVTY_P50, GRADED_DURATION, GRADED_ERROR_RATE)


async def write_inferencex_baseline(
    store: GateRecords, root_commit: str, baselines: Sequence[TestRecord], concurrency: int
) -> BaselineRecord:
    """Fewer than two baselines, or a mean eval score below the floor, fails PRELUDE."""
    results = [_root_result(record, root_commit) for record in baselines]
    readings = [inferencex_reading(result, concurrency) for result in results]
    throughputs = tuple(reading.throughput for reading in readings)
    threshold = DispersionThreshold(throughputs)
    accuracy = baseline_accuracy([reading.accuracy for reading in readings])
    record = BaselineRecord(
        root_commit,
        values={GRADED_OUTPUT: statistics.fmean(throughputs), ACCURACY_SCORE: accuracy},
        dispersion_pct={GRADED_OUTPUT: threshold.spread_pct},
        threshold_pct={GRADED_OUTPUT: threshold.threshold_pct},
        request_ids=tuple(record.request_id for record in baselines),
    )
    await store.write_baseline(record)
    return record


async def read_inferencex_baseline(store: CommitQuery) -> InferenceXBaseline:
    current = await store.baseline()
    return InferenceXBaseline(
        current.threshold_pct[GRADED_OUTPUT],
        current.values[ACCURACY_SCORE],
    )


async def write_agentx_prelude_baseline(
    store: GateRecords, root_commit: str, result: TestResult, *, source: str, wall_seconds: float
) -> BaselineRecord:
    """The baseline read from the PRELUDE child's own run of the root, ``result``; it anchors on no root test, so
    ``source`` names the run and ``wall_seconds`` is how long it took."""
    record = replace(_agentx_record(root_commit, agentx_reading(result)), source=source, wall_seconds=wall_seconds)
    await store.write_baseline(record)
    return record


def _agentx_record(root_commit: str, reading: Mapping[str, float]) -> BaselineRecord:
    missing = [key for key in _AGENTX_REQUIRED if key not in reading]
    if missing:
        raise UnreadableResult(f"the AgentX baseline does not report {', '.join(missing)}")
    return BaselineRecord(
        root_commit,
        values=dict(sorted(reading.items())),
        dispersion_pct={},
        threshold_pct={GRADED_INTVTY_P50: AGENTX_KEEP_P50_THRESHOLD_PCT},
        request_ids=(),
    )


def _root_result(record: TestRecord, root_commit: str) -> TestResult:
    if record.request.purpose is not Purpose.BASELINE or record.request.parent_commit != root_commit:
        raise ValueError(f"{record.request_id} is not a baseline test of the root {root_commit}")
    if record.result is None:
        raise UnreadableResult(f"{record.request_id} has no result yet")
    return record.result


__all__ = [
    "read_inferencex_baseline",
    "write_agentx_prelude_baseline",
    "write_inferencex_baseline",
]
