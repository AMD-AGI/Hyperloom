# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate's long-context check carries the root's reading the toolbox had when the candidate ended (#142 c3219):
the complete checks of the session's BASELINE tests, pooled, with their 95% interval; the gate applies ``refuses``."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom.contracts.long_context import LongContextResult, refuses, wilson_interval
from hyperloom.contracts.measurement import Purpose, TestRequest, TestResult
from hyperloom.contracts.tests.test_long_context_bar import _check
from hyperloom_multibranch.toolbox.attempts import TestJob
from hyperloom_multibranch.toolbox.tests.helpers import ScriptedExecutor, build, ok, req


def _answering(checks: dict[str, LongContextResult]):  # type: ignore[no-untyped-def]
    """An attempt that succeeds with the check ``checks`` names for its request's server arguments, or none."""

    def run(job: TestJob) -> TestResult:
        return ok(job, long_context=checks.get(job.request.delta.config.server_args))

    return run


async def _results(
    tmp_path: Path, requests: list[TestRequest], checks: dict[str, LongContextResult]
) -> list[TestResult]:
    """Run ``requests`` one after another on a toolbox whose tests answer with ``checks``; their recorded results."""
    rig = build(tmp_path, executor=ScriptedExecutor(script=[_answering(checks)]))
    await rig.toolbox.start()
    results = []
    try:
        for request in requests:
            request_id = await rig.toolbox.submit_test(request)
            await asyncio.wait_for(rig.toolbox.drain(), 30)
            entry = rig.toolbox._log.get(request_id)
            assert entry is not None and entry.outcome is not None
            results.append(TestResult.from_document(entry.outcome["result"]))
    finally:
        await rig.toolbox.close()
    return results


BASELINE = req(purpose=Purpose.BASELINE)
CANDIDATE = req(server_args="--max-num-seqs 300")
UNCHECKED = req(server_args="--max-num-seqs 301")
#: A candidate that ended before: its check is no part of the root's reading.
EARLIER = req(server_args="--max-num-seqs 302")


async def test_a_candidate_carries_the_roots_pooled_reading_and_the_root_tests_carry_none(tmp_path: Path) -> None:
    checks = {
        "": _check(64, 64),
        CANDIDATE.delta.config.server_args: _check(64, 58),
        EARLIER.delta.config.server_args: _check(64, 20),
    }

    *roots, _, candidate, unchecked = await _results(tmp_path, [BASELINE] * 3 + [EARLIER, CANDIDATE, UNCHECKED], checks)

    assert all(result.long_context is not None and result.long_context.root is None for result in roots)
    reading = candidate.long_context.root  # type: ignore[union-attr]
    assert reading is not None and reading.tests == tuple(result.request_id for result in roots)
    assert (reading.n, reading.passed) == (192, 192) and (reading.ci_low, reading.ci_high) == wilson_interval(192, 192)
    assert refuses(candidate.long_context, reading), "58 of 64 is more than 3 items under 0.98 of 64"
    assert unchecked.long_context is None, "a test that ran no check is given no reading"


async def test_without_a_complete_root_check_a_candidate_carries_no_reading_and_is_refused_nothing(
    tmp_path: Path,
) -> None:
    checks = {"": _check(64, 60, errored=4), CANDIDATE.delta.config.server_args: _check(64, 10)}

    *_, candidate = await _results(tmp_path, [BASELINE, CANDIDATE], checks)

    assert candidate.long_context is not None and candidate.long_context.root is None
    assert not refuses(candidate.long_context, candidate.long_context.root)


async def test_a_session_with_no_root_test_gives_no_reading(tmp_path: Path) -> None:
    checks = {CANDIDATE.delta.config.server_args: _check(64, 10)}

    (candidate,) = await _results(tmp_path, [CANDIDATE], checks)

    assert candidate.long_context is not None and candidate.long_context.root is None
