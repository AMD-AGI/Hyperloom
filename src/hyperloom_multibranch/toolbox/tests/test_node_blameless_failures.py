# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A runner that could not start or was lost leaves its node out of the quarantine the blackboard keeps; a node fault
still quarantines it, and that quarantine is read back when the toolbox starts again, while one a blameless failure
recorded before is not."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom_multibranch.contracts.blackboard.records import QuarantineEvent
from hyperloom_multibranch.contracts.blackboard.vocab import QuarantineAction
from hyperloom_multibranch.toolbox.attempts import TestJob
from hyperloom_multibranch.toolbox.tests.helpers import Rig, ScriptedExecutor, build, failed, ok, req
from hyperloom_multibranch.toolbox.tests.test_attempts import (
    CACHE_REASON,
    CREATE_FAILED_REASON,
    HUB_REFUSED_REASON,
    KEY_READ_ONLY_REASON,
    LOST_REASON,
)

NODE_FAULT_REASON = "node_fault: amdgpu: GPU reset on card 3"


def _fails_once_then_succeeds(reason: str) -> ScriptedExecutor:
    """The first attempt fails with ``reason`` on node ``v3-1``; the retry succeeds on ``v3-2``."""

    def first(job: TestJob) -> TestResult:
        return failed(job, FailureClass.INFRA, node="v3-1", reason=reason)

    def second(job: TestJob) -> TestResult:
        return ok(job, node="v3-2")

    return ScriptedExecutor(script=[first, second])


async def _run_one(rig: Rig) -> TestResult:
    await rig.toolbox.start()
    request_id = await rig.toolbox.submit_test(req())
    await asyncio.wait_for(rig.toolbox.drain(), 60)
    entry = rig.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None
    return TestResult.from_document(entry.outcome["result"])


async def _restarted_quarantine(tmp_path: Path) -> frozenset[str]:
    again = build(tmp_path)
    await again.toolbox.start()
    try:
        return again.toolbox.quarantined
    finally:
        await again.toolbox.close()


@pytest.mark.parametrize(
    "reason",
    [KEY_READ_ONLY_REASON, CACHE_REASON, LOST_REASON, CREATE_FAILED_REASON, HUB_REFUSED_REASON],
    ids=["signing_key", "runner_cache", "runner_lost", "sandbox_create_failed", "hub_rate_limited"],
)
async def test_a_runner_that_could_not_start_or_was_lost_does_not_quarantine_its_node(
    tmp_path: Path, reason: str
) -> None:
    rig = build(tmp_path, executor=_fails_once_then_succeeds(reason))
    result = await _run_one(rig)
    await rig.toolbox.close()
    assert result.validity is Validity.SUCCEEDED
    assert [a.failure_reason for a in result.attempts] == [reason, None]
    assert [a.node for a in result.attempts] == ["v3-1", "v3-2"], "the retry still went to another node"
    assert rig.toolbox.quarantined == frozenset() and await rig.blackboard.quarantined() == ()
    assert await _restarted_quarantine(tmp_path) == frozenset()


async def test_a_node_fault_still_quarantines_its_node_and_a_restart_reads_it_back(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=_fails_once_then_succeeds(NODE_FAULT_REASON))
    result = await _run_one(rig)
    await rig.toolbox.close()
    assert result.validity is Validity.SUCCEEDED
    assert rig.toolbox.quarantined == frozenset({"v3-1"})
    assert [(e.node, e.reason) for e in await rig.blackboard.quarantined()] == [("v3-1", NODE_FAULT_REASON)]
    assert await _restarted_quarantine(tmp_path) == frozenset({"v3-1"}), "the blackboard keeps it across restarts"


async def test_a_node_quarantine_recorded_for_a_blameless_failure_is_not_read_back(tmp_path: Path) -> None:
    rig = build(tmp_path)
    recorded = {
        "v3-014": KEY_READ_ONLY_REASON,
        "v3-073": CACHE_REASON,
        "v3-083": LOST_REASON,
        "v3-099": NODE_FAULT_REASON,
    }
    for node, reason in recorded.items():
        event = QuarantineEvent(node=node, action=QuarantineAction.QUARANTINE, reason=reason, request_ids=("r1",))
        await rig.blackboard.record_quarantine(event)
    await rig.toolbox.start()
    await rig.toolbox.close()
    assert rig.toolbox.quarantined == frozenset({"v3-099"}), "only the node fault still holds its node"
    assert {e.node for e in await rig.blackboard.quarantined()} == set(recorded), "the record itself stays as written"
