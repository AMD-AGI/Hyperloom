# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A lease-reaped task death must count as a failure for its action.

A holder killed from outside this process (its lease reaped by
``reap_dead_holders``) never returns a ``delegated_result``, so
``baseline_failure_streak`` used to stay 0 and the streak-3 auto-terminate
never fired — the run could not leave PRELUDE.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult
from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)

# Far above any real pid on Linux, so the liveness probe proves it dead.
_DEAD_PID = 2_147_483_646


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel_agent": MockBackend(silent, name="kernel_agent"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }


async def _running_task_with_dead_lease(
    coord: Coordinator,
    *,
    kind: str = "baseline",
    key: str,
):
    """Create a running task holding a lease from a provably dead process."""
    task = await coord.tasks.create(kind=kind, params={}, idempotency_key=key)
    await coord.tasks.transition(task.task_id, "running")
    coord.db.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            "benchmark_lane",
            task.task_id,
            task.task_id,
            kind,
            _DEAD_PID,
            "2026-01-01T00:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    coord.db.raw.commit()
    return task


@pytest.mark.asyncio
async def test_pump_counts_lease_reaped_baseline_as_failure(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        task = await _running_task_with_dead_lease(c, key="k-dead-1")
        await c._pump_dispatcher_once()
        assert (await c.tasks.get(task.task_id)).state == "failed"
        assert c.shared_state.baseline_failure_streak == 1
        assert c.shared_state.baseline_total_failures == 1
        entry = c.shared_state.last_action_failures[-1]
        assert entry["action"] == "baseline"
        assert entry["task_id"] == task.task_id
        assert entry["error_class"] == "dead_holder_reaped"
        assert c.shared_state.baseline_attempts[-1]["status"] == "failed"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_three_lease_reaped_baselines_trip_the_streak_stop(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        for i in range(3):
            await _running_task_with_dead_lease(c, key=f"k-dead-streak-{i}")
            await c._pump_dispatcher_once()
        assert c.shared_state.baseline_failure_streak == 3
        assert c.shared_state.stop_reason == "baseline_failed"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_accounting_is_idempotent_per_task(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        task = await _running_task_with_dead_lease(c, key="k-dead-once")
        await c._pump_dispatcher_once()
        await c._account_dead_holder_failures([task.task_id], reason="dead_holder_pump")
        assert c.shared_state.baseline_failure_streak == 1
        assert len(c.shared_state.last_action_failures) == 1
    finally:
        await c.stop()


class _ReapStub:
    """Minimal coordinator shell for the reap-path double-count guard."""

    def __init__(self) -> None:
        self.unpromotable: list[str] = []
        self.gpu_specialist_pool = SimpleNamespace(release=self._noop_async)
        self.bus = SimpleNamespace(append_and_seq=self._noop_async)

    async def _noop_async(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def _is_promotable_result(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    async def _handle_unpromotable_result(self, task: Any, _result: Any) -> None:
        self.unpromotable.append(task.task_id)

    async def _fact_write_hook(self, **_kwargs: Any) -> None:
        return None

    def _record_coordinator_exception(self, **_kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_reap_skips_failure_accounting_already_charged():
    stub = _ReapStub()
    disp = DispatcherCollaborator(stub)
    task = SimpleNamespace(task_id="t-dead", kind="baseline", params={})
    result = SubAgentResult(task_id=task.task_id, state="failed", result={"status": "failed"})

    await disp._reap_dispatched_task(task, result, None)
    assert stub.unpromotable == ["t-dead"]

    disp._dead_holder_accounted.add(task.task_id)
    await disp._reap_dispatched_task(task, result, None)
    assert stub.unpromotable == ["t-dead"]
