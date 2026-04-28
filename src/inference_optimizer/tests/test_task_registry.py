"""Tests for orchestrator/task_registry.py."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.task_registry import (
    IllegalTransition,
    TaskNotFound,
    TaskRegistry,
)


@pytest.mark.asyncio
async def test_create_persists_task(db):
    reg = TaskRegistry(db)
    task = await reg.create(
        kind="bench_runner",
        params={"prompt": "hi"},
        idempotency_key="bench-1",
        requires_lanes=["benchmark_lane"],
        side_effects=[],
        lease_ttl_sec=600,
    )
    assert task.state == "queued"
    assert task.attempts == 0
    fetched = await reg.get(task.task_id)
    assert fetched.task_id == task.task_id
    assert fetched.kind == "bench_runner"


@pytest.mark.asyncio
async def test_create_is_idempotent(db):
    reg = TaskRegistry(db)
    a = await reg.create(
        kind="bench_runner", params={}, idempotency_key="same-key",
    )
    b = await reg.create(
        kind="bench_runner", params={}, idempotency_key="same-key",
    )
    assert a.task_id == b.task_id


@pytest.mark.asyncio
async def test_state_machine_legal_paths(db):
    reg = TaskRegistry(db)
    t = await reg.create(kind="x", params={}, idempotency_key="k1")
    t = await reg.transition(t.task_id, "running", {"pid": 1})
    assert t.state == "running"
    t = await reg.transition(t.task_id, "succeeded", {"result_path": "/tmp/r"})
    assert t.state == "succeeded"
    assert len(t.history) == 2
    assert t.history[-1]["from"] == "running"
    assert t.history[-1]["to"] == "succeeded"


@pytest.mark.asyncio
async def test_state_machine_rejects_illegal(db):
    reg = TaskRegistry(db)
    t = await reg.create(kind="x", params={}, idempotency_key="k2")
    # queued -> succeeded is not allowed
    with pytest.raises(IllegalTransition):
        await reg.transition(t.task_id, "succeeded")
    # terminal states can't move further
    t = await reg.transition(t.task_id, "running")
    t = await reg.transition(t.task_id, "succeeded")
    with pytest.raises(IllegalTransition):
        await reg.transition(t.task_id, "running")


@pytest.mark.asyncio
async def test_failed_can_retry_then_terminal(db):
    reg = TaskRegistry(db)
    t = await reg.create(kind="x", params={}, idempotency_key="k3")
    t = await reg.transition(t.task_id, "running")
    t = await reg.transition(t.task_id, "failed", {"err": "boom"})
    t = await reg.transition(t.task_id, "running")
    t = await reg.transition(t.task_id, "needs_manual_review",
                             {"reason": "evidence_missing"})
    assert t.state == "needs_manual_review"
    with pytest.raises(IllegalTransition):
        await reg.transition(t.task_id, "running")


@pytest.mark.asyncio
async def test_bump_attempts(db):
    reg = TaskRegistry(db)
    t = await reg.create(kind="x", params={}, idempotency_key="k4")
    n1 = await reg.bump_attempts(t.task_id)
    n2 = await reg.bump_attempts(t.task_id)
    assert n1 == 1
    assert n2 == 2


@pytest.mark.asyncio
async def test_inflight_query(db):
    reg = TaskRegistry(db)
    t1 = await reg.create(kind="x", params={}, idempotency_key="a")
    t2 = await reg.create(kind="y", params={}, idempotency_key="b")
    await reg.transition(t1.task_id, "running")
    await reg.transition(t2.task_id, "running")
    await reg.transition(t1.task_id, "succeeded")
    inflight = await reg.inflight()
    ids = {t.task_id for t in inflight}
    assert t2.task_id in ids
    assert t1.task_id not in ids


@pytest.mark.asyncio
async def test_get_unknown_task_raises(db):
    reg = TaskRegistry(db)
    with pytest.raises(TaskNotFound):
        await reg.get("nonexistent")
