"""End-to-end smoke test for the SQLite-backed Conductor SoT.

Simulates: create task -> run -> crash mid-run -> reopen DB -> verify the
in-flight task is still discoverable as ``running`` so the Conductor's
resume logic can route it to the §13.6 evidence-check matrix.

Also verifies cross-table atomicity: a single transaction that does
(advance cursor + emit event + transition task) either commits all four
writes or rolls them all back.
"""

from __future__ import annotations

import json

import pytest

from inference_optimizer.orchestrator.cursor_store import CursorStore
from inference_optimizer.orchestrator.message_bus import Message, MessageBus
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.storage.connection import SqliteConnection


@pytest.mark.asyncio
async def test_resume_finds_inflight_task(db_path):
    db = SqliteConnection(db_path)
    try:
        reg = TaskRegistry(db)
        bus = MessageBus(db)
        task = await reg.create(
            kind="bench_runner",
            params={"prompt": "x"},
            idempotency_key="b-1",
        )
        await reg.transition(task.task_id, "running", {"pid": 12345})
        await bus.append_and_seq(
            Message.new("conductor", "executor", "delegated_result",
                        {"task_id": task.task_id, "status": "running"})
        )
    finally:
        # Simulate crash by closing the connection without further work.
        db.close()

    # ``Resume`` reopens the DB and finds the running task.
    db2 = SqliteConnection(db_path)
    try:
        reg2 = TaskRegistry(db2)
        bus2 = MessageBus(db2)
        inflight = await reg2.inflight()
        assert len(inflight) == 1
        assert inflight[0].task_id == task.task_id
        events = await bus2.replay_for("executor", after_seq=0)
        assert len(events) == 1
        assert events[0].topic == "delegated_result"
    finally:
        db2.close()


@pytest.mark.asyncio
async def test_cross_table_atomicity_rollback(db):
    """If we raise inside a single transaction, NONE of the writes commit."""
    reg = TaskRegistry(db)
    bus = MessageBus(db)
    cursors = CursorStore(db)
    task = await reg.create(
        kind="bench_runner", params={}, idempotency_key="atomic-1",
    )

    pre_count = (await db.fetchone("SELECT COUNT(*) AS c FROM events"))["c"]
    pre_state = (await reg.get(task.task_id)).state
    pre_cursor = (await cursors.load("executor")).last_processed_seq

    with pytest.raises(RuntimeError):
        async with db.transaction() as cur:
            cur.execute(
                "INSERT INTO events(msg_id, from_agent, to_agent, topic, "
                "in_reply_to, payload, priority, ts) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("xrollback", "x", "*", "event", None, "{}", 1, "now"),
            )
            cur.execute(
                "UPDATE tasks SET state='running' WHERE task_id=?",
                (task.task_id,),
            )
            cur.execute(
                "INSERT INTO cursors(agent, last_processed_seq, "
                "last_processed_msg_id, processed_at) "
                "VALUES ('executor', 999, 'x', 'now')"
            )
            raise RuntimeError("simulated mid-txn crash")

    # All three writes must have rolled back.
    post_count = (await db.fetchone("SELECT COUNT(*) AS c FROM events"))["c"]
    post_state = (await reg.get(task.task_id)).state
    post_cursor = (await cursors.load("executor")).last_processed_seq
    assert post_count == pre_count
    assert post_state == pre_state == "queued"
    assert post_cursor == pre_cursor == 0


@pytest.mark.asyncio
async def test_cross_table_atomicity_commit(db):
    """If the txn body succeeds, every write is visible together."""
    reg = TaskRegistry(db)
    cursors = CursorStore(db)
    task = await reg.create(
        kind="bench_runner", params={}, idempotency_key="atomic-2",
    )

    async with db.transaction() as cur:
        cur.execute(
            "INSERT INTO events(msg_id, from_agent, to_agent, topic, "
            "in_reply_to, payload, priority, ts) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("xcommit", "x", "executor", "delegated_result",
             None, json.dumps({"task_id": task.task_id}), 2, "now"),
        )
        cur.execute(
            "UPDATE tasks SET state='running', updated_at='now' WHERE task_id=?",
            (task.task_id,),
        )
        cur.execute(
            "INSERT INTO cursors(agent, last_processed_seq, "
            "last_processed_msg_id, processed_at) "
            "VALUES ('executor', 1, 'xcommit', 'now')"
        )

    state = (await reg.get(task.task_id)).state
    seq = (await cursors.load("executor")).last_processed_seq
    rows = await db.fetchall("SELECT msg_id FROM events WHERE msg_id='xcommit'")
    assert state == "running"
    assert seq == 1
    assert len(rows) == 1
