# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a long task's progress trail is allowed to cost its own row."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.state.task_registry import (
    _MAX_PROGRESS_NOTES,
    TaskRegistry,
)


async def _running_task(tmp_path, name: str) -> tuple[TaskRegistry, str]:
    """Create a registry holding one task already in ``running``."""
    registry = TaskRegistry(SqliteConnection(tmp_path / f"{name}.db"))
    task = await registry.create(kind="roofline", params={}, idempotency_key=name)
    await registry.transition(task.task_id, "running")
    return registry, task.task_id


async def _report(registry: TaskRegistry, task_id: str, indices: range) -> None:
    """Report one progress note per index, shaped like the heartbeat driver's."""
    for index in indices:
        await registry.record_progress(
            task_id,
            {"unit": "roofline_step", "index": index, "agent": "orchestration", "label": f"step-{index}"},
        )


async def _history_bytes(registry: TaskRegistry, task_id: str) -> int:
    """Size of the blob ``record_progress`` rewrites on every note."""
    row = await registry.db.fetchone("SELECT history FROM tasks WHERE task_id=?", (task_id,))
    return len(row["history"])


def _notes(history: list[dict]) -> list[dict]:
    return [entry["progress"] for entry in history if "progress" in entry]


@pytest.mark.asyncio
async def test_the_progress_trail_stops_growing_at_the_bound(tmp_path):
    """A 12-hour session at the 60s tick would otherwise leave a 160 KB blob."""
    over = _MAX_PROGRESS_NOTES + 40
    registry, task_id = await _running_task(tmp_path, "bounded")
    try:
        await _report(registry, task_id, range(over))
        at_bound = await _history_bytes(registry, task_id)
        await _report(registry, task_id, range(over, over + 40))
        later = await _history_bytes(registry, task_id)

        history = (await registry.get(task_id)).history
    finally:
        registry.db.close()

    notes = _notes(history)
    assert len(notes) == _MAX_PROGRESS_NOTES
    assert notes[0]["index"] == over + 40 - _MAX_PROGRESS_NOTES
    assert notes[-1]["index"] == over + 39
    # 40 more notes of this shape add ~4 KB to an uncapped blob; at the bound they only shift which ones are held, so
    # the size is steady.
    assert later - at_bound < 512
    assert at_bound < 32 * 1024


@pytest.mark.asyncio
async def test_no_number_of_notes_can_bury_a_state_transition(tmp_path):
    """Consumers read transitions positionally; dropping one would make them lie."""
    registry, task_id = await _running_task(tmp_path, "transitions")
    try:
        await _report(registry, task_id, range(_MAX_PROGRESS_NOTES + 5))
        await registry.transition(task_id, "failed", evidence={"failure_class": "timeout"})

        history = (await registry.get(task_id)).history
    finally:
        registry.db.close()

    transitions = [(entry.get("from"), entry.get("to")) for entry in history if "to" in entry]
    assert transitions == [("queued", "running"), ("running", "failed")]
    assert history[-1]["evidence"]["failure_class"] == "timeout"


@pytest.mark.asyncio
async def test_a_note_lands_whole_and_readable_under_the_bound(tmp_path):
    """The trail is still a trail: the retained notes keep their own timestamps."""
    registry, task_id = await _running_task(tmp_path, "readable")
    try:
        await _report(registry, task_id, range(3))
        row = await registry.db.fetchone("SELECT history FROM tasks WHERE task_id=?", (task_id,))
    finally:
        registry.db.close()

    notes = [entry for entry in json.loads(row["history"]) if "progress" in entry]
    assert [entry["progress"]["label"] for entry in notes] == ["step-0", "step-1", "step-2"]
    assert all(entry["ts"] for entry in notes)


def _terminal_runner(registry):
    from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner

    return SubAgentRunner(ResourceLockManager(SqliteLeaseBackend(registry.db)), registry)


@pytest.mark.asyncio
async def test_terminal_evidence_keeps_repeated_results_and_survives_progress_pruning(tmp_path):
    registry, task_id = await _running_task(tmp_path, "terminal-evidence")
    try:
        await registry.transition(task_id, "cancelled", evidence={"reason": "watchdog"})
        before = await registry.get(task_id)
        evidence = {"outcome": {"result": {"answer": [42]}, "state": "succeeded"}, "cleanup_confirmed": False}
        runner = _terminal_runner(registry)
        for _ in range(2):
            await runner._write_terminal(task_id, "succeeded", evidence=evidence, context="test")
        appended = await registry.get(task_id)
        assert appended.state == "cancelled"
        assert appended.updated_at == before.updated_at
        assert appended.history[:-2] == before.history
        for entry in appended.history[-2:]:
            assert set(entry) == {"ts", "evidence"}
            assert entry["ts"]
            assert entry["evidence"] == evidence
        await _report(registry, task_id, range(_MAX_PROGRESS_NOTES + 5))
        retained = await registry.get(task_id)
        assert retained.updated_at == before.updated_at
        assert [row for row in retained.history if "progress" not in row] == appended.history
        assert len(_notes(retained.history)) == _MAX_PROGRESS_NOTES
    finally:
        registry.db.close()


@pytest.mark.asyncio
async def test_terminal_evidence_successful_transition_keeps_transition_shape(tmp_path):
    registry, task_id = await _running_task(tmp_path, "terminal-transition")
    try:
        await _terminal_runner(registry)._write_terminal(task_id, "succeeded", context="test")
        task = await registry.get(task_id)
        assert task.state == "succeeded"
        assert len(task.history) == 2
        assert task.history[-1] == {"from": "running", "to": "succeeded", "ts": task.updated_at, "evidence": {}}
    finally:
        registry.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disappear_after_transition", [False, True])
async def test_terminal_evidence_does_not_hide_missing_rows(tmp_path, monkeypatch, disappear_after_transition):
    from hyperloom.orchestrator.state.task_registry import IllegalTransition, TaskNotFound

    registry, task_id = await _running_task(tmp_path, "terminal-missing")
    try:
        if disappear_after_transition:

            async def lost_race(*_args, **_kwargs):
                await registry.db.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
                raise IllegalTransition("already terminal")

            monkeypatch.setattr(registry, "transition", lost_race)
        else:
            await registry.db.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
        error = TypeError if disappear_after_transition else TaskNotFound
        with pytest.raises(error):
            await _terminal_runner(registry)._write_terminal(task_id, "succeeded", context="test")
        assert registry.db.raw.in_transaction is False
    finally:
        registry.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_append", [False, True])
async def test_terminal_evidence_second_transaction_rolls_back_on_failure(tmp_path, monkeypatch, cancel_append):
    registry, task_id = await _running_task(tmp_path, "terminal-rollback")
    try:
        await registry.transition(task_id, "cancelled")
        before = await registry.get(task_id)
        transaction = registry.db.transaction
        transactions = []

        @asynccontextmanager
        async def tracked_transaction():
            transactions.append("begin")
            async with transaction() as cursor:
                yield cursor
                if cancel_append:
                    raise asyncio.CancelledError()
            transactions.append("commit")

        monkeypatch.setattr(registry.db, "transaction", tracked_transaction)
        evidence = {"result": 42} if cancel_append else {"not_json": object()}
        with pytest.raises(asyncio.CancelledError if cancel_append else TypeError):
            await _terminal_runner(registry)._write_terminal(task_id, "succeeded", evidence=evidence, context="test")
        assert transactions == ["begin", "begin"]
        assert await registry.get(task_id) == before
        assert registry.db.raw.in_transaction is False
        monkeypatch.setattr(registry.db, "transaction", transaction)
        await registry.record_progress(task_id, {"after": "rollback"})
        assert (await registry.get(task_id)).history[-1]["progress"] == {"after": "rollback"}
    finally:
        registry.db.close()
