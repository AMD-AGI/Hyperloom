# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a long task's progress trail is allowed to cost its own row."""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.state.task_registry import (
    _MAX_PROGRESS_NOTES,
    Task,
    TaskRegistry,
    create_in_cursor,
    task_dispatch_class,
    task_dispatch_evidence,
    task_dispatch_record,
)


_ORIGIN = {"phase": "PRELUDE", "macro_cycle": 2, "tick": 7}


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


@pytest.mark.asyncio
async def test_dispatch_class_is_persisted_without_a_schema_migration(tmp_path):
    registry = TaskRegistry(SqliteConnection(tmp_path / "dispatch-class.db"))
    try:
        task = await registry.create(
            kind="baseline",
            params={},
            idempotency_key="llm-baseline",
            dispatch_class="llm",
            dispatch_origin=_ORIGIN,
        )
        reloaded = await registry.get(task.task_id)
    finally:
        registry.db.close()

    assert task_dispatch_class(reloaded) == "llm"
    assert task_dispatch_record(reloaded) == {
        "dispatch_class": "llm",
        "allowed": True,
        "denial_rule": None,
        **_ORIGIN,
    }


@pytest.mark.asyncio
async def test_unconfigured_registry_does_not_publish_partial_dispatch_evidence(tmp_path):
    registry = TaskRegistry(SqliteConnection(tmp_path / "legacy-registry.db"))
    try:
        task = await registry.create(
            kind="baseline",
            params={},
            idempotency_key="legacy-registry",
            dispatch_class="coordinator",
        )
    finally:
        registry.db.close()

    assert task_dispatch_evidence(task) is None
    assert task_dispatch_record(task) is None


@pytest.mark.asyncio
async def test_unknown_dispatch_class_fails_before_insert(tmp_path):
    registry = TaskRegistry(SqliteConnection(tmp_path / "bad-dispatch-class.db"))
    try:
        with pytest.raises(ValueError, match="unknown dispatch_class"):
            await registry.create(
                kind="baseline",
                params={},
                idempotency_key="bad",
                dispatch_class="agent",
            )
        row = await registry.db.fetchone("SELECT COUNT(*) AS count FROM tasks")
    finally:
        registry.db.close()

    assert row["count"] == 0


@pytest.mark.asyncio
async def test_idempotent_reuse_requires_the_same_dispatch_class(tmp_path):
    registry = TaskRegistry(SqliteConnection(tmp_path / "dispatch-reuse.db"))
    try:
        created, existing = await registry.create_or_return_existing(
            kind="baseline",
            params={},
            idempotency_key="same-key",
            dispatch_class="llm",
            dispatch_origin=_ORIGIN,
        )
        reused, was_existing = await registry.create_or_return_existing(
            kind="baseline",
            params={},
            idempotency_key="same-key",
            dispatch_class="llm",
        )
        with pytest.raises(ValueError, match="dispatch_class mismatch"):
            await registry.create_or_return_existing(
                kind="baseline",
                params={},
                idempotency_key="same-key",
                dispatch_class="coordinator",
            )
    finally:
        registry.db.close()

    assert existing is False
    assert was_existing is True
    assert reused.task_id == created.task_id


@pytest.mark.asyncio
async def test_cursor_reuse_validates_dispatch_class(tmp_path):
    db = SqliteConnection(tmp_path / "cursor-dispatch-reuse.db")
    try:
        async with db.transaction() as cur:
            created, existing = create_in_cursor(
                cur,
                kind="specialist",
                params={},
                idempotency_key="cursor-key",
                dispatch_class="coordinator",
                dispatch_origin=_ORIGIN,
            )
        async with db.transaction() as cur:
            reused, was_existing = create_in_cursor(
                cur,
                kind="specialist",
                params={},
                idempotency_key="cursor-key",
                dispatch_class="coordinator",
            )
        with pytest.raises(ValueError, match="dispatch_class mismatch"):
            async with db.transaction() as cur:
                create_in_cursor(
                    cur,
                    kind="specialist",
                    params={},
                    idempotency_key="cursor-key",
                    dispatch_class="inline",
                )
    finally:
        db.close()

    assert existing is False
    assert was_existing is True
    assert reused.task_id == created.task_id


def test_legacy_task_dispatch_provenance_is_unknown_not_guessed():
    task = Task(task_id="legacy", kind="baseline", state="queued", params={}, idempotency_key="legacy")

    assert task_dispatch_evidence(task) is None
    assert task_dispatch_class(task) is None


@pytest.mark.asyncio
async def test_legacy_task_reuse_stays_unknown_without_blocking_resume(tmp_path):
    registry = TaskRegistry(SqliteConnection(tmp_path / "legacy-reuse.db"))
    try:
        legacy, _ = await registry.create_or_return_existing(
            kind="baseline",
            params={},
            idempotency_key="legacy-async",
        )
        reused, was_existing = await registry.create_or_return_existing(
            kind="baseline",
            params={},
            idempotency_key="legacy-async",
            dispatch_class="coordinator",
        )
        async with registry.db.transaction() as cur:
            cursor_legacy, _ = create_in_cursor(
                cur,
                kind="specialist",
                params={},
                idempotency_key="legacy-cursor",
            )
        async with registry.db.transaction() as cur:
            cursor_reused, cursor_existing = create_in_cursor(
                cur,
                kind="specialist",
                params={},
                idempotency_key="legacy-cursor",
                dispatch_class="coordinator",
            )
    finally:
        registry.db.close()

    assert was_existing is True
    assert cursor_existing is True
    assert reused.task_id == legacy.task_id
    assert cursor_reused.task_id == cursor_legacy.task_id
    assert task_dispatch_evidence(reused) is None
    assert task_dispatch_evidence(cursor_reused) is None
