# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a long task's progress trail is allowed to cost its own row.

Every note rewrites the whole ``history`` blob inside a transaction on the one
shared connection, so an unbounded trail charges a session for its own length —
and the sessions this heartbeat exists for are the long ones. These tests pin
the bound and the thing the bound must not break: the state transitions
consumers read positionally.
"""

from __future__ import annotations

import json

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
    """A 12-hour session at the 60s tick would otherwise leave a 160 KB blob.

    The newest notes are the ones a consumer reads, so the oldest are dropped,
    and once the bound is reached each further note costs what one note costs
    rather than what the session's whole trail costs.
    """
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
    # 40 more notes of this shape add ~4 KB to an uncapped blob; at the bound
    # they only shift which ones are held, so the size is steady.
    assert later - at_bound < 512
    assert at_bound < 32 * 1024


@pytest.mark.asyncio
async def test_no_number_of_notes_can_bury_a_state_transition(tmp_path):
    """Consumers read transitions positionally; dropping one would make them lie.

    The dispatcher's policy-denied lookup scans for the newest
    ``queued -> cancelled`` entry, and the enablement path reads a failure class
    off the last one, so the cap must only ever retire progress notes.
    """
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
