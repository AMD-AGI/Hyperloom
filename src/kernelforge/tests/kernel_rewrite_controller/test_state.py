# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller import TaskState, TaskStateError, TaskStateStore


def test_task_state_round_trip_and_terminal_transition(task_dir: Path) -> None:
    store = TaskStateStore(task_dir)
    ready = store.initialize_ready()
    running = store.transition("running", workspace_dir="/workspace/operator")
    succeeded = store.transition(
        "succeeded",
        result_patch_dir="/output/result/patches/operator",
    )

    assert ready.status == "ready"
    assert running.status == "running"
    assert running.started_at
    assert succeeded.status == "succeeded"
    assert succeeded.started_at == running.started_at
    assert succeeded.finished_at
    assert succeeded.workspace_dir == "/workspace/operator"
    assert succeeded.result_patch_dir == "/output/result/patches/operator"
    assert store.load() == succeeded
    assert not list(task_dir.glob(".state.json.*.tmp"))


def test_invalid_transition_is_rejected(task_dir: Path) -> None:
    store = TaskStateStore(task_dir)
    store.initialize_ready()

    with pytest.raises(TaskStateError, match="cannot transition"):
        store.transition("succeeded")


def test_terminal_state_cannot_be_reopened(task_dir: Path) -> None:
    store = TaskStateStore(task_dir)
    store.mark_skipped("invalid task")

    with pytest.raises(TaskStateError, match="cannot transition"):
        store.transition("running")


def test_corrupt_state_is_not_silently_replaced(task_dir: Path) -> None:
    state_path = task_dir / "state.json"
    state_path.write_text("{", encoding="utf-8")

    with pytest.raises(TaskStateError, match="could not read task state"):
        TaskStateStore(task_dir).load()


def test_task_state_rejects_unknown_schema_fields() -> None:
    payload = TaskState(status="ready").to_dict()
    payload["unexpected"] = True

    with pytest.raises(TaskStateError, match="unknown fields"):
        TaskState.from_dict(payload)


def test_task_state_rejects_unknown_status() -> None:
    payload = {
        "schema_version": 1,
        "status": "paused",
        "reason": "",
        "started_at": "",
        "finished_at": "",
        "workspace_dir": "",
        "result_patch_dir": "",
    }

    with pytest.raises(TaskStateError, match="unsupported task status"):
        TaskState.from_dict(json.loads(json.dumps(payload)))
