# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified candidate key + terminal-row invariant."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hyperloom.orchestrator.framework.artifacts import candidate_key
from hyperloom.orchestrator.state.task_registry import Task

from .conftest import make_coordinator
from hyperloom.orchestrator.phases.framework_records import ProgressStamp


def test_candidate_key_precedence_and_fallbacks():
    assert candidate_key({"candidate_id": "cid", "pr_url": "u", "ref": "r"}) == "cid"
    assert candidate_key({"pr_url": "u", "ref": "r"}) == "u"
    assert candidate_key({"ref": "r"}) == "r"
    assert candidate_key({}) == ""
    assert candidate_key(None) == ""
    assert candidate_key("not-a-dict") == ""  # type: ignore[arg-type]


def test_pr_url_only_candidate_dedups_against_progress_row(tmp_path: Path):
    """A candidate with only ``pr_url`` must dedup once its terminal row (keyed on the same pr_url) is written."""
    coord = make_coordinator(tmp_path)
    phase = coord.phase_framework
    cand_pr_only = {"pr_url": "https://example.com/pr/9", "batch_id": "b1"}
    cand_other = {"candidate_id": "cid-2", "batch_id": "b1"}
    coord.shared_state.framework_agent_batches = [
        {"batch_id": "b1", "candidates": [cand_pr_only, cand_other]},
    ]
    assert len(phase._unprocessed_framework_agent_candidates()) == 2
    phase._stamp_framework_progress(
        ProgressStamp(candidate_id=phase._framework_candidate_key(cand_pr_only), batch_id="b1", status="enqueue_failed")
    )
    remaining = phase._unprocessed_framework_agent_candidates()
    assert [candidate_key(c) for c in remaining] == ["cid-2"]


def test_stamp_writes_row_and_is_idempotent(tmp_path: Path):
    coord = make_coordinator(tmp_path)
    phase = coord.phase_framework
    first = phase._stamp_framework_progress(
        ProgressStamp(
            candidate_id="cid-1",
            batch_id="b1",
            status="apply_fail_cap",
            rationale="cap reached",
            provenance="pump",
            extra={"error": "boom"},
        )
    )
    assert first is True
    rows = coord.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "cid-1"
    assert row["batch_id"] == "b1"
    assert row["status"] == "apply_fail_cap"
    assert row["kept"] is False
    assert row["rationale"] == "cap reached"
    assert row["error"] == "boom"
    assert row["ts"]
    second = phase._stamp_framework_progress(
        ProgressStamp(candidate_id="cid-1", batch_id="b1", status="no_result_failed")
    )
    assert second is False
    assert len(coord.shared_state.framework_agent_phase_progress) == 1
    assert coord.shared_state.framework_agent_phase_progress[0]["status"] == "apply_fail_cap"


def test_stamp_empty_key_is_noop(tmp_path: Path):
    coord = make_coordinator(tmp_path)
    assert coord.phase_framework._stamp_framework_progress(ProgressStamp(candidate_id="", status="x")) is False
    assert coord.shared_state.framework_agent_phase_progress == []


def test_failed_framework_task_stamps_no_result_failed(tmp_path: Path):
    """An upstream-PR task settling ``status="failed"`` routes to ``handle_unpromotable_result`` and must be stamped no_result_failed."""
    coord = make_coordinator(tmp_path)
    task = Task(
        task_id="t-1",
        kind="integrate_patch",
        state="succeeded",
        idempotency_key="t-1",
        params={
            "framework_agent_candidate_id": "https://example.com/pr/7",
            "candidate": {"pr_url": "https://example.com/pr/7"},
            "batch_id": "b1",
        },
    )
    result = {"status": "failed", "reason": "server never came up"}
    asyncio.run(coord.writeback.handle_unpromotable_result(task, result))
    rows = coord.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "https://example.com/pr/7"
    assert rows[0]["status"] == "no_result_failed"
    assert rows[0]["kept"] is False
