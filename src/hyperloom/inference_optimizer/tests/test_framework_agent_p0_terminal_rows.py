# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified candidate key + terminal-row invariant."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.framework import FrameworkPhase
from hyperloom.orchestrator.framework.artifacts import candidate_key


def test_candidate_key_precedence_and_fallbacks():
    assert candidate_key({"candidate_id": "cid", "pr_url": "u", "ref": "r"}) == "cid"
    assert candidate_key({"pr_url": "u", "ref": "r"}) == "u"
    assert candidate_key({"ref": "r"}) == "r"
    assert candidate_key({}) == ""
    assert candidate_key(None) == ""
    assert candidate_key("not-a-dict") == ""  # type: ignore[arg-type]


class _StateStub:
    def __init__(self) -> None:
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.framework_agent_review_counts: dict[str, int] = {}
        self.saves = 0

    def save(self, _session_dir: Path) -> None:
        self.saves += 1

    def record_action_failure(self, **_kw: Any) -> None:
        return None


class _MiniCoord:
    """The Coordinator surface the framework progress helpers read."""

    _handle_unpromotable_result = Coordinator._handle_unpromotable_result

    def __init__(self, tmp_path: Path) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub()
        self.phase_framework = FrameworkPhase(self)


def test_pr_url_only_candidate_dedups_against_progress_row(tmp_path: Path):
    """A candidate with only ``pr_url`` must dedup once its terminal row (keyed on the same pr_url) is written."""
    coord = _MiniCoord(tmp_path)
    cand_pr_only = {"pr_url": "https://example.com/pr/9", "batch_id": "b1"}
    cand_other = {"candidate_id": "cid-2", "batch_id": "b1"}
    coord.shared_state.framework_agent_batches = [
        {"batch_id": "b1", "candidates": [cand_pr_only, cand_other]},
    ]
    assert len(coord.phase_framework._unprocessed_framework_agent_candidates()) == 2
    coord.phase_framework._stamp_framework_progress(
        candidate_id=coord.phase_framework._framework_candidate_key(cand_pr_only),
        batch_id="b1",
        status="critic_denied",
    )
    remaining = coord.phase_framework._unprocessed_framework_agent_candidates()
    assert [candidate_key(c) for c in remaining] == ["cid-2"]


def test_stamp_writes_row_and_is_idempotent(tmp_path: Path):
    coord = _MiniCoord(tmp_path)
    first = coord.phase_framework._stamp_framework_progress(
        candidate_id="cid-1",
        batch_id="b1",
        status="reauthor_cap",
        rationale="cap reached",
        provenance="pump",
        extra={"error": "boom"},
    )
    assert first is True
    rows = coord.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "cid-1"
    assert row["batch_id"] == "b1"
    assert row["status"] == "reauthor_cap"
    assert row["kept"] is False
    assert row["rationale"] == "cap reached"
    assert row["error"] == "boom"
    assert row["ts"]
    second = coord.phase_framework._stamp_framework_progress(
        candidate_id="cid-1",
        batch_id="b1",
        status="no_result_failed",
    )
    assert second is False
    assert len(coord.shared_state.framework_agent_phase_progress) == 1
    assert coord.shared_state.framework_agent_phase_progress[0]["status"] == "reauthor_cap"


def test_stamp_empty_key_is_noop(tmp_path: Path):
    coord = _MiniCoord(tmp_path)
    assert coord.phase_framework._stamp_framework_progress(candidate_id="", status="x") is False
    assert coord.shared_state.framework_agent_phase_progress == []


def test_failed_framework_task_stamps_no_result_failed(tmp_path: Path):
    """An upstream-PR task settling ``status="failed"`` routes to ``_handle_unpromotable_result`` and must be stamped no_result_failed."""
    coord = _MiniCoord(tmp_path)
    task = SimpleNamespace(
        kind="integrate_patch",
        task_id="t-1",
        params={
            "framework_agent_candidate_id": "https://example.com/pr/7",
            "candidate": {"pr_url": "https://example.com/pr/7"},
            "batch_id": "b1",
        },
    )
    result = {"status": "failed", "reason": "server never came up"}
    asyncio.run(coord._handle_unpromotable_result(task, result))
    rows = coord.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "https://example.com/pr/7"
    assert rows[0]["status"] == "no_result_failed"
    assert rows[0]["kept"] is False


class _BusStub:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def append_and_seq(self, msg: Any) -> Any:
        self.messages.append(msg)
        return msg


class _ReviewCoord(_MiniCoord):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.bus = _BusStub()
        self.state = SimpleNamespace(pending_proposals={})

    async def _record_observation(self, *_a: Any, **_k: Any) -> None:
        return None


def _submit(coord: _ReviewCoord, cand: dict[str, Any]) -> None:
    asyncio.run(coord.phase_framework._submit_framework_agent_candidate_for_review(cand))


def test_repeated_review_aborts_after_cap(tmp_path: Path):
    coord = _ReviewCoord(tmp_path)
    cand = {"candidate_id": "cid-loop", "batch_id": "b1"}
    cap = FrameworkPhase._MAX_REPEATED_REVIEW_SUBMISSIONS
    # First ``cap`` submissions proceed (each drains its pending before the next).
    for _ in range(cap):
        coord.state.pending_proposals = {}
        _submit(coord, cand)
    assert coord.shared_state.framework_agent_review_counts["cid-loop"] == cap
    assert coord.shared_state.framework_agent_phase_progress == []
    # The (cap+1)-th submission trips the backstop: terminal row, no proposal.
    coord.state.pending_proposals = {}
    proposals_before = len(coord.bus.messages)
    _submit(coord, cand)
    rows = coord.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["status"] == "repeated_review_abort"
    assert rows[0]["candidate_id"] == "cid-loop"
    assert len(coord.bus.messages) == proposals_before
    # Once stamped, the candidate is "processed" → never re-selected.
    coord.shared_state.framework_agent_batches = [
        {"batch_id": "b1", "candidates": [cand]},
    ]
    assert coord.phase_framework._select_next_framework_agent_candidate() is None
