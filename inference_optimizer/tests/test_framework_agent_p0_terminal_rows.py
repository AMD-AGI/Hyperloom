# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P0 (framework enhance 1.0) — unified candidate key + terminal-row invariant.

Covers the four P0 sub-items:

- P0-1: :func:`candidate_key` / ``_framework_candidate_key`` unify the dedup key
  so a candidate carrying only a ``pr_url`` (no ``candidate_id``) still dedups
  against its own progress row.
- P0-2: ``_stamp_framework_progress`` is the single idempotent terminal-row
  writer.
- P0-3: silent apply/bench failure (framework_agent task settles
  ``status="failed"``) still stamps a ``no_result_failed`` terminal row.
- P0-4: a candidate submitted for review past the cap is force-aborted with a
  ``repeated_review_abort`` terminal row.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.framework_agent_artifacts import candidate_key


# ---------------------------------------------------------------------------
# P0-1 — candidate_key pure function + unified selector dedup.
# ---------------------------------------------------------------------------
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
    """Minimal binding of the P0 helpers under test."""

    _MAX_REPEATED_REVIEW_SUBMISSIONS = Coordinator._MAX_REPEATED_REVIEW_SUBMISSIONS
    _framework_candidate_key = staticmethod(Coordinator._framework_candidate_key)
    _framework_processed_candidate_keys = Coordinator._framework_processed_candidate_keys
    _stamp_framework_progress = Coordinator._stamp_framework_progress
    _unprocessed_framework_agent_candidates = Coordinator._unprocessed_framework_agent_candidates
    _select_next_framework_agent_candidate = Coordinator._select_next_framework_agent_candidate
    _handle_unpromotable_result = Coordinator._handle_unpromotable_result

    def __init__(self, tmp_path: Path) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub()


def test_pr_url_only_candidate_dedups_against_progress_row(tmp_path: Path):
    """Regression (P0-1): a candidate with only ``pr_url`` must dedup once its
    terminal row (keyed on the same pr_url) is written — the historical
    asymmetry re-selected it forever."""
    coord = _MiniCoord(tmp_path)
    cand_pr_only = {"pr_url": "https://example.com/pr/9", "batch_id": "b1"}
    cand_other = {"candidate_id": "cid-2", "batch_id": "b1"}
    coord.shared_state.framework_agent_batches = [
        {"batch_id": "b1", "candidates": [cand_pr_only, cand_other]},
    ]
    # Before any row: both are unprocessed.
    assert len(coord._unprocessed_framework_agent_candidates()) == 2
    # Stamp a terminal row for the pr_url-only candidate.
    coord._stamp_framework_progress(
        candidate_id=coord._framework_candidate_key(cand_pr_only),
        batch_id="b1",
        status="critic_denied",
    )
    remaining = coord._unprocessed_framework_agent_candidates()
    assert [candidate_key(c) for c in remaining] == ["cid-2"]


# ---------------------------------------------------------------------------
# P0-2 — _stamp_framework_progress: fields + idempotency.
# ---------------------------------------------------------------------------
def test_stamp_writes_row_and_is_idempotent(tmp_path: Path):
    coord = _MiniCoord(tmp_path)
    first = coord._stamp_framework_progress(
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
    assert row["error"] == "boom"  # extra merged onto the row
    assert row["ts"]
    # A second stamp for the same key is a no-op (idempotent).
    second = coord._stamp_framework_progress(
        candidate_id="cid-1", batch_id="b1", status="no_result_failed",
    )
    assert second is False
    assert len(coord.shared_state.framework_agent_phase_progress) == 1
    assert coord.shared_state.framework_agent_phase_progress[0]["status"] == "reauthor_cap"


def test_stamp_empty_key_is_noop(tmp_path: Path):
    coord = _MiniCoord(tmp_path)
    assert coord._stamp_framework_progress(candidate_id="", status="x") is False
    assert coord.shared_state.framework_agent_phase_progress == []


# ---------------------------------------------------------------------------
# P0-3 — silent apply/bench failure stamps no_result_failed.
# ---------------------------------------------------------------------------
def test_failed_framework_task_stamps_no_result_failed(tmp_path: Path):
    """A framework_agent task settling ``status="failed"`` routes to
    ``_handle_unpromotable_result``; without a terminal row the candidate would
    livelock (this session's death loop). It must be stamped no_result_failed."""
    coord = _MiniCoord(tmp_path)
    task = SimpleNamespace(
        kind="framework_agent",
        task_id="t-1",
        params={"candidate": {"pr_url": "https://example.com/pr/7"}, "batch_id": "b1"},
    )
    result = {"status": "failed", "reason": "server never came up"}
    asyncio.run(coord._handle_unpromotable_result(task, result))
    rows = coord.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "https://example.com/pr/7"
    assert rows[0]["status"] == "no_result_failed"
    assert rows[0]["kept"] is False


# ---------------------------------------------------------------------------
# P0-4 — repeated review backstop.
# ---------------------------------------------------------------------------
class _BusStub:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def append_and_seq(self, msg: Any) -> Any:
        self.messages.append(msg)
        return msg


class _ReviewCoord(_MiniCoord):
    _collect_framework_agent_candidate_priors = Coordinator._collect_framework_agent_candidate_priors
    _submit_framework_agent_candidate_for_review = (
        Coordinator._submit_framework_agent_candidate_for_review
    )

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.bus = _BusStub()
        self.state = SimpleNamespace(pending_proposals={})

    async def _record_observation(self, *_a: Any, **_k: Any) -> None:
        return None


def _submit(coord: _ReviewCoord, cand: dict[str, Any]) -> None:
    asyncio.run(
        Coordinator._submit_framework_agent_candidate_for_review(coord, cand)  # type: ignore[arg-type]
    )


def test_repeated_review_aborts_after_cap(tmp_path: Path):
    coord = _ReviewCoord(tmp_path)
    cand = {"candidate_id": "cid-loop", "batch_id": "b1"}
    cap = coord._MAX_REPEATED_REVIEW_SUBMISSIONS
    # First ``cap`` submissions proceed (each drains its pending before the next).
    for _ in range(cap):
        coord.state.pending_proposals = {}
        _submit(coord, cand)
    assert coord.shared_state.framework_agent_review_counts["cid-loop"] == cap
    # No terminal row yet — still within budget.
    assert coord.shared_state.framework_agent_phase_progress == []
    # The (cap+1)-th submission trips the backstop: terminal row, no proposal.
    coord.state.pending_proposals = {}
    proposals_before = len(coord.bus.messages)
    _submit(coord, cand)
    rows = coord.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["status"] == "repeated_review_abort"
    assert rows[0]["candidate_id"] == "cid-loop"
    # And the aborted submission emitted no new proposal message.
    assert len(coord.bus.messages) == proposals_before
    # Once stamped, the candidate is "processed" → never re-selected.
    coord.shared_state.framework_agent_batches = [
        {"batch_id": "b1", "candidates": [cand]},
    ]
    assert coord._select_next_framework_agent_candidate() is None
