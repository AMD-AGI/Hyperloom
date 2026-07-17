# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover the retry fix — ``fa phase-discover`` retries before flipping ``framework_agent_phase_done``.

Tests ``_discover_next_framework_batch`` (bumps the failure counter, resets
on success) and ``_pump_framework_agent_phase`` (flips done after the retry limit
or a clean empty payload) by binding them to a minimal Coordinator stub.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.loop.coordinator import Coordinator


class _StateStub:
    """SharedState minimal stub for discover retry tests."""

    def __init__(self) -> None:
        self.phase = "FRAMEWORK"
        self.framework_agent_phase_done = False
        self.framework_agent_discover_failures = 0
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.model = "test-model"
        self.framework = "sglang"
        self.gpu_type = "MI300X"
        self.model_class = "dense"
        self.precision = "fp8"
        self.framework_max_candidates = 0
        self.last_profile_kernel_breakdown = None
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1


class _CoordinatorStub:
    """Minimal stub to bind the Coordinator's discover methods to.

    Pins discovery to a single repo so one discover call == one phase_discover
    call (the per-batch failure-counter semantics under test).
    """

    _unprocessed_framework_agent_candidates = Coordinator._unprocessed_framework_agent_candidates
    _framework_candidate_key = staticmethod(Coordinator._framework_candidate_key)
    _framework_processed_candidate_keys = Coordinator._framework_processed_candidate_keys
    _stamp_framework_progress = Coordinator._stamp_framework_progress
    # Reverse-lookup called on every repo; here it resolves to the session
    # framework, so nothing is tagged (same-framework path).
    _framework_agent_repo_url_origin_framework = staticmethod(
        Coordinator._framework_agent_repo_url_origin_framework
    )

    def __init__(self, tmp_path: Path) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub()
        self.framework_agent_discover_timeout_sec = 0.0

    def _framework_agent_discover_repo_urls(self, framework: str) -> list[str]:
        return ["https://github.com/sgl-project/sglang.git"]

    def _framework_known_candidate_ids(self) -> set[str]:
        return Coordinator._framework_known_candidate_ids(self)  # type: ignore[arg-type]

    def _framework_tried_refs(self) -> list[str]:
        return Coordinator._framework_tried_refs(self)  # type: ignore[arg-type]


async def _call_discover(stub: _CoordinatorStub) -> bool:
    return await Coordinator._discover_next_framework_batch(stub)  # type: ignore[arg-type]


def test_discover_failure_bumps_counter_without_flipping_phase_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A single failure increments the counter, leaves phase_done False, and logs a ``framework_agent_discover_failed`` row."""

    async def _raise(**_: Any) -> dict[str, Any]:
        raise RuntimeError("simulated timeout")

    monkeypatch.setattr(_fa_client, "phase_discover", _raise)
    stub = _CoordinatorStub(tmp_path)

    out = asyncio.run(_call_discover(stub))

    assert out is False
    assert stub.shared_state.framework_agent_discover_failures == 1
    assert stub.shared_state.framework_agent_phase_done is False
    failed = [r for r in stub.shared_state.phase_history if r.get("event") == "framework_agent_discover_failed"]
    assert len(failed) == 1
    assert failed[0]["attempt"] == 1
    assert failed[0]["limit"] == _fa_client.DISCOVER_FAILURE_RETRY_LIMIT


def test_discover_three_consecutive_failures_reach_retry_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """After ``DISCOVER_FAILURE_RETRY_LIMIT`` failures the counter reflects the cap; discover never flips phase_done."""

    async def _raise(**_: Any) -> dict[str, Any]:
        raise RuntimeError("simulated")

    monkeypatch.setattr(_fa_client, "phase_discover", _raise)
    stub = _CoordinatorStub(tmp_path)

    for _ in range(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT):
        ok = asyncio.run(_call_discover(stub))
        assert ok is False

    assert stub.shared_state.framework_agent_discover_failures == _fa_client.DISCOVER_FAILURE_RETRY_LIMIT
    assert stub.shared_state.framework_agent_phase_done is False


def test_discover_success_resets_failure_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A success after N failures resets the counter to 0."""

    call_count = SimpleNamespace(n=0)

    async def _flaky(**_: Any) -> dict[str, Any]:
        call_count.n += 1
        if call_count.n <= 2:
            raise RuntimeError("flaky")
        return {
            "batch_id": "b-after-recovery",
            "candidates": [
                {"pr_url": "https://example.com/pr/1"},
            ],
        }

    monkeypatch.setattr(_fa_client, "phase_discover", _flaky)
    stub = _CoordinatorStub(tmp_path)

    assert asyncio.run(_call_discover(stub)) is False
    assert asyncio.run(_call_discover(stub)) is False
    assert stub.shared_state.framework_agent_discover_failures == 2

    assert asyncio.run(_call_discover(stub)) is True
    assert stub.shared_state.framework_agent_discover_failures == 0
    assert len(stub.shared_state.framework_agent_batches) == 1


def test_discover_timeout_override_is_passed_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"batch_id": "b", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)
    stub.framework_agent_discover_timeout_sec = 42.0

    asyncio.run(_call_discover(stub))

    assert captured["timeout_sec"] == 42.0


def test_discover_timeout_default_used_when_override_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def _spy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"batch_id": "b", "candidates": []}

    monkeypatch.setattr(_fa_client, "phase_discover", _spy)
    stub = _CoordinatorStub(tmp_path)
    stub.framework_agent_discover_timeout_sec = 0.0

    asyncio.run(_call_discover(stub))

    assert captured["timeout_sec"] == _fa_client.DEFAULT_FA_PHASE_TIMEOUT_SEC
    assert _fa_client.DEFAULT_FA_PHASE_TIMEOUT_SEC == 180.0


# Enqueue failure records progress row.
class _TasksStub:
    """Mimics ``Coordinator.tasks.create_or_return_existing``; raises to simulate an enqueue failure."""

    def __init__(self, *, fail: bool = True) -> None:
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    async def create_or_return_existing(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError("simulated registry failure")
        return SimpleNamespace(task_id="t-ok")


async def _call_enqueue(stub: _CoordinatorStub, cand: dict[str, Any]) -> None:
    await Coordinator._enqueue_framework_agent_task(stub, cand)  # type: ignore[arg-type]


def test_enqueue_failure_appends_progress_row(tmp_path: Path):
    """Regression: a registry failure records an ``enqueue_failed`` progress row so the next tick skips the candidate."""
    stub = _CoordinatorStub(tmp_path)
    stub.tasks = _TasksStub(fail=True)  # type: ignore[attr-defined]
    cand = {
        "candidate_id": "pr-1",
        "batch_id": "b-fail",
        "pr_url": "https://example.com/pr/1",
    }

    asyncio.run(_call_enqueue(stub, cand))

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "pr-1"
    assert rows[0]["batch_id"] == "b-fail"
    assert rows[0]["status"] == "enqueue_failed"
    assert "simulated registry failure" in rows[0]["error"]


def test_enqueue_failed_candidate_skipped_by_selector(tmp_path: Path):
    """After an enqueue_failed row, the next selector pass must NOT return that candidate."""
    stub = _CoordinatorStub(tmp_path)
    stub.tasks = _TasksStub(fail=True)  # type: ignore[attr-defined]
    cand_bad = {"candidate_id": "pr-bad", "batch_id": "b1"}
    cand_good = {"candidate_id": "pr-good", "batch_id": "b1"}
    stub.shared_state.framework_agent_batches = [
        {
            "batch_id": "b1",
            "candidates": [cand_bad, cand_good],
        },
    ]

    asyncio.run(_call_enqueue(stub, cand_bad))

    nxt = Coordinator._select_next_framework_agent_candidate(stub)  # type: ignore[arg-type]
    assert nxt is not None
    assert nxt["candidate_id"] == "pr-good"


def test_enqueue_success_does_not_append_progress_row(tmp_path: Path):
    """Belt-and-braces: the success path must NOT write any progress row."""
    stub = _CoordinatorStub(tmp_path)
    stub.tasks = _TasksStub(fail=False)  # type: ignore[attr-defined]
    cand = {"candidate_id": "pr-ok", "batch_id": "b1"}

    asyncio.run(_call_enqueue(stub, cand))

    assert stub.shared_state.framework_agent_phase_progress == []


# Gap 4 — phase_history summary row when the pump gives up on discover.
def test_record_framework_agent_phase_done_appends_history_row(tmp_path: Path):
    """The helper appends a phase_history summary row so the give-up decision is visible."""
    stub = _CoordinatorStub(tmp_path)
    stub.shared_state.framework_agent_batches = [
        {"batch_id": "b1", "candidates": []},
        {"batch_id": "b2", "candidates": []},
    ]

    Coordinator._record_framework_agent_phase_done(  # type: ignore[arg-type]
        stub,
        reason="discover_retries_exhausted",
        failure_count=3,
    )

    rows = [r for r in stub.shared_state.phase_history if r.get("event") == "framework_agent_phase_done"]
    assert len(rows) == 1
    assert rows[0]["reason"] == "discover_retries_exhausted"
    assert rows[0]["failure_count"] == 3
    assert rows[0]["retry_limit"] == _fa_client.DISCOVER_FAILURE_RETRY_LIMIT
    assert rows[0]["batches_discovered"] == 2
    assert "ts" in rows[0]


def test_record_framework_agent_phase_done_records_empty_payload_reason(
    tmp_path: Path,
):
    """The helper records ``discover_empty_payload`` when discover returned a clean empty payload."""
    stub = _CoordinatorStub(tmp_path)

    Coordinator._record_framework_agent_phase_done(  # type: ignore[arg-type]
        stub,
        reason="discover_empty_payload",
        failure_count=0,
    )

    rows = [r for r in stub.shared_state.phase_history if r.get("event") == "framework_agent_phase_done"]
    assert len(rows) == 1
    assert rows[0]["reason"] == "discover_empty_payload"
    assert rows[0]["failure_count"] == 0
    assert rows[0]["batches_discovered"] == 0
