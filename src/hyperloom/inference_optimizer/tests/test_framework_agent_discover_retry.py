# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover the source arm's enqueue and give-up bookkeeping.

Tests ``_enqueue_framework_agent_task`` and
``_record_framework_agent_phase_done`` (the give-up summary row, whether the
reason is the retry limit or a clean empty payload) by binding them to a
minimal Coordinator stub.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any


from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.framework import client as _fa_client
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator


_ACTION_REGISTRY = ACTION_CATALOGUE


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
        self.last_profile_kernel_breakdown = None
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1

    def append_phase_history_event(self, **kwargs: Any) -> dict[str, Any]:
        from hyperloom.orchestrator.phases import machine_state as _ms

        return _ms.append_phase_history_event(self, **kwargs)


def _phase_history_event_rows(history: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    from hyperloom.orchestrator.phases.machine_state import phase_history_event_name

    rows: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        if phase_history_event_name(row) == event:
            rows.append(row)
    return rows


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
    # Real lane/TTL resolution, so the enqueue tests exercise the production
    # registry lookup instead of a stub that silently yields no lanes.
    _registry_lanes_ttl = DispatcherCollaborator._registry_lanes_ttl

    def __init__(self, tmp_path: Path) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub()
        self.action_registry = _ACTION_REGISTRY
        self.framework_agent_discover_timeout_sec = 0.0

    def _framework_agent_discover_repo_urls(self, framework: str) -> list[str]:
        return ["https://github.com/sgl-project/sglang.git"]

    def _framework_known_candidate_ids(self) -> set[str]:
        return Coordinator._framework_known_candidate_ids(self)  # type: ignore[arg-type]

    def _framework_tried_refs(self) -> list[str]:
        return Coordinator._framework_tried_refs(self)  # type: ignore[arg-type]


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


def test_enqueue_sources_lanes_and_lease_ttl_from_the_action_registry(tmp_path: Path):
    """Regression: the pump enqueue must inherit ``framework_agent`` lanes and lease TTL from the registry.

    Hardcoding the lanes and omitting ``lease_ttl_sec`` left the task row at 0,
    which the dispatcher turned into a 60s lane lease for a 12-minute action and
    which also made the watchdog skip reclamation of orphaned tasks.
    """
    stub = _CoordinatorStub(tmp_path)
    stub.tasks = _TasksStub(fail=False)  # type: ignore[attr-defined]
    meta = _ACTION_REGISTRY.get("integrate_patch")
    assert meta is not None

    asyncio.run(_call_enqueue(stub, {"candidate_id": "pr-ttl", "batch_id": "b1"}))

    kwargs = stub.tasks.calls[-1]  # type: ignore[attr-defined]
    assert kwargs["kind"] == "integrate_patch"
    assert kwargs["lease_ttl_sec"] == meta.lease_ttl_sec
    assert kwargs["lease_ttl_sec"] > 0
    assert kwargs["requires_lanes"] == list(meta.requires_lanes)
    assert kwargs["requires_lanes"]


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

    rows = _phase_history_event_rows(stub.shared_state.phase_history, "framework_agent_phase_done")
    assert len(rows) == 1
    assert rows[0]["reason"] == "discover_retries_exhausted"
    assert rows[0]["evidence"]["failure_count"] == 3
    assert rows[0]["evidence"]["retry_limit"] == _fa_client.DISCOVER_FAILURE_RETRY_LIMIT
    assert rows[0]["evidence"]["batches_discovered"] == 2
    assert "ts" in rows[0]
    assert rows[0]["ts_unix"] > 0
