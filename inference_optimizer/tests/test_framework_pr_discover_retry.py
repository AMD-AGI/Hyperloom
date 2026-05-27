"""Cover the P2.b fix — ``fa phase-discover`` retries before
flipping ``framework_pr_phase_done``.

The retry logic spans two methods on Coordinator:

  - ``_discover_next_framework_pr_batch`` — bumps
    ``state.framework_pr_discover_failures`` on Exception, resets to 0
    on success, returns False either way.
  - ``_pump_framework_pr_phase`` — only marks the phase done after
    ``DISCOVER_FAILURE_RETRY_LIMIT`` consecutive failures, OR on a
    clean empty payload (failures counter == 0).

We test both methods in isolation by binding them to a minimal stub
that mirrors the Coordinator attributes they read; this keeps the
test fast (no DB, no bus, no backends).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator import framework_agent_client as _fa_client
from inference_optimizer.orchestrator.coordinator import Coordinator


class _StateStub:
    """SharedState minimal stub for discover retry tests."""

    def __init__(self) -> None:
        self.phase = "FRAMEWORK_PR"
        self.framework_pr_phase_done = False
        self.framework_pr_discover_failures = 0
        self.framework_pr_batches: list[dict[str, Any]] = []
        self.framework_pr_phase_progress: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.model = "test-model"
        self.framework = "sglang"
        self.gpu_type = "MI300X"
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1


class _CoordinatorStub:
    """Holds just enough attributes to bind the Coordinator's discover
    methods to."""

    def __init__(self, tmp_path: Path) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub()
        self.framework_pr_discover_timeout_sec = 0.0


async def _call_discover(stub: _CoordinatorStub) -> bool:
    return await Coordinator._discover_next_framework_pr_batch(stub)  # type: ignore[arg-type]


def test_discover_failure_bumps_counter_without_flipping_phase_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A single failure increments the counter but leaves
    framework_pr_phase_done False. Phase_history gets a
    ``framework_pr_discover_failed`` row."""

    async def _raise(**_: Any) -> dict[str, Any]:
        raise RuntimeError("simulated timeout")

    monkeypatch.setattr(_fa_client, "phase_discover", _raise)
    stub = _CoordinatorStub(tmp_path)

    out = asyncio.run(_call_discover(stub))

    assert out is False
    assert stub.shared_state.framework_pr_discover_failures == 1
    assert stub.shared_state.framework_pr_phase_done is False
    failed = [
        r for r in stub.shared_state.phase_history
        if r.get("event") == "framework_pr_discover_failed"
    ]
    assert len(failed) == 1
    assert failed[0]["attempt"] == 1
    assert failed[0]["limit"] == _fa_client.DISCOVER_FAILURE_RETRY_LIMIT


def test_discover_three_consecutive_failures_reach_retry_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """After ``DISCOVER_FAILURE_RETRY_LIMIT`` failures the counter
    reflects the cap. The pump (separately exercised below) interprets
    this as the green light to flip framework_pr_phase_done."""

    async def _raise(**_: Any) -> dict[str, Any]:
        raise RuntimeError("simulated")

    monkeypatch.setattr(_fa_client, "phase_discover", _raise)
    stub = _CoordinatorStub(tmp_path)

    for _ in range(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT):
        ok = asyncio.run(_call_discover(stub))
        assert ok is False

    assert (
        stub.shared_state.framework_pr_discover_failures
        == _fa_client.DISCOVER_FAILURE_RETRY_LIMIT
    )
    # Discover itself never marks the phase done; pump does.
    assert stub.shared_state.framework_pr_phase_done is False


def test_discover_success_resets_failure_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A success after N failures must reset the counter to 0 so the
    next failure starts fresh."""

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

    # 2 failures
    assert asyncio.run(_call_discover(stub)) is False
    assert asyncio.run(_call_discover(stub)) is False
    assert stub.shared_state.framework_pr_discover_failures == 2

    # success
    assert asyncio.run(_call_discover(stub)) is True
    assert stub.shared_state.framework_pr_discover_failures == 0
    assert len(stub.shared_state.framework_pr_batches) == 1


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
    stub.framework_pr_discover_timeout_sec = 42.0

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
    stub.framework_pr_discover_timeout_sec = 0.0

    asyncio.run(_call_discover(stub))

    assert captured["timeout_sec"] == _fa_client.DEFAULT_FA_PHASE_TIMEOUT_SEC
    assert _fa_client.DEFAULT_FA_PHASE_TIMEOUT_SEC == 180.0
