# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-A3 (Arbor-into-Hyperloom): multi-emit concurrent specialist dispatch."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest



# Helpers
async def _build_coord_with_capacity(
    tmp_path: Path, *, capacity: int, gpu_specialist_capacity: int = 0,
):
    """Build a minimal Coordinator with the requested research_lane capacity."""
    from inference_optimizer.orchestrator.agent_role import default_role_registry
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.shared_state import SharedState

    # SharedState pre-write so Coordinator picks the capacity at construction.
    state = SharedState(session_id=f"concurrent-{capacity}")
    state.research_lane_capacity = capacity
    state.gpu_specialist_capacity = gpu_specialist_capacity
    state.save(tmp_path)

    idle_plan = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "kernel":        MockBackend(idle_plan),
        "critic":        MockBackend(idle_plan),
        "robustness":    MockBackend(idle_plan),
    }
    coord = Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=None,
        knowledge_plane=None,
    )
    return coord


# Stub specialist executor: records concurrency timing
class _ConcurrencyProbe:
    """Captures per-task entry/exit times to detect actual parallelism."""

    def __init__(self, sleep_seconds: float = 0.2):
        self.sleep_seconds = sleep_seconds
        self.entries: list[tuple[str, float]] = []
        self.exits: list[tuple[str, float]] = []
        self.gpu_ids_by_task: dict[str, list[int]] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, ctx) -> dict:
        async with self._lock:
            self.entries.append((ctx.task.task_id, time.monotonic()))
            self.gpu_ids_by_task[ctx.task.task_id] = list(
                (ctx.extra or {}).get("gpu_ids") or []
            )
        await asyncio.sleep(self.sleep_seconds)
        async with self._lock:
            self.exits.append((ctx.task.task_id, time.monotonic()))
        return {
            "runner_status": "succeeded",
            "task_id": ctx.task.task_id,
            "domain": "serving_specialist",
            "gap_canonical_id": ctx.task.params.get("gap_canonical_id", ""),
            "specialist_done": {
                "gap_canonical_id": ctx.task.params.get("gap_canonical_id", ""),
                "domain": "serving_specialist",
                "proposal_set": [],
                "empty": True,
                "summary": "concurrency-probe noop",
                "reason": "test",
                "confidence": 0.0,
                "new_findings": [],
                "residual_questions": [],
            },
            "turns_used": 1,
            "workspace": "",
            "transcript_path": "",
            "done_path": "",
            "error": None,
            "notes": [],
        }


def _max_concurrent(entries: list[tuple[str, float]],
                    exits: list[tuple[str, float]]) -> int:
    """Compute peak concurrency from entry/exit timestamps."""
    events: list[tuple[float, int]] = []
    for _, t in entries:
        events.append((t, 1))
    for _, t in exits:
        events.append((t, -1))
    events.sort()
    peak = 0
    cur = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


# Tests
@pytest.mark.asyncio
async def test_dispatcher_runs_four_specialists_concurrently(tmp_path: Path):
    """capacity=4 with 4 queued specialists runs all 4 in one pump (peak concurrency 4)."""
    coord = await _build_coord_with_capacity(tmp_path, capacity=4)
    probe = _ConcurrencyProbe(sleep_seconds=0.3)
    coord.sub.register_executor("specialist", probe)

    for i in range(4):
        await coord.tasks.create_or_return_existing(
            kind="specialist",
            params={
                "domain": "serving_specialist",
                "gap_canonical_id": f"gap.test.{i}",
                "max_turns": 2,
            },
            idempotency_key=f"concurrent-t-{i}",
            requires_lanes=["research_lane"],
        )

    started = time.monotonic()
    await coord._pump_dispatcher_once()
    elapsed = time.monotonic() - started

    # Serial would take ≥1.2s; parallel ~0.3s. Generous budget for CI flakiness.
    assert elapsed < 1.0, (
        f"4 specialists serialised (elapsed={elapsed:.3f}s); expected "
        f"parallel dispatch with elapsed < 1.0s"
    )
    assert len(probe.entries) == 4
    assert len(probe.exits) == 4

    peak = _max_concurrent(probe.entries, probe.exits)
    assert peak == 4, (
        f"expected peak concurrency 4 (capacity=4 with 4 queued), got {peak}"
    )


@pytest.mark.asyncio
async def test_dispatcher_caps_at_capacity_when_more_queued(tmp_path: Path):
    """capacity=2 with 4 queued runs only 2 per pump; the rest stay queued."""
    coord = await _build_coord_with_capacity(tmp_path, capacity=2)
    probe = _ConcurrencyProbe(sleep_seconds=0.3)
    coord.sub.register_executor("specialist", probe)

    for i in range(4):
        await coord.tasks.create_or_return_existing(
            kind="specialist",
            params={
                "domain": "serving_specialist",
                "gap_canonical_id": f"gap.test.{i}",
                "max_turns": 2,
            },
            idempotency_key=f"capped-t-{i}",
            requires_lanes=["research_lane"],
        )

    await coord._pump_dispatcher_once()

    assert len(probe.entries) == 2
    assert len(probe.exits) == 2
    peak = _max_concurrent(probe.entries, probe.exits)
    assert peak == 2, (
        f"expected peak concurrency 2 (capacity=2), got {peak}"
    )


@pytest.mark.asyncio
async def test_dispatcher_capacity_one_falls_back_to_serial(tmp_path: Path):
    """capacity=1 makes the dispatcher behave serially (backward compatibility)."""
    coord = await _build_coord_with_capacity(tmp_path, capacity=1)
    probe = _ConcurrencyProbe(sleep_seconds=0.1)
    coord.sub.register_executor("specialist", probe)

    for i in range(3):
        await coord.tasks.create_or_return_existing(
            kind="specialist",
            params={
                "domain": "serving_specialist",
                "gap_canonical_id": f"gap.test.{i}",
                "max_turns": 2,
            },
            idempotency_key=f"serial-t-{i}",
            requires_lanes=["research_lane"],
        )

    await coord._pump_dispatcher_once()
    assert len(probe.entries) == 1


@pytest.mark.asyncio
async def test_gpu_specialist_pool_limits_dispatch_even_when_research_lane_free(
    tmp_path: Path,
):
    coord = await _build_coord_with_capacity(
        tmp_path, capacity=2, gpu_specialist_capacity=1,
    )
    probe = _ConcurrencyProbe(sleep_seconds=0.1)
    coord.sub.register_executor("specialist", probe)

    for i in range(2):
        await coord.tasks.create_or_return_existing(
            kind="specialist",
            params={
                "domain": "serving_specialist",
                "gap_canonical_id": f"gap.gpu.{i}",
                "max_turns": 2,
                "needs_gpu": True,
                "gpu_count": 1,
            },
            idempotency_key=f"gpu-t-{i}",
            requires_lanes=["research_lane"],
        )

    await coord._pump_dispatcher_once()

    assert len(probe.entries) == 1
    ran_task_id = probe.entries[0][0]
    assert probe.gpu_ids_by_task[ran_task_id] == [0]
    queued = await coord.tasks.queued()
    assert len(queued) == 1
    assert queued[0].params["needs_gpu"] is True


@pytest.mark.asyncio
async def test_gpu_specialist_lease_ttl_covers_subprocess_timeout(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SPECIALIST_PER_TURN_MAX_SECONDS", "10")
    coord = await _build_coord_with_capacity(
        tmp_path, capacity=1, gpu_specialist_capacity=1,
    )
    probe = _ConcurrencyProbe(sleep_seconds=0.01)
    coord.sub.register_executor("specialist", probe)
    captured_ttls: list[int] = []
    original_try_acquire = coord.gpu_specialist_pool.try_acquire

    async def _spy_try_acquire(**kwargs):
        captured_ttls.append(int(kwargs.get("ttl_sec") or 0))
        return await original_try_acquire(**kwargs)

    coord.gpu_specialist_pool.try_acquire = _spy_try_acquire  # type: ignore[method-assign]

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.gpu.ttl",
            "max_turns": 4,
            "needs_gpu": True,
            "gpu_count": 1,
        },
        idempotency_key="gpu-ttl",
        requires_lanes=["research_lane"],
        lease_ttl_sec=5,
    )

    await coord._pump_dispatcher_once()

    assert captured_ttls == [40]
    assert probe.gpu_ids_by_task


# CLI surface check — the default capacity is the research-lane ceiling.
def test_cli_default_research_lane_capacity_is_ceiling(monkeypatch):
    """The default ``--research-lane-capacity`` is the GPU-derived ceiling (2 × visible GPU)."""
    import inference_optimizer.cli as cli_mod
    from inference_optimizer.orchestrator import policy as policy_mod

    monkeypatch.delenv("INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY", raising=False)
    monkeypatch.setattr(policy_mod, "detect_gpu_count", lambda: 4)
    parser = cli_mod._build_parser()
    args = parser.parse_args(["optimize", "--model", "/tmp/dummy"])
    assert args.research_lane_capacity == policy_mod.research_lane_ceiling()
    assert args.research_lane_capacity == 8
    assert args.gpu_specialist_capacity == 0


def test_cli_research_lane_capacity_env_override(monkeypatch):
    """An explicit env value still wins over the GPU-derived default."""
    import inference_optimizer.cli as cli_mod

    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY", "3")
    parser = cli_mod._build_parser()
    args = parser.parse_args(["optimize", "--model", "/tmp/dummy"])
    assert args.research_lane_capacity == 3


def test_cli_clamps_research_lane_capacity_above_ceiling(tmp_path, monkeypatch):
    """An operator value above the GPU-derived ceiling is clamped down in SharedState."""
    import argparse

    from inference_optimizer.cli import _seed_shared_state
    from inference_optimizer.orchestrator import policy as policy_mod

    monkeypatch.setattr(policy_mod, "detect_gpu_count", lambda: 4)

    args = argparse.Namespace(
        research_lane_capacity=32,
        model="/tmp/dummy-model",
        model_class="",
        target_summary="clamp test",
        target_gain=0.0,
        max_hours=0,
    )
    state = _seed_shared_state(
        session_dir=tmp_path,
        args=args,
        session_id="clamp-test",
    )
    assert policy_mod.research_lane_ceiling() == 8
    assert state.research_lane_capacity == 8
