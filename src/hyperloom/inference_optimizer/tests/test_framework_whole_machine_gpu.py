# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that framework-family authoring specialists lease the whole machine.

A framework-authoring specialist (perf-framework or enablement) holds the
serving-exclusive, cap-1 ``gpu_research_lane`` and leases every visible card,
distinct from an EXPLORE GPU specialist, which leases from the carved
``gpu_specialist_pool``.

The suite checks:

1. framework authoring params (perf & enablement) carry ``needs_gpu`` +
   whole-machine ``gpu_count``;
2. dispatch for the framework family leases the whole machine even when
   ``gpu_specialist_capacity=0``;
3. holding ``gpu_research_lane`` serializes the serving lanes (mutex);
4. the EXPLORE GPU-specialist path is unchanged (carved pool, still gated by
   ``gpu_specialist_capacity``);
5. bench-capable EXPLORE specialists take the whole-machine time-shared lane,
   while non-bench GPU probes keep the carved pool;
6. serving-priority defers a GPU specialist (stays queued) and releases its
   ``gpu_research_lane`` lease.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def _build_coord(
    tmp_path: Path,
    monkeypatch,
    *,
    gpu_specialist_capacity: int,
    visible_devices: str | None = "0,1,2,3",
    tp: int = 0,
):
    """Build a minimal Coordinator with a deterministic GPU env.

    ``visible_devices`` is written into ``ROCR_VISIBLE_DEVICES`` *before*
    construction so both the carved ``gpu_specialist_pool`` and the
    whole-machine ``framework_gpu_pool`` resolve deterministically (the pools
    are baked at construction). ``tp`` sets the serving TP carve (0 = no carve).
    """
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import (
        MockBackend,
        MockTurn,
        ScriptedPlan,
    )
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.state.shared_state import SharedState

    for _var in (
        "TP",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES",
        "INFERENCE_OPTIMIZER_NODES",
    ):
        monkeypatch.delenv(_var, raising=False)
    if visible_devices is None:
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", visible_devices)

    state = SharedState(session_id="framework-gpu")
    state.gpu_specialist_capacity = gpu_specialist_capacity
    state.tp = tp
    # research_lane headroom so a GPU task is not blocked by the LLM lane.
    state.research_lane_capacity = 4
    state.save(tmp_path)

    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {name: MockBackend(idle) for name in ("orchestration", "critic", "robustness")}
    return Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )


class _GpuProbe:
    """Records the ``gpu_ids`` injected into each dispatched task's context."""

    def __init__(self, sleep_seconds: float = 0.05):
        self.sleep_seconds = sleep_seconds
        self.gpu_ids_by_task: dict[str, list[int]] = {}
        self.entries: list[str] = []

    async def __call__(self, ctx) -> dict:
        self.entries.append(ctx.task.task_id)
        self.gpu_ids_by_task[ctx.task.task_id] = list((ctx.extra or {}).get("gpu_ids") or [])
        await asyncio.sleep(self.sleep_seconds)
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
                "summary": "gpu-probe noop",
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


# ── 1. params carry needs_gpu + whole-machine gpu_count (perf & enablement) ──


def test_framework_gpu_params_request_whole_machine(tmp_path, monkeypatch):
    """The shared helper (used by BOTH the perf-framework and enablement param
    builders) requests the whole machine when GPUs are visible + single-node."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0)
    assert coord.framework_gpu_pool.capacity == 4
    gpu_params = coord._framework_gpu_params()
    assert coord._coerce_needs_gpu(gpu_params.get("needs_gpu")) is True
    assert gpu_params.get("gpu_count") == 4


def test_enablement_params_carry_whole_machine_gpu(tmp_path, monkeypatch):
    """The enablement param builder merges needs_gpu + whole-machine gpu_count."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0)
    coord.shared_state.framework = "sglang"
    coord.shared_state.model_name = "some/model"
    # A missing-model-arch log classifies to an actionable signature.
    log = "Model architecture 'FooBarForCausalLM' is not supported by this build"
    params = coord._build_enablement_specialist_params(log)
    assert params is not None
    assert params.get("framework_agent_authoring") is True
    assert coord._coerce_needs_gpu(params.get("needs_gpu")) is True
    assert params.get("gpu_count") == 4


def test_framework_gpu_params_empty_without_gpus(tmp_path, monkeypatch):
    """No visible cards → no needs_gpu (never deadlock the dispatcher)."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0, visible_devices="")
    assert coord.framework_gpu_pool.capacity == 0
    assert coord._framework_gpu_params() == {}


def test_framework_gpu_params_empty_on_multi_node(tmp_path, monkeypatch):
    """Multi-node → no whole-machine GPU request (integrate_patch is single-node)."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    assert coord._framework_gpu_params() == {}


# ── 2. dispatch leases the whole machine even when capacity=0 ────────────────


@pytest.mark.asyncio
async def test_framework_family_leases_whole_machine_when_capacity_zero(tmp_path, monkeypatch):
    """A framework-family GPU task leases every card from ``framework_gpu_pool``
    even though ``gpu_specialist_capacity=0`` empties the EXPLORE pool."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0)
    assert coord.gpu_specialist_pool.capacity == 0  # EXPLORE pool empty
    assert coord.framework_gpu_pool.capacity == 4  # whole machine
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "domain": "enablement_specialist",
            "gap_canonical_id": "gap.enablement.test",
            "framework_agent_authoring": True,
            "enablement": True,
            "needs_gpu": True,
            "gpu_count": 4,
        },
        idempotency_key="fw-gpu-wholemachine",
        requires_lanes=["research_lane", "gpu_research_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    assert probe.entries, "framework GPU task never dispatched"
    tid = probe.entries[0]
    assert probe.gpu_ids_by_task[tid] == [0, 1, 2, 3]
    assert not await coord.tasks.queued()


@pytest.mark.asyncio
async def test_framework_family_defaults_gpu_count_to_whole_machine(tmp_path, monkeypatch):
    """Omitting ``gpu_count`` defaults a framework-family task to the whole
    machine (not the serving TP)."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0)
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.framework.test",
            "framework_agent_authoring": True,
            "needs_gpu": True,
            # no explicit gpu_count → default to whole-machine capacity
        },
        idempotency_key="fw-gpu-default-count",
        requires_lanes=["research_lane", "gpu_research_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    assert probe.entries
    assert probe.gpu_ids_by_task[probe.entries[0]] == [0, 1, 2, 3]


# ── 3. gpu_research_lane serializes the serving lanes (mutex) ────────────────


@pytest.mark.asyncio
async def test_gpu_research_lane_mutexes_serving_lanes(tmp_path, monkeypatch):
    """While a framework GPU task holds gpu_research_lane, the serving lanes
    (benchmark / profile / server_lifecycle) cannot be acquired."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0)
    held = await coord.locks.try_acquire_many(
        ["gpu_research_lane"],
        holder_id="framework-holder",
        task_id="framework-holder",
        action="specialist",
        ttl_sec=3600,
    )
    assert held is not None
    try:
        for serving_lane in ("benchmark_lane", "profile_lane", "server_lifecycle"):
            blocked = await coord.locks.try_acquire_many(
                [serving_lane],
                holder_id="serving-holder",
                task_id="serving-holder",
                action="baseline",
                ttl_sec=60,
            )
            assert blocked is None, f"{serving_lane} was acquirable while gpu_research_lane held"
        # A second GPU task is also blocked (cap-1 / strictly serial).
        second = await coord.locks.try_acquire_many(
            ["gpu_research_lane"],
            holder_id="framework-holder-2",
            task_id="framework-holder-2",
            action="specialist",
            ttl_sec=3600,
        )
        assert second is None
    finally:
        await coord.locks.release(held)


# ── 4. EXPLORE GPU-specialist path is unchanged ─────────────────────────────


@pytest.mark.asyncio
async def test_explore_gpu_specialist_still_gated_by_capacity(tmp_path, monkeypatch):
    """A NON-framework needs_gpu specialist still leases from the carved
    ``gpu_specialist_pool`` — so ``gpu_specialist_capacity=0`` leaves it queued
    (the whole-machine special case must NOT leak to EXPLORE)."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0)
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.explore.gpu",
            "needs_gpu": True,
            "gpu_count": 1,
        },
        idempotency_key="explore-gpu-blocked",
        requires_lanes=["research_lane", "gpu_research_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    # Carved pool is empty (capacity=0) → no lease → task stays queued, unrun.
    assert not probe.entries
    still_queued = await coord.tasks.queued()
    assert any(t.kind == "specialist" for t in still_queued)


@pytest.mark.asyncio
async def test_explore_gpu_specialist_uses_carved_pool(tmp_path, monkeypatch):
    """With capacity>0 and no serving carve, an EXPLORE GPU specialist leases
    from the carved pool (gpu_count=1 → a single card)."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=4)
    assert coord.gpu_specialist_pool.capacity == 4
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "domain": "serving_specialist",
            "gap_canonical_id": "gap.explore.gpu2",
            "needs_gpu": True,
            "gpu_count": 1,
        },
        idempotency_key="explore-gpu-carved",
        requires_lanes=["research_lane", "gpu_research_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    assert probe.entries
    assert probe.gpu_ids_by_task[probe.entries[0]] == [0]


# ── 5. EXPLORE bench specialists take the whole-machine time-shared lane ─────


@pytest.mark.asyncio
async def test_bench_specialist_leases_whole_machine_when_serving_owns_node(tmp_path, monkeypatch):
    """A bench-capable EXPLORE specialist (mode=patch & bench=true) leases the
    whole machine from ``framework_gpu_pool`` — so serving occupying the whole
    node (TP == #GPUs, which empties the serving-disjoint pool) does not leave
    it unschedulable."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=4, tp=4)
    # Serving carve empties the disjoint pool; the whole-machine pool is full.
    assert coord.gpu_specialist_pool.capacity == 0
    assert coord.framework_gpu_pool.capacity == 4
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "scope": "freeform",
            "task_description": "start a TP-sharded server and rebench a patch",
            "mode": "patch",
            "bench": True,
            "needs_gpu": True,
            # gpu_count omitted → defaults to serving TP (floored to 4).
        },
        idempotency_key="explore-bench-wholemachine",
        requires_lanes=["research_lane", "gpu_research_lane", "benchmark_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    assert probe.entries, "bench specialist never dispatched"
    tid = probe.entries[0]
    assert probe.gpu_ids_by_task[tid] == [0, 1, 2, 3]
    assert not await coord.tasks.queued()


@pytest.mark.asyncio
async def test_non_bench_gpu_probe_still_uses_carved_pool(tmp_path, monkeypatch):
    """A non-bench GPU probe (bench=false) keeps the serving-disjoint pool — the
    whole-machine route must NOT leak to ordinary microbench/profiling probes."""
    # 8 visible cards, serving TP=4 → carved pool = cards [4..7].
    coord = _build_coord(
        tmp_path,
        monkeypatch,
        gpu_specialist_capacity=8,
        visible_devices="0,1,2,3,4,5,6,7",
        tp=4,
    )
    assert coord.gpu_specialist_pool.capacity == 4
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "scope": "freeform",
            "task_description": "microbench the decode attention kernel",
            "mode": "patch",
            "bench": False,
            "needs_gpu": True,
            "gpu_count": 1,
        },
        idempotency_key="explore-nonbench-carved",
        requires_lanes=["research_lane", "gpu_research_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    assert probe.entries
    # First card of the carved (serving-disjoint) pool, not card 0.
    assert probe.gpu_ids_by_task[probe.entries[0]] == [4]


# ── 6. serving-priority defer: GPU specialist stays queued, lane released ─────


@pytest.mark.asyncio
async def test_serving_priority_defers_gpu_specialist_and_releases_lane(tmp_path, monkeypatch):
    """§3.4 dispatcher E2E: when serving_slot_busy()==True the GPU specialist is
    deferred (stays queued), its executor is never called, and the SQLite
    gpu_research_lane lease is released so it leaves no held holder."""
    import hyperloom.orchestrator.actions.executors._ray_backend as _rb

    # Enable serving-priority and make the slot always appear busy.
    monkeypatch.setattr(_rb, "ray_serving_priority_enabled", lambda: True)
    monkeypatch.setattr(_rb, "serving_slot_busy", lambda: True)

    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=4)
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "scope": "freeform",
            "task_description": "probe a kernel while serving is active",
            "needs_gpu": True,
            "gpu_count": 1,
        },
        idempotency_key="serving-priority-defer",
        requires_lanes=["research_lane", "gpu_research_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    # Executor must NOT have run — the task should still be queued.
    assert not probe.entries, "executor must not run while serving slot is busy"
    still_queued = await coord.tasks.queued()
    assert any(t.idempotency_key == "serving-priority-defer" for t in still_queued), (
        "GPU specialist must remain queued when serving slot is busy"
    )

    # The SQLite lane lease must have been released — no residual holder.
    holders = await coord.locks.lane_holders()
    assert holders.get("gpu_research_lane", 0) == 0, (
        "gpu_research_lane must not remain held after defer"
    )
    assert holders.get("research_lane", 0) == 0, (
        "research_lane must not remain held after defer"
    )


@pytest.mark.asyncio
async def test_serving_priority_defers_on_second_probe_racing_admit(tmp_path, monkeypatch):
    """§3.4 immediate-probe regression: a serving start that races between the
    pass start and the per-task admit must still trigger a defer.

    This test uses a call-count-based side_effect so the first call
    (ray_serving_priority_enabled check) returns True, and the first
    serving_slot_busy() probe (at admit time) returns True — simulating a
    serving start that happened between the pass beginning and admit.
    """
    import hyperloom.orchestrator.actions.executors._ray_backend as _rb

    # serving-priority enabled; the slot was free at the start of the pass
    # but became busy by the time we probe at admit.
    busy_calls: list[bool] = []

    def _slot_busy_racing() -> bool:
        busy_calls.append(True)
        # Returns True on every call — simulates serving starting mid-pass.
        return True

    monkeypatch.setattr(_rb, "ray_serving_priority_enabled", lambda: True)
    monkeypatch.setattr(_rb, "serving_slot_busy", _slot_busy_racing)

    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=4)
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "scope": "freeform",
            "task_description": "probe right as serving started",
            "needs_gpu": True,
            "gpu_count": 1,
        },
        idempotency_key="serving-priority-race",
        requires_lanes=["research_lane", "gpu_research_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    # The per-task probe must have been called at least once.
    assert busy_calls, "serving_slot_busy must be called at admit time"

    # Task must be deferred.
    assert not probe.entries, "executor must not run when slot is busy at admit"
    still_queued = await coord.tasks.queued()
    assert any(t.idempotency_key == "serving-priority-race" for t in still_queued)

    # Lane must be released.
    holders = await coord.locks.lane_holders()
    assert holders.get("gpu_research_lane", 0) == 0
    assert holders.get("research_lane", 0) == 0


@pytest.mark.asyncio
async def test_serving_priority_admits_gpu_specialist_when_slot_free(tmp_path, monkeypatch):
    """§3.4 inverse: when serving_slot_busy()==False the GPU specialist IS
    admitted (executor runs), so the serving-priority gate is not overly
    aggressive."""
    import hyperloom.orchestrator.actions.executors._ray_backend as _rb

    monkeypatch.setattr(_rb, "ray_serving_priority_enabled", lambda: True)
    monkeypatch.setattr(_rb, "serving_slot_busy", lambda: False)

    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=4)
    probe = _GpuProbe()
    coord.sub.register_executor("specialist", probe)

    await coord.tasks.create_or_return_existing(
        kind="specialist",
        params={
            "scope": "freeform",
            "task_description": "probe a kernel while slot is free",
            "needs_gpu": True,
            "gpu_count": 1,
        },
        idempotency_key="serving-priority-free",
        requires_lanes=["research_lane", "gpu_research_lane"],
        lease_ttl_sec=3600,
    )

    await coord._pump_dispatcher_once()

    assert probe.entries, "executor must run when serving slot is free"
    assert not await coord.tasks.queued()
