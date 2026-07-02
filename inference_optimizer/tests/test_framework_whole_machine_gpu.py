# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Item J (framework-stage2.md §P1.5): framework-family authoring specialists
lease the *whole machine* on demand.

A framework-authoring specialist (perf-framework or enablement) holds the
serving-exclusive, cap-1 ``gpu_research_lane``, so while it runs no
serving/bench/profile step (and no other GPU specialist) runs — making it safe
to lease every visible card. This is distinct from an EXPLORE GPU specialist,
which leases from the serving-disjoint carved ``gpu_specialist_pool``.

The four assertions mirror the doc:

1. framework authoring params (perf & enablement) carry ``needs_gpu`` +
   whole-machine ``gpu_count``;
2. dispatch for the framework family leases the whole machine even when
   ``gpu_specialist_capacity=0``;
3. holding ``gpu_research_lane`` serializes the serving lanes (mutex works);
4. the EXPLORE GPU-specialist path is unchanged (still serving-disjoint carve,
   still gated by ``gpu_specialist_capacity``).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest


def _build_coord(
    tmp_path: Path,
    monkeypatch,
    *,
    gpu_specialist_capacity: int,
    visible_devices: str | None = "0,1,2,3",
):
    """Build a minimal Coordinator with a deterministic GPU env.

    ``visible_devices`` is written into ``ROCR_VISIBLE_DEVICES`` *before*
    construction so both the carved ``gpu_specialist_pool`` and the
    whole-machine ``framework_gpu_pool`` resolve deterministically (the pools
    are baked at construction). ``TP`` is cleared so the serving carve is 0.
    """
    from inference_optimizer.orchestrator.agent_role import default_role_registry
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend,
        MockTurn,
        ScriptedPlan,
    )
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.shared_state import SharedState

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
    # research_lane headroom so a GPU task is never blocked by the LLM lane.
    state.research_lane_capacity = 4
    state.save(tmp_path)

    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        name: MockBackend(idle)
        for name in ("orchestration", "kernel_agent", "critic", "robustness")
    }
    return Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=None,
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
        self.gpu_ids_by_task[ctx.task.task_id] = list(
            (ctx.extra or {}).get("gpu_ids") or []
        )
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
    coord = _build_coord(
        tmp_path, monkeypatch, gpu_specialist_capacity=0, visible_devices=""
    )
    assert coord.framework_gpu_pool.capacity == 0
    assert coord._framework_gpu_params() == {}


def test_framework_gpu_params_empty_on_multi_node(tmp_path, monkeypatch):
    """Multi-node → no whole-machine GPU request (integrate_patch is single-node)."""
    coord = _build_coord(tmp_path, monkeypatch, gpu_specialist_capacity=0)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    assert coord._framework_gpu_params() == {}


# ── 2. dispatch leases the whole machine even when capacity=0 ────────────────


@pytest.mark.asyncio
async def test_framework_family_leases_whole_machine_when_capacity_zero(
    tmp_path, monkeypatch
):
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
async def test_framework_family_defaults_gpu_count_to_whole_machine(
    tmp_path, monkeypatch
):
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
    (benchmark / profile / server_lifecycle) cannot be acquired — the exact
    exclusivity item J relies on to safely hand over the whole machine."""
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
            assert blocked is None, (
                f"{serving_lane} was acquirable while gpu_research_lane held"
            )
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
    from the carved pool (gpu_count=1 → a single card), unchanged by item J."""
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
