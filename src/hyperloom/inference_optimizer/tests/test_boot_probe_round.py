# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cheap half of a bring-up round, played without a GPU.

What is asserted here is the separation the boot probe exists to make: the
verdict it produces is about whether a server came up, it is recorded as the
same boot observation a baseline records, and it carries no throughput at all.
Then the elision: a round that already watched a boot reach a serving server
learns nothing from booting one again, and must not.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from hyperloom.common.bringup import LadderStage
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.actions.executors import boot_probe as bp
from hyperloom.orchestrator.bringup import env_preflight as ep
from hyperloom.orchestrator.bringup import load_boot_observation
from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager, SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema
from hyperloom.orchestrator.enablement.lane import EnablementLane
from hyperloom.orchestrator.rehearsal import LaunchScenario, ScriptedLaunchBackend
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound
from hyperloom.orchestrator.state.round_store import RoundStore
from hyperloom.orchestrator.state.task_registry import TaskRegistry

#: One attempt that comes up and serves.
_SERVES = {"name": "serves", "attempts": [{"name": "clean-boot", "outcome": "ready", "duration_sec": 90.0}]}

#: One attempt that dies where a capability gap dies.
_STOPS = {
    "name": "stops-at-engine-init",
    "attempts": [{"name": "wall", "outcome": "stage_failed", "stage": "engine_init", "duration_sec": 120.0}],
}


@pytest.fixture
def probe_slot(tmp_path, monkeypatch):
    """A session root, a round slot, a config, and a host that is not at fault.

    The environment preflight is answered rather than run: it probes the
    interpreter that would serve, and this round serves nothing.
    """
    session = tmp_path / "session"
    session.mkdir()
    slot = tmp_path / "round"
    slot.mkdir()
    config = session / "bench.yaml"
    config.write_text("benchmark:\n  framework: vllm\n  model: scripted/Model\n", encoding="utf-8")
    monkeypatch.setattr(bp, "check_environment", lambda **_kwargs: ep.EnvVerdict(status=ep.OK))
    return session, slot, config


def _ctx(session: Path, slot: Path, config: Path, task_id: str):
    """Return the runner context one boot probe runs under."""
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
    from hyperloom.orchestrator.state.task_registry import Task

    return RunnerContext(
        task=Task(
            task_id=task_id,
            kind="boot_probe",
            state="running",
            params={"framework": "vllm", "config_path": str(config), "output_dir": str(slot)},
            idempotency_key=task_id,
            requires_lanes=(),
        ),
        lease=None,
        extra={},
    )


@pytest.mark.asyncio
async def test_a_probe_of_a_server_that_serves_reports_a_boot_and_no_throughput(
    probe_slot,
    launch_backend,
    virtual_clock,
):
    """The probe answers the binary question and measures nothing else."""
    session, slot, config = probe_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SERVES), clock=virtual_clock)
    launch_backend(launches)

    result = await bp.BootProbeExecutor(session_dir=session)(_ctx(session, slot, config, "probe-0"))

    assert result["booted"] is True
    assert result["status"] == "succeeded"
    # A boot verdict, not a measurement: nothing here reports a rate.
    assert "output_throughput" not in result
    loaded = load_boot_observation(result["boot_observation_path"])
    assert loaded.observation is not None, loaded.degraded
    assert loaded.observation.booted
    assert launches.served == 1


@pytest.mark.asyncio
async def test_a_probe_of_a_server_that_never_comes_up_places_the_wall(
    probe_slot,
    launch_backend,
    virtual_clock,
):
    """The probe's observation is the classifier's, at the stage the boot stopped."""
    session, slot, config = probe_slot
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_STOPS), clock=virtual_clock)
    launch_backend(launches)

    result = await bp.BootProbeExecutor(session_dir=session)(_ctx(session, slot, config, "probe-0"))

    assert result["booted"] is False
    loaded = load_boot_observation(result["boot_observation_path"])
    assert loaded.observation is not None, loaded.degraded
    assert loaded.observation.stage_failed == LadderStage.ENGINE_INIT
    assert not loaded.observation.booted


@pytest.mark.asyncio
async def test_a_host_that_cannot_run_the_round_is_a_named_outcome_not_a_boot_failure(
    probe_slot,
    launch_backend,
    monkeypatch,
):
    """A fault in the host stops the probe before anything is launched."""
    session, slot, config = probe_slot
    monkeypatch.setattr(
        bp,
        "check_environment",
        lambda **_kwargs: ep.EnvVerdict(
            status=ep.FAULT,
            fault="framework_not_installed",
            detail="the serving interpreter cannot resolve vllm",
        ),
    )
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_SERVES))
    launch_backend(launches)

    result = await bp.BootProbeExecutor(session_dir=session)(_ctx(session, slot, config, "probe-0"))

    assert result["error_class"] == bp.ENV_FAULT
    assert result["booted"] is False
    assert launches.served == 0, "a host that cannot serve was still asked to"
    loaded = load_boot_observation(result["boot_observation_path"])
    assert loaded.observation is not None, loaded.degraded
    assert loaded.observation.env_fault == "framework_not_installed"


def test_the_probe_is_elided_once_the_round_has_an_answer():
    """The probe's ceiling bounds what it can add to what is already recorded."""
    from hyperloom.orchestrator.bringup.ladder import classify
    from hyperloom.orchestrator.rehearsal import boot_log_for

    assert bp.probe_would_inform(None), "nothing on disk knew whether the combo boots"

    # A boot already witnessed at a serving server is as deep as the probe sees.
    served = classify(server_log=boot_log_for(None), server_elapsed_sec=1.0)
    assert served.stage_reached >= bp.PROBE_CEILING
    assert not bp.probe_would_inform(served)

    # A wall already placed is a wall a second boot of the same combo re-finds.
    stopped = classify(server_log=boot_log_for(LadderStage.ENGINE_INIT), server_elapsed_sec=1.0)
    assert stopped.stage_failed == LadderStage.ENGINE_INIT
    assert not bp.probe_would_inform(stopped)


@pytest.fixture
def registry(tmp_path):
    """A real task registry and round store over one database."""
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    yield TaskRegistry(db), RoundStore(db), ResourceLockManager(SqliteLeaseBackend(db))
    db.close()


def _lane(session: Path, tasks: TaskRegistry, rounds: RoundStore, *, observation_path: str):
    """Build the collaborator surface the boot-probe dispatch runs against."""
    state = types.SimpleNamespace(
        framework="vllm",
        enablement_mode="all",
        enablement=EnablementRound(launch_observation_path=observation_path),
        baseline_tput=0.0,
        baseline_failure_streak=1,
        stop_reason="",
        save=lambda *a, **k: None,
    )
    shim = types.SimpleNamespace(
        shared_state=state,
        tasks=tasks,
        rounds=rounds,
        session_dir=str(session),
        action_registry=ACTION_CATALOGUE,
    )
    shim._maybe_enqueue_boot_probe = types.MethodType(EnablementLane._maybe_enqueue_boot_probe, shim)
    return shim


@pytest.mark.asyncio
async def test_the_lane_asks_the_cheap_question_once_and_then_stops_asking(
    probe_slot,
    registry,
    launch_backend,
    virtual_clock,
    monkeypatch,
):
    """Tick one dispatches a probe; tick two has an observation and does not."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env as multi_node

    monkeypatch.setattr(multi_node, "is_multi_node", lambda: False)
    session, slot, config = probe_slot
    tasks, rounds, _locks = registry
    launches = ScriptedLaunchBackend(scenario=LaunchScenario.from_dict(_STOPS), clock=virtual_clock)
    launch_backend(launches)

    first = await _lane(session, tasks, rounds, observation_path="")._maybe_enqueue_boot_probe()
    assert first, "the lane asked nothing while nothing knew whether the combo boots"

    result = await bp.BootProbeExecutor(session_dir=session)(_ctx(session, slot, config, first))
    recorded = result["boot_observation_path"]
    assert recorded

    second = await _lane(session, tasks, rounds, observation_path=recorded)._maybe_enqueue_boot_probe()
    assert second == "", "the lane re-booted a server to re-derive an observation already on disk"
    assert launches.served == 1
