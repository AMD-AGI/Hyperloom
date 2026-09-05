# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fault in the host ends the run instead of buying an authoring round.

The enablement lane exists to repair framework source, and it installs
frameworks and builds extensions on the way. So an import that fails now is not
a fault: the next round may be the one that makes it import. Only a checkpoint
path this host does not hold is beyond every lever the loop owns, and that is
the one condition asserted here to end the run.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.stop_reasons import (
    INFRASTRUCTURE_STOP_REASONS,
    outcome_status,
)
from hyperloom.orchestrator.bringup import ENV_FAULT, env_preflight as ep
from hyperloom.orchestrator.enablement.lane import EnablementLane
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound


def _lane(session: Path, verdict, *, observation_path: str = ""):
    """Build the collaborator surface the terminal is decided on."""
    state = types.SimpleNamespace(
        framework="vllm",
        model_path="/weights/absent",
        enablement=EnablementRound(launch_observation_path=observation_path),
        stop_reason="",
        save=lambda *a, **k: None,
    )
    state.set_stop_reason = lambda value, **_kw: setattr(state, "stop_reason", str(value or ""))
    shim = types.SimpleNamespace(shared_state=state, session_dir=str(session))
    shim._environment_verdict = lambda: verdict
    shim._environment_fault_is_terminal = types.MethodType(EnablementLane._environment_fault_is_terminal, shim)
    return shim


def test_a_host_fault_stops_the_run_before_a_round_is_opened(tmp_path):
    """The lane refuses to author against something no patch could change."""
    lane = _lane(
        tmp_path,
        ep.EnvVerdict(status=ep.FAULT, fault=ep.CHECKPOINT_UNRESOLVED, detail="/weights/absent does not exist"),
    )
    assert lane._environment_fault_is_terminal() is True
    assert lane.shared_state.stop_reason == ENV_FAULT


def test_a_healthy_host_opens_the_round_as_before(tmp_path):
    """Nothing about a host that can serve stops the lane."""
    lane = _lane(tmp_path, ep.EnvVerdict(status=ep.OK))
    assert lane._environment_fault_is_terminal() is False
    assert lane.shared_state.stop_reason == ""


def test_a_check_that_could_not_be_made_never_stops_the_run(tmp_path):
    """An unavailable verdict is not evidence of a healthy host, nor of a broken one."""
    lane = _lane(tmp_path, ep.EnvVerdict(status=ep.UNAVAILABLE, fault=ep.INTERPRETER_UNPROVEN))
    assert lane._environment_fault_is_terminal() is False
    assert lane.shared_state.stop_reason == ""


def test_a_preflight_that_raises_does_not_end_the_run(tmp_path):
    """A broken check must not be able to terminate a session on its own."""
    lane = _lane(tmp_path, None)
    assert lane._environment_fault_is_terminal() is False
    assert lane.shared_state.stop_reason == ""


@pytest.mark.parametrize(
    "fault",
    [ep.FRAMEWORK_NOT_INSTALLED, ep.EXTENSION_MISSING, ep.EXTENSION_UNBUILT, ep.PORT_ALREADY_BOUND],
)
def test_a_condition_a_later_round_could_change_never_ends_the_run(tmp_path, fault):
    """An extension still compiling, or a port an orphan still holds, is not a fault.

    The preflight runs on every tick, including the ticks while a targeted build
    it asked for is still queued. Ending the run on what that build exists to
    fix would make the lane unable to use its own repair.
    """
    lane = _lane(tmp_path, ep.EnvVerdict(status=ep.UNAVAILABLE, fault=fault, detail="undefined symbol"))
    assert lane._environment_fault_is_terminal() is False
    assert lane.shared_state.stop_reason == ""


def test_a_fault_a_round_already_recorded_is_read_back_rather_than_re_derived(tmp_path):
    """The observation the round wrote is the evidence, when there is one."""
    from hyperloom.orchestrator.bringup import write_boot_observation

    observation = ep.env_fault_observation(
        ep.EnvVerdict(status=ep.FAULT, fault=ep.CHECKPOINT_UNRESOLVED, detail="/weights/absent does not exist"),
    )
    slot = tmp_path / "round"
    slot.mkdir()
    path = write_boot_observation(observation, session_dir=tmp_path, output_dir=slot, attempt=0)

    def _never_called():
        raise AssertionError("the recorded observation already answered")

    lane = _lane(tmp_path, None, observation_path=path)
    lane._environment_verdict = _never_called
    assert lane._environment_fault_is_terminal() is True
    assert lane.shared_state.stop_reason == ENV_FAULT


@pytest.mark.parametrize("reason", sorted(INFRASTRUCTURE_STOP_REASONS))
def test_the_terminal_is_reported_as_infrastructure(reason):
    """The run ended without judging the model, and says so."""
    assert outcome_status(reason) != "failed"
