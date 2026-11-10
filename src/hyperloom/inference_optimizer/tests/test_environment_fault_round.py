# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fault in the host ends the run instead of buying an authoring round."""

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


def _lane(session: Path, verdict):
    """Build the collaborator surface the terminal is decided on."""
    state = types.SimpleNamespace(
        framework="vllm",
        model_path="/weights/absent",
        enablement=EnablementRound(),
        stop_reason="",
        save=lambda *a, **k: None,
    )
    state.set_stop_reason = lambda value, **_kw: setattr(state, "stop_reason", str(value or ""))
    shim = types.SimpleNamespace(shared_state=state, session_dir=str(session))
    shim._environment_verdict = lambda: verdict
    shim._check_environment_terminal = types.MethodType(EnablementLane._check_environment_terminal, shim)
    return shim


def test_a_host_fault_stops_the_run_before_a_round_is_opened(tmp_path):
    """The lane refuses to author against something no patch could change."""
    lane = _lane(
        tmp_path,
        ep.EnvVerdict(status=ep.FAULT, fault=ep.CHECKPOINT_UNRESOLVED, detail="/weights/absent does not exist"),
    )
    assert lane._check_environment_terminal() is True
    assert lane.shared_state.stop_reason == ENV_FAULT


def test_a_healthy_host_opens_the_round_as_before(tmp_path):
    """Nothing about a host that can serve stops the lane."""
    lane = _lane(tmp_path, ep.EnvVerdict(status=ep.OK))
    assert lane._check_environment_terminal() is False
    assert lane.shared_state.stop_reason == ""


def test_a_preflight_that_raises_does_not_end_the_run(tmp_path):
    """A broken check must not be able to terminate a session on its own."""
    lane = _lane(tmp_path, None)
    assert lane._check_environment_terminal() is False
    assert lane.shared_state.stop_reason == ""


@pytest.mark.parametrize("reason", sorted(INFRASTRUCTURE_STOP_REASONS))
def test_the_terminal_is_reported_as_infrastructure(reason):
    """The run ended without judging the model, and says so."""
    assert outcome_status(reason, 1.0) != "failed"
