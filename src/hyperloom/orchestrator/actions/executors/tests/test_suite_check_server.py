# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The check server of an AgentX point: its measured launch less the golden layer's acceptance simulation, booted only.

The points are composed the way the baseline composes them (the accuracy round's tests' points): the session base and
its golden layer seeded in place of the recipe's argv, the slot's own layer applied above.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hyperloom.orchestrator.actions.executors import CHECK_SERVER_ENVS
from hyperloom.orchestrator.actions.executors._suite.config import check_server_config, suite_config
from hyperloom.orchestrator.actions.executors._suite.rounds import _server_phase_point
from hyperloom.orchestrator.actions.executors.tests.test_suite_accuracy_round import _agentx_base, _agentx_point
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _launch, _settings
from inference_suite.contracts import ConfigError, GoldenUse, LifecycleMode, WorkloadKind

_TIMEOUT_SEC = 1200.0
_BOOT_SEC = 1800.0


def _point(base, **envs: str) -> dict:
    point = _agentx_point(base)
    bench = point["benchmark"]
    bench["timeout_seconds"] = _TIMEOUT_SEC
    bench["server_lifecycle"] = {"server_ready_timeout_s": _BOOT_SEC}
    bench["envs"] = {**bench["envs"], **envs}
    return point


def _booted(base, **envs: str):
    return suite_config(_point(base, HYPERLOOM_RUN_PHASE="server", **envs), base, _settings(LifecycleMode.BOOT))


def test_the_check_server_is_the_measured_server_with_only_the_golden_use_changed():
    base = _agentx_base()

    measured = _booted(base)
    check = check_server_config(_point(base), base, _settings(LifecycleMode.BOOT))

    assert check.server == replace(measured.server, golden=GoldenUse.REAL_ACCEPTANCE)
    assert (check.workload, check.eval.enabled, check.lifecycle) == (WorkloadKind.NONE, False, measured.lifecycle)
    assert check.deadlines.boot_sec == _BOOT_SEC, "the point's own boot timeout"
    measured_launch, check_launch = _launch(measured, base), _launch(check, base)
    assert check_launch.argv == measured_launch.argv, "the speculative flags stay"
    assert "--speculative-algorithm" in check_launch.argv
    assert "SGLANG_SIMULATE_ACC_LEN" in measured_launch.env and "SGLANG_SIMULATE_ACC_LEN" not in check_launch.env
    assert check_launch.to_document()["golden_use"] == "real_acceptance"


def test_a_point_in_the_check_server_phase_boots_the_check_server():
    base = _agentx_base()

    config = suite_config(_point(base, **CHECK_SERVER_ENVS), base, _settings(LifecycleMode.BOOT))

    assert config == check_server_config(_point(base), base, _settings(LifecycleMode.BOOT))
    assert "HYPERLOOM_RUN_PHASE" not in _launch(config, base).env, "the phase is the config's, not the server's"


def test_a_boot_round_keeps_the_check_server_phase_and_turns_any_other_point_into_a_server_boot():
    base = _agentx_base()

    kept = _server_phase_point(_point(base, **CHECK_SERVER_ENVS))
    plain = _server_phase_point(_point(base))

    assert kept["benchmark"]["envs"]["HYPERLOOM_RUN_PHASE"] == "check_server"
    assert plain["benchmark"]["envs"]["HYPERLOOM_RUN_PHASE"] == "server"


def test_a_base_without_a_golden_layer_has_no_check_server():
    base = _agentx_base(golden=None)

    with pytest.raises(ConfigError, match="no golden layer"):
        check_server_config(_point(base), base, _settings(LifecycleMode.BOOT))


def test_a_point_that_is_not_agentx_has_no_check_server():
    base = _agentx_base()
    point = _point(base)
    point["benchmark"]["agentx"] = {"enabled": False}

    with pytest.raises(ConfigError, match="AgentX point"):
        check_server_config(point, base, _settings(LifecycleMode.BOOT))
