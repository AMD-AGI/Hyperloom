# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A scriptable point (xDiT, or an operator's own script) on inference_suite: the round runs the session base's
script with the point's environment, and every framework's point of a frozen-base session translates to a round."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._suite_config import suite_config
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import (
    _MODEL,
    _MODEL_ID,
    _base,
    _point,
    _settings,
)
from inference_suite import LaunchQuery, scriptable_base
from inference_suite.contracts import (
    BaseLaunch,
    ConfigError,
    LaunchModel,
    Lifecycle,
    LifecycleMode,
    SuiteConfig,
    Topology,
    Workload,
    WorkloadKind,
)
from inference_suite.frameworks import Framework, FrameworkKind, framework_facts


def _script_base(framework: str, script: str | None = None) -> BaseLaunch:
    query = LaunchQuery(_MODEL_ID, framework, "mi355x", Workload.SCRIPTABLE, (1,), tp=2)
    return scriptable_base(query, script, LaunchModel(_MODEL_ID, _MODEL), Topology(tp=2))


def _scriptable_point(framework: str, **envs: str) -> dict:
    return {
        "benchmark": {
            "framework": framework,
            "workload_kind": "scriptable",
            "model": _MODEL,
            "precision": "bf16",
            "runner_type": "mi355x",
            "envs": {
                "TP": "2", "CONC": "1", "ROCR_VISIBLE_DEVICES": "4,5", "MODEL": "/elsewhere", "PATH": "/opt/venv/bin",
                "XDIT_HEIGHT": "768", "HYPERLOOM_QUALITY_REF": "/s/ref.png", "XDIT_QUALITY_REF": "/s/ref.png", **envs,
            },
            "timeout_seconds": 7200,
            "profiler": {"torch_profiler": {"enabled": False}},
        }
    }  # fmt: skip


def test_an_xdit_point_runs_its_base_script_with_the_point_environment_less_the_names_the_round_sets() -> None:
    base = _script_base("xdit")
    config = suite_config(_scriptable_point("xdit"), base, _settings())

    assert config.workload is WorkloadKind.SCRIPTABLE and config.fixed is None
    assert config.scriptable.params == {
        "RUNNER_TYPE": "mi355x",
        "PRECISION": "bf16",
        "CONC": "1",
        "PATH": "/opt/venv/bin",
        "XDIT_HEIGHT": "768",
        "HYPERLOOM_QUALITY_REF": "/s/ref.png",
        "XDIT_QUALITY_REF": "/s/ref.png",
    }
    assert config.scriptable.quality_reference == "/s/ref.png"
    assert config.server.gpus == (4, 5) and config.server.base.digest == base.digest()
    assert not config.eval.enabled and not config.profile.enabled
    config.check_base(base)
    assert SuiteConfig.from_document(json.loads(json.dumps(config.to_document()))) == config


def test_a_custom_point_runs_the_operator_script_its_session_froze(tmp_path: Path) -> None:
    script = tmp_path / "custom_mi355x.sh"
    script.write_text("exit 0\n")
    point = _scriptable_point("custom", HYPERLOOM_QUALITY_REF="", XDIT_QUALITY_REF="", CUSTOM_KNOB="3")

    config = suite_config(point, _script_base("custom", str(script)), _settings())

    assert config.scriptable.params["CUSTOM_KNOB"] == "3" and config.scriptable.quality_reference is None


def test_a_scriptable_profile_point_captures_the_script_run() -> None:
    point = _scriptable_point("xdit", PROFILE="1")
    point["benchmark"]["profiler"]["torch_profiler"]["enabled"] = True

    config = suite_config(point, _script_base("xdit"), _settings())

    assert config.profile.enabled and "PROFILE" not in config.scriptable.params


@pytest.mark.parametrize(
    ("envs", "settings", "message"),
    [
        ({"MAGPIE_RUN_PHASE": "server"}, _settings(), "neither boots one nor reuses one"),
        ({}, _settings(LifecycleMode.REUSE), "neither boots one nor reuses one"),
        ({"BENCHMARK_BASE_URL": "http://head:8888"}, _settings(), "not the external server at http://head:8888"),
        ({}, replace(_settings(), server_url="http://head:8888"), "not the external server at http://head:8888"),
        ({"XDIT_QUALITY_LPIPS_MAX": "0.15"}, _settings(), "is a threshold; the caller judges the result"),
        ({"TP": "4"}, _settings(), "the point runs TP 4, its base runs 2"),
    ],
)
def test_a_scriptable_point_no_round_carries_is_refused(envs, settings, message) -> None:
    with pytest.raises(ConfigError, match=message):
        suite_config(_scriptable_point("xdit", **envs), _script_base("xdit"), settings)


def test_a_scriptable_point_on_another_framework_base_is_refused() -> None:
    with pytest.raises(ConfigError, match="the point runs xdit, its base runs sglang"):
        suite_config(_scriptable_point("xdit"), _base("sglang"), _settings())


def _session_point(framework: Framework, tmp_path: Path) -> tuple[dict, BaseLaunch]:
    if framework_facts(framework).kind is FrameworkKind.SERVING:
        return _point(framework.value), _base(framework.value)
    script = None
    if framework is Framework.CUSTOM:
        script = tmp_path / "custom_mi355x.sh"
        script.write_text("exit 0\n")
    return _scriptable_point(framework.value), _script_base(framework.value, script and str(script))


@pytest.mark.parametrize("framework", list(Framework))
def test_a_point_of_every_framework_runs_on_its_frozen_base(framework, tmp_path):
    point, base = _session_point(framework, tmp_path)

    config = suite_config(point, base, replace(_settings(), lifecycle=Lifecycle(LifecycleMode.OWN)))

    config.check_base(base)
    scriptable = framework_facts(framework).kind is FrameworkKind.SCRIPTABLE
    assert config.workload is (WorkloadKind.SCRIPTABLE if scriptable else WorkloadKind.FIXED)
