# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An explore variant Magpie would refuse to launch is dropped before it is enqueued."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.loop._explore_delegate import _keep_launchable_variants

_MODULE = "Magpie.modes.benchmark.agentx_launch"
_GOLDEN_REFUSAL = "Changing AgentX golden acceptance inputs requires a new resolved recipe"
_PROTECTED_REFUSAL = "append_args cannot change model, topology, host or port; profiling is also protected"
_RECIPE_ARGV = [
    "python3",
    "-m",
    "sglang.launch_server",
    "--model-path",
    "/models/glm",
    "--tp",
    "8",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-num-draft-tokens",
    "4",
]


def _fake_agentx_launch() -> types.ModuleType:
    """The subset of Magpie's launch contract these tests exercise, for an environment without Magpie."""

    def apply_launch_args(argv, overrides, framework):
        appended = list(overrides.get("append_args", []))
        if any(token.partition("=")[0] == "--tp" for token in appended):
            raise ValueError(_PROTECTED_REFUSAL)
        return [*argv, *appended]

    def _validate_golden_launch(spec, effective):
        def steps(argv):
            return [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "--speculative-num-steps"]

        if spec.get("golden_acceptance_overrides") and steps(spec["argv"]) != steps(effective):
            raise ValueError(_GOLDEN_REFUSAL)

    module = types.ModuleType(_MODULE)
    module.apply_launch_args = apply_launch_args
    module._validate_golden_launch = _validate_golden_launch
    return module


@pytest.fixture(autouse=True)
def agentx_launch(monkeypatch):
    try:
        return importlib.import_module(_MODULE)
    except ImportError:
        module = _fake_agentx_launch()
        monkeypatch.setitem(sys.modules, _MODULE, module)
        return module


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "benchmark_sglang_baseline"
    path.mkdir()
    spec = {
        "version": 1,
        "framework": "sglang",
        "argv": _RECIPE_ARGV,
        "env": {},
        "golden_acceptance_overrides": [{"name": "eagle"}],
    }
    config = {
        "framework": "sglang",
        "agentx": {
            "enabled": True,
            "launch_overrides": {"version": 1, "append_args": [], "env": {"AITER_LOG_TUNED_CONFIG": "1"}},
            "resolved": {"server-launch-spec": spec},
        },
    }
    (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _variant(name: str, args: str) -> dict:
    return {"name": name, "extra_args": args}


def _narrow(workspace, grid):
    """Run the launch-contract narrowing on ``grid``; the outcome, the grid left and the observations."""
    observations: list[dict] = []

    async def record_observation(sender, kind, payload):
        observations.append(payload)

    router = SimpleNamespace(
        shared_state=SimpleNamespace(baseline_workspace=str(workspace)),
        bus=SimpleNamespace(record_observation=record_observation),
    )
    params = {"grid": list(grid)}
    admitted = asyncio.run(_keep_launchable_variants(router, "orchestration", params))
    return admitted, params["grid"], observations


def test_a_variant_changing_golden_acceptance_inputs_is_refused_with_magpies_reason(workspace):
    spec = _variant("spec_steps_4", "--speculative-num-steps 4 --speculative-num-draft-tokens 5")
    chunked = _variant("chunked_16k", "--chunked-prefill-size 16384")

    admitted, grid, observations = _narrow(workspace, [spec, chunked])

    assert admitted is True
    assert grid == [chunked]
    [observation] = observations
    assert observation["kind"] == "explore_variants_refused_by_launch_contract"
    assert observation["refused"] == [{"name": "spec_steps_4", "reason": _GOLDEN_REFUSAL}]
    assert "recipe" in observation["hint"]


def test_a_grid_of_only_protected_args_is_not_enqueued(workspace):
    admitted, grid, observations = _narrow(workspace, [_variant("tp8", "--tp 8")])

    assert admitted is False
    assert grid == []
    [observation] = observations
    assert observation["refused"] == [{"name": "tp8", "reason": _PROTECTED_REFUSAL}]


def test_a_variant_on_other_knobs_is_kept_without_an_observation(workspace):
    chunked = _variant("chunked_16k", "--chunked-prefill-size 16384")

    admitted, grid, observations = _narrow(workspace, [chunked])

    assert admitted is True
    assert grid == [chunked]
    assert observations == []


def test_a_session_without_a_resolved_spec_keeps_every_variant(tmp_path):
    grid = [_variant("spec_steps_4", "--speculative-num-steps 4"), _variant("tp8", "--tp 8")]

    admitted, kept, observations = _narrow(tmp_path, grid)

    assert admitted is True
    assert kept == grid
    assert observations == []
