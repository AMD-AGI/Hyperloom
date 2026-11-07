# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shipped baseline and profile points: read as inference_suite runs them, with nothing only Magpie read."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from hyperloom.common.benchmark_config import BenchmarkPoint
from hyperloom.inference_optimizer.framework_registry import is_scriptable, shipped_config_name
from hyperloom.inference_optimizer.session.paths import asset_root
from hyperloom.orchestrator.actions.executors._suite_config import backend_reason

_FRAMEWORKS = ("atom", "sglang", "vllm", "xdit")
_SERVING = tuple(framework for framework in _FRAMEWORKS if not is_scriptable(framework))
_KINDS = ("baseline", "profile")

#: Keys of the benchmark block only Magpie's runner read; ``profiler`` and ``server_lifecycle`` are pinned below.
_MAGPIE_KEYS = ("run_mode", "gpu_selection", "runner_type", "benchmark_script")
#: The HIP runtime switches that make the profiler record every graph replay (outside_facts hip_graph_packet_capture).
_GRAPH_REPLAY_ENV = {
    "DEBUG_CLR_GRAPH_PACKET_CAPTURE": "0",
    "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "0",
    "DEBUG_HIP_GRAPH_CLASSIC_PATH": "1",
}


def _text(kind: str, framework: str) -> str:
    return (asset_root() / "assets" / "configs" / shipped_config_name(kind, framework)).read_text("utf-8")


def _block(kind: str, framework: str) -> dict[str, Any]:
    return yaml.safe_load(_text(kind, framework))["benchmark"]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("framework", _FRAMEWORKS)
def test_the_point_reads_as_a_point_of_its_framework(kind, framework):
    point = BenchmarkPoint.from_block(_block(kind, framework))

    assert point.framework == framework
    assert point.model
    assert not point.agentx
    assert point.torch_profiler is (kind == "profile")


@pytest.mark.parametrize("framework", _SERVING)
def test_a_serving_baseline_is_a_point_inference_suite_carries(framework):
    assert backend_reason({"benchmark": _block("baseline", framework)}) is None


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("framework", _SERVING)
def test_a_serving_point_leaves_path_to_the_host_environment(kind, framework):
    assert "PATH" not in _block(kind, framework)["envs"]


@pytest.mark.parametrize("kind", _KINDS)
def test_the_xdit_point_is_a_scriptable_point(kind):
    block = _block(kind, "xdit")

    assert BenchmarkPoint.from_block(block).workload_kind == "scriptable"
    assert backend_reason({"benchmark": block}) == "the point runs xdit, which serves nothing"


@pytest.mark.parametrize("framework", _FRAMEWORKS)
def test_a_profile_point_marks_its_round_as_a_profile(framework):
    assert str(_block("profile", framework)["envs"]["PROFILE"]) == "1"


@pytest.mark.parametrize("framework", _SERVING)
def test_a_serving_profile_records_every_graph_replay(framework):
    envs = _block("profile", framework)["envs"]

    assert {name: envs.get(name) for name in _GRAPH_REPLAY_ENV} == _GRAPH_REPLAY_ENV


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("framework", _FRAMEWORKS)
def test_the_point_carries_nothing_only_magpie_read(kind, framework):
    block = _block(kind, framework)

    assert not set(_MAGPIE_KEYS) & set(block)
    assert set(block.get("profiler", {})) == {"torch_profiler"}
    assert "enabled" not in block.get("server_lifecycle", {})
    assert "XDIT_SUPPORTS_PROFILER" not in block["envs"]
    assert "magpie" not in _text(kind, framework).lower()
