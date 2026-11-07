# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shipped baseline points: read as inference_suite runs them, with nothing only Magpie read."""

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

#: Keys of the benchmark block only Magpie's runner read; ``profiler`` and ``server_lifecycle`` are pinned below.
_MAGPIE_KEYS = ("run_mode", "gpu_selection", "runner_type", "benchmark_script")


def _text(framework: str) -> str:
    return (asset_root() / "assets" / "configs" / shipped_config_name("baseline", framework)).read_text("utf-8")


def _block(framework: str) -> dict[str, Any]:
    return yaml.safe_load(_text(framework))["benchmark"]


@pytest.mark.parametrize("framework", _FRAMEWORKS)
def test_the_baseline_reads_as_a_point_of_its_framework(framework):
    point = BenchmarkPoint.from_block(_block(framework))

    assert point.framework == framework
    assert point.model
    assert not point.agentx
    assert not point.torch_profiler


@pytest.mark.parametrize("framework", _SERVING)
def test_a_serving_baseline_is_a_point_inference_suite_carries(framework):
    assert backend_reason({"benchmark": _block(framework)}) is None


@pytest.mark.parametrize("framework", _SERVING)
def test_a_serving_baseline_leaves_path_to_the_host_environment(framework):
    assert "PATH" not in _block(framework)["envs"]


def test_the_xdit_baseline_is_a_scriptable_point():
    block = _block("xdit")

    assert BenchmarkPoint.from_block(block).workload_kind == "scriptable"
    assert backend_reason({"benchmark": block}) == "the point runs xdit, which serves nothing"


@pytest.mark.parametrize("framework", _FRAMEWORKS)
def test_the_baseline_carries_nothing_only_magpie_read(framework):
    block = _block(framework)

    assert not set(_MAGPIE_KEYS) & set(block)
    assert set(block.get("profiler", {})) == {"torch_profiler"}
    assert "enabled" not in block.get("server_lifecycle", {})
    assert "magpie" not in _text(framework).lower()
