# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a rendered point carries the recipe's launch, which the long-context and evaluation checks read."""

from __future__ import annotations

import pytest

from hyperloom.common.benchmark_config import BenchmarkPointError
from hyperloom.inference_optimizer.framework_registry import FRAMEWORKS
from hyperloom.measure.points import launch_carriers

_SGLANG_ARGS = FRAMEWORKS["sglang"].extra_args_env


def test_a_fixed_point_carries_its_launch_in_the_benchmark_envs() -> None:
    envs = {"TP": "8", _SGLANG_ARGS: "--context-length 8192"}

    present, carried = launch_carriers({"benchmark": {"framework": "SGLang", "envs": envs}})

    assert (dict(present), carried) == (envs, "--context-length 8192")


def test_a_framework_the_registry_does_not_know_has_no_argument_text() -> None:
    present, carried = launch_carriers({"benchmark": {"framework": "other", "envs": {"TP": "1"}}})

    assert (dict(present), carried) == ({"TP": "1"}, None)


def test_an_agentx_point_carries_its_launch_in_the_overrides_not_the_envs() -> None:
    overrides = {"env": {"MAX_MODEL_LEN": "4096"}, "append_args": ["--context-length", 4096]}
    config = {"benchmark": {"framework": "sglang", "envs": {"TP": "8"}, "agentx": {"launch_overrides": overrides}}}

    present, carried = launch_carriers(config)

    assert (dict(present), carried) == ({"MAX_MODEL_LEN": "4096"}, "--context-length 4096")


def test_a_rendered_point_without_a_benchmark_block_is_refused() -> None:
    with pytest.raises(BenchmarkPointError):
        launch_carriers({})
