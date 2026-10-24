# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shipped profile configurations replay captured graphs node by node; the baselines do not."""

from __future__ import annotations

import pytest
import yaml

from hyperloom.inference_optimizer.session.paths import asset_root

_CONFIGS = asset_root() / "assets" / "configs"
_GRAPH_REPLAY_ENV = "DEBUG_CLR_GRAPH_PACKET_CAPTURE"


def _envs(name: str) -> dict[str, str]:
    return yaml.safe_load((_CONFIGS / name).read_text())["benchmark"]["envs"]


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
def test_a_profile_replays_graphs_node_by_node_and_a_baseline_replays_as_shipped(framework):
    """The setting is the HIP runtime's, so every framework's profile carries it and no baseline does."""
    assert _envs(f"profile_{framework}.yaml")[_GRAPH_REPLAY_ENV] == "0"
    assert _GRAPH_REPLAY_ENV not in _envs(f"baseline_{framework}.yaml")
