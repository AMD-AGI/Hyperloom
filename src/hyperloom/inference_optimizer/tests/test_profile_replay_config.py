# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shipped profile configurations replay captured graphs node by node; the baselines do not."""

from __future__ import annotations

import pytest
import yaml

from hyperloom.inference_optimizer.session.paths import asset_root

_CONFIGS = asset_root() / "assets" / "configs"
# One switch per HIP runtime line; each runtime reads only the one it defines.
_GRAPH_REPLAY_ENVS = {
    "DEBUG_CLR_GRAPH_PACKET_CAPTURE": "0",  # ROCm 7.1.0-7.2.3
    "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING": "0",  # ROCm 7.2.4
    "DEBUG_HIP_GRAPH_CLASSIC_PATH": "1",  # ROCm 10.x
}


def _envs(name: str) -> dict[str, str]:
    return yaml.safe_load((_CONFIGS / name).read_text())["benchmark"]["envs"]


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
@pytest.mark.parametrize(("variable", "value"), sorted(_GRAPH_REPLAY_ENVS.items()))
def test_a_profile_replays_graphs_node_by_node_and_a_baseline_replays_as_shipped(framework, variable, value):
    """The switches are the HIP runtime's, so every framework's profile carries each of them and no baseline does."""
    assert _envs(f"profile_{framework}.yaml")[variable] == value
    assert variable not in _envs(f"baseline_{framework}.yaml")
