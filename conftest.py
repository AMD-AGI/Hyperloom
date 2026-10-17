# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Repository-wide pytest bootstrap."""

from __future__ import annotations

import os
import pytest
from pathlib import Path


def _pin_child_processes_to_this_tree() -> None:
    """Put this checkout's ``src`` first on the child ``PYTHONPATH``.

    ``pythonpath`` in ``pyproject.toml`` only reaches the pytest process. Tests
    that shell out inherit the ambient one, so a Hyperloom installed elsewhere
    on the host silently answers their imports instead of the tree under test.
    """
    src = str(Path(__file__).resolve().parent / "src")
    existing = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    os.environ["PYTHONPATH"] = os.pathsep.join([src, *(entry for entry in existing if entry != src)])


_pin_child_processes_to_this_tree()


@pytest.fixture
def native_benchmark(tmp_path):
    """A canonical Magpie workload; individual tests choose whether to activate it."""
    import yaml

    path = tmp_path / "agentx.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "Qwen/Qwen3-0.6B",
                    "precision": "bf16",
                    "runner_type": "mi355x",
                    "docker_image": "example/sglang:pinned",
                    "agentx": {"enabled": True, "mode": "canonical"},
                    "envs": {"TP": "1", "EP_SIZE": "1", "CONC": "8"},
                }
            }
        )
    )
    return path
