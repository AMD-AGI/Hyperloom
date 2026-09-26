# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The benchmark launch synchronizes inner and outer caps for every workload."""

from __future__ import annotations

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._grid_runner import sync_benchmark_timeout
from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts


@pytest.mark.parametrize("declared", [1800, 2400, 36000])
def test_launch_replaces_stale_yaml_cap_with_invoking_policy(tmp_path, monkeypatch, declared):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "123.5")
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "timeout_seconds": declared,
                    "envs": {"PYTHONUNBUFFERED": "0", "AGENTX_PHASE_WAIT_TIMEOUT_S": "50000"},
                }
            }
        ),
        encoding="utf-8",
    )
    _, hard = resolve_benchmark_timeouts()
    sync_benchmark_timeout(path, hard)
    bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
    assert bench["timeout_seconds"] == hard
    assert bench["envs"]["AGENTX_PHASE_WAIT_TIMEOUT_S"] == str(hard)
    assert bench["envs"]["PYTHONUNBUFFERED"] == "1"
