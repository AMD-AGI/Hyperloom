# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The benchmark launch synchronizes inner and outer caps for every workload."""

from __future__ import annotations

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._grid_runner import sync_benchmark_timeout
from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts
from hyperloom.orchestrator.actions.executors._workload_envs import apply_agentx_switch


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
    assert bench["envs"]["AGENTX_PHASE_WAIT_TIMEOUT_S"] == "124"
    assert bench["envs"]["PYTHONUNBUFFERED"] == "1"


def test_the_stock_cap_reaches_the_client_as_an_integer(tmp_path, monkeypatch):
    """aiperf_client.sh rejects anything but digits, so the stock 7800.0 must arrive as 7800."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", raising=False)
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump({"benchmark": {"timeout_seconds": 1800, "envs": {"AGENTX_PHASE_WAIT_TIMEOUT_S": "1800"}}}),
        encoding="utf-8",
    )
    sync_benchmark_timeout(path, resolve_benchmark_timeouts({})[1])
    bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
    assert bench["envs"]["AGENTX_PHASE_WAIT_TIMEOUT_S"] == "7800"


def test_the_agentx_switch_exports_a_float_cap_as_an_integer(monkeypatch):
    monkeypatch.delenv("AGENTX_WARMUP_GRACE_PERIOD", raising=False)
    bench = {"framework": "vllm", "model": "/m", "timeout_seconds": 7800.0, "envs": {}}
    apply_agentx_switch(bench, "/m", active=True)
    assert bench["envs"]["AGENTX_PHASE_WAIT_TIMEOUT_S"] == "7800"


def _mlperf_env(**over: str) -> dict[str, str]:
    return {"HYPERLOOM_AGENTIC_BACKEND": "mlperf", **over}


def test_an_mlperf_run_gets_a_cap_sized_to_its_trajectories():
    """The stock 7800s default reaped a healthy 150-trajectory baseline at 38%."""
    smoke = resolve_benchmark_timeouts(_mlperf_env(MLPERF_AGENTIC_FLOW="smoke_test"))[1]
    full = resolve_benchmark_timeouts(_mlperf_env(MLPERF_AGENTIC_FLOW="full"))[1]
    assert smoke > 7800.0
    assert full > 613 * 26
    assert full > smoke


def test_a_small_mlperf_run_never_tightens_the_stock_cap():
    assert resolve_benchmark_timeouts(_mlperf_env(AGENTIC_NUM_TRAJECTORIES="5"))[1] == 7800.0


def test_an_operator_cap_wins_over_the_mlperf_derivation():
    env = _mlperf_env(MLPERF_AGENTIC_FLOW="full", INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC="1234")
    assert resolve_benchmark_timeouts(env)[1] == 1234.0


def test_the_aiperf_backend_keeps_the_stock_cap():
    assert resolve_benchmark_timeouts({"MLPERF_AGENTIC_FLOW": "full"})[1] == 7800.0
