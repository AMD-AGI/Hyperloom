# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixed benchmark caps remain independent of AgentX client warmup settings."""

from __future__ import annotations


import pytest

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor
from hyperloom.orchestrator.actions.executors._workload_envs import apply_agentx_switch


def _clear(monkeypatch):
    for key in (
        "HYPERLOOM_AGENTX",
        "AGENTX_WARMUP_GRACE_PERIOD",
        "AGENTX_WARMUP_GRACE_CONC",
        "CONC",
        "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("agentx", ["0", "1"])
def test_baseline_cap_does_not_expand_with_workload(monkeypatch, tmp_path, agentx):
    _clear(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", agentx)
    monkeypatch.setenv("AGENTX_DURATION", "50000")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "20000")
    executor = BaselineExecutor(session_dir=tmp_path)
    assert executor._resolve_timeout({"timeout_sec": 99}) == 7800
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "123.5")
    assert executor._resolve_timeout({}) == 123.5


def test_profile_retains_nonbenchmark_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "99")
    executor = ProfileExecutor(session_dir=tmp_path)
    assert executor.benchmark_watchdog is False
    assert executor._resolve_timeout({}) == 14400
    assert executor._resolve_timeout({"timeout_sec": 15000}) == 15000


def test_agentx_switch_keeps_profile_yaml_cap(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("AGENTX_DURATION", "50000")
    bench = {"framework": "sglang", "timeout_seconds": 14400}
    apply_agentx_switch(bench, active=True)
    assert bench["timeout_seconds"] == 14400


# --- the grace declares which concurrency it was measured at -------------------


@pytest.mark.parametrize("executor_class", [BaselineExecutor, ProfileExecutor])
async def test_a_round_of_a_point_that_names_no_timeout_runs_under_the_executors_cap(
    monkeypatch, tmp_path, executor_class
):
    """A session's AgentX ``benchmark.yaml`` names no ``timeout_seconds``; the round the suite runs carries the cap."""
    from types import SimpleNamespace

    import yaml

    from hyperloom.orchestrator.actions.executors.benchmark_run import launch
    from hyperloom.orchestrator.actions.executors.benchmark_run.records import BenchmarkAttempt

    _clear(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "7800")
    config = tmp_path / "benchmark.yaml"
    config.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "agentx": {"enabled": True}}}))
    seen = {}

    def run_suite_round(_route, request, **_kwargs):
        seen.update(yaml.safe_load(request.config_path.read_text())["benchmark"])
        raise launch.SuiteRoundRefused("test_stop", "the config is read")

    monkeypatch.setattr(launch, "run_suite_round", run_suite_round)
    executor = executor_class(session_dir=tmp_path)
    executor.shared_state = SimpleNamespace()
    attempt = BenchmarkAttempt(
        timeout_sec=executor._resolve_timeout({}),
        override_result_dir=None,
        resolved_model="m",
        materialized_config_path=config,
        inferencex_path="",
        effective_extra_server_args="",
        params={},
        ctx=SimpleNamespace(extra={}),
    )

    await executor._run_single_benchmark(attempt, config_path=config, output_dir=tmp_path / "round")

    assert seen["timeout_seconds"] == attempt.timeout_sec
