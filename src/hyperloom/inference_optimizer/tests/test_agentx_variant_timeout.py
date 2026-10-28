# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The benchmark launch synchronizes inner and outer caps for every workload."""

from __future__ import annotations

import pytest
import yaml

from hyperloom.common.agentx_workload import agentx_warmup_requests
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    sync_benchmark_timeout,
    variant_benchmark_timeout_sec,
)
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    resolve_benchmark_timeouts,
    sync_agentx_phase_wait,
    workload_benchmark_timeout_sec,
)
from hyperloom.orchestrator.kernel.conc_sweep import _budget_skip_result


@pytest.mark.parametrize("declared", [1800, 2400, 36000])
def test_launch_replaces_stale_yaml_cap_with_invoking_policy(tmp_path, monkeypatch, declared):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "123.5")
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "timeout_seconds": declared,
                    "envs": {"PYTHONUNBUFFERED": "0"},
                }
            }
        ),
        encoding="utf-8",
    )
    _, hard = resolve_benchmark_timeouts()
    sync_benchmark_timeout(path, hard)
    bench = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
    assert bench["timeout_seconds"] == hard
    assert bench["envs"]["PYTHONUNBUFFERED"] == "1"


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


def test_native_magpie_keeps_the_stock_cap():
    assert resolve_benchmark_timeouts({"MLPERF_AGENTIC_FLOW": "full"})[1] == 7800.0


def _canonical_env(**over: str) -> dict[str, str]:
    return {"HYPERLOOM_AGENTX": "1", **over}


def test_a_canonical_rung_is_granted_the_phases_it_was_composed_for():
    """A CONC 20 rung spent 4151.88s warming up and 3640s profiling, and was reaped at 7800s."""
    cap = resolve_benchmark_timeouts(_canonical_env(CONC="20"))[1]
    assert agentx_warmup_requests(20) == 221
    assert cap > 4151.88 + 3640.0 + 300.0


def test_a_canonical_cap_grows_with_the_concurrency_it_measures():
    """Warmup is one set of requests per lane, so the rung that measures more lanes needs longer."""
    caps = [resolve_benchmark_timeouts(_canonical_env(CONC=str(conc)))[1] for conc in (14, 20, 28)]
    assert caps == sorted(caps) and caps[0] < caps[-1]


def test_a_canonical_run_that_pins_no_concurrency_keeps_the_stock_cap():
    assert resolve_benchmark_timeouts(_canonical_env())[1] == 7800.0


def test_synthetic_magpie_keeps_the_stock_cap_at_any_concurrency():
    """Only canonical replay warms up per lane; the synthetic client's warmup is capped at 8 requests."""
    assert resolve_benchmark_timeouts({"CONC": "28"})[1] == 7800.0


def test_an_operator_cap_wins_over_the_canonical_derivation():
    env = _canonical_env(CONC="28", INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC="1234")
    assert resolve_benchmark_timeouts(env)[1] == 1234.0


def test_rungs_of_one_grid_are_capped_by_their_own_workload(monkeypatch):
    """The ladder's rungs share a process environment: the cap has to follow the variant, not that."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("CONC", "14")
    monkeypatch.delenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", raising=False)
    low = workload_benchmark_timeout_sec({"CONC": "14", "NUM_PROMPTS": "70"})
    high = workload_benchmark_timeout_sec({"CONC": "28", "NUM_PROMPTS": "140"})
    assert high > low == resolve_benchmark_timeouts()[1]


def test_a_rung_that_clears_its_concurrency_falls_back_to_the_stock_cap(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("CONC", "28")
    monkeypatch.delenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", raising=False)
    assert workload_benchmark_timeout_sec(None, ["CONC"]) == 7800.0


def test_a_rung_is_granted_the_cap_its_whole_spec_asks_for(monkeypatch):
    """A variant's clears move the size of its work as much as its pins do, so the grant has to read both."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "60")
    variant = GridVariant(
        "optimized_conc20",
        extra_envs={"CONC": "20", "NUM_PROMPTS": "100"},
        unset_envs=["INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC"],
    )
    assert variant_benchmark_timeout_sec(variant) == resolve_benchmark_timeouts(_canonical_env(CONC="20"))[1]
    # What the rung's pins alone say, which is the cap of some other run.
    assert workload_benchmark_timeout_sec(variant.extra_envs) == 60.0


def test_a_rung_publishes_the_cap_it_was_granted(monkeypatch):
    """Whoever grants the cap writes it on the row; a reader that derives its own reports a run nobody made."""
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "60")
    variant = GridVariant(
        "optimized_conc20",
        extra_envs={"CONC": "20", "NUM_PROMPTS": "100"},
        unset_envs=["INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC"],
    )
    assert _budget_skip_result(variant).granted_cap_sec == variant_benchmark_timeout_sec(variant)


def test_the_client_phase_wait_is_exported_at_the_rungs_own_cap():
    """The client's phase wait and the orchestrator's watchdog have to agree per rung, not per grid."""
    cap = resolve_benchmark_timeouts(_canonical_env(CONC="28"))[1]
    envs = {"AGENTX_PHASE_WAIT_TIMEOUT_S": "7800"}
    sync_agentx_phase_wait(envs, cap)
    assert envs["AGENTX_PHASE_WAIT_TIMEOUT_S"] == str(int(cap))
