# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The benchmark launch synchronizes inner and outer caps for every workload."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.common.agentx_workload import agentx_warmup_requests
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridRecipe,
    GridVariant,
    _build_variant_yaml,
    sync_benchmark_timeout,
    variant_conc,
)
from hyperloom.orchestrator.actions.executors.benchmark_config import composed_benchmark_timeout_sec
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    resolve_benchmark_timeouts,
    sync_agentx_phase_wait,
)
from hyperloom.orchestrator.actions.executors._workload_envs import LaunchLayer
from hyperloom.orchestrator.kernel._rung_outcomes import _budget_skip_result


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


@pytest.fixture
def granted_cap(tmp_path, monkeypatch):
    """Compose a variant's config the way the runner does, and read back the cap its rounds get.

    The invoking environment is stripped of every name that sizes a workload, so what a
    test asserts is what the base YAML, the recipe's base layer and the variant put into
    the composition between them -- which is all the rounds launch with.
    """
    for name in ("CONC", "NUM_PROMPTS", "HYPERLOOM_AGENTX", "INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC"):
        monkeypatch.delenv(name, raising=False)
    counter = iter(range(100))

    def compose(base_envs: dict[str, str], variant: GridVariant, base: LaunchLayer = LaunchLayer()) -> float:
        nth = next(counter)
        base_yaml = tmp_path / f"base_{nth:02d}.yaml"
        config = {
            "benchmark": {
                "framework": "sglang",
                "model": "Qwen/Qwen3-0.6B",
                "agentx": {"enabled": True, "mode": "canonical"},
                "envs": {"HYPERLOOM_AGENTX": "1", "TP": "1", **base_envs},
            }
        }
        base_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
        cfg_path = _build_variant_yaml(
            GridRecipe(base_yaml_path=base_yaml, base=base), variant, output_subdir=tmp_path / f"variant_{nth:02d}"
        )
        return composed_benchmark_timeout_sec(Path(cfg_path))

    return compose


def test_a_rung_is_granted_the_cap_the_workload_it_inherits_asks_for(granted_cap):
    """The pin that sizes the work is as often inherited as set: this rung moves only server args."""
    variant = GridVariant("args_only", extra_server_args="--chunked-prefill-size 2048")
    assert variant_conc(variant) is None, "nothing in the variant spec names this rung's workload"
    assert granted_cap({"CONC": "20"}, variant) == resolve_benchmark_timeouts(_canonical_env(CONC="20"))[1] > 7800.0


def test_a_pin_from_the_recipes_base_layer_sizes_the_grant_too(granted_cap):
    """``GridRecipe.base`` is composed into the launch env the same way the base YAML is."""
    variant = GridVariant("args_only", extra_server_args="--chunked-prefill-size 2048")
    cap = granted_cap({}, variant, LaunchLayer(extra_envs={"CONC": "20"}))
    assert cap == resolve_benchmark_timeouts(_canonical_env(CONC="20"))[1] > 7800.0


def test_rungs_of_one_grid_are_capped_by_their_own_workload(granted_cap):
    """The ladder's rungs share one base config: the cap has to follow the rung, not that."""
    low = granted_cap({"CONC": "14"}, GridVariant("low", extra_envs={"CONC": "14", "NUM_PROMPTS": "70"}))
    high = granted_cap({"CONC": "14"}, GridVariant("high", extra_envs={"CONC": "28", "NUM_PROMPTS": "140"}))
    assert high > low == resolve_benchmark_timeouts(_canonical_env(CONC="14"))[1]


def test_a_rung_that_clears_an_inherited_concurrency_falls_back_to_the_stock_cap(granted_cap):
    """A variant's clears move the size of its work as much as its pins do, so the grant reads both."""
    assert granted_cap({"CONC": "28"}, GridVariant("no_conc", unset_envs=["CONC"])) == 7800.0


def test_a_rung_cannot_grant_itself_time_through_its_config(granted_cap):
    """The override is the operator's, and preflight validates it once against the process env.

    What a rung's config carries under that name went through no such validation, and
    parsing it here would raise out of the rung and end the whole grid.
    """
    variant = GridVariant("capped", extra_envs={"INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC": "two hours"})
    assert granted_cap({"CONC": "28"}, variant) == resolve_benchmark_timeouts(_canonical_env(CONC="28"))[1]


def test_the_operators_cap_holds_every_rung(granted_cap, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "1234")
    variant = GridVariant("args_only", extra_server_args="--chunked-prefill-size 2048")
    assert granted_cap({"CONC": "28"}, variant) == 1234.0


def test_a_rung_that_clears_the_operators_cap_falls_back_to_its_workloads_size(granted_cap, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "1234")
    variant = GridVariant("uncapped", unset_envs=["INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC"])
    assert granted_cap({"CONC": "28"}, variant) == resolve_benchmark_timeouts(_canonical_env(CONC="28"))[1]


def test_a_rung_that_never_ran_publishes_no_grant(monkeypatch):
    """No config was composed for it and no round of it started, so it was granted no time at all.

    A number here would be the cap of a run nobody made, and the reader cannot tell that
    apart from the cap a rung was really held to.
    """
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    variant = GridVariant("optimized_conc20", extra_envs={"CONC": "20", "NUM_PROMPTS": "100"})
    assert _budget_skip_result(variant).granted_cap_sec is None


def test_the_client_phase_wait_is_exported_at_the_rungs_own_cap():
    """The client's phase wait and the orchestrator's watchdog have to agree per rung, not per grid."""
    cap = resolve_benchmark_timeouts(_canonical_env(CONC="28"))[1]
    envs = {"AGENTX_PHASE_WAIT_TIMEOUT_S": "7800"}
    sync_agentx_phase_wait(envs, cap)
    assert envs["AGENTX_PHASE_WAIT_TIMEOUT_S"] == str(int(cap))
