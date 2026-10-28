# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A native AgentX KEEP must replay the server configuration that was measured."""

from copy import deepcopy
import os
import shlex
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.inference_optimizer.agentx.report import native_launch_evidence
from hyperloom.orchestrator.actions.executors import (
    _grid_runner,
    _integrate_bench,
    _integrate_results,
    _ray_serving,
    explore,
    integrate_patch,
)
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant, VariantResult
from hyperloom.orchestrator.actions.executors._launch_evidence import persist_launch_evidence
from hyperloom.orchestrator.actions.executors._workload_envs import (
    materialize_benchmark_config,
    prepare_grid_benchmark_config,
    LaunchLayer,
    BenchmarkTarget,
)
from hyperloom.orchestrator.actions.executors.magpie_agentx import load_benchmark
from hyperloom.orchestrator.state.shared_state import SharedState, stack_base_params


_MODEL = "Qwen/Qwen3-0.6B"
_PREFIX = ["python3", "-m", "sglang.launch_server", "--model-path", _MODEL]
_OPTION = "--cuda-graph-max-bs"
_ENV = "SGLANG_USE_AITER"


def _evidence(config, recipe, effective, env):
    return native_launch_evidence(
        {
            "agentx_metrics": {
                "server_launch": {"base_argv": recipe, "effective_argv": effective, "effective_env": env},
                "recipe": {"tp": 1, "ep": 1, "conc": 8},
            }
        },
        config,
    )


@pytest.fixture
def native_session(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("HYPERLOOM_", "INFERENCE_OPTIMIZER_", "SGLANG_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("MODEL_PATH", _MODEL)
    monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda **kwargs: None)

    def build(*, appended="--mem-fraction-static 0.85", recipe=None, env=None):
        canonical = list(recipe if recipe is not None else _PREFIX + [_OPTION, "256", "--enable-hierarchical-cache"])
        initial_env = dict(env or {})
        source = tmp_path / "workload.yaml"
        source.write_text(
            yaml.safe_dump(
                {
                    "benchmark": {
                        "model": _MODEL,
                        "framework": "sglang",
                        "precision": "bf16",
                        "runner_type": "mi355x",
                        "agentx": {"enabled": True, "mode": "canonical"},
                        "envs": {"TP": "1", "EP_SIZE": "1", "CONC": "8"},
                    }
                }
            )
        )
        baseline = materialize_benchmark_config(
            source, tmp_path / "baseline", launch=LaunchLayer(extra_server_args=appended, extra_envs=initial_env)
        )
        evidence = _evidence(baseline, canonical, canonical + shlex.split(appended), initial_env)
        persist_launch_evidence(evidence, slot=baseline.parent)
        state = SharedState(
            framework="sglang",
            benchmark_mode="agentx",
            agentx_backend="magpie",
            baseline_config_path=str(baseline),
            baseline_tput=100,
            baseline_perf={
                "workload_identity": "fixed-workload",
                "output_throughput": 100,
                "total_throughput": 200,
                "e2e_norm_intvty_p50": 100,
                "e2e_norm_intvty_p90": 100,
                "duration_seconds": 3600,
                "request_error_rate": 0,
            },
            grading={"objective": "e2e_norm_intvty_p90", "noise_pct": 2},
        )
        return SimpleNamespace(config=baseline, recipe=canonical, evidence=evidence, state=state)

    return build


def _stub_gpu(monkeypatch, session, *, effective):
    """Keep executor/materializer behavior real; substitute only benchmark execution."""
    observed = []

    async def run_grid(**kwargs):
        variant = kwargs["grid"][0]
        config = _grid_runner._build_variant_yaml(
            _grid_runner.GridRecipe(
                base_yaml_path=kwargs["recipe"].base_yaml_path,
                base=kwargs["recipe"].base,
                target=BenchmarkTarget(model_path=kwargs["recipe"].target.model_path),
            ),
            variant,
            output_subdir=kwargs["output_root"] / "variant_00_native",
        )
        launch = load_benchmark(config)["agentx"]["launch_overrides"]
        observed.append(deepcopy(launch))
        evidence = _evidence(config, session.recipe, effective, launch["env"])
        evidence_path = persist_launch_evidence(evidence, slot=config.parent)
        return [
            VariantResult(
                name=variant.name,
                extra_server_args=variant.extra_server_args,
                extra_envs=variant.extra_envs,
                status="succeeded",
                output_throughput=100 + 20 * len(observed),
                total_token_throughput=200 + 40 * len(observed),
                intvty_p50=100 + 20 * len(observed),
                intvty_p90=100 + 20 * len(observed),
                duration_seconds=3600,
                completed_requests=100,
                request_error_rate=0,
                benchmark_backend="magpie_agentx",
                benchmark_valid=True,
                diagnostic_only=False,
                workload_identity="fixed-workload",
                launch_evidence=evidence,
                launch_evidence_path=evidence_path,
            )
        ]

    monkeypatch.setattr(explore, "run_grid", run_grid)
    monkeypatch.setattr(_integrate_bench, "run_grid", run_grid)
    return observed


async def _explore(session, tmp_path, *, name, candidate, base=None):
    params = {
        "config_path": str(session.config),
        "output_dir": str(tmp_path / name),
        "base_tput": 100,
        "grid": [{"name": name, **candidate}],
        **(base or {}),
    }
    ctx = SimpleNamespace(task=SimpleNamespace(params=params, task_id=name), extra={"shared_state": session.state})
    result = await explore.ExploreExecutor(session_dir=tmp_path)(ctx)
    assert result["status"] == "succeeded", result
    return result


@pytest.mark.asyncio
async def test_env_only_keep_replays_unrelated_recipe_and_operator_options(native_session, tmp_path, monkeypatch):
    session = native_session(env={_ENV: "1"})
    baseline_bytes = session.config.read_bytes()
    observed = _stub_gpu(monkeypatch, session, effective=session.recipe + ["--mem-fraction-static", "0.85"])
    result = await _explore(
        session,
        tmp_path,
        name="unset-only",
        candidate={"unset_envs": [_ENV]},
        base={"base_extra_envs": {_ENV: "1"}},
    )
    assert len(result["winners"]) == 1, result
    winner = result["winners"][0]
    session.state.current_best = winner
    await _explore(
        session,
        tmp_path,
        name="next-candidate",
        candidate={"extra_envs": {"SGLANG_TEST": "1"}},
        base=stack_base_params(winner),
    )
    assert len(observed) == 2
    assert session.config.read_bytes() == baseline_bytes
    for launch in observed:
        assert _ENV not in launch["env"]
        assert _ENV in launch["unset_env"]
        assert "--mem-fraction-static" in launch["append_args"]
        assert launch["append_args"][launch["append_args"].index("--mem-fraction-static") + 1] == "0.85"
        assert _OPTION not in launch["remove_args"]
        assert "--enable-hierarchical-cache" not in launch["remove_args"]
        if launch["replace_args"]:
            assert _OPTION in launch["append_args"]
            assert launch["append_args"][launch["append_args"].index(_OPTION) + 1] == "256"
            assert "--enable-hierarchical-cache" in launch["append_args"]


@pytest.mark.asyncio
async def test_explore_ablation_removes_recipe_and_appended_copies(native_session, tmp_path, monkeypatch):
    session = native_session(appended=f"{_OPTION} 512")
    observed = _stub_gpu(monkeypatch, session, effective=_PREFIX + ["--enable-hierarchical-cache"])
    result = await _explore(session, tmp_path, name="ablate-both", candidate={"remove_args": [_OPTION]})
    assert len(result["winners"]) == 1, result
    assert len(observed) == 1
    launch = observed[0]
    assert _OPTION not in launch["append_args"]
    assert _OPTION in launch["remove_args"] or launch["replace_args"]


@pytest.mark.asyncio
async def test_accepted_appended_only_ablation_survives_integrate_patch(native_session, tmp_path, monkeypatch):
    session = native_session(appended="--disable-cuda-graph", recipe=_PREFIX)
    observed = _stub_gpu(monkeypatch, session, effective=_PREFIX)
    result = await _explore(
        session, tmp_path, name="ablate-appended", candidate={"remove_args": ["--disable-cuda-graph"]}
    )
    assert len(result["winners"]) == 1, result
    winner = result["winners"][0]
    session.state.current_best = winner
    next_round = await _explore(
        session,
        tmp_path,
        name="after-ablation",
        candidate={"extra_envs": {"SGLANG_TEST": "1"}},
        base=stack_base_params(winner),
    )
    assert len(next_round["winners"]) == 1, next_round
    winner = next_round["winners"][0]
    bench, _ = await integrate_patch.IntegratePatchExecutor(session_dir=tmp_path)._bench_patch(
        _integrate_results._PatchBench(
            params={"config_path": str(session.config), **stack_base_params(winner)},
            output_root=tmp_path / "integrate",
            specialist_task_id="accepted-ablation",
            benchmark_mode="agentx",
        ),
        extra_server_args_applied="",
        extra_envs_applied={},
    )
    assert bench["status"] == "succeeded", bench
    assert len(observed) == 3
    for launch in observed:
        assert "--disable-cuda-graph" not in launch["append_args"]
        assert "--disable-cuda-graph" not in launch["remove_args"]


@pytest.mark.parametrize("option", ["--unrecognized-server-option", "--model-path", "--port"])
def test_unknown_and_protected_removals_reach_magpie_validation(native_session, tmp_path, option):
    session = native_session(recipe=_PREFIX + ["--port", "30000"])
    intermediate = prepare_grid_benchmark_config(session.config, tmp_path / "explore")
    variant = _grid_runner._build_variant_yaml(
        _grid_runner.GridRecipe(base_yaml_path=intermediate),
        GridVariant("remove", remove_args=[option]),
        output_subdir=tmp_path / "candidate",
    )
    launch = load_benchmark(variant)["agentx"]["launch_overrides"]
    assert option in launch["remove_args"]


@pytest.mark.asyncio
async def test_explicit_recipe_replacement_survives_following_keep_and_replay(native_session, tmp_path, monkeypatch):
    session = native_session()
    replacement = ["--mem-fraction-static", "0.8"]
    observed = _stub_gpu(monkeypatch, session, effective=_PREFIX + replacement)
    first = await _explore(
        session,
        tmp_path,
        name="explicit-replacement",
        candidate={"extra_args": "--mem-fraction-static 0.8", "args_mode": "replace", "remove_args": [_OPTION]},
    )
    assert len(first["winners"]) == 1, first
    winner = first["winners"][0]
    assert winner["args_mode"] == "replace"
    session.state.current_best = winner
    second = await _explore(
        session,
        tmp_path,
        name="env-after-replacement",
        candidate={"extra_envs": {"SGLANG_TEST": "1"}},
        base=stack_base_params(winner),
    )
    assert len(second["winners"]) == 1, second
    winner = second["winners"][0]
    assert winner["args_mode"] == "replace"
    session.state.current_best = winner
    await _explore(
        session,
        tmp_path,
        name="replay-replaced-stack",
        candidate={"extra_envs": {"SGLANG_TEST": "2"}},
        base=stack_base_params(winner),
    )
    bench, _ = await integrate_patch.IntegratePatchExecutor(session_dir=tmp_path)._bench_patch(
        _integrate_results._PatchBench(
            params={"config_path": str(session.config), **stack_base_params(winner)},
            output_root=tmp_path / "integrate-replaced-stack",
            specialist_task_id="after-replacement",
            benchmark_mode="agentx",
        ),
        extra_server_args_applied="",
        extra_envs_applied={"SGLANG_TEST": "3"},
    )
    assert bench["status"] == "succeeded", bench
    assert len(observed) == 4
    for launch in observed:
        assert launch["replace_args"] is True
        assert launch["append_args"] == [*replacement, "--watchdog-timeout", "1800"]
        assert "--enable-hierarchical-cache" not in launch["append_args"]
        assert _OPTION not in launch["append_args"]
