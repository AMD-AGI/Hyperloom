# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance through the real hosted-measurement compile, points and surface check."""

from __future__ import annotations

from pathlib import Path

import pytest

import yaml

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import ComponentChange, ConfigDelta, RecipeDelta
from hyperloom.inference_optimizer.cli.benchmark_config import AgentXWorkload, write_agentx_benchmark
from hyperloom.measure.compile import EnvironmentCompiler, LocalBlobStore, PackageBase
from hyperloom.measure.points import AcceptedMeasurement, default_canonical
from hyperloom.measure.tests._recipes import IMAGE, recipe_with
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
from hyperloom_multibranch.contracts.toolbox import RequestRejected, WorkloadKind
from hyperloom_multibranch.toolbox.planner import HostedPlanner
from hyperloom_multibranch.toolbox.tests.helpers import FakeRepo, build, req


@pytest.fixture(autouse=True)
def clean_process_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "CONC",
        "ISL",
        "OSL",
        "RANDOM_RANGE_RATIO",
        "RUN_EVAL",
        "TP",
        "EP",
        "PORT",
        "MAX_MODEL_LEN",
        "PRECISION",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "MAGPIE_EVAL_TASKS",
        "MAGPIE_EVAL_LIMIT",
        "MAGPIE_EVAL_TOKENIZED_REQUESTS",
        "INFERENCE_OPTIMIZER_SERVER_ARGS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "userdata"))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")


def _planner(tmp_path: Path) -> HostedPlanner:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    canonical = default_canonical(
        tmp_path / "canonical", primary_concurrency=32, config_path=default_baseline_config(), model_path=str(model)
    )
    base = PackageBase("sglang", tmp_path, None, None)
    compiler = EnvironmentCompiler(LocalBlobStore(tmp_path / "blobs"), IMAGE, lambda _layout: base)
    return HostedPlanner(compiler, canonical, tmp_path / "points")


async def test_a_request_is_accepted_with_its_compiled_environment_and_points(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))
    request_id = await rig.toolbox.submit_test(req(server_args="--max-running-requests 77", concurrency=(8, 64)))
    await rig.toolbox.drain()
    accepted = rig.executor.jobs[0].environment
    assert isinstance(accepted, AcceptedMeasurement) and len(accepted.points) == 2
    assert (await rig.notebook.record(request_id)).result is not None


async def test_a_launch_variable_that_changes_the_workload_is_a_surface_refusal(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))
    change = RecipeDelta(config=ConfigDelta(envs={"ISL": "4096"}))
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=change))
    assert refused.value.code == "surface_mismatch"
    assert rig.notebook._state().requests == {}


async def test_a_component_that_cannot_be_applied_is_refused_by_acceptance(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))
    change = RecipeDelta(components=(ComponentChange("nonexistent"),))
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=change))
    assert refused.value.code == "delta_not_applicable"


async def test_an_accuracy_test_whose_launch_serves_less_context_than_its_evaluation_is_refused_with_the_minimum(
    tmp_path: Path,
) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(server_args="--context-length 2304"))  # an InferenceX explore: accuracy on
    assert refused.value.code == "context_below_eval"
    assert "Serve at least 8192 tokens" in refused.value.reason and "--context-length 2304" in refused.value.reason
    assert rig.notebook._state().requests == {} and rig.toolbox._log.entries() == []


async def test_the_minimum_context_follows_the_point_the_evaluation_runs_on(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))
    change = RecipeDelta(config=ConfigDelta(server_args="--context-length 2304", envs={"MAX_MODEL_LEN": "4096"}))
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=change))
    assert refused.value.code == "context_below_eval" and "Serve at least 4096 tokens" in refused.value.reason


@pytest.mark.parametrize(
    "server_args", ["--context-length 8192", "--context-length=16384", "--max-running-requests 77"]
)
async def test_a_launch_that_serves_the_evaluation_context_or_sets_none_is_accepted(
    tmp_path: Path, server_args: str
) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))
    await rig.toolbox.submit_test(req(server_args=server_args))
    await rig.toolbox.drain()
    assert len(rig.executor.jobs) == 1


async def test_a_profile_test_carries_no_evaluation_and_is_not_held_to_its_context(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))
    await rig.toolbox.submit_test(
        req(kind=TestKind.PROFILE, purpose=Purpose.PROFILE, server_args="--context-length 2304")
    )
    await rig.toolbox.drain()
    assert len(rig.executor.jobs) == 1 and rig.executor.jobs[0].request.accuracy is False


async def test_an_agentx_session_plans_points_through_the_agentx_launch_overrides(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    source = tmp_path / "benchmark.yaml"
    write_agentx_benchmark(
        source, AgentXWorkload(framework="sglang", model="Qwen/Qwen3-8B", precision="fp8", tp=2, ep=1, conc=32)
    )
    canonical = default_canonical(
        tmp_path / "canonical", primary_concurrency=32, config_path=source, model_path=str(model)
    )
    base = PackageBase("sglang", tmp_path, None, None)
    compiler = EnvironmentCompiler(LocalBlobStore(tmp_path / "blobs"), IMAGE, lambda _layout: base)
    rig = build(
        tmp_path,
        repo=FakeRepo(recipe_with()),
        planner=HostedPlanner(compiler, canonical, tmp_path / "points"),
        workload_kind=WorkloadKind.AGENTX,
    )

    await rig.toolbox.submit_test(req(server_args="--max-running-requests 77", concurrency=(8, 64)))
    await rig.toolbox.drain()

    points = [yaml.safe_load(path.read_text()) for path in rig.executor.jobs[0].environment.points]
    assert [point["benchmark"]["envs"]["CONC"] for point in points] == [8, 64]
    for point in points:
        bench = point["benchmark"]
        assert bench["agentx"]["launch_overrides"]["append_args"] == ["--max-running-requests", "77"]
        assert bench["envs"]["RUN_EVAL"] == "false"
        assert not {"benchmark_script", "workload_spec"} & bench.keys()
        assert not {"NUM_PROMPTS", "NUM_WARMUPS", "EXTRA_SGLANG_ARGS"} & bench["envs"].keys()
