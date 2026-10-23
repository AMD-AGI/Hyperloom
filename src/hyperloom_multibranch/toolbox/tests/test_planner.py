# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance through the real hosted-measurement compile, points and surface check."""

from __future__ import annotations

from pathlib import Path

import pytest

import yaml

from hyperloom.common.outside_facts import AITER_LOG_TUNED_CONFIG_ENV
from hyperloom.contracts.measurement import (
    TABLE_EXERCISED,
    TABLE_NOT_EXERCISED,
    TABLE_NOT_OBSERVABLE,
    Purpose,
    TestKind,
)
from hyperloom.contracts.recipe import ComponentChange, ConfigDelta, LaunchConfig, RecipeDelta
from hyperloom.inference_optimizer.cli.benchmark_config import AgentXWorkload, write_agentx_benchmark
from hyperloom.measure.compile import EnvironmentCompiler, LocalBlobStore, PackageBase
from hyperloom.measure.farm import server_environment
from hyperloom.measure.points import AcceptedMeasurement, default_canonical
from hyperloom.measure.tests._recipes import IMAGE, recipe_with
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
from hyperloom_multibranch.contracts.toolbox import RequestRejected, WorkloadKind
from hyperloom_multibranch.toolbox.planner import HIT_LOGGING_ON, HostedPlanner
from hyperloom_multibranch.toolbox.table_hits import read_table_hits
from hyperloom_multibranch.toolbox.tests.helpers import FakeRepo, build, req
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import B01_NEW, csv_text
from hyperloom_multibranch.toolbox.tests.test_table_hits import MERGED, MISS, ROW_16_34816, Session, a8w8_hit


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


async def test_a_check_refuses_what_acceptance_refuses_and_accepts_nothing_either_way(tmp_path: Path) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))

    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.check_test(req(server_args="--context-length 2304"))
    await rig.toolbox.check_test(req(server_args="--max-running-requests 77"))

    assert refused.value.code == "context_below_eval"
    assert rig.notebook._state().requests == {} and rig.toolbox._log.entries() == []
    assert rig.executor.jobs == [] and not (await rig.toolbox.queue_state()).queued_by_branch


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


def _agentx_planner(tmp_path: Path) -> HostedPlanner:
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
    return HostedPlanner(compiler, canonical, tmp_path / "points")


async def test_an_agentx_session_plans_points_through_the_agentx_launch_overrides(tmp_path: Path) -> None:
    rig = build(
        tmp_path,
        repo=FakeRepo(recipe_with()),
        planner=_agentx_planner(tmp_path),
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


ENV = AITER_LOG_TUNED_CONFIG_ENV


def _server_env_values(accepted: AcceptedMeasurement) -> list[object]:
    """What each planned point launches its server with for ``AITER_LOG_TUNED_CONFIG``, as the executor reads it."""
    return [server_environment(yaml.safe_load(path.read_text())).get(ENV) for path in accepted.points]


@pytest.mark.parametrize(
    "fields",
    [{"concurrency": (8, 64)}, {"purpose": Purpose.BASELINE}, {"kind": TestKind.PROFILE, "purpose": Purpose.PROFILE}],
    ids=["explore", "baseline", "profile"],
)
async def test_every_point_of_every_test_asks_its_server_for_aiter_hits(tmp_path: Path, fields: dict) -> None:
    rig = build(tmp_path, repo=FakeRepo(recipe_with()), planner=_planner(tmp_path))
    await rig.toolbox.submit_test(req(**fields))
    await rig.toolbox.drain()
    accepted = rig.executor.jobs[0].environment
    assert len(accepted.points) == len(fields.get("concurrency", (0,)))
    assert _server_env_values(accepted) == [HIT_LOGGING_ON] * len(accepted.points)
    assert accepted.environment.envs[ENV] == HIT_LOGGING_ON, "the variable is the environment's, not the recipe's"
    assert ENV not in accepted.environment.recipe.launch.envs


@pytest.mark.parametrize("purpose", [Purpose.EXPLORE, Purpose.BASELINE], ids=["explore", "baseline"])
async def test_an_agentx_point_launches_its_server_with_aiter_hits_which_its_renderer_does_not_set(
    tmp_path: Path, purpose: Purpose
) -> None:
    rig = build(
        tmp_path, repo=FakeRepo(recipe_with()), planner=_agentx_planner(tmp_path), workload_kind=WorkloadKind.AGENTX
    )
    await rig.toolbox.submit_test(req(purpose=purpose, concurrency=(8, 64)))
    await rig.toolbox.drain()
    accepted = rig.executor.jobs[0].environment
    for path in accepted.points:
        bench = yaml.safe_load(path.read_text())["benchmark"]
        assert bench["agentx"]["launch_overrides"]["env"][ENV] == HIT_LOGGING_ON
        assert ENV not in bench["envs"], "an AgentX server is launched from the overrides, not from benchmark.envs"
    assert len(accepted.points) == 2


@pytest.mark.parametrize(
    ("parent_envs", "change"),
    [
        ({ENV: HIT_LOGGING_ON}, RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256"))),
        ({}, RecipeDelta(config=ConfigDelta(envs={ENV: HIT_LOGGING_ON}))),
        ({ENV: HIT_LOGGING_ON}, RecipeDelta(config=ConfigDelta(unset_envs=(ENV,)))),
    ],
    ids=["the parent sets it", "the delta sets it", "the delta unsets it"],
)
async def test_a_recipe_that_sets_the_variable_to_the_same_value_or_unsets_it_is_measured_as_any_other(
    tmp_path: Path, parent_envs: dict[str, str], change: RecipeDelta
) -> None:
    repo = FakeRepo(recipe_with(launch=LaunchConfig(envs=parent_envs)))
    rig = build(tmp_path, repo=repo, planner=_planner(tmp_path))
    await rig.toolbox.submit_test(req(change=change))
    await rig.toolbox.drain()
    assert _server_env_values(rig.executor.jobs[0].environment) == [HIT_LOGGING_ON]


@pytest.mark.parametrize("where", ["parent", "delta"])
async def test_a_recipe_that_launches_with_another_value_is_refused_and_the_reason_names_the_variable(
    tmp_path: Path, where: str
) -> None:
    parent = recipe_with(launch=LaunchConfig(envs={ENV: "0"} if where == "parent" else {}))
    change = RecipeDelta(config=ConfigDelta(envs={ENV: "0"} if where == "delta" else {}, server_args="--x 1"))
    rig = build(tmp_path, repo=FakeRepo(parent), planner=_planner(tmp_path))
    with pytest.raises(RequestRejected) as refused:
        await rig.toolbox.submit_test(req(change=change))
    assert refused.value.code == "delta_not_applicable"
    assert f"{ENV}=0" in refused.value.reason and "every measured point runs with" in refused.value.reason
    assert rig.notebook._state().requests == {} and rig.executor.jobs == []


async def test_a_table_no_hit_names_is_not_exercised_under_the_planner_and_not_observable_without_it(
    tmp_path: Path,
) -> None:
    """A planned test's points ask for hits, so a log with none says "not exercised"; a point that did not ask cannot."""
    session = Session(tmp_path)
    artifact = session.put(csv_text(B01_NEW))
    delta = RecipeDelta(add_kernels=(artifact,))
    accepted = _agentx_planner(tmp_path).plan(req(change=delta), recipe_with())
    session.log(MISS)

    (planned,) = read_table_hits(accepted.environment, accepted.points, session.output)
    (unasked,) = read_table_hits(accepted.environment, (session.point(),), session.output)

    assert (planned.status, planned.rows, planned.rows_hit) == (TABLE_NOT_EXERCISED, 4, 0), planned
    assert planned.reason == "aiter logged no hit at all"
    assert unasked.status == TABLE_NOT_OBSERVABLE and "logs a hit only under" in unasked.reason


async def test_a_table_a_hit_names_is_exercised_under_the_planner(tmp_path: Path) -> None:
    session = Session(tmp_path)
    artifact = session.put(csv_text(B01_NEW))
    accepted = _agentx_planner(tmp_path).plan(req(change=RecipeDelta(add_kernels=(artifact,))), recipe_with())
    session.log(MISS, a8w8_hit(ROW_16_34816, MERGED, asked_m=12))

    (planned,) = read_table_hits(accepted.environment, accepted.points, session.output)

    assert (planned.status, planned.rows_hit, planned.hit_lines) == (TABLE_EXERCISED, 1, 1), planned
