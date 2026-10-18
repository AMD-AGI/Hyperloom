# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance through the real hosted-measurement compile, points and surface check."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ComponentChange, ConfigDelta, RecipeDelta
from hyperloom.measure.compile import EnvironmentCompiler, LocalBlobStore, PackageBase
from hyperloom.measure.points import AcceptedMeasurement, default_canonical
from hyperloom.measure.tests._recipes import IMAGE, recipe_with
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
from hyperloom_multibranch.contracts.toolbox import RequestRejected
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
