# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, KernelKind, RecipeDelta
from hyperloom.measure.compile import (
    BaseImageMismatchError,
    EnvironmentCompiler,
    LocalBlobStore,
    PackageBase,
    compile_environment,
)
from hyperloom.measure.points import (
    PointError,
    accept_measurement,
    default_canonical,
    generate_points,
    generate_profile_point,
)
from hyperloom.measure.surface import SurfaceViolation
from hyperloom.measure.tests._recipes import IMAGE, recipe_with, sha
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config

SERVER_ARGS = "--max-running-requests 77 --enable-dp-attention"


@pytest.fixture(autouse=True)
def clean_process_env(monkeypatch, tmp_path):
    """The renderer reads the workload from the process environment; start from a known, empty one."""

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


@pytest.fixture()
def canonical(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    return default_canonical(
        tmp_path / "canonical", primary_concurrency=32, config_path=default_baseline_config(), model_path=str(model)
    )


@pytest.fixture()
def blobs(tmp_path):
    return LocalBlobStore(tmp_path / "blobs")


def environment(blobs, tmp_path, recipe=None, delta=None):
    base = PackageBase("sglang", tmp_path, None, None)
    return compile_environment(
        recipe or recipe_with(), delta or RecipeDelta(), blobs=blobs, active_image_digest=IMAGE, locator=lambda _l: base
    )


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_one_point_per_concurrency_with_run_eval_and_launch_from_the_request(canonical, blobs, tmp_path):
    delta = RecipeDelta(config=ConfigDelta(server_args=SERVER_ARGS, envs={"HSA_FORCE_FINE_GRAIN_PCIE": "1"}))
    env = environment(blobs, tmp_path, delta=delta)

    points = generate_points(env, canonical, tmp_path / "points", concurrency=(8, 64), accuracy=True)

    configs = [load(path) for path in points]
    assert [c["benchmark"]["envs"]["CONC"] for c in configs] == ["8", "64"]
    assert {c["benchmark"]["envs"]["RUN_EVAL"] for c in configs} == {"true"}
    envs = configs[0]["benchmark"]["envs"]
    assert envs["HSA_FORCE_FINE_GRAIN_PCIE"] == "1"
    assert "--max-running-requests 77" in envs["EXTRA_SGLANG_ARGS"]


def test_run_eval_follows_the_request_and_concurrency_defaults_to_the_primary_point(canonical, blobs, tmp_path):
    (point,) = generate_points(environment(blobs, tmp_path), canonical, tmp_path / "points")

    envs = load(point)["benchmark"]["envs"]
    assert (envs["RUN_EVAL"], envs["CONC"]) == ("false", "32")


def test_kernel_table_reaches_the_yaml_as_an_attachment_placeholder(canonical, blobs, tmp_path):
    blobs.put(b"table")
    artifact = KernelArtifact(
        KernelKind.GEMM,
        "bf16",
        "kernel/g.csv",
        sha(b"table"),
        {"load": {"via": "env_file", "env": "AITER_CONFIG_GEMM_BF16"}},
    )
    env = environment(blobs, tmp_path, delta=RecipeDelta(add_kernels=(artifact,)))

    (point,) = generate_points(env, canonical, tmp_path / "points")

    value = load(point)["benchmark"]["envs"]["AITER_CONFIG_GEMM_BF16"]
    assert value == "${HL_ATTACH:" + env.attachments[0].id + "}"


@pytest.mark.parametrize("name", ["ISL", "MAGPIE_EVAL_LIMIT", "RUN_EVAL", "NUM_PROMPTS"])
def test_recipe_that_sets_a_workload_or_eval_variable_is_refused_before_rendering(canonical, blobs, tmp_path, name):
    delta = RecipeDelta(config=ConfigDelta(envs={name: "1"}))
    env = environment(blobs, tmp_path, delta=delta)

    with pytest.raises(SurfaceViolation, match=name):
        generate_points(env, canonical, tmp_path / "points")
    assert not list((tmp_path / "points").glob("*.yaml"))


@pytest.mark.parametrize(("name", "value"), [("ISL", "16"), ("MAGPIE_EVAL_LIMIT", "5"), ("OSL", "8")])
def test_a_point_whose_surface_drifted_from_the_canonical_one_is_refused(
    canonical, blobs, tmp_path, monkeypatch, name, value
):
    env = environment(blobs, tmp_path)
    monkeypatch.setenv(name, value)

    with pytest.raises(SurfaceViolation, match="changes the measurement surface"):
        generate_points(env, canonical, tmp_path / "points")


def test_a_launch_variable_the_renderer_drops_is_refused(canonical, blobs, tmp_path):
    delta = RecipeDelta(config=ConfigDelta(envs={"LD_PRELOAD": "/x.so"}))
    env = environment(blobs, tmp_path, delta=delta)

    with pytest.raises(PointError, match="dropped or changed launch variables LD_PRELOAD"):
        generate_points(env, canonical, tmp_path / "points")


@pytest.mark.parametrize("levels", [(0,), (4, 4), (-1,)])
def test_concurrency_must_be_distinct_positive_integers(canonical, blobs, tmp_path, levels):
    with pytest.raises(PointError, match="distinct positive integers"):
        generate_points(environment(blobs, tmp_path), canonical, tmp_path / "points", concurrency=levels)


def request_for(delta: RecipeDelta, **kwargs) -> TestRequest:
    return TestRequest(
        kind=TestKind.TEST, purpose=Purpose.EXPLORE, branch_id="b1", parent_commit="c" * 40, delta=delta, **kwargs
    )


def test_acceptance_compiles_and_generates_points_for_a_request(canonical, blobs, tmp_path):
    compiler = EnvironmentCompiler(blobs, IMAGE, locator=lambda _l: PackageBase("sglang", tmp_path, None, None))
    delta = RecipeDelta(config=ConfigDelta(server_args=SERVER_ARGS))
    request = request_for(delta, concurrency=(16,), accuracy=True)

    accepted = accept_measurement(request, recipe_with(), compiler, canonical, tmp_path / "points")

    assert accepted.environment.recipe.launch.server_args == SERVER_ARGS
    envs = load(accepted.points[0])["benchmark"]["envs"]
    assert (envs["CONC"], envs["RUN_EVAL"]) == ("16", "true")


def test_acceptance_refuses_a_request_on_another_image_before_generating_anything(canonical, blobs, tmp_path):
    compiler = EnvironmentCompiler(blobs, "sha256:" + "e" * 64)

    with pytest.raises(BaseImageMismatchError):
        accept_measurement(request_for(RecipeDelta()), recipe_with(), compiler, canonical, tmp_path / "points")
    assert not (tmp_path / "points").exists()


# --- profile points ------------------------------------------------------------------------------------------------


def test_a_profile_point_carries_the_profiler_settings_and_the_recipes_launch(canonical, blobs, tmp_path):
    delta = RecipeDelta(config=ConfigDelta(server_args=SERVER_ARGS, envs={"HSA_FORCE_FINE_GRAIN_PCIE": "1"}))

    (point,) = generate_profile_point(environment(blobs, tmp_path, delta=delta), canonical, tmp_path / "points")

    config = load(point)
    envs = config["benchmark"]["envs"]
    assert config["benchmark"]["profiler"]["torch_profiler"]["enabled"] is True
    assert envs["PROFILE"] == "1" and json.loads(envs["PROFILE_EXTRA_BODY"])["num_steps"] > 0
    assert "--enable-profile-cuda-graph" in envs["EXTRA_SGLANG_ARGS"]
    assert "--max-running-requests 77" in envs["EXTRA_SGLANG_ARGS"], "the profile runs the candidate's launch"
    assert envs["HSA_FORCE_FINE_GRAIN_PCIE"] == "1"
    assert (envs["CONC"], envs["RUN_EVAL"]) == ("32", "false")


def test_a_normal_point_does_not_turn_the_profiler_on(canonical, blobs, tmp_path):
    (point,) = generate_points(environment(blobs, tmp_path), canonical, tmp_path / "points")

    config = load(point)
    assert "PROFILE" not in config["benchmark"]["envs"]
    assert config["benchmark"].get("profiler", {}).get("torch_profiler", {}).get("enabled") is not True


def test_a_profile_point_may_lower_the_output_length_but_not_change_the_rest_of_the_surface(
    canonical, blobs, tmp_path, monkeypatch
):
    env = environment(blobs, tmp_path)
    monkeypatch.setenv("OSL", "8")
    generate_profile_point(env, canonical, tmp_path / "ok")  # the output length is the profile's own to size
    monkeypatch.setenv("ISL", "16")

    with pytest.raises(SurfaceViolation, match="envs.ISL"):
        generate_profile_point(env, canonical, tmp_path / "bad")


def test_acceptance_generates_the_profile_point_for_a_profile_request(canonical, blobs, tmp_path):
    compiler = EnvironmentCompiler(blobs, IMAGE, locator=lambda _l: PackageBase("sglang", tmp_path, None, None))
    request = TestRequest(
        kind=TestKind.PROFILE, purpose=Purpose.PROFILE, branch_id="b1", parent_commit="c" * 40, delta=RecipeDelta()
    )

    accepted = accept_measurement(request, recipe_with(), compiler, canonical, tmp_path / "points")

    assert len(accepted.points) == 1
    assert load(accepted.points[0])["benchmark"]["envs"]["PROFILE"] == "1"
