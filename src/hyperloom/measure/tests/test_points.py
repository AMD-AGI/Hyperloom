# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, KernelKind, RecipeDelta
from hyperloom.inference_optimizer.cli.benchmark_config import AgentXWorkload, write_agentx_benchmark
from hyperloom.measure.compile import (
    BaseImageMismatchError,
    EnvironmentCompiler,
    LocalBlobStore,
    PackageBase,
    compile_environment,
)
from hyperloom.inference_optimizer.session.paths import asset_root
from hyperloom.measure.points import (
    AGENTX_PROFILE_STEPS,
    CANONICAL_FILE,
    PointError,
    accept_measurement,
    default_canonical,
    generate_points,
    generate_profile_point,
)
from hyperloom.measure.points import _limit_agentx_capture
from hyperloom.measure.surface import SurfaceViolation
from hyperloom.measure.tests._recipes import IMAGE, recipe_with, sha
from hyperloom.orchestrator.actions.executors._workload_envs import (
    BenchmarkTarget,
    LaunchLayer,
    default_baseline_config,
    materialize_benchmark_config,
    materialize_config_with_envs,
)
from hyperloom.orchestrator.actions.executors.magpie_agentx import enable_profile

SERVER_ARGS = "--max-running-requests 77 --enable-dp-attention"
#: The scheduler watchdog the main flow gives an sglang AgentX launch that sets none.
WATCHDOG = ["--watchdog-timeout", "1800"]


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
def model(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text("{}")
    return path


@pytest.fixture()
def canonical(tmp_path, model):
    return default_canonical(
        tmp_path / "canonical", primary_concurrency=32, config_path=default_baseline_config(), model_path=str(model)
    )


@pytest.fixture()
def blobs(tmp_path):
    return LocalBlobStore(tmp_path / "blobs")


def agentx_source(path: Path, conc: int) -> Path:
    """The benchmark YAML an AgentX session at ``conc`` binds, written as ``hyperloom optimize`` writes it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_agentx_benchmark(
        path,
        AgentXWorkload(framework="sglang", model="Qwen/Qwen3-8B", precision="fp8", tp=2, ep=1, conc=conc),
    )
    return path


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


def test_a_profile_point_replays_graphs_node_by_node_so_the_profiler_records_every_replay(canonical, blobs, tmp_path):
    (profile,) = generate_profile_point(environment(blobs, tmp_path), canonical, tmp_path / "profile")
    (normal,) = generate_points(environment(blobs, tmp_path), canonical, tmp_path / "normal")

    assert load(profile)["benchmark"]["envs"]["DEBUG_CLR_GRAPH_PACKET_CAPTURE"] == "0"
    assert "DEBUG_CLR_GRAPH_PACKET_CAPTURE" not in load(normal)["benchmark"]["envs"], (
        "a measured point replays as shipped"
    )


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
def test_every_shipped_profile_configuration_replays_graphs_node_by_node_and_no_baseline_does(framework):
    """The setting is the HIP runtime's, so the vLLM and ATOM profiles carry it as the SGLang one does."""

    configs = asset_root() / "assets" / "configs"
    profile = load(configs / f"profile_{framework}.yaml")["benchmark"]["envs"]
    baseline = load(configs / f"baseline_{framework}.yaml")["benchmark"]["envs"]

    assert profile["DEBUG_CLR_GRAPH_PACKET_CAPTURE"] == "0"
    assert "DEBUG_CLR_GRAPH_PACKET_CAPTURE" not in baseline


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


# --- AgentX points -------------------------------------------------------------------------------------------------


@pytest.fixture()
def agentx_canonical(tmp_path, model):
    return default_canonical(
        tmp_path / "canonical",
        primary_concurrency=32,
        config_path=agentx_source(tmp_path / "bench" / "benchmark.yaml", 32),
        model_path=str(model),
    )


def test_an_agentx_canonical_is_what_the_main_flow_renders_for_the_workload(tmp_path, model, agentx_canonical):
    source = agentx_source(tmp_path / "single" / "benchmark.yaml", 32)

    main_flow = materialize_benchmark_config(source, tmp_path / "main", target=BenchmarkTarget(model_path=str(model)))

    rendered = load(tmp_path / "canonical" / "canonical.yaml")
    assert CANONICAL_FILE == "canonical.yaml", "the name callers read the rendered canonical by"
    assert rendered == load(main_flow)
    bench = rendered["benchmark"]
    assert bench["envs"]["RUN_EVAL"] == "false"
    assert not {"benchmark_script", "workload_spec"} & bench.keys()
    assert agentx_canonical.surface.fields["benchmark.benchmark_script"] is None
    assert json.loads(agentx_canonical.surface.fields["model_files"]) == {"config.json": sha(b"{}")}


@pytest.mark.parametrize("level", [8, 64])
def test_an_agentx_point_is_what_the_main_flow_hands_magpie_for_the_recipe_and_concurrency(
    agentx_canonical, blobs, tmp_path, model, level
):
    delta = RecipeDelta(config=ConfigDelta(server_args=SERVER_ARGS, envs={"HSA_FORCE_FINE_GRAIN_PCIE": "1"}))
    env = environment(blobs, tmp_path, delta=delta)
    source = agentx_source(tmp_path / "single" / "benchmark.yaml", level)

    (point,) = generate_points(env, agentx_canonical, tmp_path / "points", concurrency=(level,))

    main_flow = materialize_benchmark_config(
        source,
        tmp_path / "main",
        launch=LaunchLayer(SERVER_ARGS, {"HSA_FORCE_FINE_GRAIN_PCIE": "1", "RUN_EVAL": "false"}),
        target=BenchmarkTarget(model_path=str(model)),
    )
    rendered = load(point)
    assert rendered == load(main_flow)
    envs = rendered["benchmark"]["envs"]
    assert (envs["CONC"], envs["RUN_EVAL"]) == (level, "false")
    assert not {"NUM_PROMPTS", "NUM_WARMUPS", "EXTRA_SGLANG_ARGS", "HSA_FORCE_FINE_GRAIN_PCIE"} & envs.keys()
    launch = rendered["benchmark"]["agentx"]["launch_overrides"]
    assert launch["append_args"] == [*SERVER_ARGS.split(), *WATCHDOG]
    assert launch["env"] == {"HSA_FORCE_FINE_GRAIN_PCIE": "1"}


#: An option of the launch Magpie resolves from the InferenceX recipe, which the recipe's own arguments lack.
RESOLVED_OPTION = "--enable-linear-replayssm-spec"


def test_an_agentx_point_removes_the_options_of_the_resolved_launch_the_recipe_removes(
    agentx_canonical, blobs, tmp_path
):
    delta = RecipeDelta(config=ConfigDelta(server_args=SERVER_ARGS, remove_args=(RESOLVED_OPTION,)))
    env = environment(blobs, tmp_path, delta=delta)

    (point,) = generate_points(env, agentx_canonical, tmp_path / "points", concurrency=(8,))
    (profile,) = generate_profile_point(env, agentx_canonical, tmp_path / "profile")

    for config in (point, profile):
        assert load(config)["benchmark"]["agentx"]["launch_overrides"]["remove_args"] == [RESOLVED_OPTION]


def test_a_point_that_resolves_no_launch_of_its_own_refuses_a_removal_it_cannot_make(canonical, blobs, tmp_path):
    delta = RecipeDelta(config=ConfigDelta(remove_args=(RESOLVED_OPTION,)))
    env = environment(blobs, tmp_path, delta=delta)

    with pytest.raises(PointError, match=f"only a Magpie AgentX point .* so {RESOLVED_OPTION} would stay"):
        generate_points(env, canonical, tmp_path / "points")


def test_an_agentx_workload_has_no_accuracy_test(agentx_canonical, blobs, tmp_path):
    with pytest.raises(PointError, match="no accuracy test"):
        generate_points(environment(blobs, tmp_path), agentx_canonical, tmp_path / "points", accuracy=True)
    assert not (tmp_path / "points").exists()


@pytest.mark.parametrize("name", ["LD_PRELOAD", "PYTHONPATH"])
def test_an_agentx_launch_variable_a_variant_may_not_set_is_refused_as_on_the_inferencex_path(
    agentx_canonical, blobs, tmp_path, name
):
    delta = RecipeDelta(config=ConfigDelta(envs={name: "/x.so"}))
    env = environment(blobs, tmp_path, delta=delta)

    with pytest.raises(PointError, match=f"dropped or changed launch variables {name}"):
        generate_points(env, agentx_canonical, tmp_path / "points")


def test_an_agentx_profile_point_is_the_main_flows_agentx_profile_of_the_recipe(
    agentx_canonical, blobs, tmp_path, model
):
    env = environment(blobs, tmp_path, delta=RecipeDelta(config=ConfigDelta(server_args=SERVER_ARGS)))
    source = agentx_source(tmp_path / "single" / "benchmark.yaml", 32)

    (point,) = generate_profile_point(env, agentx_canonical, tmp_path / "points")

    main_flow = materialize_benchmark_config(
        source,
        tmp_path / "main",
        launch=LaunchLayer(SERVER_ARGS, {"RUN_EVAL": "false"}),
        target=BenchmarkTarget(model_path=str(model)),
    )
    _limit_agentx_capture(main_flow)  # the one difference: the window is sized to the result's cap
    enable_profile(main_flow)
    rendered = load(point)
    assert rendered == load(main_flow)
    assert rendered["benchmark"]["profiler"]["torch_profiler"]["enabled"] is True
    assert rendered["benchmark"]["profiler"]["torch_profiler"]["num_steps"] == AGENTX_PROFILE_STEPS
    assert "PROFILE" not in rendered["benchmark"]["envs"]
    assert rendered["benchmark"]["agentx"]["launch_overrides"]["append_args"] == [*SERVER_ARGS.split(), *WATCHDOG]


def test_an_agentx_profile_captures_half_the_main_flows_steps_unless_its_point_names_a_count(tmp_path):
    named, unnamed = tmp_path / "named.yaml", tmp_path / "unnamed.yaml"
    named.write_text(
        yaml.safe_dump({"benchmark": {"profiler": {"torch_profiler": {"num_steps": 64}}}}), encoding="utf-8"
    )
    unnamed.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang"}}), encoding="utf-8")

    _limit_agentx_capture(named)
    _limit_agentx_capture(unnamed)

    assert load(named)["benchmark"]["profiler"]["torch_profiler"]["num_steps"] == 64
    assert load(unnamed)["benchmark"]["profiler"]["torch_profiler"]["num_steps"] == AGENTX_PROFILE_STEPS == 256 // 2


def test_acceptance_generates_the_agentx_profile_point_for_a_profile_request(agentx_canonical, blobs, tmp_path):
    compiler = EnvironmentCompiler(blobs, IMAGE, locator=lambda _l: PackageBase("sglang", tmp_path, None, None))
    request = TestRequest(
        kind=TestKind.PROFILE, purpose=Purpose.PROFILE, branch_id="b1", parent_commit="c" * 40, delta=RecipeDelta()
    )

    accepted = accept_measurement(request, recipe_with(), compiler, agentx_canonical, tmp_path / "points")

    (point,) = accepted.points
    assert load(point)["benchmark"]["profiler"]["torch_profiler"]["enabled"] is True


def test_an_inferencex_point_is_what_the_inferencex_renderer_hands_magpie(canonical, blobs, tmp_path, model):
    delta = RecipeDelta(config=ConfigDelta(server_args=SERVER_ARGS, envs={"HSA_FORCE_FINE_GRAIN_PCIE": "1"}))
    env = environment(blobs, tmp_path, delta=delta)

    (point,) = generate_points(env, canonical, tmp_path / "points", concurrency=(8,), accuracy=True)

    expected = materialize_config_with_envs(
        default_baseline_config(),
        tmp_path / "direct",
        launch=LaunchLayer(SERVER_ARGS, {"HSA_FORCE_FINE_GRAIN_PCIE": "1", "CONC": "8", "RUN_EVAL": "true"}),
        target=BenchmarkTarget(model_path=str(model)),
    )
    rendered = load(point)
    assert rendered == load(expected)
    assert rendered["benchmark"]["envs"]["RUN_EVAL"] == "true" and "NUM_PROMPTS" in rendered["benchmark"]["envs"]


# --- the launcher on a cold InferenceX checkout ---------------------------------------------------------------------

LAUNCHER = "sglang_mi355x.sh"
MAGPIE_LAUNCHER = b"magpie launcher\n"


@pytest.fixture()
def magpie(tmp_path, monkeypatch):
    """The Magpie the session runs, which ships the launcher; the process names no checkout of its own."""

    shipped = tmp_path / "magpie" / "Magpie" / "scripts" / "benchmark"
    shipped.mkdir(parents=True)
    (shipped / LAUNCHER).write_bytes(MAGPIE_LAUNCHER)
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path / "magpie"))
    for name in ("INFERENCEX_PATH", "HYPERLOOM_BENCHMARK_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    return shipped / LAUNCHER


@pytest.fixture()
def cold_checkout(tmp_path):
    """A fresh InferenceX cache: the pinned commit does not track the launcher, so it is not in the tree yet."""

    benchmarks = tmp_path / "InferenceX" / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text("# library\n")
    return benchmarks.parent


def cold_canonical(tmp_path, model, checkout):
    """The canonical measurement as a session takes it at start, on the runner type that names the launcher."""

    return default_canonical(
        tmp_path / "canonical",
        primary_concurrency=32,
        config_path=default_baseline_config(),
        model_path=str(model),
        gpu_type="mi355x",
        inferencex_path=str(checkout),
    )


def magpie_first_benchmark(launcher: Path, checkout: Path) -> Path:
    """What Magpie does when its first benchmark starts: it copies the launchers it ships into the checkout."""

    placed = checkout / "benchmarks" / LAUNCHER
    shutil.copy2(launcher, placed)
    return placed


def test_a_baseline_test_on_a_cold_checkout_has_the_surface_the_canonical_took_at_session_start(
    tmp_path, model, magpie, cold_checkout, blobs
):
    canonical = cold_canonical(tmp_path, model, cold_checkout)
    magpie_first_benchmark(magpie, cold_checkout)

    (point,) = generate_points(environment(blobs, tmp_path), canonical, tmp_path / "points")

    assert load(point)["benchmark"]["benchmark_script"] == LAUNCHER
    assert canonical.surface.fields["benchmark_script_sha256"] == sha(MAGPIE_LAUNCHER)


def test_the_launcher_a_candidate_changed_in_the_checkout_is_still_a_surface_mismatch(
    tmp_path, model, magpie, cold_checkout, blobs
):
    canonical = cold_canonical(tmp_path, model, cold_checkout)
    placed = magpie_first_benchmark(magpie, cold_checkout)
    placed.write_bytes(b"magpie launcher\nexport SGLANG_FAST=1\n")

    with pytest.raises(SurfaceViolation, match=rf"benchmark_script_sha256: canonical '{sha(MAGPIE_LAUNCHER)}', test '"):
        generate_points(environment(blobs, tmp_path), canonical, tmp_path / "points")


def test_a_launcher_that_is_nowhere_refuses_the_canonical_instead_of_recording_none(
    tmp_path, model, magpie, cold_checkout
):
    magpie.unlink()

    with pytest.raises(PointError, match=rf"{LAUNCHER}.*{cold_checkout.name}.*no such file"):
        cold_canonical(tmp_path, model, cold_checkout)


def test_a_launcher_the_pinned_commit_tracks_is_hashed_as_it_is_when_magpie_ships_none(
    tmp_path, model, magpie, cold_checkout
):
    magpie.unlink()
    (cold_checkout / "benchmarks" / LAUNCHER).write_bytes(b"tracked launcher\n")

    canonical = cold_canonical(tmp_path, model, cold_checkout)

    assert canonical.surface.fields["benchmark_script_sha256"] == sha(b"tracked launcher\n")


def test_the_bypass_backend_never_has_magpie_copy_a_launcher_so_the_checkout_is_left_as_it_is(
    tmp_path, model, magpie, cold_checkout, monkeypatch
):
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")

    canonical = cold_canonical(tmp_path, model, cold_checkout)

    assert canonical.surface.fields["benchmark_script_sha256"] is None
    assert not (cold_checkout / "benchmarks" / LAUNCHER).exists()


def test_a_checkout_with_nowhere_to_put_the_launcher_is_left_to_the_host_check(tmp_path, model, magpie):
    """A test cannot see a launcher there either, and the host refuses a tree without the benchmark library."""

    canonical = cold_canonical(tmp_path, model, tmp_path / "not-a-checkout")

    assert canonical.surface.fields["benchmark_script_sha256"] is None
