# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An AgentX point runs the agentx-harness replay on its launch over the session base, golden layer applied.

Every key the Magpie AgentX path read from the point is either carried into the config or refused by name.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hyperloom.common.agentx_workload import CANON_WARMUP_PER_LANE, agentx_profiling_requests
from hyperloom.orchestrator.actions.executors import _suite_config
from hyperloom.orchestrator.actions.executors._suite_config import (
    MEASURED_SEED,
    TOOL_CACHE_ENV,
    WARMUP_SEED,
    SuiteRound,
    run_suite_round,
    suite_config,
    suite_route,
)
from hyperloom.orchestrator.actions.executors.magpie_agentx import LaunchChanges
from hyperloom.orchestrator.actions.executors.tests.test_suite_accuracy_round import _agentx_base, _agentx_point
from hyperloom.orchestrator.actions.executors.tests.test_suite_config import _settings
from hyperloom.orchestrator.state.tests._session_base import freeze_base
from inference_suite.compose import RoundFacts, compose
from inference_suite.contracts import (
    ConfigError,
    GoldenUse,
    LaunchDelta,
    Lifecycle,
    LifecycleMode,
    SuiteConfig,
    WorkloadKind,
)
from inference_suite.contracts.launch import Launch, VersionCheck

_SHA = "0" * 64
_CORPUS = "semianalysis_cc_traces_weka_062126"


def _base():
    """The session base as resolving freezes it for AgentX: its client env names the corpus InferenceX replays."""
    base = _agentx_base()
    return replace(base, client=replace(base.client, env={"WEKA_LOADER_OVERRIDE": _CORPUS}))


def _stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "tools"
    cache.mkdir()
    (cache / "staging_manifest.json").write_text('{"schema": "inference-suite-staging/1", "tools": {}}\n')
    monkeypatch.setenv(TOOL_CACHE_ENV, str(cache))
    monkeypatch.setattr(_suite_config, "_resolve_probe_python", lambda framework: f"/venv/{framework}/python")
    monkeypatch.setattr(_suite_config, "client_only_server_url", lambda: None)


def _point(**changes) -> dict:
    point = _agentx_point(_base())
    bench = point["benchmark"]
    bench["timeout_seconds"] = 7200
    for key, value in changes.items():
        if key == "envs":
            bench["envs"] = {**bench["envs"], **value}
        elif key == "agentx":
            bench["agentx"] = {**bench["agentx"], **value}
        else:
            bench[key] = value
    return point


def test_an_agentx_point_replays_on_its_launch_over_the_base_with_the_golden_layer_applied() -> None:
    base = _base()

    config = suite_config(_point(), base, _settings())

    assert (config.workload, config.fixed, config.eval.enabled) == (WorkloadKind.AGENTX, None, False)
    assert config.server.golden is GoldenUse.APPLY
    assert config.server.model_path == "/models/glm"
    assert config.server.user == LaunchDelta(
        args_add=("--mem-fraction-static", "0.85", "--cuda-graph-max-bs", "64"),
        args_remove=("--kv-cache-dtype",),
        env_set={"AITER_LOG_TUNED_CONFIG": "1"},
        env_unset=("SGLANG_USE_AITER",),
    )
    agentx = config.agentx
    assert (agentx.concurrency, agentx.dataset, agentx.num_prompts) == (8, _CORPUS, agentx_profiling_requests(8))
    assert (agentx.warmup_requests_per_lane, agentx.seed, agentx.warmup_seed) == (CANON_WARMUP_PER_LANE, 0, 0)
    assert agentx.failed_request_threshold == 0.1
    assert agentx.client_env == {"AIPERF_EXPERIMENTAL_FAST": "0"}
    assert not config.profile.enabled
    config.check_base(base)
    launch = compose(config, base, RoundFacts(_SHA, "0.0.0", VersionCheck.NOT_APPLICABLE))
    assert isinstance(launch, Launch), launch
    assert "--speculative-algorithm" in [arg.flag for arg in launch.args]
    assert SuiteConfig.from_document(json.loads(json.dumps(config.to_document()))) == config


def test_the_replay_takes_the_bases_client_knobs_under_the_points() -> None:
    base = _base()
    client = {"AIPERF_WARMUP_REQUESTS_PER_LANE": "4", "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS": "300"}
    base = replace(base, client=replace(base.client, env={**client, "WEKA_LOADER_OVERRIDE": "corpus_a"}))
    point = _point(envs={"AIPERF_WARMUP_REQUESTS_PER_LANE": "2", "NUM_PROMPTS": "640"})

    agentx = suite_config(point, base, _settings()).agentx

    assert (agentx.dataset, agentx.warmup_requests_per_lane, agentx.num_prompts) == ("corpus_a", 2, 640)
    assert agentx.client_env["AIPERF_TRACE_IDLE_GAP_CAP_SECONDS"] == "300"
    assert suite_config(_point(envs={"WEKA_LOADER_OVERRIDE": "corpus_b"}), base, _settings()).agentx.dataset == (
        "corpus_b"
    )


def test_a_replay_whose_base_and_point_name_no_corpus_is_refused() -> None:
    with pytest.raises(ConfigError, match="WEKA_LOADER_OVERRIDE"):
        suite_config(_point(), _agentx_base(), _settings())


def test_the_server_sees_the_points_environment_under_its_launchs() -> None:
    point = _point(envs={"SGLANG_X": "1", "AITER_LOG_TUNED_CONFIG": "0", "SGLANG_USE_AITER": "1", "AIPERF_Y": "2"})

    user = suite_config(point, _base(), _settings()).server.user

    assert user.env_set == {"SGLANG_X": "1", "AITER_LOG_TUNED_CONFIG": "1"}
    assert user.env_unset == ("SGLANG_USE_AITER",)


def test_the_server_runs_on_the_interpreter_and_source_files_the_launch_names() -> None:
    point = _point()
    overrides = point["benchmark"]["agentx"]["launch_overrides"]
    overrides.update(executable="/venv/candidate/bin/python", source_files={"/src/a.py": "a" * 64})
    overrides["absent_source_files"] = ["/src/b.py"]

    server = suite_config(point, _base(), _settings()).server

    assert server.framework_python == "/venv/candidate/bin/python"
    assert (dict(server.source_files), server.absent_source_files) == ({"/src/a.py": "a" * 64}, ("/src/b.py",))


@pytest.mark.parametrize(("warmup", "seed"), [(False, MEASURED_SEED), (True, WARMUP_SEED)])
def test_a_discarded_warmup_replay_draws_its_trajectories_with_the_warmup_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warmup: bool, seed: int
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    freeze_base(session, _base())
    _stage(tmp_path, monkeypatch)
    ran = []
    monkeypatch.setattr(_suite_config, "_run_config", lambda config, lease, on_poll: ran.append(config))
    point = tmp_path / "slot" / "point.yaml"
    point.parent.mkdir()
    point.write_text(yaml.safe_dump(_point()), encoding="utf-8")
    route = suite_route(session, point)
    assert route is not None

    run_suite_round(route, SuiteRound(point, point.parent, Lifecycle(LifecycleMode.OWN), warmup), on_poll=lambda: None)

    (config,) = ran
    assert (config.agentx.seed, config.agentx.warmup_seed) == (seed, seed)


def test_a_server_phase_point_boots_the_launch_and_replays_nothing() -> None:
    point = _point(envs={"MAGPIE_RUN_PHASE": "server"})
    config = suite_config(point, _base(), _settings(LifecycleMode.BOOT))

    assert (config.workload, config.agentx, config.profile.enabled) == (WorkloadKind.NONE, None, False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"agentx": {"mode": "fast"}}, "fast mode"),
        ({"agentx": {"failed_request_threshold": 0.2}}, "failed_request_threshold 0.2"),
        ({"agentx": {"concurrency": 16}}, "concurrency 16 is not its CONC"),
        ({"agentx": {"recipe": "glm5-fp8"}}, "agentx recipe"),
        ({"envs": {"DURATION": "3600"}}, "DURATION"),
        ({"envs": {"AGENTX_EXTRA": "1"}}, "AGENTX_EXTRA"),
        ({"envs": {"RUN_EVAL": "true"}}, "accuracy is a round of its own"),
        ({"envs": {"EP_SIZE": "8"}}, "EP_SIZE 8, its base runs 1"),
        ({"envs": {"AIPERF_EXPERIMENTAL_FAST": "1"}}, "AIPERF_EXPERIMENTAL_FAST=0"),
        ({"envs": {"BENCHMARK_BASE_URL": "http://head:8888"}}, "serves its own launch"),
    ],
)
def test_an_agentx_setting_no_round_carries_is_refused(changes: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        suite_config(_point(**changes), _base(), _settings())


def test_a_launch_override_magpie_did_not_define_is_refused() -> None:
    point = _point()
    point["benchmark"]["agentx"]["launch_overrides"]["prepend_args"] = ["--x"]

    with pytest.raises(ConfigError, match="launch_overrides prepend_args"):
        suite_config(point, _base(), _settings())


def _profiled(framework: str = "sglang", **settings) -> dict:
    point = _point(framework=framework)
    window = {"enabled": True, "num_steps": 256, "num_profiles": 1, "start_seconds": 300, "interval_seconds": 900}
    point["benchmark"]["profiler"] = {"torch_profiler": {**window, "detailed_annotations": True, **settings}}
    return point


def test_an_agentx_profile_point_opens_its_window_on_the_replays_measured_phase() -> None:
    config = suite_config(_profiled(), _base(), _settings())

    profile = config.profile
    assert (profile.enabled, profile.delay_iters, profile.max_iters) == (True, 0, 256)
    assert (profile.agentx_open_after_sec, profile.agentx_duration_sec) == (300.0, 300.0)
    assert profile.start_body == {"detailed_annotations": True}
    assert config.workload is WorkloadKind.AGENTX


_SGLANG_ANNOTATION_ENV = {
    "SGLANG_PROFILE_WITH_STACK": "True",
    "SGLANG_PROFILE_RECORD_SHAPES": "True",
    "SGLANG_GRAPH_BATCH_CAPTURE": "True",
}


def test_an_annotated_sglang_agentx_profile_records_shapes_stacks_and_graph_replays() -> None:
    user = suite_config(_profiled(), _base(), _settings()).server.user

    assert _SGLANG_ANNOTATION_ENV.items() <= user.env_set.items()
    assert user.args_add[-1] == "--enable-profile-cuda-graph"


def test_an_eager_sglang_agentx_profile_profiles_no_graph_capture() -> None:
    point = _profiled()
    point["benchmark"]["agentx"]["launch_overrides"]["append_args"].append("--disable-cuda-graph")

    user = suite_config(point, _base(), _settings()).server.user

    assert _SGLANG_ANNOTATION_ENV.items() <= user.env_set.items()
    assert "--enable-profile-cuda-graph" not in user.args_add


def test_an_unannotated_or_unprofiled_sglang_round_adds_no_instrumentation() -> None:
    for point in (_profiled(detailed_annotations=False), _point()):
        user = suite_config(point, _base(), _settings()).server.user

        assert not _SGLANG_ANNOTATION_ENV.keys() & user.env_set.keys()
        assert "--enable-profile-cuda-graph" not in user.args_add


def test_a_vllm_agentx_profile_passes_the_profiler_fields_its_annotations_need() -> None:
    base = _base()
    base = replace(base, framework="vllm", query=replace(base.query, framework="vllm"), entrypoint=("vllm", "serve"))
    point = _profiled("vllm", capture_timeout_seconds=600)
    point["benchmark"]["agentx"]["launch_overrides"].update(append_args=list(base.args), env={})

    config = suite_config(point, base, _settings())

    args = config.server.user.args_add
    for field in ("ignore_frontend", "capture_torch_profiler", "detailed_trace_annotation"):
        assert args[args.index(f"--profiler-config.{field}") + 1] == "True"
    assert args[args.index("--profiler-config.torch_profiler_dump_cuda_time_total") + 1] == "False"
    assert config.profile.start_body is None
    assert config.profile.agentx_duration_sec == 600.0


@pytest.mark.parametrize(
    ("settings", "message"),
    [({"num_profiles": 2}, "2 captures"), ({"flush_timeout_seconds": 1800}, "flush_timeout_seconds")],
)
def test_an_agentx_profile_setting_no_round_carries_is_refused(settings: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        suite_config(_profiled(**settings), _base(), _settings())


def test_the_accuracy_round_serves_the_same_launch_as_the_measured_round() -> None:
    from hyperloom.orchestrator.actions.executors._suite_config import accuracy_config

    point = _point()
    point["benchmark"]["agentx"]["launch_overrides"]["executable"] = "/venv/candidate/bin/python"
    base = _base()

    measured = suite_config(point, base, _settings()).server
    accuracy = accuracy_config(point, base, _settings(), 900.0).server

    assert (accuracy.user, accuracy.framework_python) == (measured.user, measured.framework_python)
    assert (measured.golden, accuracy.golden) == (GoldenUse.APPLY, GoldenUse.EXCLUDE)


def test_a_slot_running_the_base_unchanged_replays_it_with_no_layer_of_its_own() -> None:
    point = _agentx_point(_base(), LaunchChanges())
    point["benchmark"]["timeout_seconds"] = 7200

    assert suite_config(point, _base(), _settings()).server.user == LaunchDelta()
