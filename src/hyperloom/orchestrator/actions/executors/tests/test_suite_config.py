# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A point YAML translated into a suite config: the server it launches, the workload it sends and what it refuses.

The launch the translation must produce is spelled out per framework: the base's defaults with the point's
``EXTRA_<FW>_ARGS`` on top, its knobs as config fields rather than server environment, and the GPUs masked by
``HIP_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._suite.config import RoundSettings, fixed_workload, suite_config
from hyperloom.orchestrator.actions.executors._suite.rounds import point_server
from hyperloom.orchestrator.actions.executors._workload_envs import (
    BenchmarkTarget,
    LaunchLayer,
    materialize_config_with_envs,
)
from hyperloom.orchestrator.state.tests._session_base import freeze_base
from inference_suite import LaunchQuery, SourceSnapshot, resolve_base
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, normalize_argv
from inference_suite.compose import RoundFacts, compose
from inference_suite.contracts import (
    BaseLaunch,
    ConfigError,
    GpuTelemetry,
    Lifecycle,
    LifecycleMode,
    RunPaths,
    SuiteConfig,
    Telemetry,
    Workload,
    WorkloadKind,
)
from inference_suite.contracts.launch import Launch, VersionCheck
from inference_suite.recipe.resolve import UnreadSource

_MODEL = "/models/Qwen3-8B"
_MODEL_ID = "Qwen/Qwen3-8B"
_SHA = "0" * 64
_EXTRA = {
    "sglang": ("EXTRA_SGLANG_ARGS", "--context-length 8192 --cuda-graph-max-bs 64"),
    "vllm": ("EXTRA_VLLM_ARGS", "--max-num-seqs 64 --gpu-memory-utilization 0.9"),
    "atom": ("EXTRA_ATOM_ARGS", "--trust-remote-code"),
}
_PYTHON = "/opt/venv/bin/python"
_SGLANG_ENTRY = (_PYTHON, "-m", "sglang.launch_server")
_VLLM_ENTRY = (_PYTHON, "-m", "vllm.entrypoints.cli.main", "serve")
_ATOM_ENTRY = (_PYTHON, "-m", "atom.entrypoints.openai_server")
_SERVED = ("--model", _MODEL, "--host", "127.0.0.1", "--port", "8888", "--served-model-name", _MODEL)
_SGLANG_ARGV = (
    *("--trust-remote-code", "--mem-fraction-static", "0.8", "--tensor-parallel-size", "2"),
    *("--context-length", "8192", "--cuda-graph-max-bs", "64", "--model-path", _MODEL, "--host", "127.0.0.1"),
    *("--port", "8888", "--served-model-name", _MODEL, "--enable-metrics"),
)
_VLLM_ARGV = (
    *("--trust-remote-code", "--tensor-parallel-size", "2", "--max-model-len", "8192", "--max-num-seqs", "64"),
    *("--gpu-memory-utilization", "0.9", *_SERVED),
)
_ATOM_ARGV = (
    *("-tp", "2", "--max-model-len", "8192", "--trust-remote-code", "--model", _MODEL, "--host", "127.0.0.1"),
    *("--server-port", "8888"),
)
#: The server environment every framework gets from ``_point``: its non-knob variables and the round's GPU masks.
_POINT_ENV = {
    "PATH": "/opt/venv/bin:/usr/bin",
    "AITER_LOG_TUNED_CONFIG": "1",
    "ROCR_VISIBLE_DEVICES": "4,5",
    "HIP_VISIBLE_DEVICES": "0,1",
}
#: (server command under the framework python, argv after it, server environment) per framework and board.
_LAUNCH = {
    ("sglang", "mi300x"): (_SGLANG_ENTRY, _SGLANG_ARGV, {"SGLANG_USE_AITER": "1", "SGLANG_AITER_MLA_PERSIST": "1"}),
    ("sglang", "mi355x"): (_SGLANG_ENTRY, _SGLANG_ARGV, {"SGLANG_USE_AITER": "1"}),
    ("vllm", "mi300x"): (_VLLM_ENTRY, _VLLM_ARGV, {"VLLM_ROCM_USE_AITER": "1"}),
    ("vllm", "mi355x"): (_VLLM_ENTRY, _VLLM_ARGV, {"VLLM_ROCM_USE_AITER": "1"}),
    ("atom", "mi300x"): (_ATOM_ENTRY, _ATOM_ARGV, {}),
    ("atom", "mi355x"): (_ATOM_ENTRY, _ATOM_ARGV, {}),
}


def _point_envs(framework: str) -> dict[str, str]:
    extra_name, extra = _EXTRA[framework]
    return {
        "TP": "2", "CONC": "64", "ISL": "1024", "OSL": "1024", "RANDOM_RANGE_RATIO": "1", "MAX_MODEL_LEN": "8192",
        "NUM_PROMPTS": "640", "NUM_WARMUPS": "8", "RUN_EVAL": "true", "ROCR_VISIBLE_DEVICES": "4,5",
        "PORT": "8888", "BENCH_TRUST_REMOTE_CODE": "1", "PATH": "/opt/venv/bin:/usr/bin",
        "AITER_LOG_TUNED_CONFIG": "1", extra_name: extra,
    }  # fmt: skip


def _point(framework: str, **envs: str) -> dict:
    return {
        "benchmark": {
            "framework": framework,
            "model": _MODEL,
            "envs": {**_point_envs(framework), **envs},
            "timeout_seconds": 7200,
            "profiler": {"torch_profiler": {"enabled": False}},
        }
    }


def _seeded(point: dict, base: BaseLaunch) -> dict:
    """``point`` as the main flow writes it: the base's args below its own, the base's variables under its own."""
    envs = point["benchmark"]["envs"]
    name = _EXTRA[point["benchmark"]["framework"]][0]
    envs[name] = " ".join((*base.args, envs.get(name, "")))
    for variable, value in base.env.items():
        envs.setdefault(variable, value)
    return point


def _base(framework: str, board: str = "mi355x", tp: int = 2) -> BaseLaunch:
    snapshot = SourceSnapshot(None, 1_793_000_000, unread=UnreadSource("InferenceX", "offline"))
    query = LaunchQuery(_MODEL_ID, framework, board, Workload.FIXED, (64,), isl=1024, osl=1024, tp=tp)
    base = resolve_base(query, snapshot, gpus=8)
    assert isinstance(base, BaseLaunch), base
    return base


def _settings(mode: LifecycleMode = LifecycleMode.OWN) -> RoundSettings:
    return RoundSettings(
        base_path="/session/base_launch.json",
        paths=RunPaths("/session/runs/r0", "/tools", _SHA),
        framework_python=_PYTHON,
        lifecycle=Lifecycle(mode, None if mode is LifecycleMode.OWN else "/session/server"),
        telemetry=Telemetry(GpuTelemetry(True, 1.0), kv=True),
        silence_sec=600,
        env_inherit=("HOME",),
    )


def _launch(config: SuiteConfig, base: BaseLaunch) -> Launch:
    launch = compose(config, base, RoundFacts(_SHA, "0.0.0", VersionCheck.NOT_APPLICABLE))
    assert isinstance(launch, Launch), launch
    return launch


def _flags(argv: tuple[str, ...]) -> set[tuple[str, tuple[str, ...]]]:
    """Each flag the server applies -- the last of its spellings -- with its values."""
    tokens = normalize_argv(argv)
    flags: dict[str, tuple[str, ...]] = {}
    starts = [index for index, token in enumerate(tokens) if argv_flag_occurrences([token])]
    for start, end in zip(starts, [*starts[1:], len(tokens)], strict=True):
        flags[canonical_flag_name(tokens[start])] = tokens[start + 1 : end]
    return set(flags.items())


@pytest.mark.parametrize(("framework", "board"), sorted(_LAUNCH))
def test_the_translated_launch_is_the_base_under_the_points_server_layer(framework: str, board: str) -> None:
    base = _base(framework, board)
    launch = _launch(suite_config(_seeded(_point(framework), base), base, _settings()), base)
    entry, argv, env = _LAUNCH[(framework, board)]

    assert launch.entrypoint == entry
    assert _flags(launch.argv[len(entry) :]) == _flags(argv)
    assert dict(launch.env) == {**env, **_POINT_ENV}


def test_the_point_knobs_become_the_config_fields() -> None:
    config = suite_config(
        _point("sglang", HYPERLOOM_EVAL_PROBE_MIN_SAMPLES="64", HYPERLOOM_EVAL_LIMIT="200"),
        _base("sglang"),
        _settings(),
    )

    assert config.workload is WorkloadKind.FIXED
    fixed = config.fixed
    assert (fixed.concurrency, fixed.isl, fixed.osl, fixed.random_range_ratio) == (64, 1024, 1024, 1.0)
    assert (fixed.num_prompts, fixed.num_warmups, fixed.seed, fixed.warmup_seed) == (640, 8, 0, 1)
    assert (fixed.tokenizer_mode, fixed.trust_remote_code) == ("auto", True)
    assert (config.eval.enabled, config.eval.limit, config.eval.concurrency, config.eval.num_fewshot) == (
        True,
        200,
        64,
        5,
    )
    assert (config.eval.probe.enabled, config.eval.probe.min_samples, config.eval.probe.length_ratio) == (
        True,
        64,
        0.75,
    )
    assert (config.server.port, config.server.gpus, config.server.framework_python) == (
        8888,
        (4, 5),
        "/opt/venv/bin/python",
    )
    assert (config.deadlines.boot_sec, config.deadlines.round_sec, config.deadlines.silence_sec) == (7200, 7200, 600)
    assert config.server.base.digest == _base("sglang").digest()
    assert SuiteConfig.from_document(json.loads(json.dumps(config.to_document()))) == config


def test_the_fixed_workload_of_a_point_is_the_one_its_round_sends() -> None:
    point = _point("vllm", NUM_WARMUPS="16", HYPERLOOM_CLIENT_TOKENIZER_MODE="slow")

    assert fixed_workload(point) == suite_config(point, _base("vllm"), _settings()).fixed


def test_a_server_phase_point_boots_the_server_and_runs_no_load_or_eval() -> None:
    config = suite_config(_point("vllm", HYPERLOOM_RUN_PHASE="server"), _base("vllm"), _settings(LifecycleMode.BOOT))

    assert (config.workload, config.fixed, config.eval.enabled) == (WorkloadKind.NONE, None, False)
    assert "HYPERLOOM_RUN_PHASE" not in config.server.user.env_set


def test_an_eval_only_point_runs_the_eval_and_no_load_on_the_held_server() -> None:
    point = _point("vllm", RUN_EVAL="true", EVAL_ONLY="true")
    config = suite_config(point, _base("vllm"), _settings(LifecycleMode.REUSE))

    assert (config.workload, config.fixed, config.eval.enabled) == (WorkloadKind.NONE, None, True)
    assert "EVAL_ONLY" not in config.server.user.env_set


def test_without_a_gpu_mask_the_round_takes_the_first_tp_gpus() -> None:
    point = _point("sglang")
    del point["benchmark"]["envs"]["ROCR_VISIBLE_DEVICES"]

    assert suite_config(point, _base("sglang"), _settings()).server.gpus == (0, 1)


def test_a_round_takes_the_cards_its_topology_needs_from_the_front_of_a_wider_mask() -> None:
    point = _point("sglang", TP="1", ROCR_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", HYPERLOOM_ROCR_AUTOFILL="0,1,2,3,4,5,6,7")

    assert suite_config(point, _base("sglang", tp=1), _settings()).server.gpus == (0,)


def test_each_vllm_data_parallel_replica_takes_tp_cards_of_its_own() -> None:
    base = _base("vllm")
    base = replace(base, args=(*base.args, "--data-parallel-size", "2"))
    point = _point("vllm", ROCR_VISIBLE_DEVICES="2,3,4,5,6,7")

    assert suite_config(point, base, _settings()).server.gpus == (2, 3, 4, 5)


def test_a_mask_narrower_than_the_launch_is_refused() -> None:
    base = _base("vllm")
    base = replace(base, args=(*base.args, "--data-parallel-size", "2"))

    with pytest.raises(ConfigError, match="ROCR_VISIBLE_DEVICES='4,5' holds 2 GPUs; the launch needs 4"):
        suite_config(_point("vllm", ROCR_VISIBLE_DEVICES="4,5"), base, _settings())


def test_hierarchical_cache_drops_a_base_radix_cache_switch_as_sglangs_launcher_does() -> None:
    base = _base("sglang")
    base = replace(base, args=(*base.args, "--disable-radix-cache"))
    point = _seeded(_point("sglang", EXTRA_SGLANG_ARGS="--context-length 8192 --enable-hierarchical-cache"), base)
    launch = _launch(suite_config(point, base, _settings()), base)

    assert "--disable-radix-cache" not in launch.argv
    assert "--enable-hierarchical-cache" in launch.argv


def test_an_explicit_context_length_replaces_the_launcher_default() -> None:
    base = _base("vllm")
    point = _point("vllm", EXTRA_VLLM_ARGS="--max-model-len=16384")
    del point["benchmark"]["envs"]["MAX_MODEL_LEN"]
    launch = _launch(suite_config(point, base, _settings()), base)

    assert ("max-model-len", ("16384",)) in _flags(launch.argv[len(launch.entrypoint) :])
    assert launch.argv.count("--max-model-len") == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"framework": "vllm"}, "the point runs vllm, its base runs sglang"),
        ({"framework": "xdit"}, "the point runs xdit, its base runs sglang"),
        ({"benchmark_script": "mlperf_agentic_client.sh"}, "the harness scores its accuracy inline"),
        ({"profiler": {"torch_profiler": {"enabled": True}}}, "bounds no capture window"),
        ({"envs": {"PROFILE": "1"}}, "PROFILE without profiler.torch_profiler.enabled"),
        ({"envs": {"BENCHMARK_BASE_URL": "https://head:8888"}}, "is not http://<host>"),
        ({"envs": {"TP": "4"}}, "the point runs TP 4, its base runs 2"),
        ({"envs": {"CONC": None}}, "the point sets no CONC"),
        ({"envs": {"HYPERLOOM_EVAL_TASKS": "mmlu"}}, "evaluates 'mmlu'"),
        ({"timeout_seconds": 0}, "timeout_seconds must be a positive number"),
    ],
)
def test_a_point_the_translation_does_not_carry_is_refused(change: dict, message: str) -> None:
    point = _point("sglang")
    bench = point["benchmark"]
    for key, value in change.items():
        bench[key] = {**bench[key], **value} if key == "envs" else value

    with pytest.raises(ConfigError, match=message):
        suite_config(point, _base("sglang"), _settings())


def _profile_point(framework: str, **envs: str) -> dict:
    point = _point(framework, PROFILE="1", NUM_PROMPTS="96", **envs)
    point["benchmark"]["profiler"]["torch_profiler"]["enabled"] = True
    return point


def test_a_sglang_profile_point_opens_the_window_its_start_body_bounds() -> None:
    body = {"shape_discovery": True, "detailed_annotations": True, "start_step": 40, "num_steps": 16}
    config = suite_config(_profile_point("sglang", PROFILE_EXTRA_BODY=json.dumps(body)), _base("sglang"), _settings())

    profile = config.profile
    assert (profile.enabled, profile.num_prompts, profile.delay_iters, profile.max_iters) == (True, 96, 40, 16)
    assert profile.start_body == {"shape_discovery": True, "detailed_annotations": True}
    assert {"PROFILE", "PROFILE_EXTRA_BODY"}.isdisjoint(config.server.user.env_set)
    assert SuiteConfig.from_document(json.loads(json.dumps(config.to_document()))) == config


def test_a_vllm_profile_point_opens_the_window_its_profiler_flags_bound() -> None:
    args = "--max-num-seqs 64 --profiler-config.delay_iterations 8 --profiler-config.max_iterations=128"
    config = suite_config(_profile_point("vllm", EXTRA_VLLM_ARGS=args), _base("vllm"), _settings())

    assert (config.profile.delay_iters, config.profile.max_iters, config.profile.start_body) == (8, 128, None)


def test_an_atom_profile_point_captures_its_whole_profiled_pass() -> None:
    config = suite_config(_profile_point("atom"), _base("atom"), _settings())

    assert (config.profile.delay_iters, config.profile.max_iters, config.profile.num_prompts) == (0, 96, 96)


def test_a_point_naming_an_unknown_framework_is_refused_not_routed() -> None:
    point = _point("sglang")
    point["benchmark"]["framework"] = "tgi"

    with pytest.raises(ConfigError, match="not one inference_suite runs"):
        suite_config(point, _base("sglang"), _settings())


_PID_DIR = "/session/arm"


@pytest.mark.parametrize(
    ("block", "finds_server", "mode", "stop_after"),
    [
        (None, False, LifecycleMode.OWN, None),
        (None, True, LifecycleMode.OWN, None),
        ({"cleanup": False}, False, LifecycleMode.BOOT, None),
        ({"cleanup": True}, False, LifecycleMode.OWN, None),
        ({"cleanup": False}, True, LifecycleMode.REUSE, None),
        ({"cleanup": True}, True, LifecycleMode.REUSE, Path(_PID_DIR)),
    ],
)
def test_a_points_server_lifecycle_block_says_how_its_round_holds_the_server(
    tmp_path, block, finds_server, mode, stop_after
):
    point = _point("sglang")
    if block is not None:
        point["benchmark"]["server_lifecycle"] = {"enabled": True, "pid_dir": _PID_DIR, **block}
    path = tmp_path / "point.yaml"
    path.write_text(json.dumps(point), encoding="utf-8")

    server = point_server(path, finds_server=finds_server)

    assert server.lifecycle.mode is mode
    assert server.lifecycle.pid_dir == (_PID_DIR if mode in (LifecycleMode.BOOT, LifecycleMode.REUSE) else None)
    assert server.stop_after == stop_after


@pytest.mark.parametrize(
    ("point_url", "session_url", "address"),
    [
        (None, "http://10.0.0.2:30000", ("10.0.0.2", 30000)),
        ("http://head:8888/", "http://10.0.0.2:30000", ("head", 8888)),
        ("http://head", None, ("head", 80)),
    ],
)
def test_a_client_only_round_measures_the_external_server_and_samples_no_local_gpu(point_url, session_url, address):
    point = _point("sglang", **({} if point_url is None else {"BENCHMARK_BASE_URL": point_url}))

    config = suite_config(point, _base("sglang"), replace(_settings(LifecycleMode.BOOT), server_url=session_url))

    assert config.lifecycle == Lifecycle(LifecycleMode.EXTERNAL)
    assert (config.server.host, config.server.port) == address
    assert not config.telemetry.gpu.enabled
    assert "BENCHMARK_BASE_URL" not in config.server.user.env_set


def test_a_round_without_an_external_server_keeps_its_own_lifecycle_and_telemetry() -> None:
    config = suite_config(_point("sglang"), _base("sglang"), _settings(LifecycleMode.BOOT))

    assert config.lifecycle == _settings(LifecycleMode.BOOT).lifecycle
    assert (config.server.host, config.telemetry.gpu.enabled) == ("127.0.0.1", True)


#: What the AgentX switch writes into an MLPerf point (``_workload_envs.apply_agentx_switch``).
_MLPERF_ENVS = {
    "HYPERLOOM_AGENTIC_BACKEND": "mlperf",
    "MLPERF_AGENTIC_MODEL": "kimi-k3",
    "MLPERF_AGENTIC_FLOW": "smoke_test",
    "MLPERF_AGENTIC_HARDWARE": "mi355x",
    "MLPERF_ENDPOINTS_DIR": "/opt/mlperf-endpoints",
    "MLPERF_TOKENIZER_DIR": "/data/tokenizer",
    "AGENTIC_DATASET_PATH": "/data/agentic_combined_v6.jsonl",
    "AGENTIC_NUM_TRAJECTORIES": "150",
    "AGENTIC_CONCURRENCY": "16",
    "AGENTX_FAILED_REQUEST_THRESHOLD": "0.1",
    "MODEL": _MODEL,
    "FRAMEWORK": "sglang",
    "CONC": "16",
    "PORT": "30000",
    "RUN_EVAL": "false",
    "EXTRA_SGLANG_ARGS": "--context-length 8192 --served-model-name kimi-k3",
}


def _mlperf_point(**envs: str | None) -> dict:
    point = _point("sglang", **_MLPERF_ENVS)
    point["benchmark"]["benchmark_script"] = "mlperf_agentic_client.sh"
    for name, value in envs.items():
        if value is None:
            point["benchmark"]["envs"].pop(name)
        else:
            point["benchmark"]["envs"][name] = value
    return point


def test_an_mlperf_point_runs_the_harness_against_a_server_serving_the_name_it_requests() -> None:
    base = _base("sglang")
    config = suite_config(_mlperf_point(), base, _settings())

    assert config.workload is WorkloadKind.MLPERF and config.mlperf is not None
    mlperf = config.mlperf
    assert (mlperf.concurrency, mlperf.trajectories, mlperf.flow, mlperf.hardware) == (16, 150, "smoke_test", "mi355x")
    assert (mlperf.dataset, mlperf.tokenizer) == ("/data/agentic_combined_v6.jsonl", "/data/tokenizer")
    assert mlperf.endpoints_dir == "/opt/mlperf-endpoints"
    assert config.server.served_model_name == mlperf.model == "kimi-k3"
    assert (config.server.port, config.lifecycle.mode, config.eval.enabled) == (30000, LifecycleMode.OWN, False)
    env = config.server.user.env_set
    assert not [name for name in env if name.startswith(("MLPERF_", "AGENTIC_", "AGENTX_"))]
    assert not {"HYPERLOOM_AGENTIC_BACKEND", "MODEL", "FRAMEWORK"} & set(env)
    assert env["SGLANG_TIMEOUT_KEEP_ALIVE"] == "900"
    assert "--served-model-name" not in config.server.user.args_add
    launch = _launch(config, base)
    assert launch.argv[launch.argv.index("--served-model-name") + 1] == "kimi-k3"


def test_an_operator_keep_alive_is_kept() -> None:
    config = suite_config(_mlperf_point(SGLANG_TIMEOUT_KEEP_ALIVE="60"), _base("sglang"), _settings())

    assert config.server.user.env_set["SGLANG_TIMEOUT_KEEP_ALIVE"] == "60"


def test_an_mlperf_server_phase_point_boots_the_server_and_runs_no_harness() -> None:
    config = suite_config(_mlperf_point(HYPERLOOM_RUN_PHASE="server"), _base("sglang"), _settings(LifecycleMode.BOOT))

    assert (config.workload, config.mlperf) == (WorkloadKind.NONE, None)


def test_an_mlperf_server_phase_point_boots_the_server_its_measured_round_reuses() -> None:
    base = _base("sglang")
    boot = suite_config(_mlperf_point(HYPERLOOM_RUN_PHASE="server"), base, _settings(LifecycleMode.BOOT))
    reuse = suite_config(_mlperf_point(), base, _settings(LifecycleMode.REUSE))

    assert boot.server.served_model_name == reuse.server.served_model_name == "kimi-k3"
    assert _launch(boot, base).argv == _launch(reuse, base).argv


@pytest.mark.parametrize(
    ("envs", "bench", "message"),
    [
        (
            {"AGENTIC_DATASET_PATH": None, "MLPERF_TOKENIZER_DIR": None},
            {},
            "sets no AGENTIC_DATASET_PATH, MLPERF_TOKENIZER_DIR",
        ),
        ({"AGENTIC_CONCURRENCY": "8"}, {}, "AGENTIC_CONCURRENCY is not its CONC 16"),
        ({"RUN_EVAL": "true"}, {}, "scores its accuracy inline"),
        ({"BENCHMARK_BASE_URL": "http://head:30000"}, {}, "not the external server"),
        ({}, {"profiler": {"torch_profiler": {"enabled": True}}}, "an MLPerf round is not profiled"),
    ],
)
def test_an_mlperf_point_a_round_cannot_run_is_refused(envs: dict, bench: dict, message: str) -> None:
    point = _mlperf_point(**envs)
    point["benchmark"].update(bench)

    with pytest.raises(ConfigError, match=message):
        suite_config(point, _base("sglang"), _settings())


def test_the_points_rocr_mask_is_the_servers_gpus_and_its_fill_record_stays_out_of_the_server_env() -> None:
    point = _point("sglang", ROCR_VISIBLE_DEVICES="6,7", HYPERLOOM_ROCR_AUTOFILL="6,7")
    server = suite_config(point, _base("sglang"), _settings()).server

    assert server.gpus == (6, 7)
    assert "HYPERLOOM_ROCR_AUTOFILL" not in server.user.env_set


@pytest.mark.parametrize(
    ("layer", "gone", "kept"),
    [
        (LaunchLayer(remove_args=["--mem-fraction-static"]), "mem-fraction-static", "trust-remote-code"),
        (
            LaunchLayer(extra_server_args="--trust-remote-code --tensor-parallel-size 2", args_mode="replace"),
            "mem-fraction-static",
            "tensor-parallel-size",
        ),
    ],
    ids=["remove_args", "args_mode_replace"],
)
def test_a_point_that_drops_a_base_flag_launches_without_it(tmp_path, monkeypatch, layer, gone, kept) -> None:
    launch = _materialized_launch(tmp_path, monkeypatch, layer)
    flags = {name for name, _ in _flags(launch.argv[len(launch.entrypoint) :])}

    assert (gone in flags, kept in flags) == (False, True)


def test_a_point_that_unsets_a_base_variable_launches_without_it(tmp_path, monkeypatch) -> None:
    launch = _materialized_launch(tmp_path, monkeypatch, LaunchLayer(unset_envs=["SGLANG_USE_AITER"]))

    assert "SGLANG_USE_AITER" not in dict(launch.env)


def _materialized_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: LaunchLayer) -> Launch:
    """The launch a round composes from a point the main flow materialized on the session's frozen base."""
    session = tmp_path / "session"
    base = freeze_base(session, _base("sglang"))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SERVER_ARGS", raising=False)
    recipe = _point("sglang")
    del recipe["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
    source = tmp_path / "recipe.yaml"
    source.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    point = materialize_config_with_envs(
        source, tmp_path / "point", launch=layer, target=BenchmarkTarget(model_path=_MODEL)
    )
    return _launch(suite_config(yaml.safe_load(point.read_text(encoding="utf-8")), base, _settings()), base)
