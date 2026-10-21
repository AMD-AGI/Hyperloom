# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI workloads and explicit YAML converge on one native Magpie contract."""

import pytest

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV, benchmark_config_path
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli.benchmark_config import bind_benchmark, configure_benchmark
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.orchestrator.actions.executors._workload_envs import (
    default_baseline_config,
    materialize_benchmark_config,
)
from hyperloom.orchestrator.actions.executors.magpie_agentx import enable_profile, load_benchmark


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for key in (
        BENCHMARK_CONFIG_ENV,
        "HYPERLOOM_AGENTX",
        "HYPERLOOM_AGENTIC_BACKEND",
        "FRAMEWORK",
        "FRAMEWORK_VERSION",
        "GPU_TYPE",
        "TARGET_GPU_TYPE",
        "MODEL_PATH",
        "TP",
        "EP",
        "CONC",
        "ISL",
        "OSL",
        "PRECISION",
        "MAX_MODEL_LEN",
        "INFERENCE_OPTIMIZER_EXTRA_ENV",
        "INFERENCE_OPTIMIZER_SERVER_ARGS",
        "TEST_SERVER_KNOB",
        "INFERENCE_OPTIMIZER_NODES",
        "SKIP_VARIANTS",
        "PD_MODE",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")


def _launch(tmp_path, monkeypatch, *flags):
    args = _build_parser().parse_args(["optimize", "--model", "Qwen/Qwen3-8B", *flags])
    configure_benchmark(args)
    cli._resolve_run_topology(args)
    cli._resolve_precision(args)
    monkeypatch.setenv("FRAMEWORK", args.framework or "sglang")
    session = tmp_path / "session"
    session.mkdir()
    bind_benchmark(session, args)
    return args, session / "benchmark.yaml"


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_standard_cli_generates_native_yaml_and_candidate_controls(tmp_path, monkeypatch, framework):
    args, source = _launch(
        tmp_path,
        monkeypatch,
        "--framework",
        framework,
        "--tp",
        "4",
        "--ep",
        "4",
        "--conc",
        "8",
        "--precision",
        "fp8",
        "--gpu-type",
        "mi355x",
        "--docker-image",
        "example/serving:pinned",
        "--server-args=--disable-custom-all-reduce",
        "--extra-env",
        "TEST_SERVER_KNOB=1",
    )
    bench = load_benchmark(source)
    assert bench["framework"] == framework
    assert bench["model"] == "Qwen/Qwen3-8B"
    assert bench["precision"] == "fp8"
    assert bench["runner_type"] == "mi355x"
    assert bench["docker_image"] == "example/serving:pinned"
    assert bench["run_mode"] == "local"
    assert bench["envs"] == {"TP": 4, "EP_SIZE": 4, "CONC": 8}
    assert bench["agentx"]["mode"] == "canonical"
    assert "benchmark_script" not in bench
    assert args.enable_conc_sweep is False
    assert agentx_enabled()
    assert default_baseline_config() == source
    candidate = materialize_benchmark_config(source, tmp_path / "round", extra_envs={"TEST_SERVER_KNOB": "2"})
    launch = load_benchmark(candidate)["agentx"]["launch_overrides"]
    assert launch["append_args"] == ["--disable-custom-all-reduce"]
    assert launch["env"] == {"TEST_SERVER_KNOB": "2"}
    enable_profile(candidate)
    assert load_benchmark(candidate)["profiler"]["torch_profiler"]["enabled"] is True
    assert load_benchmark(source)["profiler"]["torch_profiler"]["enabled"] is False


@pytest.mark.parametrize(("board", "runner"), [("mi308x", "mi300x"), ("mi325x", "mi300x"), ("mi355x", "mi355x")])
def test_generated_yaml_pins_the_magpie_runner_label_not_the_board(tmp_path, monkeypatch, board, runner):
    """Magpie ships no mi308x/mi325x launch script, so those boards must resolve to mi300x."""
    _, source = _launch(tmp_path, monkeypatch, "--gpu-type", board)
    assert load_benchmark(source)["runner_type"] == runner


def test_cli_defaults_and_operator_workload_pins_use_existing_resolver(tmp_path, monkeypatch):
    monkeypatch.setenv("CONC", "32")
    _, source = _launch(tmp_path, monkeypatch, "--extra-env", "TP=2", "--extra-env", "EP=2")
    bench = load_benchmark(source)
    assert bench["precision"] == "bf16"
    assert bench["envs"] == {"TP": 2, "EP_SIZE": 2, "CONC": 32}
    assert "docker_image" not in bench
    round_path = materialize_benchmark_config(source, tmp_path / "round")
    assert load_benchmark(round_path)["agentx"]["launch_overrides"]["env"] == {}


def test_generated_yaml_pins_local_checkpoint(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setenv("MODEL_PATH", str(model))
    _, source = _launch(tmp_path, monkeypatch)
    assert load_benchmark(source)["envs"]["MODEL_PATH"] == str(model.resolve())


def test_resume_reuses_saved_workload_without_switch_or_yaml_flag(tmp_path, monkeypatch):
    _, source = _launch(tmp_path, monkeypatch, "--conc", "8", "--docker-image", "example/image:pinned")
    before = source.read_bytes()
    monkeypatch.setenv("HYPERLOOM_AGENTX", "")
    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, "")
    monkeypatch.setenv("CONC", "128")
    args = _build_parser().parse_args(["optimize", "--resume-from", str(source.parent)])
    configure_benchmark(args)
    assert args.conc == 8
    assert args.docker_image == "example/image:pinned"
    assert benchmark_config_path() == source
    assert source.read_bytes() == before
    assert agentx_enabled()


@pytest.mark.parametrize("override", [["--conc", "16"], ["--extra-env", "TP=8"], ["--docker-image", "other:image"]])
def test_resume_rejects_workload_changes(tmp_path, monkeypatch, override):
    _, source = _launch(tmp_path, monkeypatch, "--conc", "8")
    args = _build_parser().parse_args(["optimize", "--resume-from", str(source.parent), *override])
    with pytest.raises(ValueError, match="conflicts"):
        configure_benchmark(args)


def test_native_resume_does_not_regenerate_missing_snapshot(tmp_path):
    args = _build_parser().parse_args(["optimize", "--resume-from", str(tmp_path)])
    with pytest.raises(ValueError, match="requires the session's benchmark.yaml"):
        configure_benchmark(args)
    assert not (tmp_path / "benchmark.yaml").exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--nodes", "2"],
        ["--framework", "atom"],
        ["--isl", "1024"],
        ["--osl", "512"],
        ["--enable-conc-sweep"],
        ["--quantize", "fp8"],
        ["--extra-env", "AIPERF_BIN=/tmp/client"],
        ["--extra-env", "CONC=0", "--extra-env", "ISL=2048"],
        ["--extra-env", "EP_SIZE=4"],
    ],
)
def test_unsupported_native_options_fail_before_session_creation(flags):
    args = _build_parser().parse_args(["optimize", "--model", "/model", *flags])
    with pytest.raises(ValueError):
        configure_benchmark(args)


def test_operator_switch_uses_same_native_validation(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "")
    args = _build_parser().parse_args(["optimize", "--extra-env", "HYPERLOOM_AGENTX=1", "--nodes", "2"])
    with pytest.raises(ValueError, match="one node"):
        configure_benchmark(args)


@pytest.mark.parametrize("backend", ["synthetic", "mlperf"])
def test_other_workloads_do_not_generate_native_yaml(tmp_path, monkeypatch, backend):
    if backend == "synthetic":
        monkeypatch.setenv("HYPERLOOM_AGENTX", "0")
    else:
        monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")
    args = _build_parser().parse_args(["optimize", "--model", "/model", "--isl", "512"])
    configure_benchmark(args)
    bind_benchmark(tmp_path, args)
    assert benchmark_config_path() is None
    assert not (tmp_path / "benchmark.yaml").exists()


def test_yaml_input_can_use_recipe_image(tmp_path, monkeypatch):
    _, source = _launch(tmp_path, monkeypatch)
    args = _build_parser().parse_args(["optimize", "--benchmark-config", str(source)])
    configure_benchmark(args)
    assert args.docker_image is None
    assert args.tp == 1


def test_generated_snapshot_precedes_manifest(tmp_path, monkeypatch):
    import asyncio

    args = _build_parser().parse_args(["optimize", "--model", "Qwen/Qwen3-8B", "--tp", "2", "--conc", "8"])
    configure_benchmark(args)
    cli._resolve_run_topology(args)
    for name in (
        "_enforce_expected_framework",
        "_apply_operator_supplied_paths",
        "_require_custom_entrypoint",
        "_check_gfx_arch_resolvable",
    ):
        monkeypatch.setattr(cli, name, lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_autodetect_gpu_type", lambda: "mi355x")
    monkeypatch.setattr(cli, "_should_remote_probe_gpu", lambda args: False)
    monkeypatch.setattr(cli, "resolve_framework_version", lambda args: "test-version")
    monkeypatch.setattr(cli, "make_session_dir", lambda **kw: tmp_path)
    monkeypatch.setattr(cli, "_acquire_session_lock_or_exit", lambda path: None)

    class ManifestReached(Exception):
        pass

    def manifest(session_dir, *, args):
        assert benchmark_config_path() == session_dir / "benchmark.yaml"
        assert load_benchmark(benchmark_config_path())["envs"]["TP"] == 2
        raise ManifestReached

    monkeypatch.setattr(cli, "write_manifest", manifest)
    with pytest.raises(ManifestReached):
        asyncio.run(cli._seed_fresh_session(args, nodes_resolved=1))


def test_operator_argument_can_be_ablated(tmp_path, monkeypatch):
    _, source = _launch(tmp_path, monkeypatch, "--server-args=--cuda-graph-max-bs 256")
    candidate = materialize_benchmark_config(source, tmp_path / "ablate", remove_args=["--cuda-graph-max-bs"])
    launch = load_benchmark(candidate)["agentx"]["launch_overrides"]
    assert launch["append_args"] == launch["remove_args"] == []


def test_operator_env_can_be_removed_or_replaced_by_a_candidate(tmp_path, monkeypatch):
    _, source = _launch(tmp_path, monkeypatch, "--extra-env", "TEST_SERVER_KNOB=1")
    candidate = materialize_benchmark_config(source, tmp_path / "removed", unset_envs=["TEST_SERVER_KNOB"])
    launch = load_benchmark(candidate)["agentx"]["launch_overrides"]
    assert "TEST_SERVER_KNOB" not in launch["env"]
    assert launch["unset_env"] == ["TEST_SERVER_KNOB"]
    candidate = materialize_benchmark_config(
        source, tmp_path / "replaced", unset_envs=["TEST_SERVER_KNOB"], extra_envs={"TEST_SERVER_KNOB": "2"}
    )
    launch = load_benchmark(candidate)["agentx"]["launch_overrides"]
    assert launch["env"]["TEST_SERVER_KNOB"] == "2"
    assert launch["unset_env"] == []


def test_operator_switch_can_generate_native_workload(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "")
    _, source = _launch(tmp_path, monkeypatch, "--extra-env", "HYPERLOOM_AGENTX=1")
    rendered = materialize_benchmark_config(source, tmp_path / "round")
    assert load_benchmark(rendered)["agentx"]["enabled"] is True
    assert "HYPERLOOM_AGENTX" not in load_benchmark(rendered)["agentx"]["launch_overrides"]["env"]


@pytest.mark.parametrize(
    "flag,value,error",
    [("--tp", "-1", SystemExit), ("--tp", "0", SystemExit), ("--ep", "-1", ValueError), ("--conc", "-1", SystemExit)],
)
def test_nonpositive_topology_is_rejected_before_startup(flag, value, error):
    with pytest.raises(error):
        args = _build_parser().parse_args(["optimize", flag, value])
        configure_benchmark(args)
