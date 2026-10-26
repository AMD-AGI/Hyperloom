# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared instrumentation preserves each benchmark backend's launch contract."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.session.paths import asset_root
from hyperloom.orchestrator.actions.executors import _server_patcher
from hyperloom.orchestrator.actions.executors._profile_setup import prepare_profile, speculative_tokens_per_step
from hyperloom.orchestrator.actions.executors._workload_envs import LaunchLayer, materialize_config_with_envs
from hyperloom.orchestrator.actions.executors.magpie_agentx import apply_native_runtime, enable_profile, load_benchmark


@pytest.fixture(autouse=True)
def profile_environment(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "patched")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "1")
    for name in (
        "PROFILE_OSL",
        "HYPERLOOM_PROFILE_MAX_ITERS",
        "HYPERLOOM_PROFILE_DELAY_ITERS",
        "HYPERLOOM_PROFILE_SHAPE_DISCOVERY",
        "ISL",
        "OSL",
        "CONC",
        "TP",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_server_patcher, "ensure_vllm_patched_for_tracelens", lambda: True)
    monkeypatch.setattr(_server_patcher, "ensure_sglang_patched_for_tracelens", lambda: True)


def _fixed_profile(framework: str) -> dict:
    source = asset_root() / "assets" / "configs" / f"profile_{framework}.yaml"
    bench = yaml.safe_load(source.read_text(encoding="utf-8"))["benchmark"]
    bench["model"] = "Qwen/Qwen3-0.6B"
    bench["envs"].update({"TP": 1, "CONC": 4, "ISL": 256, "OSL": 8192})
    prepare_profile(bench, bench["envs"])
    return bench


def _native_config(tmp_path: Path, framework: str, **fields) -> Path:
    path = tmp_path / f"agentx_{framework}.yaml"
    bench = {
        "framework": framework,
        "model": "Qwen/Qwen3-0.6B",
        "precision": "bf16",
        "agentx": {"enabled": True},
        "envs": {"TP": "1", "CONC": "4"},
        "profiler": {"torch_profiler": {"enabled": False}},
        **fields,
    }
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_native_and_fixed_profiles_share_patches_not_launch_controls(tmp_path, monkeypatch, framework):
    calls = []
    monkeypatch.setattr(
        _server_patcher, f"ensure_{framework}_patched_for_tracelens", lambda: calls.append(framework) or True
    )
    fixed = _fixed_profile(framework)
    path = _native_config(tmp_path, framework)
    original = load_benchmark(path)
    enable_profile(path)
    native = load_benchmark(path)
    assert calls == [framework, framework]
    assert fixed["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "ok"
    assert native["profiler"]["torch_profiler"] == {
        "enabled": True,
        "num_steps": 256,
        "num_profiles": 1,
        "start_seconds": 300,
        "interval_seconds": 900,
        "detailed_annotations": True,
    }
    assert native["envs"]["TP"] == original["envs"]["TP"]
    assert native["envs"]["CONC"] == original["envs"]["CONC"]
    assert (
        not {"OSL", "NUM_PROMPTS", "PROFILE_EXTRA_BODY", "EXTRA_VLLM_ARGS", "EXTRA_SGLANG_ARGS"} & native["envs"].keys()
    )
    launch = native["agentx"].get("launch_overrides", {})
    assert launch.get("append_args", []) == []
    assert (
        not {
            "SGLANG_PROFILE_WITH_STACK",
            "SGLANG_PROFILE_RECORD_SHAPE",
            "SGLANG_PROFILE_RECORD_SHAPES",
            "SGLANG_GRAPH_BATCH_CAPTURE",
        }
        & launch.get("env", {}).keys()
    )
    if framework == "vllm":
        assert launch["env"]["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "1200"
        assert "--profiler-config.max_iterations 128" in fixed["envs"]["EXTRA_VLLM_ARGS"]
    else:
        assert json.loads(fixed["envs"]["PROFILE_EXTRA_BODY"])["num_steps"] == 128


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.parametrize("detailed", [False, True])
def test_native_profile_preserves_explicit_capture_settings(tmp_path, monkeypatch, framework, detailed):
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "7")
    monkeypatch.setenv("HYPERLOOM_PROFILE_DELAY_ITERS", "2")
    monkeypatch.setenv("PROFILE_OSL", "3")
    requested = {
        "enabled": False,
        "num_steps": 80,
        "num_profiles": 3,
        "start_seconds": 60,
        "interval_seconds": 45,
        "detailed_annotations": detailed,
    }
    path = _native_config(tmp_path, framework, profiler={"torch_profiler": deepcopy(requested)})
    enable_profile(path)
    assert load_benchmark(path)["profiler"]["torch_profiler"] == {**requested, "enabled": True}


def test_fixed_sequence_materialization_keeps_its_capture_plan(tmp_path):
    fixed = _fixed_profile("sglang")
    source = asset_root() / "assets" / "configs" / "profile_sglang.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["benchmark"]["model"] = "Qwen/Qwen3-0.6B"
    config["benchmark"]["envs"].update({"TP": 1, "CONC": 4, "ISL": 256, "OSL": 8192})
    path = tmp_path / "fixed.yaml"
    path.write_text(yaml.safe_dump(config))
    rendered = load_benchmark(materialize_config_with_envs(path, tmp_path / "fixed-run"))
    assert json.loads(rendered["envs"]["PROFILE_EXTRA_BODY"]) == json.loads(fixed["envs"]["PROFILE_EXTRA_BODY"])
    assert int(rendered["envs"]["OSL"]) == 512


def test_shape_tool_preserves_the_selected_native_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "sitecustomize")
    tool_dir = tmp_path / "kernel_shape_tool"
    tool_dir.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    monkeypatch.setattr(_server_patcher, "kernel_shape_tool_dir", lambda: tool_dir)
    launch = {
        "version": 1,
        "env": {"PYTHONPATH": "/candidate/overlay:/inherited/server", "SGLANG_TEST": "1"},
        "append_args": ["--chunked-prefill-size", "2048"],
        "unset_env": ["TRACELENS_SHAPE_DISCOVERY", "UNRELATED_ENV"],
    }
    path = _native_config(tmp_path, "sglang", agentx={"enabled": True, "launch_overrides": launch})
    apply_native_runtime(path, {"pythonpath_prefix": str(runtime_dir)})
    before = load_benchmark(path)["agentx"]["launch_overrides"]
    fixed = _fixed_profile("sglang")
    enable_profile(path)
    once = load_benchmark(path)
    enable_profile(path)
    assert load_benchmark(path) == once
    actual = once["agentx"]["launch_overrides"]
    assert actual["env"]["PYTHONPATH"] == f"{tool_dir}:{before['env']['PYTHONPATH']}"
    assert actual["env"]["TRACELENS_SHAPE_DISCOVERY"] == "1"
    assert fixed["envs"]["TRACELENS_SHAPE_DISCOVERY"] == "1"
    assert actual["env"]["SGLANG_TEST"] == "1"
    assert actual["append_args"] == before["append_args"]
    assert actual["unset_env"] == ["UNRELATED_ENV"]
    assert "shape-discovery-for-cuda-graph-profile" not in str(actual)


def test_missing_shape_tool_keeps_native_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "sitecustomize")
    monkeypatch.setattr(_server_patcher, "kernel_shape_tool_dir", lambda: None)
    launch = {"version": 1, "env": {"PYTHONPATH": "/candidate/overlay:/runtime"}}
    path = _native_config(tmp_path, "sglang", agentx={"enabled": True, "launch_overrides": launch})
    enable_profile(path)
    environment = load_benchmark(path)["agentx"]["launch_overrides"]["env"]
    assert environment["PYTHONPATH"] == launch["env"]["PYTHONPATH"]
    assert environment["TRACELENS_SHAPE_DISCOVERY"] == "0"


def test_shape_tool_does_not_reintroduce_an_explicitly_unset_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "sitecustomize")
    monkeypatch.setenv("PYTHONPATH", "/ambient/client")
    tool_dir = tmp_path / "kernel_shape_tool"
    tool_dir.mkdir()
    monkeypatch.setattr(_server_patcher, "kernel_shape_tool_dir", lambda: tool_dir)
    launch = {"version": 1, "unset_env": ["PYTHONPATH", "TRACELENS_SHAPE_DISCOVERY"]}
    path = _native_config(tmp_path, "sglang", agentx={"enabled": True, "launch_overrides": launch})
    enable_profile(path)
    actual = load_benchmark(path)["agentx"]["launch_overrides"]
    assert actual["env"]["PYTHONPATH"] == str(tool_dir)
    assert actual["unset_env"] == []


@pytest.mark.parametrize("timeout,expected", [("600", "1200"), ("2400", "2400")])
def test_native_vllm_timeout_preserves_a_longer_server_timeout(tmp_path, timeout, expected):
    launch = {
        "version": 1,
        "env": {"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": timeout},
        "unset_env": ["UNRELATED_ENV"],
    }
    path = _native_config(tmp_path, "vllm", agentx={"enabled": True, "launch_overrides": launch})
    enable_profile(path)
    actual = load_benchmark(path)["agentx"]["launch_overrides"]
    assert actual["env"]["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == expected
    assert actual["unset_env"] == ["UNRELATED_ENV"]


@pytest.mark.parametrize("detailed", [False, True])
def test_native_shape_setup_preserves_annotations_without_rewriting_client_body(tmp_path, monkeypatch, detailed):
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "sitecustomize")
    tool_dir = tmp_path / "kernel_shape_tool"
    tool_dir.mkdir()
    monkeypatch.setattr(_server_patcher, "kernel_shape_tool_dir", lambda: tool_dir)
    client_body = json.dumps({"client_option": "retained", "detailed_annotations": not detailed})
    requested = {"enabled": False, "detailed_annotations": detailed, "num_steps": 21}
    path = _native_config(
        tmp_path,
        "sglang",
        envs={"CONC": "4", "PROFILE_EXTRA_BODY": client_body},
        profiler={"torch_profiler": requested},
    )
    enable_profile(path)
    native = load_benchmark(path)
    assert native["profiler"]["torch_profiler"] == {
        **requested,
        "enabled": True,
        "num_profiles": 1,
        "start_seconds": 300,
        "interval_seconds": 900,
    }
    assert native["envs"]["PROFILE_EXTRA_BODY"] == client_body
    server_env = native["agentx"]["launch_overrides"]["env"]
    assert server_env["TRACELENS_SHAPE_DISCOVERY"] == "1"
    assert "PROFILE_EXTRA_BODY" not in server_env


def test_fixed_shape_setup_preserves_request_options_and_capture_window(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "sitecustomize")
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "9")
    monkeypatch.setenv("HYPERLOOM_PROFILE_DELAY_ITERS", "2")
    tool_dir = tmp_path / "kernel_shape_tool"
    tool_dir.mkdir()
    monkeypatch.setattr(_server_patcher, "kernel_shape_tool_dir", lambda: tool_dir)
    envs = {
        "CONC": 4,
        "OSL": 512,
        "PROFILE_EXTRA_BODY": json.dumps(
            {"shape_discovery": True, "detailed_annotations": False, "client_option": "retained"}
        ),
    }
    prepare_profile({"framework": "sglang", "model": "Qwen/Qwen3-0.6B"}, envs)
    assert json.loads(envs["PROFILE_EXTRA_BODY"]) == {
        "detailed_annotations": False,
        "client_option": "retained",
        "start_step": 2,
        "num_steps": 9,
    }
    assert envs["TRACELENS_SHAPE_DISCOVERY"] == "1"
    assert "--enable-shape-discovery-for-cuda-graph-profile" not in envs.get("EXTRA_SGLANG_ARGS", "")


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_native_profile_drops_annotations_when_the_patch_did_not_apply(tmp_path, monkeypatch, framework):
    """Magpie raises rather than degrading when the server lacks the fields, so an unreached patch means false."""
    monkeypatch.setattr(_server_patcher, f"ensure_{framework}_patched_for_tracelens", lambda: False)
    path = _native_config(tmp_path, framework)
    enable_profile(path)
    native = load_benchmark(path)
    assert native["envs"]["HYPERLOOM_PROFILE_DEGRADED_REASON"] == "tracelens_runtime_patch_unavailable"
    assert native["profiler"]["torch_profiler"]["detailed_annotations"] is False


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_native_profile_keeps_the_operator_annotation_choice_through_a_failed_patch(tmp_path, monkeypatch, framework):
    monkeypatch.setattr(_server_patcher, f"ensure_{framework}_patched_for_tracelens", lambda: False)
    requested = {"enabled": False, "detailed_annotations": True}
    path = _native_config(tmp_path, framework, profiler={"torch_profiler": deepcopy(requested)})
    enable_profile(path)
    assert load_benchmark(path)["profiler"]["torch_profiler"]["detailed_annotations"] is True


@pytest.mark.parametrize("enable_patch,expected_status", [("1", "ok"), ("0", "not_attempted")])
def test_native_patch_retry_clears_its_stale_failure(tmp_path, monkeypatch, enable_patch, expected_status):
    path = _native_config(tmp_path, "sglang")
    monkeypatch.setattr(_server_patcher, "ensure_sglang_patched_for_tracelens", lambda: False)
    enable_profile(path)
    failed = load_benchmark(path)["envs"]
    assert failed["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "unavailable"
    assert failed["HYPERLOOM_PROFILE_DEGRADED_REASON"] == "tracelens_runtime_patch_unavailable"
    calls = []
    monkeypatch.setattr(_server_patcher, "ensure_sglang_patched_for_tracelens", lambda: calls.append(True) or True)
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", enable_patch)
    enable_profile(path)
    retried = load_benchmark(path)["envs"]
    assert retried["HYPERLOOM_TRACELENS_PATCH_STATUS"] == expected_status
    assert "HYPERLOOM_PROFILE_DEGRADED_REASON" not in retried
    assert calls == ([True] if enable_patch == "1" else [])


@pytest.mark.parametrize("enable_patch", ["1", "0"])
def test_native_patch_preparation_preserves_other_degraded_reasons(tmp_path, monkeypatch, enable_patch):
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", enable_patch)
    reason = "trace_export_unavailable"
    path = _native_config(tmp_path, "sglang", envs={"HYPERLOOM_PROFILE_DEGRADED_REASON": reason})
    enable_profile(path)
    assert load_benchmark(path)["envs"]["HYPERLOOM_PROFILE_DEGRADED_REASON"] == reason


_SGLANG_MTP = (
    "--speculative-algorithm NEXTN --speculative-num-steps 5 --speculative-eagle-topk 1 "
    "--speculative-num-draft-tokens 6"
)


@pytest.mark.parametrize(
    ("server_args", "expected"),
    [
        ("", 1),
        ("--tp 8 --mem-fraction-static 0.8", 1),
        (_SGLANG_MTP, 6),
        ("--speculative-algorithm EAGLE --speculative-num-steps=3 --speculative-num-draft-tokens=4", 4),
        ("--speculative-algorithm EAGLE --speculative-eagle-topk 4 --speculative-num-draft-tokens 8", 8),
        ("--speculative-algorithm NEXTN", 8),
        ('--speculative-config {"method":"mtp","num_speculative_tokens":3}', 4),
        ("--speculative-config.method mtp --speculative-config.num_speculative_tokens=2", 3),
    ],
)
def test_speculative_tokens_per_step_reads_the_upper_bound(server_args, expected):
    assert speculative_tokens_per_step(server_args) == expected


def _materialized_profile(tmp_path: Path, framework: str, server_args: str, args_mode: str = "append") -> dict:
    source = asset_root() / "assets" / "configs" / f"profile_{framework}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["benchmark"]["model"] = "Qwen/Qwen3-0.6B"
    config["benchmark"]["envs"].update({"TP": 1, "CONC": 64, "ISL": 1024, "OSL": 1024})
    path = tmp_path / f"{framework}.yaml"
    path.write_text(yaml.safe_dump(config))
    out = tmp_path / f"{framework}-{args_mode}-{len(server_args)}"
    launch = LaunchLayer(extra_server_args=server_args, args_mode=args_mode)
    rendered = materialize_config_with_envs(path, out, launch=launch)
    return load_benchmark(rendered)["envs"]


def _steps_at_full_acceptance(envs: dict, tokens_per_step: int) -> int:
    """Decode forward steps the run lasts if every draft token is accepted, the shortest it can be."""
    return int(envs["NUM_PROMPTS"]) * int(envs["OSL"]) // (int(envs["CONC"]) * tokens_per_step)


def test_sglang_speculative_launch_reaches_the_profile_window(tmp_path):
    """W186 shape: 776 prompts x OSL 1024 at CONC 64 put start_step at 6080, past an MTP run's last step."""
    plain = json.loads(_materialized_profile(tmp_path, "sglang", "")["PROFILE_EXTRA_BODY"])
    envs = _materialized_profile(tmp_path, "sglang", _SGLANG_MTP)
    body = json.loads(envs["PROFILE_EXTRA_BODY"])
    assert plain["start_step"] == 6080
    assert body["start_step"] == 6080 // 6
    assert body["num_steps"] == plain["num_steps"] == 128
    assert body["start_step"] + body["num_steps"] <= _steps_at_full_acceptance(envs, 6) // 2


def test_sglang_plain_launch_keeps_its_capture_plan(tmp_path):
    envs = _materialized_profile(tmp_path, "sglang", "--mem-fraction-static 0.8")
    body = json.loads(envs["PROFILE_EXTRA_BODY"])
    assert (body["start_step"], body["num_steps"], int(envs["NUM_PROMPTS"])) == (6080, 128, 776)


@pytest.mark.parametrize("args_mode", ["append", "replace"])
def test_vllm_speculative_launch_reaches_the_profile_window(tmp_path, args_mode):
    """``replace`` drops the profiler flags, so the delay that ships is the one the restore re-adds."""
    spec = '--speculative-config {"method":"mtp","num_speculative_tokens":3}'
    plain = _materialized_profile(tmp_path, "vllm", "")
    envs = _materialized_profile(tmp_path, "vllm", spec, args_mode)
    assert "--profiler-config.delay_iterations 6080" in plain["EXTRA_VLLM_ARGS"]
    assert int(plain["NUM_PROMPTS"]) == 776
    args = envs["EXTRA_VLLM_ARGS"]
    assert args.count("delay_iterations") == 1
    assert f"--profiler-config.delay_iterations {6080 // 4}" in args
    assert int(envs["NUM_PROMPTS"]) == 824
    assert 6080 // 4 + 128 <= _steps_at_full_acceptance(envs, 4) // 2


def test_an_operator_capture_window_is_left_as_written(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "16")
    monkeypatch.setenv("HYPERLOOM_PROFILE_DELAY_ITERS", "40")
    body = json.loads(_materialized_profile(tmp_path, "sglang", _SGLANG_MTP)["PROFILE_EXTRA_BODY"])
    assert (body["start_step"], body["num_steps"]) == (40, 16)
