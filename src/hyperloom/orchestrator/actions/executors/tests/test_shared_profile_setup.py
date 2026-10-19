# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixed-sequence and native AgentX profiles share one TraceLens setup."""

import json
from pathlib import Path

import yaml

from hyperloom.inference_optimizer.session.paths import asset_root
from hyperloom.orchestrator.actions.executors._profile_setup import prepare_profile
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs
from hyperloom.orchestrator.actions.executors.magpie_agentx import enable_profile


def _pin_profile_env(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "patched")
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


def _fixed_profile(framework: str) -> dict:
    source = asset_root() / "assets" / "configs" / f"profile_{framework}.yaml"
    bench = yaml.safe_load(source.read_text(encoding="utf-8"))["benchmark"]
    bench["model"] = "Qwen/Qwen3-0.6B"
    bench["envs"].update({"TP": 1, "CONC": 4, "ISL": 256, "OSL": 8192})
    prepare_profile(bench, bench["envs"], write_workload=True)
    return bench


def _agentx_profile(tmp_path: Path, framework: str) -> dict:
    path = tmp_path / f"agentx_{framework}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": framework,
                    "model": "Qwen/Qwen3-0.6B",
                    "precision": "bf16",
                    "agentx": {"enabled": True},
                    "envs": {"TP": "1", "CONC": "4", "ISL": "256", "OSL": "8192"},
                    "profiler": {"torch_profiler": {"enabled": False}},
                }
            }
        ),
        encoding="utf-8",
    )
    enable_profile(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]


def test_sglang_agentx_profile_matches_fixed_sequence(tmp_path, monkeypatch):
    _pin_profile_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    fixed = _fixed_profile("sglang")
    agentx = _agentx_profile(tmp_path, "sglang")
    shipped = yaml.safe_load((asset_root() / "assets" / "configs" / "profile_sglang.yaml").read_text(encoding="utf-8"))
    shipped["benchmark"]["model"] = "Qwen/Qwen3-0.6B"
    shipped["benchmark"]["envs"].update({"TP": 1, "CONC": 4, "ISL": 256, "OSL": 8192})
    source = tmp_path / "shipped.yaml"
    source.write_text(yaml.safe_dump(shipped), encoding="utf-8")
    rendered = yaml.safe_load(materialize_config_with_envs(source, tmp_path / "fixed-run").read_text(encoding="utf-8"))
    assert json.loads(rendered["benchmark"]["envs"]["PROFILE_EXTRA_BODY"]) == json.loads(
        fixed["envs"]["PROFILE_EXTRA_BODY"]
    )

    assert json.loads(agentx["envs"]["PROFILE_EXTRA_BODY"]) == json.loads(fixed["envs"]["PROFILE_EXTRA_BODY"])
    assert agentx["envs"]["EXTRA_SGLANG_ARGS"] == fixed["envs"]["EXTRA_SGLANG_ARGS"]
    assert agentx["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"] == fixed["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"]
    for key in ("SGLANG_PROFILE_WITH_STACK", "SGLANG_PROFILE_RECORD_SHAPE", "SGLANG_GRAPH_BATCH_CAPTURE"):
        assert agentx["envs"][key] == fixed["envs"][key]
        assert agentx["agentx"]["launch_overrides"]["env"][key] == str(fixed["envs"][key])
    body = json.loads(agentx["envs"]["PROFILE_EXTRA_BODY"])
    torch_profiler = agentx["profiler"]["torch_profiler"]
    assert torch_profiler["num_steps"] == body["num_steps"] == 128
    assert torch_profiler["detailed_annotations"] is body["detailed_annotations"] is True
    assert torch_profiler["num_profiles"] == 1
    assert "--enable-profile-cuda-graph" in agentx["agentx"]["launch_overrides"]["append_args"]
    assert int(agentx["envs"]["OSL"]) == 8192
    assert int(fixed["envs"]["OSL"]) == 512


def test_vllm_agentx_profile_matches_fixed_sequence_and_leaves_profiler_config_to_magpie(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _server_patcher

    _pin_profile_env(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    counts = {"vllm": 0}

    def _patched() -> bool:
        counts["vllm"] += 1
        return True

    monkeypatch.setattr(_server_patcher, "ensure_vllm_patched_for_tracelens", _patched)
    fixed = _fixed_profile("vllm")
    agentx = _agentx_profile(tmp_path, "vllm")

    assert counts == {"vllm": 2}
    assert agentx["envs"]["EXTRA_VLLM_ARGS"] == fixed["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.max_iterations 128" in agentx["envs"]["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.detailed_trace_annotation True" in agentx["envs"]["EXTRA_VLLM_ARGS"]
    assert agentx["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "ok"
    torch_profiler = agentx["profiler"]["torch_profiler"]
    assert torch_profiler["num_steps"] == 128
    assert torch_profiler["detailed_annotations"] is True
    launch = agentx["agentx"]["launch_overrides"]
    assert all(not str(token).startswith("--profiler-config") for token in launch.get("append_args") or [])
    assert launch["env"]["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] == "1200"
    assert int(agentx["envs"]["OSL"]) == 8192
    assert int(fixed["envs"]["OSL"]) == 512


def test_sglang_shape_tool_is_shared_and_published_to_the_agentx_server(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _server_patcher

    _pin_profile_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "sitecustomize")
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    tool_dir = tmp_path / "kernel_shape_tool"
    tool_dir.mkdir()
    monkeypatch.setattr(_server_patcher, "kernel_shape_tool_dir", lambda: tool_dir)
    monkeypatch.setattr(_server_patcher, "ensure_sglang_patched_for_tracelens", lambda: True)

    fixed = _fixed_profile("sglang")
    agentx = _agentx_profile(tmp_path, "sglang")

    assert fixed["envs"]["TRACELENS_SHAPE_DISCOVERY"] == "1"
    assert agentx["envs"]["TRACELENS_SHAPE_DISCOVERY"] == "1"
    assert str(tool_dir) in agentx["envs"]["PYTHONPATH"]
    assert agentx["agentx"]["launch_overrides"]["env"]["PYTHONPATH"] == agentx["envs"]["PYTHONPATH"]
    assert agentx["agentx"]["launch_overrides"]["env"]["TRACELENS_SHAPE_DISCOVERY"] == "1"
    assert "shape_discovery" not in json.loads(agentx["envs"]["PROFILE_EXTRA_BODY"])
    assert json.loads(agentx["envs"]["PROFILE_EXTRA_BODY"]) == json.loads(fixed["envs"]["PROFILE_EXTRA_BODY"])
    assert "shape-discovery-for-cuda-graph-profile" not in str(agentx["envs"].get("EXTRA_SGLANG_ARGS", ""))
