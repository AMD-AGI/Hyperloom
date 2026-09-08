# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK must replay the saved AgentX client rather than synthetic traffic."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from hyperloom.inference_optimizer.agentx.deploy import deploy_agentx_assets
from hyperloom.orchestrator.actions.executors import _workload_envs
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _isolate_workload_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith(("AGENTX_", "HYPERLOOM_PROFILE_")) or key in {
            "HYPERLOOM_AGENTX",
            "AIPERF_BIN",
            "WEKA_LOADER_OVERRIDE",
            "MAGPIE_RUN_PHASE",
            "INFERENCEX_PATH",
            "INFERENCE_OPTIMIZER_EXTRA_ENV",
            "INFERENCE_OPTIMIZER_SERVER_ARGS",
            "CONC",
            "ISL",
            "OSL",
            "NUM_PROMPTS",
            "NUM_WARMUPS",
            "RANDOM_RANGE_RATIO",
            "SEED",
            "TP",
            "PORT",
            "MAX_MODEL_LEN",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(_workload_envs, "_visible_gpu_count", lambda: 2)
    monkeypatch.setattr(_workload_envs, "resolve_reference_base", lambda: ("", {}))
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")


def _materialized_coord(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    framework: str = "sglang",
    agentx: bool = True,
    server_override: bool = True,
    absolute_client: bool = False,
) -> tuple[Coordinator, Path, Path, Path, dict]:
    monkeypatch.setenv("FRAMEWORK", framework)
    if agentx:
        monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
        monkeypatch.setenv("AGENTX_DATASET", "semianalysis-cc-traces-weka-with-subagents")
        monkeypatch.setenv("WEKA_LOADER_OVERRIDE", "semianalysis_cc_traces_weka_062126")
        monkeypatch.setenv("AGENTX_MAX_CTX", "98304")
        monkeypatch.setenv("AGENTX_NUM_ENTRIES", "12")
        monkeypatch.setenv("AGENTX_DURATION", "3600")
        monkeypatch.setenv("AGENTX_PROFILE_WINDOW_S", "9")
        monkeypatch.setenv("AGENTX_TRACE_FLUSH_TIMEOUT_S", "123")
        monkeypatch.setenv("AIPERF_BIN", "/opt/agentx/bin/aiperf")
        if server_override:
            monkeypatch.setenv("AGENTX_SERVER_SCRIPT", f"{framework}_mi355x_pinned.sh")
    inferencex = tmp_path / "materialized_inferencex"
    benchmarks = inferencex / "benchmarks"
    benchmarks.mkdir(parents=True)
    deploy_agentx_assets(benchmarks)
    client = benchmarks / "aiperf_client.sh"
    server = benchmarks / f"{framework}_mi355x{'_pinned' if agentx and server_override else ''}.sh"
    server.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": framework,
                    "model": "/models/accepted",
                    "precision": "bf16",
                    "envs": {
                        "CONC": 6,
                        "ISL": 2048,
                        "OSL": 1536,
                        "NUM_PROMPTS": 73,
                        "NUM_WARMUPS": 3,
                        "RANDOM_RANGE_RATIO": 0.25,
                        "SEED": 41,
                        "TP": 2,
                        "RUN_EVAL": "false",
                        "PROFILE_EXTRA_BODY": json.dumps({"start_step": 0, "num_steps": 5}),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    recipe = _workload_envs.materialize_config_with_envs(
        source,
        tmp_path / "accepted_run",
        gpu_type="mi355x",
        inferencex_path=str(inferencex),
        agentx_mode=agentx,
    )
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    if absolute_client:
        config["benchmark"]["benchmark_script"] = str(client)
        recipe.write_text(yaml.safe_dump(config), encoding="utf-8")
    benchmark = config["benchmark"]
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_config_path=str(recipe),
        benchmark_mode="agentx" if agentx else "synthetic",
        framework=framework,
        model_path="/models/accepted",
        gpu_type="mi355x",
        tp=2,
        isl=2048,
        osl=1536,
        conc=6,
        baseline_tput=100.0,
        current_best={
            "action": "explore",
            "tput": 140.0,
            "extra_server_args": "--accepted-flag 1",
            "extra_envs": {"ACCEPTED_SETTING": "1"},
            "remove_args": ["--obsolete-flag"],
            "unset_envs": ["OBSOLETE_SETTING"],
            "args_mode": "replace",
            "optimization_stack": [],
            "measurement": {
                "tput": 140.0,
                "resolved_server_launch_flags": "--accepted-flag 1",
                "benchmark_workspace": str(tmp_path / "accepted_run"),
            },
        },
    )
    coord.shared_state.current_best["measurement"]["launch_identity"] = coord.build_env_spec()["launch_identity"]
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    return coord, recipe, client, server, benchmark


async def _handoff(coord: Coordinator, monkeypatch: pytest.MonkeyPatch) -> dict:
    stop = Mock(side_effect=RuntimeError("stop after handoff write"))
    monkeypatch.setattr("hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path", stop)
    await coord._run_geak_kernel_phase(from_phase="KERNEL")
    path = coord.session_dir / "geak" / "handoff.json"
    assert path.is_file(), coord.shared_state.geak_result
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("metric_override", "expected_metric"),
    [(None, "total"), ("output", "output")],
    ids=["agentx_default_total", "explicit_output"],
)
@pytest.mark.asyncio
async def test_handoff_and_runner_share_effective_grading_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric_override: str | None, expected_metric: str
) -> None:
    coord, _recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    if metric_override is not None:
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", metric_override)
    monkeypatch.setenv("E2E_METRIC", "stale_metric")
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        lambda _name: tmp_path / "mock_geak_runner.py",
    )
    coord.phase_kernel._geak_timeouts = lambda: (60, 90, False)
    captured_env = {}

    def _start_runner(_cmd, *, env, **_kwargs):
        captured_env.update(env)
        process = Mock(returncode=0)
        process.communicate.return_value = ("", "")
        return process

    monkeypatch.setattr("hyperloom.orchestrator.phases.kernel.subprocess.Popen", _start_runner)

    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    assert (handoff["e2e_metric"], captured_env.get("E2E_METRIC")) == (expected_metric, expected_metric)
    assert coord.shared_state.geak_result["error_class"] == "no_result_json"


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.parametrize("absolute_client", [False, True], ids=["script_name", "absolute_script"])
@pytest.mark.asyncio
async def test_handoff_replays_materialized_agentx_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framework: str, absolute_client: bool
) -> None:
    coord, recipe, client, server, benchmark = _materialized_coord(
        tmp_path, monkeypatch, framework=framework, absolute_client=absolute_client
    )

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["bench_client"] == "agentx"
    assert handoff["benchmark_mode"] == "agentx"
    assert handoff["schema_version"] == 3
    replay = handoff["bench_client_config"]
    assert replay["argv"] == ["bash", str(client.resolve())]
    assert replay["cwd"] == str(client.parent.resolve())
    assert replay["env"] == {key: str(value) for key, value in benchmark["envs"].items()}
    assert "MAGPIE_RUN_PHASE" not in replay["env"]
    assert handoff["launch_server_script"] == str(server.resolve())
    assert handoff["launch_server_script"] != replay["argv"][1]
    assert handoff["launch_recipe"] == str(recipe)
    assert handoff["workload"] == {"isl": 2048, "osl": 1536, "conc": 6}
    assert handoff["workload"]["conc"] == int(replay["env"]["CONC"])
    assert replay["env"]["NUM_PROMPTS"] == "73"
    assert replay["env"]["ISL"] == "2048"
    assert replay["env"]["OSL"] == "1536"
    assert "bench_protocol" not in handoff
    assert replay["workload_identity"]
    identity_text = json.dumps(replay["workload_identity"], sort_keys=True)
    assert hashlib.sha256(recipe.read_bytes()).hexdigest() in identity_text
    assert hashlib.sha256(client.read_bytes()).hexdigest() in identity_text
    assert handoff["same_config_reference_status"] == "verified"
    assert handoff["orchestrator_best_tput_same_config"] == pytest.approx(140.0)
    spec = handoff["baseline_env_spec"]
    assert spec["launch_identity"] == handoff["same_config_reference_identity"]
    assert spec["config"]["remove_args"] == ["--obsolete-flag"]
    assert spec["config"]["unset_envs"] == ["OBSOLETE_SETTING"]
    assert spec["config"]["args_mode"] == "replace"


@pytest.mark.asyncio
async def test_handoff_derives_default_server_from_saved_framework_and_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, _recipe, client, server, _benchmark = _materialized_coord(tmp_path, monkeypatch, server_override=False)
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", "ambient_wrong_server.sh")
    monkeypatch.setenv("GPU_TYPE", "mi300x")

    handoff = await _handoff(coord, monkeypatch)

    assert handoff.get("launch_server_script") == str(server.resolve())
    assert handoff["launch_server_script"] != str(client.resolve())


@pytest.mark.asyncio
async def test_handoff_uses_saved_workload_not_ambient_or_placeholder_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, _recipe, _client, _server, benchmark = _materialized_coord(tmp_path, monkeypatch)
    original = await _handoff(coord, monkeypatch)
    coord.shared_state.conc = 99
    coord.shared_state.isl = 7
    coord.shared_state.osl = 8
    for key, value in {
        "HYPERLOOM_AGENTX": "0",
        "AGENTX_DATASET": "different-corpus",
        "WEKA_LOADER_OVERRIDE": "different-loader",
        "AGENTX_MAX_CTX": "512",
        "AIPERF_BIN": "/wrong/aiperf",
        "INFERENCEX_PATH": str(tmp_path / "wrong_checkout"),
        "AGENTX_SERVER_SCRIPT": "wrong_server.sh",
        "CONC": "42",
        "ISL": "64",
        "OSL": "32",
        "NUM_PROMPTS": "2",
        "SEED": "999",
    }.items():
        monkeypatch.setenv(key, value)

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["workload"] == {"isl": 2048, "osl": 1536, "conc": 6}
    assert handoff["bench_client_config"] == original["bench_client_config"]
    assert handoff["bench_client_config"]["env"] == {key: str(value) for key, value in benchmark["envs"].items()}
    assert handoff["launch_server_script"] == original["launch_server_script"]


@pytest.mark.parametrize("changed_source", ["recipe", "client"])
@pytest.mark.asyncio
async def test_handoff_workload_identity_tracks_real_recipe_and_client_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_source: str
) -> None:
    coord, recipe, client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    original = await _handoff(coord, monkeypatch)
    if changed_source == "recipe":
        config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        config["benchmark"]["envs"]["WEKA_LOADER_OVERRIDE"] = "different-saved-loader"
        recipe.write_text(yaml.safe_dump(config), encoding="utf-8")
    else:
        client.write_text(client.read_text(encoding="utf-8") + "\n# Different materialized client.\n", encoding="utf-8")

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["bench_client_config"]["workload_identity"] != original["bench_client_config"]["workload_identity"]


@pytest.mark.asyncio
async def test_handoff_workload_identity_survives_identical_recipe_repath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    original = await _handoff(coord, monkeypatch)
    copied = tmp_path / "accepted_run" / "baseline.resumed.yaml"
    copied.write_bytes(recipe.read_bytes())
    coord.shared_state.baseline_config_path = str(copied)

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["bench_client_config"]["workload_identity"] == original["bench_client_config"]["workload_identity"]


@pytest.mark.parametrize(
    "missing",
    ["recipe", "client", "server", "inferencex_path", "client_as_server", "conc", "invalid_recipe"],
)
@pytest.mark.asyncio
async def test_handoff_fails_closed_without_materialized_agentx_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    coord, recipe, client, server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    if missing in {"recipe", "client", "server"}:
        {"recipe": recipe, "client": client, "server": server}[missing].unlink()
    elif missing == "invalid_recipe":
        recipe.write_text("benchmark: [unterminated\n", encoding="utf-8")
    else:
        config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        benchmark = config["benchmark"]
        if missing == "inferencex_path":
            benchmark.pop("inferencex_path")
        elif missing == "client_as_server":
            benchmark["envs"]["AGENTX_SERVER_SCRIPT"] = "aiperf_client.sh"
        else:
            benchmark["envs"].pop("CONC")
        recipe.write_text(yaml.safe_dump(config), encoding="utf-8")
    runner_lookup = Mock(side_effect=RuntimeError("must not resolve runner for invalid replay"))
    monkeypatch.setattr("hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path", runner_lookup)

    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    runner_lookup.assert_not_called()
    assert coord.shared_state.geak_result["status"] in {"error", "skipped"}
    assert coord.shared_state.geak_result["error_class"] == "agentx_handoff_unavailable"
    assert coord.shared_state.geak_result["error"]
    assert not (tmp_path / "geak" / "handoff.json").exists()
    assert coord.shared_state.current_best["tput"] == 140.0


@pytest.mark.asyncio
async def test_handoff_projects_shared_profile_policy_without_changing_saved_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, recipe, client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    envs = config["benchmark"]["envs"]
    envs.pop("PROFILE_EXTRA_BODY")
    recipe.write_text(yaml.safe_dump(config), encoding="utf-8")
    original_recipe = recipe.read_bytes()
    monkeypatch.setenv("PROFILE_EXTRA_BODY", '{"num_steps": 100000}')
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "100000")
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_STEPS_CAP", "1")
    monkeypatch.setenv("MAGPIE_RUN_PHASE", "client")

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["bench_client"] == "agentx"
    replay = handoff["bench_client_config"]
    expected_env = {key: str(value) for key, value in envs.items()}
    assert {key: value for key, value in replay["env"].items() if key != "PROFILE_EXTRA_BODY"} == expected_env
    assert json.loads(replay["env"]["PROFILE_EXTRA_BODY"]) == {"start_step": 0, "num_steps": 8}
    assert "PROFILE" not in replay["env"]
    assert "MAGPIE_RUN_PHASE" not in replay["env"]
    assert recipe.read_bytes() == original_recipe
    identity = replay["workload_identity"]
    assert identity["recipe_sha256"] == hashlib.sha256(original_recipe).hexdigest()
    assert identity["client_sha256"] == hashlib.sha256(client.read_bytes()).hexdigest()
    client_payload = {key: replay[key] for key in ("argv", "cwd", "env")}
    client_digest = hashlib.sha256(
        json.dumps(client_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert identity["client_config_digest"] == f"sha256:{client_digest}"


@pytest.mark.parametrize(
    ("saved_policy", "expected_steps"),
    [
        ({}, 8),
        ({"HYPERLOOM_PROFILE_MAX_STEPS_CAP": "4"}, 4),
        ({"HYPERLOOM_PROFILE_MAX_STEPS_CAP": "64"}, 8),
        ({"HYPERLOOM_PROFILE_MAX_ITERS": "12"}, 12),
    ],
    ids=["default", "smaller_cap", "agentx_cap", "explicit_steps"],
)
@pytest.mark.asyncio
async def test_handoff_profile_policy_uses_saved_overrides_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_policy: dict, expected_steps: int
) -> None:
    coord, recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    envs = config["benchmark"]["envs"]
    envs["PROFILE_EXTRA_BODY"] = '{"merge_profiles": false}'
    envs.update(saved_policy)
    recipe.write_text(yaml.safe_dump(config), encoding="utf-8")
    original = recipe.read_bytes()
    monkeypatch.setenv("HYPERLOOM_PROFILE_MAX_ITERS", "999")

    handoff = await _handoff(coord, monkeypatch)

    replay_env = handoff["bench_client_config"]["env"]
    assert json.loads(replay_env["PROFILE_EXTRA_BODY"]) == {
        "merge_profiles": False,
        "start_step": 0,
        "num_steps": expected_steps,
    }
    assert {key: value for key, value in replay_env.items() if key != "PROFILE_EXTRA_BODY"} == {
        key: str(value) for key, value in envs.items() if key != "PROFILE_EXTRA_BODY"
    }
    assert recipe.read_bytes() == original


@pytest.mark.parametrize(
    "invalid_body",
    ["{invalid", "[]", '{"num_steps": 0}', '{"num_steps": -1}', '{"num_steps": true}', '{"num_steps": "8"}'],
    ids=["invalid_json", "not_mapping", "zero", "negative", "boolean", "string"],
)
@pytest.mark.asyncio
async def test_handoff_rejects_explicit_invalid_profile_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_body: str
) -> None:
    coord, recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["benchmark"]["envs"]["PROFILE_EXTRA_BODY"] = invalid_body
    recipe.write_text(yaml.safe_dump(config), encoding="utf-8")
    runner_lookup = Mock(side_effect=RuntimeError("must not resolve runner for invalid profile bounds"))
    monkeypatch.setattr("hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path", runner_lookup)

    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    runner_lookup.assert_not_called()
    assert coord.shared_state.geak_result["error_class"] == "agentx_handoff_unavailable"
    assert "PROFILE_EXTRA_BODY" in coord.shared_state.geak_result["error"]


@pytest.mark.asyncio
async def test_handoff_profile_projection_is_bound_by_client_config_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["benchmark"]["envs"].pop("PROFILE_EXTRA_BODY")
    recipe.write_text(yaml.safe_dump(config), encoding="utf-8")
    original = await _handoff(coord, monkeypatch)
    monkeypatch.setattr(_workload_envs, "_AGENTX_PROFILE_MAX_ITERS", 6)

    handoff = await _handoff(coord, monkeypatch)

    original_identity = original["bench_client_config"]["workload_identity"]
    identity = handoff["bench_client_config"]["workload_identity"]
    assert identity["recipe_sha256"] == original_identity["recipe_sha256"]
    assert identity["client_sha256"] == original_identity["client_sha256"]
    assert identity["client_config_digest"] != original_identity["client_config_digest"]
    assert json.loads(handoff["bench_client_config"]["env"]["PROFILE_EXTRA_BODY"])["num_steps"] == 6


@pytest.mark.asyncio
async def test_handoff_does_not_project_sglang_profile_body_for_vllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch, framework="vllm")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config["benchmark"]["envs"].pop("PROFILE_EXTRA_BODY")
    recipe.write_text(yaml.safe_dump(config), encoding="utf-8")

    handoff = await _handoff(coord, monkeypatch)

    assert "PROFILE_EXTRA_BODY" not in handoff["bench_client_config"]["env"]


@pytest.mark.asyncio
async def test_handoff_keeps_saved_model_and_framework_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, _recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch)
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("MODEL_PATH", "/models/wrong")
    monkeypatch.setenv("GPU_TYPE", "mi300x")

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["framework"] == "sglang"
    assert handoff["model_path"] == "/models/accepted"
    assert handoff["gpu_type"] == "mi355x"


@pytest.mark.asyncio
async def test_saved_synthetic_mode_is_not_reclassified_by_ambient_agentx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, _recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch, agentx=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["bench_client"] == "auto"
    assert "bench_client_config" not in handoff


@pytest.mark.asyncio
async def test_synthetic_handoff_keeps_existing_benchmark_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord, _recipe, _client, _server, _benchmark = _materialized_coord(tmp_path, monkeypatch, agentx=False)

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["schema_version"] == 3
    assert handoff["bench_client"] == "auto"
    assert "bench_client_config" not in handoff
    assert "benchmark_mode" not in handoff
    assert "launch_server_script" not in handoff
    assert handoff["workload"] == {"isl": 2048, "osl": 1536, "conc": 6}
    assert handoff["bench_protocol"] == {
        "random_range_ratio": 0.25,
        "num_prompts": 73,
        "num_warmups": 3,
        "seed": 41,
    }
