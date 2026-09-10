# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK uses its supported clients; Hyperloom owns canonical AgentX validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _isolate_workload_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith(("AGENTX_", "HYPERLOOM_PROFILE_")) or key in {
            "HYPERLOOM_AGENTX",
            "HYPERLOOM_PERF_METRIC",
            "E2E_METRIC",
            "FRAMEWORK",
            "MODEL_PATH",
            "GPU_TYPE",
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
            "MAX_MODEL_LEN",
            "GPU_MEMORY_UTILIZATION",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        }:
            monkeypatch.delenv(key, raising=False)


def _coord(tmp_path: Path, *, framework: str = "sglang", agentx: bool = True, metric: str = "total") -> Coordinator:
    benchmarks = tmp_path / "InferenceX" / "benchmarks"
    benchmarks.mkdir(parents=True)
    for name in ("benchmark_lib.sh", f"{framework}_mi355x.sh", "aiperf_client.sh"):
        (benchmarks / name).write_text("# stub\n", encoding="utf-8")
    workload_spec = (
        {
            "workload_spec": {
                "kind": "agentx_trace_replay",
                "client": "aiperf",
                "scenario": "inferencex-agentx-mvp",
                "corpus": "semianalysis_cc_traces_weka_062126",
                "duration_s": 3600,
                "geak_loop_duration_s": 900,
                "concurrency": 6,
                "metric_basis": "aggregate_total_token_tok_s" if metric == "total" else "aggregate_output_tok_s",
            }
        }
        if agentx
        else {}
    )
    recipe = tmp_path / "baseline.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": framework,
                    "model": "/models/accepted",
                    "runner_type": "mi355x",
                    "inferencex_path": str(benchmarks.parent),
                    "benchmark_script": "aiperf_client.sh" if agentx else f"{framework}_mi355x.sh",
                    **workload_spec,
                    "envs": {
                        "CONC": 6,
                        "ISL": 2048,
                        "OSL": 1536,
                        "NUM_PROMPTS": 73,
                        "NUM_WARMUPS": 3,
                        "RANDOM_RANGE_RATIO": 0.25,
                        "SEED": 41,
                        "TP": 2,
                        "ROCR_VISIBLE_DEVICES": "6,7",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
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
    return coord


async def _handoff(coord: Coordinator, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        Mock(side_effect=RuntimeError("stop after handoff write")),
    )
    await coord._run_geak_kernel_phase(from_phase="KERNEL")
    return json.loads((coord.session_dir / "geak" / "handoff.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.asyncio
async def test_agentx_handoff_keeps_supported_schema_and_frozen_launch_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framework: str
) -> None:
    coord = _coord(tmp_path, framework=framework)
    monkeypatch.setenv("FRAMEWORK", "vllm" if framework == "sglang" else "sglang")
    monkeypatch.setenv("MODEL_PATH", "/models/wrong")
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("CONC", "99")

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["schema_version"] == 3
    assert handoff["bench_client"] == "auto"
    assert handoff["bench_launcher"] == "native"
    assert not {"bench_client_config", "benchmark_mode", "workload_identity"} & handoff.keys()
    assert handoff["launch_server_script"] == str(tmp_path / "InferenceX" / "benchmarks" / f"{framework}_mi355x.sh")
    recipe = yaml.safe_load(Path(coord.shared_state.baseline_config_path).read_text(encoding="utf-8"))
    assert handoff["workload_spec"] == recipe["benchmark"]["workload_spec"]
    assert handoff["workload_spec"]["metric_basis"] == "aggregate_total_token_tok_s"
    assert handoff["e2e_metric"] == "total"
    assert handoff["framework"] == framework
    assert handoff["model_path"] == "/models/accepted"
    assert handoff["gpu_type"] == "mi355x"
    assert handoff["launch_recipe"] == coord.shared_state.baseline_config_path
    assert handoff["workload"] == {"isl": 2048, "osl": 1536, "conc": 6}
    assert handoff["gpu_pin"]["var"] == "ROCR_VISIBLE_DEVICES"
    assert handoff["gpu_pin"]["ids"] == [6, 7]
    assert handoff["gpu_ids"] == "0,1"
    assert handoff["gpu_ids_space"] == "logical"
    assert handoff["tp"] == 2
    assert handoff["accepted_flags"] == "--accepted-flag 1"
    assert "ACCEPTED_SETTING=1" in handoff["accepted_env"]
    assert handoff["same_config_reference_status"] == "unverified"
    assert handoff["same_config_reference_verification_status"] == "unverified_workload"
    assert handoff["orchestrator_best_tput_same_config"] == 0.0
    assert handoff["raw_baseline_tput"] == 0.0
    spec = handoff["baseline_env_spec"]
    assert spec["launch_identity"] == handoff["same_config_reference_identity"]
    assert spec["config"]["remove_args"] == ["--obsolete-flag"]
    assert spec["config"]["unset_envs"] == ["OBSOLETE_SETTING"]
    assert spec["config"]["args_mode"] == "replace"


@pytest.mark.parametrize(
    ("metric_override", "expected_metric"),
    [(None, "total"), ("intvty_v1", "total"), ("composite_v1", "output"), ("output", "output")],
)
@pytest.mark.asyncio
async def test_agentx_geak_metric_aligned_result_is_only_a_proposal_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    metric_override: str | None,
    expected_metric: str,
) -> None:
    if metric_override is not None:
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", metric_override)
    coord = _coord(tmp_path, metric=expected_metric)
    monkeypatch.setenv("E2E_METRIC", "output" if expected_metric == "total" else "total")
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
    with caplog.at_level("INFO", logger="hyperloom.orchestrator.phases.kernel"):
        await coord._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    assert (handoff["e2e_metric"], captured_env["E2E_METRIC"]) == (expected_metric, expected_metric)
    expected_basis = "aggregate_total_token_tok_s" if expected_metric == "total" else "aggregate_output_tok_s"
    assert handoff["workload_spec"]["metric_basis"] == expected_basis
    assert handoff["same_config_reference_status"] == "unverified"
    assert handoff["same_config_reference_verification_status"] == "unverified_workload"
    assert handoff["orchestrator_best_tput_same_config"] == 0.0
    assert handoff["raw_baseline_tput"] == 0.0
    assert "canonical AgentX validation remains in Hyperloom" in caplog.text
    assert coord.shared_state.benchmark_mode == "agentx"
    assert coord.shared_state.current_best["tput"] == 140.0
    assert coord.shared_state.geak_result["error_class"] == "no_result_json"


@pytest.mark.parametrize(
    ("metric_override", "expected_metric"),
    [(None, "output"), ("intvty_v1", "total"), ("composite_v1", "output"), ("output", "output")],
)
@pytest.mark.asyncio
async def test_synthetic_handoff_keeps_existing_protocol_and_metric_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric_override: str | None, expected_metric: str
) -> None:
    coord = _coord(tmp_path, agentx=False)
    if metric_override is not None:
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", metric_override)

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["bench_client"] == "auto"
    assert "bench_client_config" not in handoff
    assert "bench_launcher" not in handoff
    assert "launch_server_script" not in handoff
    assert "workload_spec" not in handoff
    assert handoff["e2e_metric"] == expected_metric
    assert handoff["same_config_reference_status"] == "verified"
    assert handoff["same_config_reference_verification_status"] == "verified_observed"
    assert handoff["orchestrator_best_tput_same_config"] == pytest.approx(140.0)
    assert handoff["raw_baseline_tput"] == pytest.approx(100.0)
    assert handoff["bench_protocol"] == {
        "random_range_ratio": 0.25,
        "num_prompts": 73,
        "num_warmups": 3,
        "seed": 41,
    }
