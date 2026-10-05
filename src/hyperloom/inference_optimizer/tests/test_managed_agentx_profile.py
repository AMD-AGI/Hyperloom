# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Managed diagnostic captures preserve candidates and cannot become KEEP evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.inference_optimizer.agentx import identity, managed, native
from hyperloom.inference_optimizer.tests.test_agentx_optimization_identity import _candidate_artifacts
from hyperloom.orchestrator.actions.executors import baseline, profile
from hyperloom.orchestrator.actions.executors._native_candidate import apply_native_candidate
from hyperloom.orchestrator.actions.executors._managed_profile_result import diagnostic_profile_result
from hyperloom.orchestrator.actions.executors._native_profile import prepare_managed_profile
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement


def _accepted(tmp_path):
    config, report = _candidate_artifacts(tmp_path / "accepted-artifacts", env={"SGLANG_USE_AITER": "1"})
    evidence = report["agentx_metrics"]["server_launch"]
    checkout = tmp_path / "inferencex"
    (checkout / "benchmarks").mkdir(parents=True)
    for name in ("srt_agentic.sh", "benchmark_lib.sh"):
        (checkout / "benchmarks" / name).write_text("# fixture\n")
    spec = {"version": 1, "argv": evidence["base_argv"], "source_files": {}}
    config["agentx"]["resolved"]["server-launch-spec"] = spec
    config.update(benchmark_script="srt_agentic.sh", inferencex_path=str(checkout), run_mode="local")
    config["profiler"] = {"torch_profiler": {"enabled": False}}
    config.setdefault("envs", {})["RUN_EVAL"] = "false"
    apply_native_candidate(config)
    evidence["overrides_sha256"] = identity.canonical_sha256(config["agentx"]["launch_overrides"])
    evidence.update(owner="magpie", server_spec_sha256=identity.canonical_sha256(spec), recipe_source_files={})
    evidence["evidence_sha256"] = identity.canonical_sha256(
        {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    )
    config["workload_spec"]["server_launch"] = evidence
    config["workload_spec"]["resolved_topology"] = {"gpu_count": 2, "tp": 2, "pp": 1, "pcp_size": 1, "ep": 1, "conc": 8}
    config["workload_spec"]["execution"]["workload_fingerprint"] = identity.native_workload_fingerprint(
        config, "b" * 64
    )
    accepted = tmp_path / "accepted.yaml"
    accepted.write_text(yaml.safe_dump({"benchmark": config}))
    state = SimpleNamespace(
        benchmark_mode="agentx",
        agentx_backend="native",
        agentx_epoch=4,
        baseline_config_path=str(accepted),
        baseline_double_run=False,
        current_best_measurement={"materialized_config": str(accepted), "agentx_server_launch": evidence},
    )
    return config, evidence, state


def _write_capture(workspace, evidence, *, profiles=2, num_steps=20):
    workspace.mkdir(parents=True, exist_ok=True)
    root = workspace / "torch_trace" / ("a" * 32)
    rows = []
    for index in range(1, profiles + 1):
        directory = root / f"profile_{index:03d}"
        directory.mkdir(parents=True)
        # Names intentionally have identical TP components: the manifest's global ranks are authoritative.
        files = [directory / f"worker-{rank}-TP-0-EP-0.trace.json" for rank in range(2)]
        for rank, path in enumerate(files):
            path.write_text(
                json.dumps(
                    {
                        "distributedInfo": {"rank": rank},
                        "traceEvents": [{"cat": "kernel", "ph": "X", "name": "kernel", "dur": 1}],
                    }
                )
            )
        rows.append(
            {
                "status": "complete",
                "capture_id": root.name,
                "framework": evidence["framework"],
                "num_steps": num_steps,
                "profile_index": index,
                "expected_ranks": 2,
                "trace_dir": str(directory),
                "trace_files": list(map(str, files)),
                "rank_trace_files": {str(rank): [str(path)] for rank, path in enumerate(files)},
            }
        )
    capture = {
        "status": "complete",
        "capture_id": root.name,
        "framework": evidence["framework"],
        "num_steps": num_steps,
        "expected_ranks": 2,
        "requested_profiles": 99,
        "max_profiles": 18,
        "planned_profiles": 18,
        "effective_profiles": profiles,
        "completed_profiles": profiles,
        "stop_reason": "insufficient_measurement_time",
        "profiles": rows,
        "trace_files": [path for row in rows for path in row["trace_files"]],
    }
    (root / "capture.json").write_text(json.dumps(capture))
    evidence = deepcopy(evidence)
    evidence["torch_profiler"] = {"capture_id": root.name, "trace_dir": str(root), "num_steps": num_steps}
    evidence["evidence_sha256"] = identity.canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    report = {
        "success": True,
        "errors": [],
        "benchmark_valid": False,
        "publishable": False,
        "agentx_metrics": {
            "diagnostic_only": True,
            "server_launch": evidence,
            "profile_capture": capture,
            "profile_analyses": [{"profile_index": index} for index in range(1, profiles + 1)],
        },
    }
    (workspace / "agentx_server_launch.json").write_text(json.dumps(evidence))
    (workspace / "benchmark_report.json").write_text(json.dumps(report))
    return report


def _write_step_trace(path, num_steps):
    events = []
    for index in range(num_steps):
        events.extend(
            [
                {"ph": "X", "cat": "user_annotation", "name": "step[DECODE bs=8]", "ts": index * 10, "dur": 10},
                {"ph": "X", "cat": "kernel", "name": "fixture_kernel", "ts": index * 10 + 1, "dur": 5},
            ]
        )
    path.write_text(json.dumps({"traceEvents": events}))


def test_profile_derivation_keeps_accepted_candidate_and_structured_profiler_options(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "_validate_managed_receipt", lambda *args: [])
    config, _, state = _accepted(tmp_path)
    accepted = Path(state.baseline_config_path)
    original = accepted.read_bytes()
    params = {
        "base_extra_args": "--must-not-reapply",
        "base_extra_envs": {"SGLANG_USE_AITER": "0"},
        "num_steps": 12,
        "num_profiles": 99,
        "start_seconds": 30,
        "interval_seconds": 200,
        "capture_timeout_seconds": 400,
        "flush_timeout_seconds": 1900,
        "detailed_annotations": True,
    }
    prepare_managed_profile(params, state, tmp_path / "profile")
    actual = yaml.safe_load(Path(params["config_path"]).read_text())["benchmark"]
    assert actual["agentx"] == config["agentx"]
    assert actual["envs"] == config["envs"]
    assert actual["profiler"]["torch_profiler"] == {
        "enabled": True,
        "num_steps": 12,
        "num_profiles": 99,
        "start_seconds": 30,
        "interval_seconds": 200,
        "capture_timeout_seconds": 400,
        "flush_timeout_seconds": 1900,
        "detailed_annotations": True,
    }
    assert actual["workload_spec"]["profile_parent"]["config_sha256"] == hashlib.sha256(original).hexdigest()
    assert params["native_launch_overrides"] == config["agentx"]["launch_overrides"]
    assert accepted.read_bytes() == original
    assert (
        managed.identity_benchmark(actual, config["workload_spec"]["execution"]["workload_fingerprint"])["profiler"]
        == config["profiler"]
    )
    with pytest.raises(ValueError, match="only.*ProfileExecutor"):
        materialize_config_with_envs(Path(params["config_path"]), tmp_path / "baseline")


@pytest.mark.parametrize(
    "change",
    [
        None,
        "returncode",
        "runtime",
        "partial",
        "rank",
        "missing",
        "manifest",
        "receipt",
        "stale",
        "capture_id",
        "capture_dir",
        "gpu_count",
        "framework",
        "num_steps",
        "round_num_steps",
    ],
)
def test_diagnostic_report_requires_complete_runtime_and_bound_round_artifacts(tmp_path, monkeypatch, change):
    config, evidence, _ = _accepted(tmp_path)
    config["profiler"]["torch_profiler"]["enabled"] = True
    config_path = tmp_path / "profile.yaml"
    config_path.write_text(yaml.safe_dump({"benchmark": config}))
    workspace = tmp_path / "benchmark_profile"
    report = _write_capture(workspace, evidence)
    monkeypatch.setattr(identity, "_validate_managed_receipt", lambda *args: [])
    capture = report["agentx_metrics"]["profile_capture"]
    if change == "runtime":
        report["success"] = False
    if change == "partial":
        capture["status"] = "failed"
    if change == "rank":
        capture["profiles"][-1]["rank_trace_files"].pop("1")
    if change == "missing":
        Path(capture["profiles"][-1]["trace_files"][0]).unlink()
    if change == "manifest":
        capture["completed_profiles"] = 88
    if change == "gpu_count":
        config["workload_spec"]["resolved_topology"]["gpu_count"] = 4
        config_path.write_text(yaml.safe_dump({"benchmark": config}))
    if change == "framework":
        capture["framework"] = "vllm"
    if change == "num_steps":
        capture["num_steps"] += 1
    if change == "round_num_steps":
        capture["profiles"][-1]["num_steps"] += 1
    if change in {"capture_id", "capture_dir"}:
        launch = report["agentx_metrics"]["server_launch"]
        key = "capture_id" if change == "capture_id" else "trace_dir"
        launch["torch_profiler"][key] = "b" * 32 if change == "capture_id" else str(workspace / "other")
        launch["evidence_sha256"] = identity.canonical_sha256(
            {key: value for key, value in launch.items() if key != "evidence_sha256"}
        )
        (workspace / "agentx_server_launch.json").write_text(json.dumps(launch))
    (workspace / "torch_trace" / capture["capture_id"] / "capture.json").write_text(json.dumps(capture))
    if change == "receipt":
        (workspace / "agentx_server_launch.json").write_text("{}")
    if change == "stale":
        os.utime(workspace / "benchmark_report.json", (1, 1))
    result = diagnostic_profile_result(
        report,
        workspace=workspace,
        config_path=config_path,
        returncode=1 if change == "returncode" else 0,
        subprocess_started_unix=time.time(),
    )
    assert result["status"] == ("failed" if change else "succeeded"), result
    assert result["trace_input_ready"] is (not bool(change))
    assert result["valid_measurement"] is False
    assert result["submission_valid"] is False
    if change in {"capture_id", "capture_dir", "gpu_count", "framework", "num_steps", "round_num_steps"}:
        expected = {
            "capture_id": "capture ID",
            "capture_dir": "capture directory",
            "gpu_count": "rank count",
            "framework": "capture framework",
            "num_steps": "step count",
            "round_num_steps": "round num_steps",
        }
        assert expected[change] in result["error"]
    if not change:
        assert result["selected_profile_index"] == 2
        assert result["trace_files"] == capture["profiles"][1]["trace_files"]
        assert result["main_trace_path"] == capture["profiles"][1]["rank_trace_files"]["0"][0]
        assert result["profile_capture"]["effective_profiles"] == 2
        assert not extract_benchmark_measurement(report, workspace=workspace, materialized_config_path=config_path)[
            "valid_measurement"
        ]


def test_single_profile_manifest_keeps_the_same_kernel_input_contract(tmp_path, monkeypatch):
    config, evidence, _ = _accepted(tmp_path)
    config["profiler"]["torch_profiler"]["enabled"] = True
    config_path = tmp_path / "profile.yaml"
    config_path.write_text(yaml.safe_dump({"benchmark": config}))
    workspace = tmp_path / "benchmark_profile"
    report = _write_capture(workspace, evidence, profiles=1)
    capture = dict(report["agentx_metrics"]["profile_capture"]["profiles"][0])
    capture.pop("profile_index")
    capture.pop("trace_dir")
    report["agentx_metrics"]["profile_capture"] = capture
    (workspace / "torch_trace" / capture["capture_id"] / "capture.json").write_text(json.dumps(capture))
    monkeypatch.setattr(identity, "_validate_managed_receipt", lambda *args: [])
    result = diagnostic_profile_result(
        report, workspace=workspace, config_path=config_path, returncode=0, subprocess_started_unix=time.time()
    )
    assert result["trace_input_ready"] is True
    assert result["selected_profile_index"] == 1
    assert result["main_trace_path"] == capture["rank_trace_files"]["0"][0]
    assert result["trace_files"] == capture["trace_files"]


@pytest.mark.asyncio
@pytest.mark.parametrize("returncode", [0, 1])
async def test_managed_profile_executor_runs_native_materialization_and_diagnostic_gate(
    tmp_path, monkeypatch, returncode
):
    config, evidence, state = _accepted(tmp_path)
    fingerprint = config["workload_spec"]["execution"]["workload_fingerprint"]
    for key in list(os.environ):
        if key.startswith(("AGENTX_", "HYPERLOOM_AGENTX_EXPECTED_")) or key in {
            "HYPERLOOM_AGENTX_GPU_COUNT",
            "HYPERLOOM_AGENTIC_BACKEND",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(tmp_path / "no-leaks"))
    monkeypatch.setenv("INFERENCEX_PATH", config["inferencex_path"])
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setattr(identity, "_validate_managed_receipt", lambda *args: [])
    resolved = []

    def resolve(bench, **kwargs):
        managed.identity_benchmark(bench, fingerprint)
        resolved.append(deepcopy(bench))
        return bench["workload_spec"]["resolved_topology"]

    monkeypatch.setattr(native, "resolve_native_recipe", resolve)
    captured = {}

    def run(cmd, **kwargs):
        path = Path(cmd[cmd.index("--benchmark-config") + 1])
        bench = yaml.safe_load(path.read_text())["benchmark"]
        captured.update(bench)
        launched = deepcopy(evidence)
        launched["overrides_sha256"] = identity.canonical_sha256(bench["agentx"]["launch_overrides"])
        launched["evidence_sha256"] = identity.canonical_sha256(
            {k: v for k, v in launched.items() if k != "evidence_sha256"}
        )
        workspace = Path(cmd[cmd.index("--output-dir") + 1]) / "benchmark_sglang_profile"
        report = _write_capture(workspace, launched, num_steps=bench["profiler"]["torch_profiler"]["num_steps"])
        for capture in report["agentx_metrics"]["profile_capture"]["profiles"]:
            directory = Path(capture["trace_dir"])
            # A misplaced neighbouring config must not override the verified materialized workload.
            (directory.parent / "config.yaml").write_text("framework: vllm\n  num_steps: 999\n  CONC: 99\n")
            for trace in capture["trace_files"]:
                _write_step_trace(Path(trace), 20)
        return subprocess.CompletedProcess(cmd, returncode, "", "runtime failed" if returncode else "")

    monkeypatch.setattr(baseline, "run_with_session_kill", run)
    executor = profile.ProfileExecutor(magpie_python="python3", session_dir=tmp_path)
    executor.shared_state = state
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            kind="profile",
            task_id="native-profile",
            params={
                "output_dir": str(tmp_path / "run"),
                "num_steps": 20,
                "num_profiles": 99,
            },
        ),
        extra={},
    )
    result = await executor(ctx)
    assert result["status"] == ("failed" if returncode else "succeeded"), result
    assert result["trace_input_ready"] is (returncode == 0)
    assert captured["benchmark_script"] == "srt_agentic.sh"
    assert captured["profiler"]["torch_profiler"]["num_profiles"] == 99
    assert "PROFILE_EXTRA_BODY" not in captured["envs"]
    assert "EXTRA_SGLANG_ARGS" not in captured["envs"]
    assert not (tmp_path / "run" / ".agentx-profile-inferencex").exists()
    assert len(resolved) == 2
    if returncode == 0:
        assert result["valid_measurement"] is False
        assert "trace_validate" in result
        assert all("trace_validate" in item for item in result["profile_rounds"])
        for capture in result["profile_rounds"]:
            assert Path(capture["trace_dir"]).name == f"profile_{capture['profile_index']:03d}"
            validation = capture["trace_validate"]
            assert validation["probe_status"] == "ok"
            assert validation["workload_params"] == {
                "source": result["materialized_config"],
                "framework": "sglang",
                "num_steps": 20,
                "conc": 8,
                "osl": None,
                "r": 1.0,
            }
            assert validation["rank_level"][0]["split_forecast"]["num_steps_param"] == 20
            assert any(check["status"] == "failed" for check in validation["checks"])
            assert json.loads(Path(capture["trace_validate_path"]).read_text()) == validation


def test_legacy_profile_certification_keeps_adjacent_config_context(tmp_path):
    directory = tmp_path / "torch_trace"
    directory.mkdir()
    _write_step_trace(directory / "rank0.trace.json", 20)
    (tmp_path / "config.yaml").write_text(
        "framework: sglang\nprofiler:\n  torch_profiler:\n    num_steps: 12\n"
        "envs:\n  CONC: 3\n  OSL: 256\n  RANDOM_RANGE_RATIO: 0.5\n"
    )
    certificate = profile._certify_trace_dir(directory, "sglang")
    assert certificate["workload_params"] == {
        "source": str(tmp_path / "config.yaml"),
        "framework": "sglang",
        "num_steps": 12,
        "conc": 3,
        "osl": 256.0,
        "r": 0.5,
    }
    assert certificate["rank_level"][0]["split_forecast"]["num_steps_param"] == 12
