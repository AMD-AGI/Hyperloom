# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native candidates must prove their launch without changing the replay workload."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from hyperloom.common.perf_metric import graded_axes_of, perf_snapshot_from_mapping, rounds_are_comparable
from hyperloom.inference_optimizer.agentx.identity import canonical_sha256, native_workload_fingerprint
from hyperloom.inference_optimizer.agentx.identity import validate_server_launch
from hyperloom.inference_optimizer.tests.test_benchmark_result import (
    _native_agentx_report,
    _write_native_config_snapshot,
)
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement
from hyperloom.orchestrator.state.shared_state import resolve_graded_comparison


def _write_candidate_config(workspace, config):
    (workspace / "hyperloom-input.yaml").write_text(json.dumps({"benchmark": config}))
    (workspace / "config.yaml").write_text(
        json.dumps({key: value for key, value in config.items() if key != "workload_spec"})
    )


def _candidate_artifacts(workspace, *, env=None):
    workspace.mkdir(parents=True, exist_ok=True)
    _write_native_config_snapshot(workspace)
    config = json.loads((workspace / "config.yaml").read_text())
    overrides = {"version": 1, "env": env or {}}
    config["agentx"]["launch_overrides"] = overrides
    static = "b" * 64
    config["workload_spec"]["execution"] = {
        "static_execution_fingerprint": static,
        "workload_fingerprint": native_workload_fingerprint(config, static),
    }
    _write_candidate_config(workspace, config)
    launch = {
        "version": 1,
        "framework": "sglang",
        "base_argv": ["python3", "-m", "sglang.launch_server"],
        "effective_argv": ["python3", "-m", "sglang.launch_server"],
        "base_env": dict.fromkeys(env or {}),
        "effective_env": env or {},
        "source_files": {},
        "resolved_executable": "/usr/bin/python3",
        "runtime_environment": env or {},
        "overrides_sha256": canonical_sha256(overrides),
    }
    launch["evidence_sha256"] = canonical_sha256(launch)
    (workspace / "agentx_server_launch.json").write_text(json.dumps(launch))
    report = _native_agentx_report()
    report["agentx_metrics"]["server_launch"] = launch
    return config, report


def test_native_server_candidates_retain_comparable_workload_and_distinct_provenance(tmp_path):
    baseline_dir, candidate_dir = tmp_path / "baseline", tmp_path / "candidate"
    _, baseline_report = _candidate_artifacts(baseline_dir)
    _, candidate_report = _candidate_artifacts(candidate_dir, env={"SGLANG_USE_AITER": "1"})
    baseline = extract_benchmark_measurement(
        baseline_report, workspace=baseline_dir, materialized_config_path=baseline_dir / "hyperloom-input.yaml"
    )
    candidate = extract_benchmark_measurement(
        candidate_report, workspace=candidate_dir, materialized_config_path=candidate_dir / "hyperloom-input.yaml"
    )
    assert baseline["submission_valid"] is True
    assert candidate["submission_valid"] is True
    assert candidate["agentx_workload_fingerprint"] == baseline["agentx_workload_fingerprint"]
    assert candidate["agentx_candidate_fingerprint"] != baseline["agentx_candidate_fingerprint"]
    persisted = graded_axes_of(baseline)
    persisted["output_throughput"] = baseline["output_throughput"]
    assert rounds_are_comparable(perf_snapshot_from_mapping(candidate), perf_snapshot_from_mapping(persisted))


@pytest.mark.parametrize("change", ["model", "corpus", "concurrency", "dependencies", "environment"])
def test_native_workload_identity_changes_when_fixed_inputs_change(tmp_path, change):
    config, _ = _candidate_artifacts(tmp_path)
    altered = deepcopy(config)
    static = "b" * 64
    if change == "model":
        altered["model"] = "different/model"
    elif change == "corpus":
        altered["agentx"]["recipe"] = "different-replay"
    elif change == "concurrency":
        altered["agentx"]["resolved"]["conc"] = 64
    elif change == "dependencies":
        static = "c" * 64
    else:
        altered["envs"] = {"MODEL_PATH": "/other/checkpoint"}
    assert native_workload_fingerprint(altered, static) != native_workload_fingerprint(config, "b" * 64)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("missing_report", "server_launch_missing"),
        ("missing_artifact", "server_launch_artifact_missing"),
        ("candidate", "server_launch_candidate_mismatch"),
        ("env", "server_launch_environment_mismatch"),
        ("source", "server_launch_source_mismatch"),
        ("workload", "workload_fingerprint_invalid"),
    ],
)
def test_native_candidate_without_matching_launch_evidence_cannot_be_promoted(tmp_path, change, error):
    config, report = _candidate_artifacts(tmp_path, env={"SGLANG_USE_AITER": "1"})
    evidence = report["agentx_metrics"]["server_launch"]
    if change == "missing_report":
        report["agentx_metrics"].pop("server_launch")
    elif change == "missing_artifact":
        (tmp_path / "agentx_server_launch.json").unlink()
    elif change == "candidate":
        evidence["overrides_sha256"] = "c" * 64
    elif change == "env":
        evidence["effective_env"] = {}
    elif change == "source":
        config["agentx"]["launch_overrides"]["source_files"] = {"/framework/kernel.py": "c" * 64}
    else:
        config["workload_spec"]["execution"]["workload_fingerprint"] = "c" * 64
    _write_candidate_config(tmp_path, config)
    measurement = extract_benchmark_measurement(
        report, workspace=tmp_path, materialized_config_path=tmp_path / "hyperloom-input.yaml"
    )
    assert measurement["submission_valid"] is False
    assert error in measurement["native_agentx_protocol_errors"]
    assert "agentx_workload_fingerprint" not in measurement


def test_epoch_three_grading_rejects_missing_or_different_workload_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    _, report = _candidate_artifacts(tmp_path)
    baseline = extract_benchmark_measurement(
        report, workspace=tmp_path, materialized_config_path=tmp_path / "hyperloom-input.yaml"
    )
    state = SimpleNamespace(
        benchmark_mode="agentx", agentx_epoch=3, framework="sglang", baseline_perf=baseline, current_best={}
    )
    candidate = {**baseline, "e2e_norm_intvty_p50": baseline["e2e_norm_intvty_p50"] * 1.1}
    assert resolve_graded_comparison(state, candidate).verdict == "KEEP"
    candidate["agentx_workload_fingerprint"] = "c" * 64
    assert resolve_graded_comparison(state, candidate).verdict == "REVERT"
    candidate.pop("agentx_workload_fingerprint")
    result = resolve_graded_comparison(state, candidate)
    assert result.verdict == "REVERT"
    assert result.degrade_reason == "native_workload_identity_missing"


def test_custom_native_result_binds_the_real_model_context_and_config(tmp_path):
    config, report = _candidate_artifacts(tmp_path)
    declared = {"native_context_length": 32768, "max_model_len": 16384, "model_config_sha256": "d" * 64}
    resolved = config["agentx"]["resolved"]
    resolved.update({"custom": True, **{key.replace("_", "-"): value for key, value in declared.items()}})
    config["workload_spec"]["execution"]["workload_fingerprint"] = native_workload_fingerprint(config, "b" * 64)
    _write_candidate_config(tmp_path, config)
    report["agentx_metrics"]["recipe"].update(custom_recipe=True, **declared)
    raw_path = tmp_path / "inferencex_result.json"
    raw = json.loads(raw_path.read_text())
    raw.update(custom_recipe=True, **declared)
    raw_path.write_text(json.dumps(raw))
    measurement = extract_benchmark_measurement(
        report, workspace=tmp_path, materialized_config_path=tmp_path / "hyperloom-input.yaml"
    )
    assert measurement["submission_valid"] is True

    raw["max_model_len"] = 8192
    raw_path.write_text(json.dumps(raw))
    measurement = extract_benchmark_measurement(
        report, workspace=tmp_path, materialized_config_path=tmp_path / "hyperloom-input.yaml"
    )
    assert measurement["submission_valid"] is False
    assert "max_model_len_mismatch" in measurement["native_agentx_protocol_errors"]


@pytest.mark.parametrize("change", ["missing", "candidate", "model", "drop_contract"])
def test_native_snapshot_must_match_the_submitted_materialized_input(tmp_path, change):
    config, report = _candidate_artifacts(tmp_path)
    expected_path = tmp_path / "hyperloom-input.yaml"
    if change == "missing":
        expected_path = None
    else:
        snapshot = json.loads((tmp_path / "config.yaml").read_text())
        if change == "candidate":
            snapshot["agentx"]["launch_overrides"]["env"]["SGLANG_USE_AITER"] = "1"
        elif change == "model":
            snapshot["model"] = "other/model"
        else:
            snapshot["agentx"].pop("launch_overrides")
        (tmp_path / "config.yaml").write_text(json.dumps(snapshot))
    measurement = extract_benchmark_measurement(report, workspace=tmp_path, materialized_config_path=expected_path)
    assert measurement["submission_valid"] is False
    assert "agentx_workload_fingerprint" not in measurement
    expected_error = "materialized_config_missing" if change == "missing" else "materialized_config_snapshot_mismatch"
    assert expected_error in measurement["native_agentx_protocol_errors"]


@pytest.mark.parametrize("control", ["append", "remove", "replace", "unrequested"])
def test_self_consistent_receipt_cannot_hide_ignored_argument_changes(tmp_path, control):
    config, report = _candidate_artifacts(tmp_path)
    overrides = config["agentx"]["launch_overrides"]
    evidence = report["agentx_metrics"]["server_launch"]
    evidence["base_argv"] += ["--model-path", "test/model", "--disable-cuda-graph"]
    evidence["effective_argv"] = list(evidence["base_argv"])
    if control == "append":
        overrides["append_args"] = ["--json-model-override-args", '{"key": "space value"}']
    elif control == "remove":
        overrides["remove_args"] = ["--disable-cuda-graph"]
    elif control == "replace":
        overrides["replace_args"] = True
    else:
        evidence["effective_argv"].append("--unexpected-change")
    evidence["overrides_sha256"] = canonical_sha256(overrides)
    evidence["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    assert validate_server_launch(config, evidence) == ["server_launch_arguments_mismatch"]


@pytest.mark.parametrize("change", ["missing_identity", "wrong_workload", "slower_interactivity"])
def test_warm_native_replay_rejects_before_promoting_source_or_kernel(tmp_path, monkeypatch, change):
    from hyperloom.inference_optimizer.tests.test_warm_replay import _make_coord, _replay_task

    _, report = _candidate_artifacts(tmp_path)
    baseline = extract_benchmark_measurement(
        report, workspace=tmp_path, materialized_config_path=tmp_path / "hyperloom-input.yaml"
    )
    coord = _make_coord(tmp_path)
    state = coord.shared_state
    state.benchmark_mode = "agentx"
    state.agentx_backend = "magpie"
    state.agentx_epoch = 3
    state.baseline_perf = baseline
    state.baseline_tput = baseline["output_throughput"]
    result = {**baseline, "status": "succeeded", "output_throughput": state.baseline_tput * 1.2}
    if change == "missing_identity":
        result.pop("agentx_workload_fingerprint")
    elif change == "wrong_workload":
        result["agentx_workload_fingerprint"] = "d" * 64
    else:
        result["e2e_norm_intvty_p50"] *= 0.9

    def unexpected(*args, **kwargs):
        pytest.fail("invalid native replay reached source/kernel promotion")

    monkeypatch.setattr(coord.phase_prelude, "_resolve_promoted_recipe_checkout", unexpected)
    monkeypatch.setattr(coord.phase_prelude, "_book_combined_kernel_keep", unexpected)
    coord.phase_prelude._settle_warm_replay(result, task=_replay_task(), recorder=None)
    assert state.warm_replay_outcome["status"] == "drift"
    assert not state.optimization_stack
    assert not state.current_best


def test_warm_native_replay_keeps_interactivity_winner_and_its_launch_identity(tmp_path):
    from hyperloom.inference_optimizer.tests.test_warm_replay import _make_coord, _replay_task

    _, report = _candidate_artifacts(tmp_path)
    input_path = tmp_path / "hyperloom-input.yaml"
    baseline = extract_benchmark_measurement(report, workspace=tmp_path, materialized_config_path=input_path)
    coord = _make_coord(tmp_path)
    state = coord.shared_state
    state.benchmark_mode = "agentx"
    state.agentx_backend = "magpie"
    state.agentx_epoch = 3
    state.baseline_perf = baseline
    state.baseline_tput = baseline["output_throughput"]
    result = {
        **baseline,
        "status": "succeeded",
        "materialized_config": str(input_path),
        "e2e_norm_intvty_p50": baseline["e2e_norm_intvty_p50"] * 1.1,
    }
    coord.phase_prelude._settle_warm_replay(result, task=_replay_task(), recorder=None)
    assert state.warm_replay_outcome["status"] == "reproduced"
    assert state.optimization_stack
    assert state.current_best["materialized_config"] == str(input_path)
    assert state.current_best["agentx_server_launch"] == baseline["agentx_server_launch"]
    assert state.current_best["agentx_workload_fingerprint"] == baseline["agentx_workload_fingerprint"]
