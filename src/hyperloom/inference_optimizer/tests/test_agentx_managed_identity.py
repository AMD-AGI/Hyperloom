# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Managed AgentX binds native recipes and nested dependencies without shell manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace
from copy import deepcopy

import pytest
import yaml

from hyperloom.inference_optimizer.agentx import identity, managed, native
from hyperloom.inference_optimizer.agentx.identity import canonical_sha256, validate_server_launch
from hyperloom.inference_optimizer.cli import preflight
from hyperloom.inference_optimizer.tests.test_agentx_native import _commit_all, _init_git_repo
from hyperloom.inference_optimizer.tests.test_agentx_optimization_identity import _candidate_artifacts


def _managed_checkout(tmp_path, *, nested=True, custom=False):
    repository = tmp_path / "Inference X"
    _init_git_repo(repository)
    project = repository / "inferencex-e2e" if nested else repository
    files = ["benchmarks/srt_agentic.sh", "benchmarks/benchmark_lib.sh", "configs/runners.yaml", "infx/client.py"]
    if not custom:
        files.append("configs/model.yaml")
    for name in files:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# immutable recipe input\n")
    dependency = project / "utils/aiperf"
    _init_git_repo(dependency)
    (dependency / "pyproject.toml").write_text("[project]\nname = 'aiperf'\n")
    aiperf = _commit_all(dependency)
    revision = _commit_all(repository)
    spec = {
        "version": 1,
        "framework": "sglang",
        "argv": ["python3", "-m", "sglang.launch_server"],
        "env": {},
        "client_env": {},
        "source_files": {
            str(project / name): hashlib.sha256((project / name).read_bytes()).hexdigest() for name in files
        },
        "client_revision": revision,
        "aiperf_revision": aiperf,
    }
    benchmark = {
        "framework": "sglang",
        "benchmark_script": "srt_agentic.sh",
        "agentx": {"enabled": True, "launch_overrides": {"version": 1}, "resolved": {"server-launch-spec": spec}},
    }
    arguments = {
        "inferencex_path": repository,
        "benchmark_script": "srt_agentic.sh",
        "config_file": "magpie:custom" if custom else "configs/model.yaml",
        "resolved_benchmark": benchmark,
        "expected_ref": revision,
        "magpie_execution": {"source_commit": "a" * 40, "fingerprint": "b" * 64},
        "expected_magpie_ref": "a" * 40,
        "launch_env": {},
    }
    return project, arguments


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_managed_identity_accepts_both_project_roots_and_custom_recipe(tmp_path, nested, custom):
    project, arguments = _managed_checkout(tmp_path, nested=nested, custom=custom)
    repository_identity = native.native_execution_identity(**arguments)
    arguments["inferencex_path"] = project
    project_identity = native.native_execution_identity(**arguments)
    assert repository_identity == project_identity
    assert project_identity["execution_owner"] == "magpie"
    assert project_identity["launcher"] == "benchmarks/srt_agentic.sh"
    assert project_identity["config_file"] == arguments["config_file"]
    assert project_identity["aiperf_commit"] == managed.aiperf_revision(project)
    assert (
        native.resolve_native_launcher(inferencex_path=project, benchmark_script="srt_agentic.sh")
        == project / "benchmarks/srt_agentic.sh"
    )
    assert preflight._inferencex_checkout_ok(project, ref=arguments["expected_ref"], require_agentx_submodule=True)
    assert preflight._inferencex_checkout_ok(
        arguments["inferencex_path"], ref=arguments["expected_ref"], require_agentx_submodule=True
    )


@pytest.mark.parametrize(
    "change", ["client", "untracked_python", "dependency", "gitlink", "spec_revision", "source_omission"]
)
def test_managed_identity_rejects_modified_execution_inputs(tmp_path, change):
    project, arguments = _managed_checkout(tmp_path)
    spec = arguments["resolved_benchmark"]["agentx"]["resolved"]["server-launch-spec"]
    if change == "client":
        (project / "benchmarks/srt_agentic.sh").write_text("# changed\n")
    elif change == "untracked_python":
        (project / "infx/injected.py").write_text("# untracked import input\n")
    elif change == "dependency":
        (project / "utils/aiperf/pyproject.toml").write_text("# modified harness\n")
    elif change == "gitlink":
        (project / "utils/aiperf/pyproject.toml").write_text("# other harness revision\n")
        _commit_all(project / "utils/aiperf")
    elif change == "spec_revision":
        spec["client_revision"] = "d" * 40
    else:
        spec["source_files"].pop(str(project / "benchmarks/srt_agentic.sh"))
    with pytest.raises(ValueError):
        native.native_execution_identity(**arguments)


def test_managed_candidate_changes_do_not_change_static_workload_identity(tmp_path):
    _, arguments = _managed_checkout(tmp_path)
    baseline = native.native_execution_identity(**arguments)
    arguments["resolved_benchmark"]["agentx"]["launch_overrides"] = {"version": 1, "env": {"SGLANG_USE_AITER": "1"}}
    candidate = native.native_execution_identity(**arguments)
    assert candidate["static_execution_fingerprint"] == baseline["static_execution_fingerprint"]
    assert candidate["workload_fingerprint"] == baseline["workload_fingerprint"]
    assert candidate["execution_fingerprint"] != baseline["execution_fingerprint"]


@pytest.mark.parametrize("change", [None, "owner", "source", "spec", "argv", "unexpected_profile"])
def test_managed_receipt_binds_magpie_recipe_before_accepting_candidate(tmp_path, monkeypatch, change):
    monkeypatch.setattr(identity, "_validate_managed_receipt", lambda *_args: [])
    config, report = _candidate_artifacts(tmp_path)
    evidence = report["agentx_metrics"]["server_launch"]
    spec = {"version": 1, "argv": list(evidence["base_argv"]), "source_files": {"/recipe.yaml": "a" * 64}}
    config["agentx"]["resolved"]["server-launch-spec"] = spec
    evidence.update(
        owner="magpie", server_spec_sha256=canonical_sha256(spec), recipe_source_files=deepcopy(spec["source_files"])
    )
    if change == "owner":
        evidence["owner"] = "shell"
    elif change == "source":
        evidence["recipe_source_files"] = {}
    elif change == "spec":
        evidence["server_spec_sha256"] = "b" * 64
    elif change == "argv":
        evidence["base_argv"] = evidence["effective_argv"] = ["unrelated-server"]
    elif change == "unexpected_profile":
        evidence["torch_profiler"] = {"enabled": True}
    evidence["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    errors = validate_server_launch(config, evidence)
    assert bool(errors) is bool(change)


def test_managed_profiler_receipt_requires_magpie_validation(tmp_path, monkeypatch):
    config, report = _candidate_artifacts(tmp_path)
    evidence = report["agentx_metrics"]["server_launch"]
    spec = {"version": 1, "argv": list(evidence["base_argv"]), "source_files": {}}
    config["agentx"]["resolved"]["server-launch-spec"] = spec
    config["profiler"] = {"torch_profiler": {"enabled": True}}
    evidence.update(owner="magpie", recipe_source_files={}, torch_profiler={"capture_id": "a" * 32})
    evidence["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    assert "server_launch_profile_workspace_missing" in validate_server_launch(config, evidence)


def _diagnostic_arguments(tmp_path):
    project, arguments = _managed_checkout(tmp_path)
    accepted = arguments["resolved_benchmark"]
    accepted["profiler"] = {"torch_profiler": {"enabled": False}}
    execution = native.native_execution_identity(**arguments)
    accepted["workload_spec"] = {"execution": execution}
    parent = tmp_path / "accepted.yaml"
    parent.write_text(yaml.safe_dump({"benchmark": accepted}))
    diagnostic = deepcopy(accepted)
    diagnostic["profiler"] = {"torch_profiler": {"enabled": True, "num_profiles": 3, "num_steps": 20}}
    diagnostic["workload_spec"]["profile_parent"] = {
        "materialized_config": str(parent),
        "config_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "workload_fingerprint": execution["workload_fingerprint"],
    }
    arguments["resolved_benchmark"] = diagnostic
    arguments["launch_env"] = {"HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT": execution["workload_fingerprint"]}
    return arguments, execution, parent


def test_managed_diagnostics_retain_verified_parent_identity_and_bind_candidate_overrides(tmp_path):
    arguments, accepted_execution, _ = _diagnostic_arguments(tmp_path)
    diagnostic_execution = native.native_execution_identity(**arguments)
    assert diagnostic_execution == accepted_execution
    diagnostic = arguments["resolved_benchmark"]
    assert diagnostic["profiler"]["torch_profiler"]["enabled"] is True
    diagnostic["agentx"]["launch_overrides"] = {"version": 1, "env": {"SGLANG_USE_AITER": "1"}}
    candidate_execution = native.native_execution_identity(**arguments)
    assert candidate_execution["workload_fingerprint"] == accepted_execution["workload_fingerprint"]
    assert candidate_execution["static_execution_fingerprint"] == accepted_execution["static_execution_fingerprint"]
    assert candidate_execution["execution_fingerprint"] != accepted_execution["execution_fingerprint"]


@pytest.mark.parametrize("change", ["model", "concurrency", "parent", "session_pin", "profile_flag", "missing_parent"])
def test_managed_diagnostic_derivation_cannot_change_parent_workload(tmp_path, change):
    arguments, _, parent = _diagnostic_arguments(tmp_path)
    diagnostic = arguments["resolved_benchmark"]
    if change == "model":
        diagnostic["model"] = "other/model"
    elif change == "concurrency":
        diagnostic["envs"] = {"CONC": 999}
    elif change == "parent":
        parent.write_text(parent.read_text() + "# changed after acceptance\n")
    elif change == "session_pin":
        arguments["launch_env"]["HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT"] = "c" * 64
    elif change == "profile_flag":
        diagnostic["profiler"]["torch_profiler"]["enabled"] = False
    else:
        diagnostic["workload_spec"].pop("profile_parent")
    with pytest.raises(ValueError, match="profile|diagnostic"):
        native.native_execution_identity(**arguments)


@pytest.mark.parametrize(
    "framework, flag",
    [
        ("vllm", "--profiler-config"),
        ("sglang", "--enable-profile-cuda-graph"),
        ("sglang", "--enable_shape_discovery_for_cuda_graph_profile"),
    ],
)
def test_managed_replace_profiler_argv_uses_audited_magpie_contract(tmp_path, monkeypatch, framework, flag):
    config, report = _candidate_artifacts(tmp_path)
    evidence = report["agentx_metrics"]["server_launch"]
    config["framework"] = evidence["framework"] = framework
    config["profiler"] = {"torch_profiler": {"enabled": True}}
    config["agentx"]["launch_overrides"] = {"version": 1, "replace_args": True}
    spec = {"version": 1, "argv": evidence["base_argv"], "source_files": {}}
    config["agentx"]["resolved"]["server-launch-spec"] = spec
    module = "vllm.entrypoints.openai.api_server" if framework == "vllm" else "sglang.launch_server"
    options = [flag, '{"profiler": "torch"}'] if framework == "vllm" else [flag]
    evidence["base_argv"] = ["python3", "-m", module, *options, "--optional-candidate", "1"]
    evidence["effective_argv"] = evidence["base_argv"][:-2]
    evidence.update(
        owner="magpie", recipe_source_files={}, overrides_sha256=canonical_sha256(config["agentx"]["launch_overrides"])
    )
    evidence["evidence_sha256"] = canonical_sha256({k: v for k, v in evidence.items() if k != "evidence_sha256"})
    calls = []

    def verify(command, **kwargs):
        payload = json.loads(kwargs["input"])
        assert payload["profile"] is True
        assert payload["workspace"] == str(tmp_path)
        assert payload["evidence"]["effective_argv"] == evidence["effective_argv"]
        assert "_validate_magpie_execution_tree(package, commit)" in command[-1]
        assert 'read_launch_evidence(config, Path(payload["workspace"]))' in command[-1]
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(identity.subprocess, "run", verify)
    assert validate_server_launch(config, evidence, workspace=tmp_path) == []
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [None, "nonzero", "timeout"])
def test_managed_accepted_launch_uses_magpie_argv_validation_without_workspace(tmp_path, monkeypatch, failure):
    config, report = _candidate_artifacts(tmp_path)
    evidence = report["agentx_metrics"]["server_launch"]
    spec = {"version": 1, "argv": list(evidence["base_argv"]), "source_files": {}}
    config["agentx"]["resolved"]["server-launch-spec"] = spec
    evidence.update(owner="magpie", recipe_source_files={}, server_spec_sha256=canonical_sha256(spec))
    evidence["evidence_sha256"] = canonical_sha256({k: v for k, v in evidence.items() if k != "evidence_sha256"})

    def verify(command, **kwargs):
        payload = json.loads(kwargs["input"])
        assert payload["profile"] is False
        assert 'apply_launch_args(evidence["base_argv"]' in command[-1]
        assert "_validate_golden_launch(" in command[-1]
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        return SimpleNamespace(returncode=int(failure is not None))

    monkeypatch.setattr(identity.subprocess, "run", verify)
    errors = validate_server_launch(config, evidence)
    assert errors == (["server_launch_managed_verification_failed"] if failure else [])
