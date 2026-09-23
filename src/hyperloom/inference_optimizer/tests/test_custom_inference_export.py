# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Portable custom inference, relocation, validation and refused-export contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from hyperloom.inference_optimizer.breakdown.session_package import package_session_artifacts
from hyperloom.inference_optimizer.deployment.export import export_custom_inference
from hyperloom.inference_optimizer.reference_script import render_reference_script


@pytest.fixture
def workload(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "adapter.py").write_text("""import json
import os
import sys
from pathlib import Path

def load_model(weights):
    assert not any(name == "hyperloom" or name.startswith("hyperloom.") for name in sys.modules)
    factor = int(Path(weights).read_text())
    bias = int(Path(os.environ["BIAS_FILE"]).read_text())
    return lambda value: value * factor + bias

def read_input(path):
    return json.loads(Path(path).read_text())

def write_output(value, path):
    Path(path).write_text(json.dumps(value))

def validate(model):
    return {"passed": model(3) == 13}
""")
    asset = tmp_path / "bias.txt"
    asset.write_text("1")
    contract = {
        "schema_version": 1,
        "adapter": "adapter",
        "source_files": ["adapter.py"],
        "assets": [str(asset)],
        "input_contract": "JSON scalar",
        "output_contract": "JSON scalar",
        "interpreter": sys.executable,
        "runtime": "test interpreter",
    }
    (root / "hyperloom_inference.json").write_text(json.dumps(contract))
    session = tmp_path / "session"
    session.mkdir()
    state = {
        "framework": "custom",
        "framework_repo_path": str(root),
        "current_best": {"extra_envs": {"BIAS_FILE": str(asset)}},
    }
    return session, state, contract


def _run(deployment, *args):
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "BIAS_FILE"}}
    return subprocess.run(
        [sys.executable, "-I", str(deployment / "inference.py"), *map(str, args)],
        cwd=deployment.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_export_runs_after_relocation_without_checkout_or_hyperloom(workload, tmp_path):
    session, state, _ = workload
    result = export_custom_inference(session, state)
    assert result["status"] == "exported", result
    assert result["validation"] == "not_run"
    deployment = tmp_path / "relocated"
    shutil.move(session / "deployment", deployment)
    shutil.rmtree(state["framework_repo_path"])
    (tmp_path / "bias.txt").unlink()
    weights = tmp_path / "weights"
    weights.write_text("4")
    inputs = tmp_path / "input.json"
    inputs.write_text("5")
    output = tmp_path / "output.json"
    run = _run(deployment, "--weights", weights, "--input", inputs, "--output", output)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text()) == 21
    report = tmp_path / "validation.json"
    run = _run(deployment, "--weights", weights, "--validate", "--report", report)
    assert run.returncode == 0, run.stderr
    assert json.loads(report.read_text())["passed"] is True
    weights.write_text("5")
    run = _run(deployment, "--weights", weights, "--validate", "--report", report)
    assert run.returncode != 0
    assert json.loads(report.read_text())["passed"] is False


def test_export_is_in_session_package(workload, tmp_path):
    session, state, _ = workload
    assert export_custom_inference(session, state)["status"] == "exported"
    package = package_session_artifacts(session, dest_root=tmp_path / "packages")
    with zipfile.ZipFile(package) as archive:
        assert "deployment/inference.py" in archive.namelist()
        assert "deployment/source/files/adapter.py" in archive.namelist()


def test_tuning_file_family_through_session_symlink(workload, tmp_path):
    session, state, contract = workload
    tuning = tmp_path / "tuning"
    tuning.mkdir()
    (tuning / "results_0.csv").write_text("tuned")
    (tuning / "unrelated.csv").write_text("not required")
    alias = tmp_path / "alias"
    alias.symlink_to(tuning, target_is_directory=True)
    family = str(alias / "results_%d.csv")
    contract["assets"].append(family)
    root = Path(state["framework_repo_path"])
    (root / "hyperloom_inference.json").write_text(json.dumps(contract))
    state["current_best"]["extra_envs"]["TUNING_FILENAME"] = family
    result = export_custom_inference(session, state)
    assert result["status"] == "exported", result
    assert result["env"]["TUNING_FILENAME"] == "${RUNTIME_ROOT}/assets/1/files/results_%d.csv"
    assert (session / "deployment/assets/1/files/results_0.csv").read_text() == "tuned"
    assert not (session / "deployment/assets/1/files/unrelated.csv").exists()


def test_declared_cache_directory_can_be_absent_from_original_environment(workload):
    session, state, contract = workload
    contract["cache_env"] = ["LIBRARY_JIT_DIR"]
    root = Path(state["framework_repo_path"])
    (root / "hyperloom_inference.json").write_text(json.dumps(contract))
    result = export_custom_inference(session, state)
    assert result["status"] == "exported", result
    assert result["env"]["LIBRARY_JIT_DIR"] == "${RUNTIME_ROOT}/cache/LIBRARY_JIT_DIR"
    assert result["cache_env"] == ["LIBRARY_JIT_DIR"]


def test_export_command_refreshes_the_session_launcher(workload, monkeypatch, capsys):
    from hyperloom.inference_optimizer.deployment.export import main

    session, state, _ = workload
    (session / "state.json").write_text(json.dumps(state))
    (session / "current_setting.sh").write_text("obsolete serving command")
    monkeypatch.setattr(sys, "argv", ["export", str(session)])
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "exported"
    assert "deployment" in (session / "current_setting.sh").read_text()
    assert "obsolete serving command" not in (session / "current_setting.sh").read_text()


def test_export_refuses_package_truncation(workload, monkeypatch):
    from hyperloom.inference_optimizer.breakdown import session_package

    session, state, _ = workload
    monkeypatch.setattr(session_package, "_MAX_FILES", 1)
    result = export_custom_inference(session, state)
    assert result["status"] == "incomplete"
    assert "artifact_not_self_contained" in result["reasons"][0]


def test_export_refuses_missing_dependency_observation(workload, monkeypatch):
    from hyperloom.inference_optimizer.deployment import export

    session, state, _ = workload
    monkeypatch.setattr(export, "probe_environment_closure", lambda **kwargs: ({}, {}))
    result = export_custom_inference(session, state)
    assert result["status"] == "incomplete"
    assert "closure is unavailable" in result["reasons"][0]


@pytest.mark.parametrize("missing", [False, True])
def test_enablement_recipe_keeps_its_payloads_and_verdict(workload, missing):
    session, state, _ = workload
    ref = "optimization_stack/enablement/root"
    payload = session / ref / "files/model.py"
    payload.parent.mkdir(parents=True)
    if not missing:
        payload.write_text("# accepted source\n")
    section = {
        "recipe_steps": [],
        "replay_sufficiency": {"status": "insufficient", "reasons": [{"code": "closure_scope_incomplete"}]},
        "source_snapshots": [{"root_id": "root", "snapshot_ref": ref, "files": [{"rel": "model.py", "op": "upsert"}]}],
    }
    (session / "session_breakdown.json").write_text(json.dumps({"enablement": section}))
    result = export_custom_inference(session, state)
    if missing:
        assert result["status"] == "incomplete"
        assert "unavailable payloads" in result["reasons"][0]
    else:
        assert result["status"] == "exported", result
        recipe = session / "deployment/recipe"
        assert (recipe / ref / "files/model.py").read_text() == payload.read_text()
        assert (
            json.loads((recipe / "enablement.json").read_text())["replay_sufficiency"] == section["replay_sufficiency"]
        )


def test_overlay_is_loaded_by_library_api_and_validated(workload, tmp_path):
    session, state, _ = workload
    overlay = tmp_path / "kernel-overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("import builtins\nbuiltins.optimized_bias = 3\n")
    root = Path(state["framework_repo_path"])
    adapter = root / "adapter.py"
    adapter.write_text(
        adapter.read_text().replace(
            'bias = int(Path(os.environ["BIAS_FILE"]).read_text())',
            "import builtins\n    bias = builtins.optimized_bias",
        )
    )
    state["current_best"]["final_overlay"] = str(overlay)
    assert export_custom_inference(session, state)["status"] == "exported"
    weights = tmp_path / "weights"
    weights.write_text("4")
    # Exercise the public Python API rather than only the command-line wrapper.
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import inference; print(inference.load_model(sys.argv[2])(5))"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-c", script, str(session / "deployment"), str(weights)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "23"


def test_validation_refuses_changed_dependencies(workload, tmp_path):
    session, state, _ = workload
    export_custom_inference(session, state)
    manifest_path = session / "deployment/deployment.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["environment_closure"]["distributions"]["nonexistent-required-model-library"] = "1.0"
    manifest_path.write_text(json.dumps(manifest))
    weights = tmp_path / "weights"
    weights.write_text("4")
    report = tmp_path / "report.json"
    run = _run(session / "deployment", "--weights", weights, "--validate", "--report", report)
    assert run.returncode != 0
    evidence = json.loads(report.read_text())
    assert evidence["quality_gate"]["passed"] is True
    assert evidence["passed"] is False
    assert "nonexistent-required-model-library" in evidence["environment_mismatches"][0]


def test_tuning_writes_and_cache_paths_do_not_mutate_export(workload, tmp_path):
    session, state, contract = workload
    root = Path(state["framework_repo_path"])
    contract["cache_env"] = ["MODEL_CACHE"]
    (root / "hyperloom_inference.json").write_text(json.dumps(contract))
    state["current_best"]["extra_envs"]["MODEL_CACHE"] = "/old/cache"
    adapter = root / "adapter.py"
    adapter.write_text(
        adapter.read_text().replace(
            "return lambda value: value * factor + bias",
            'Path(os.environ["BIAS_FILE"]).write_text("99")\n'
            '    (Path(os.environ["MODEL_CACHE"]) / "compiled").write_text("data")\n'
            "    return lambda value: value * factor + bias",
        )
    )
    result = export_custom_inference(session, state)
    assert result["status"] == "exported", result
    weights = tmp_path / "weights"
    weights.write_text("4")
    for _ in range(2):
        run = _run(session / "deployment", "--weights", weights, "--validate", "--report", tmp_path / "validation.json")
        assert run.returncode == 0, run.stderr
    assert (session / "deployment/assets/0/files/bias.txt").read_text() == "1"


@pytest.mark.parametrize(
    "failure",
    ["missing_contract", "invalid_contract", "missing_file", "secret", "absolute_env", "escape", "stale_source"],
)
def test_export_refuses_incomplete_inputs_and_removes_stale_launcher(workload, failure, tmp_path):
    session, state, contract = workload
    assert export_custom_inference(session, state)["status"] == "exported"
    root = Path(state["framework_repo_path"])
    if failure == "missing_contract":
        (root / "hyperloom_inference.json").unlink()
    elif failure == "invalid_contract":
        (root / "hyperloom_inference.json").write_text("null")
    elif failure == "missing_file":
        (root / "adapter.py").unlink()
    elif failure == "secret":
        state["current_best"]["extra_envs"]["ANTHROPIC_API_KEY"] = "must-not-be-shipped"
    elif failure == "absolute_env":
        state["current_best"]["extra_envs"]["CACHE_DIR"] = "/unknown/cache"
    elif failure == "escape":
        (root / "adapter.py").unlink()
        (root / "adapter.py").symlink_to(tmp_path / "bias.txt")
    elif failure == "stale_source":
        snap = tmp_path / "accepted"
        (snap / "files").mkdir(parents=True)
        (snap / "files/adapter.py").write_text("accepted version")
        (snap / "manifest.json").write_text(
            json.dumps({"complete": True, "files": [{"rel": "adapter.py", "op": "upsert"}]})
        )
        state["optimization_stack"] = [{"source_snapshot": str(snap)}]
    result = export_custom_inference(session, state)
    assert result["status"] == "incomplete"
    assert result["reasons"]
    assert not (session / "deployment/inference.py").exists()
    assert "must-not-be-shipped" not in (session / "deployment/deployment.json").read_text()


def test_tampered_export_is_not_executed(workload, tmp_path):
    session, state, _ = workload
    export_custom_inference(session, state)
    (session / "deployment/source/files/adapter.py").write_text("raise RuntimeError('executed')")
    run = _run(session / "deployment", "--weights", "unused", "--validate")
    assert run.returncode != 0
    assert "Deployment file changed" in run.stderr
    assert "RuntimeError: executed" not in run.stderr


def test_custom_reference_script_never_starts_a_serving_framework(tmp_path):
    script = tmp_path / "current_setting.sh"
    script.write_text(render_reference_script(framework="custom", server_args=""))
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Standalone inference export is unavailable" in result.stderr
    assert "sglang" not in script.read_text()


def test_enablement_custom_script_resolves_session_deployment(tmp_path):
    from hyperloom.orchestrator.enablement.artifacts import write_setting_script
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

    relative = write_setting_script(tmp_path, EnablementRound(), "custom")
    script = (tmp_path / relative).read_text()
    assert 'DEPLOYMENT_DIR="$SCRIPT_DIR"/../../deployment' in script


def test_export_does_not_capture_orchestrator_pythonpath(workload, tmp_path, monkeypatch):
    metadata = tmp_path / "coordinator" / "coordinator_only-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Name: coordinator-only\nVersion: 1.0\n")
    monkeypatch.setenv("PYTHONPATH", str(metadata.parent))
    session, state, _ = workload
    result = export_custom_inference(session, state)
    assert result["status"] == "exported"
    assert "coordinator-only" not in result["environment_closure"]["distributions"]


def test_environment_capture_matches_runtime_distribution_precedence(tmp_path):
    from hyperloom.orchestrator.enablement.recipe.keep_probe import probe_environment_closure

    roots = [tmp_path / "preferred", tmp_path / "shadowed"]
    for root, version in zip(roots, ("2.0", "1.0")):
        metadata = root / f"example_runtime-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(f"Name: example-runtime\nVersion: {version}\n")
    closure, _ = probe_environment_closure(
        sys.executable, env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, roots))}
    )
    assert closure["distributions"]["example-runtime"] == "2.0"
