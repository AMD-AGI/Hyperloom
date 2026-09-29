# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native candidate launch materialization and deployed-source provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.agentx.identity import native_workload_fingerprint
from hyperloom.orchestrator.actions.executors._grid_runner import GridVariant, _build_variant_yaml
from hyperloom.orchestrator.actions.executors._native_candidate import (
    apply_native_candidate,
    record_launch_evidence,
    update_native_candidate_file,
)
from hyperloom.orchestrator.actions.executors._native_source import (
    applied_source_evidence,
    source_file_hashes,
    verify_native_source_imports,
    warm_source_evidence,
)


def _benchmark():
    topology = {"gpu_count": 4, "tp": 4, "pp": 1, "pcp_size": 1, "ep": 4, "conc": 8, "recipe_fingerprint": "a" * 64}
    return {
        "framework": "sglang",
        "model": "amd/GLM-5.2-MXFP4",
        "benchmark_script": "single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh",
        "inferencex_path": "/pinned/inferencex",
        "agentx": {"enabled": True, "mode": "canonical", "launch_overrides": {"version": 1}},
        "envs": {"CONC": "8", "TP": "4", "RUN_EVAL": "false", "MODEL_PATH": "/models/glm", "PATH": os.environ["PATH"]},
        "workload_spec": {
            "resolved_topology": topology,
            "server_launch": {
                "base_argv": [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    "/models/glm",
                    "--disable-cuda-graph",
                ]
            },
        },
    }


def test_candidate_changes_only_server_contract_and_keeps_workload_identity():
    bench = _benchmark()
    before = copy.deepcopy(bench)
    fingerprint = native_workload_fingerprint(bench, "b" * 64)
    apply_native_candidate(
        bench,
        extra_server_args="--mem-fraction-static 0.75",
        extra_envs={"SGLANG_ENABLE_JIT_DEEPGEMM": "1", "CONC": "8"},
        remove_args=["--disable-cuda-graph"],
        unset_envs=["SGLANG_ENABLE_FLASHINFER"],
    )
    override = bench["agentx"]["launch_overrides"]
    assert override["append_args"] == ["--mem-fraction-static", "0.75"]
    assert override["remove_args"] == ["--disable-cuda-graph"]
    assert override["env"] == {"SGLANG_ENABLE_JIT_DEEPGEMM": "1"}
    assert override["unset_env"] == ["SGLANG_ENABLE_FLASHINFER"]
    assert bench["envs"] == before["envs"]
    assert native_workload_fingerprint(bench, "b" * 64) == fingerprint


def test_native_argv_preserves_json_and_does_not_silently_drop_flags():
    bench = _benchmark()
    apply_native_candidate(
        bench,
        extra_server_args='--compilation-config \'{"mode": "max-autotune"}\' --no-enable-prefix-caching',
    )
    assert bench["agentx"]["launch_overrides"]["append_args"] == [
        "--compilation-config",
        '{"mode": "max-autotune"}',
        "--no-enable-prefix-caching",
    ]


@pytest.mark.parametrize(
    "env", [{"CONC": "16"}, {"AGENTX_DATASET": "other"}, {"NUM_PROMPTS": "1"}, {"RUN_EVAL": "true"}]
)
def test_native_candidate_rejects_workload_retargeting(env):
    with pytest.raises(ValueError, match="canonical workload"):
        apply_native_candidate(_benchmark(), extra_envs=env)


def test_old_native_config_cannot_gain_optimizer_contract():
    bench = _benchmark()
    bench["agentx"].pop("launch_overrides")
    with pytest.raises(ValueError, match="accepted launch_overrides"):
        apply_native_candidate(bench, extra_server_args="--disable-cuda-graph")


def test_layered_removals_distinguish_base_flags_from_previous_candidate():
    bench = _benchmark()
    apply_native_candidate(bench, extra_server_args="--mem-fraction-static 0.7")
    apply_native_candidate(bench, remove_args=["--mem-fraction-static", "--disable-cuda-graph"])
    override = bench["agentx"]["launch_overrides"]
    assert override["append_args"] == []
    assert override["remove_args"] == ["--disable-cuda-graph"]


def test_inherited_candidate_removal_requires_observed_base():
    bench = _benchmark()
    bench["workload_spec"].pop("server_launch")
    apply_native_candidate(bench, extra_server_args="--mem-fraction-static 0.7")
    with pytest.raises(ValueError, match="verified base_argv"):
        apply_native_candidate(bench, remove_args=["--mem-fraction-static"])


def test_runtime_is_server_only_and_selects_exact_interpreter(tmp_path):
    bench = _benchmark()
    fixed = copy.deepcopy(bench["envs"])
    apply_native_candidate(
        bench,
        runtime_override={"runtime_python_exe": sys.executable, "pythonpath_prefix": str(tmp_path)},
    )
    override = bench["agentx"]["launch_overrides"]
    assert override["executable"] == sys.executable
    assert override["env"]["PYTHONPATH"].split(":")[0] == str(tmp_path)
    assert bench["envs"] == fixed


@pytest.mark.parametrize(
    "runtime", [{"unknown_runtime_knob": True}, {"pythonpath_prefix": "/does-not-exist"}, {"runtime_env": {"TP": "8"}}]
)
def test_runtime_override_cannot_be_silently_dropped(runtime):
    with pytest.raises(ValueError):
        apply_native_candidate(_benchmark(), runtime_override=runtime)


def _package(root: Path, value: str) -> Path:
    package = root / "sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    module = package / "candidate.py"
    module.write_text(f"VALUE = {value!r}\n", encoding="utf-8")
    return module


def test_source_probe_uses_candidate_import_root_and_rejects_other_install(tmp_path):
    target = _package(tmp_path / "candidate", "patched")
    _package(tmp_path / "original", "original")
    bench = _benchmark()
    apply_native_candidate(
        bench,
        runtime_override={"runtime_python_exe": sys.executable, "pythonpath_prefix": str(target.parent.parent)},
        source_files=source_file_hashes([target]),
    )
    verify_native_source_imports(bench)
    assert bench["workload_spec"]["source_imports"]["sglang"]["origin"] == str(target.parent / "__init__.py")
    bench["agentx"]["launch_overrides"]["env"]["PYTHONPATH"] = str(tmp_path / "original")
    with pytest.raises(ValueError, match="not imported"):
        verify_native_source_imports(bench)


def test_source_probe_respects_launcher_runtime_before_candidate_overrides(tmp_path):
    target = _package(tmp_path / "candidate", "patched")
    _package(tmp_path / "launcher-install", "original")
    bench = _benchmark()
    bench["envs"]["PYTHONPATH"] = str(target.parent.parent)
    bench["workload_spec"]["server_launch"]["runtime_environment"] = {"PYTHONPATH": str(tmp_path / "launcher-install")}
    apply_native_candidate(bench, source_files=source_file_hashes([target]))
    with pytest.raises(ValueError, match="not imported"):
        verify_native_source_imports(bench)
    apply_native_candidate(bench, runtime_override={"pythonpath_prefix": str(target.parent.parent)})
    verify_native_source_imports(bench)


@pytest.mark.parametrize("fails_during_startup", [False, True])
def test_overlay_source_probe_requires_successful_startup_import(tmp_path, fails_during_startup):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    source = 'raise RuntimeError("overlay startup failed")\n' if fails_during_startup else "LOADED = True\n"
    (overlay / "sitecustomize.py").write_text(source, encoding="utf-8")
    bench = _benchmark()
    apply_native_candidate(bench, overlay_pythonpath=str(overlay))

    if fails_during_startup:
        with pytest.raises(ValueError, match="overlay did not load successfully"):
            verify_native_source_imports(bench)
        assert "source_imports" not in bench["workload_spec"]
    else:
        verify_native_source_imports(bench)
        assert bench["workload_spec"]["source_imports"]["sitecustomize"]["loaded"] is True


def test_source_transaction_attests_writes_and_deletions(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    kept = root / "keep.py"
    kept.write_text("VALUE = 2\n", encoding="utf-8")
    patch = tmp_path / "applied.patch"
    patch.write_text(
        "diff --git a/keep.py b/keep.py\n--- a/keep.py\n+++ b/keep.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
        "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n",
        encoding="utf-8",
    )
    evidence = applied_source_evidence(root, [patch])
    assert evidence["source_files"] == {str(kept): hashlib.sha256(kept.read_bytes()).hexdigest()}
    assert evidence["absent_source_files"] == [str(root / "old.py")]
    (root / "old.py").write_text("not deleted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="deletion was not applied"):
        applied_source_evidence(root, [patch])


def test_grid_uses_native_override_without_retargeting_benchmark(tmp_path, monkeypatch):
    bench = _benchmark()
    base = tmp_path / "baseline.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.native.resolve_native_recipe",
        lambda benchmark, **kwargs: benchmark["workload_spec"]["resolved_topology"],
    )
    variant = GridVariant("candidate", extra_server_args="--mem-fraction-static 0.72", extra_envs={"SGLANG_TEST": "1"})
    variant.runtime_override = {"runtime_python_exe": sys.executable}
    output = _build_variant_yaml(base, "", variant, output_subdir=tmp_path / "candidate")
    actual = yaml.safe_load(output.read_text(encoding="utf-8"))["benchmark"]
    assert actual["envs"] == bench["envs"]
    assert actual["agentx"]["launch_overrides"]["append_args"] == ["--mem-fraction-static", "0.72"]
    assert actual["agentx"]["launch_overrides"]["env"] == {"SGLANG_TEST": "1"}
    assert actual["agentx"]["launch_overrides"]["executable"] == sys.executable


def test_native_grid_restores_snapshot_without_reapplying_display_stack(tmp_path, monkeypatch):
    bench = _benchmark()
    apply_native_candidate(bench, extra_server_args='--json \'{"mode": "accepted"}\'')
    snapshot = copy.deepcopy(bench["agentx"]["launch_overrides"])
    base = tmp_path / "baseline.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}))
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.native.resolve_native_recipe",
        lambda benchmark, **kw: benchmark["workload_spec"]["resolved_topology"],
    )
    output = _build_variant_yaml(
        base,
        '--json \'{"mode": "accepted"}\'',
        GridVariant("next", extra_server_args="--mem-fraction-static 0.75"),
        output_subdir=tmp_path / "next",
        base_args_mode="replace",
        base_extra_envs={"SGLANG_DUPLICATE": "1"},
        base_native_launch_overrides=snapshot,
    )
    actual = yaml.safe_load(output.read_text())["benchmark"]["agentx"]["launch_overrides"]
    assert actual["append_args"] == ["--json", '{"mode": "accepted"}', "--mem-fraction-static", "0.75"]
    assert actual["replace_args"] is False
    assert "SGLANG_DUPLICATE" not in actual["env"]


def test_native_warm_replay_attests_kernel_and_final_recipe_layer(tmp_path):
    source = _package(tmp_path / "source", "recreated by recipe")
    manifest = tmp_path / "kernel-manifest.json"
    manifest.write_text(json.dumps({"source_backups": [{"path": str(source), "disposition": "deleted"}]}))
    patches = tmp_path / "warm_patches"
    patches.mkdir()
    (patches / "000_recreate.diff").write_text(
        "--- /dev/null\n+++ b/sglang/candidate.py\n@@ -0,0 +1 @@\n+VALUE = 'recreated by recipe'\n"
    )
    params = {
        "warm_kernel_apply_results": [{"manifest_path": str(manifest)}],
        "patches": [{"patch_ref": "recreate.patch"}],
        "_warm_patch_statuses": [
            {"status": "applied", "patch_ref": "recreate.patch", "target_repo": str(source.parent.parent)}
        ],
    }
    assert warm_source_evidence(params, tmp_path) == {
        "source_files": source_file_hashes([source]),
        "absent_source_files": [],
    }


def test_verified_launch_evidence_survives_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("hyperloom.inference_optimizer.agentx.native.resolve_native_recipe", lambda *a, **kw: None)
    path = tmp_path / "native.yaml"
    path.write_text(yaml.safe_dump({"benchmark": _benchmark()}), encoding="utf-8")
    evidence = {"base_argv": [sys.executable, "-m", "sglang.launch_server"], "evidence_sha256": "a" * 64}
    record_launch_evidence(
        path, {"valid_measurement": False, "agentx_launch_contract": 1, "agentx_server_launch": evidence}
    )
    assert yaml.safe_load(path.read_text())["benchmark"]["workload_spec"]["server_launch"] != evidence
    record_launch_evidence(
        path,
        {
            "valid_measurement": True,
            "agentx_launch_contract": 1,
            "agentx_server_launch": evidence,
            "agentx_candidate_fingerprint": "a" * 64,
        },
    )
    update_native_candidate_file(path, extra_server_args="--disable-cuda-graph")
    assert yaml.safe_load(path.read_text())["benchmark"]["workload_spec"]["server_launch"] == evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("correct_import", [True, False])
async def test_epoch_three_baseline_applies_and_attests_warm_source_transaction(tmp_path, monkeypatch, correct_import):
    import subprocess

    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
    from hyperloom.orchestrator.state.shared_state import SharedState
    from hyperloom.orchestrator.state.task_registry import Task

    target = _package(tmp_path / "candidate", "before")
    root = target.parent.parent
    other = _package(tmp_path / "other", "unmodified")
    import_root = root if correct_import else other.parent.parent
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        check=True,
    )
    patch = tmp_path / "change.patch"
    patch.write_text(
        "--- a/sglang/candidate.py\n+++ b/sglang/candidate.py\n@@ -1 +1 @@\n-VALUE = 'before'\n+VALUE = 'after'\n",
        encoding="utf-8",
    )
    bench = _benchmark()
    bench["envs"]["PYTHONPATH"] = str(root)
    base = tmp_path / "baseline.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    monkeypatch.setenv("AGENTX_MODEL_ID", bench["model"])
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", bench["benchmark_script"])
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path / "ix"))
    monkeypatch.setenv("TP", "4")
    monkeypatch.setenv("CONC", "8")
    state = SharedState(
        benchmark_mode="agentx",
        agentx_epoch=3,
        agentx_backend="native",
        baseline_double_run=False,
        model_path="/models/glm",
    )
    executor = BaselineExecutor(default_config_path=base, session_dir=tmp_path, shared_state=state)
    captured = {}

    def resolve(benchmark, **kwargs):
        benchmark.setdefault("workload_spec", {})["resolved_topology"] = copy.deepcopy(
            bench["workload_spec"]["resolved_topology"]
        )
        return benchmark["workload_spec"]["resolved_topology"]

    async def measure(**kwargs):
        actual = yaml.safe_load(kwargs["config_path"].read_text())["benchmark"]
        captured.update(actual)
        assert target.read_text() == "VALUE = 'after'\n"
        return {"status": "succeeded", "output_throughput": 100.0, "submission_valid": True}

    async def cleanup(**kwargs):
        return None

    monkeypatch.setattr("hyperloom.inference_optimizer.agentx.native.resolve_native_recipe", resolve)
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.baseline.prepare_agentx_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(executor, "_run_single_benchmark", measure)
    monkeypatch.setattr(executor, "_pre_start_cleanup", cleanup)
    task = Task(
        task_id="warm",
        kind="replay_warm_recipe",
        state="running",
        idempotency_key="warm",
        params={
            "output_dir": str(tmp_path / "run"),
            "extra_server_args": "--mem-fraction-static 0.73",
            "runtime_override": {"runtime_python_exe": sys.executable, "pythonpath_prefix": str(import_root)},
            "patches": [{"framework_root": str(root), "patch_file": "change.patch", "patch_ref": str(patch)}],
            "required_patch_timeline": True,
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={"shared_state": state})
    result = await executor(ctx)
    if not correct_import:
        assert result["status"] == "failed"
        assert result["error_class"] == "native_source_attestation_failed"
        assert result["rollback"]["ok"] is True
        assert target.read_text() == "VALUE = 'before'\n"
        assert not captured
        return
    assert result["status"] == "succeeded"
    override = captured["agentx"]["launch_overrides"]
    assert override["source_files"][str(target)] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert override["append_args"] == ["--mem-fraction-static", "0.73"]
    assert captured["workload_spec"]["source_imports"]["sglang"]["origin"] == str(root / "sglang/__init__.py")


def test_grid_rejects_environment_previously_dropped_by_generic_filter(tmp_path):
    bench = _benchmark()
    base = tmp_path / "baseline.yaml"
    base.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    variant = GridVariant("unsafe", extra_envs={"PYTHONPATH": "/not-an-attested-runtime"})
    with pytest.raises(ValueError, match="unsupported environment"):
        _build_variant_yaml(base, "", variant, output_subdir=tmp_path / "candidate")


def test_native_profile_replays_observed_flags_and_runtime(tmp_path):
    from types import SimpleNamespace

    from hyperloom.inference_optimizer.agentx.identity import canonical_sha256
    from hyperloom.orchestrator.actions.executors._native_profile import project_native_profile

    bench = _benchmark()
    evidence = {
        "effective_argv": [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "/models/glm",
            "--port",
            "8000",
            "--tp",
            "4",
            "--mem-fraction-static",
            "0.73",
        ],
        "runtime_environment": {"PATH": os.environ["PATH"], "PYTHONPATH": str(tmp_path), "SGLANG_TEST": "1"},
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    bench["workload_spec"]["server_launch"] = evidence
    source = tmp_path / "accepted.yaml"
    source.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    state = SimpleNamespace(current_best_measurement={}, baseline_config_path=str(source))
    params = {"extra_envs": {"PROFILE": "1"}}
    project_native_profile(params, state)
    assert params["extra_server_args"] == "--tp 4 --mem-fraction-static 0.73"
    assert params["extra_envs"]["PROFILE"] == "1"
    assert params["extra_envs"]["SGLANG_TEST"] == "1"
    assert params["runtime_override"]["runtime_python_exe"] == sys.executable
    assert params["runtime_override"]["pythonpath_prefixes"] == [str(tmp_path)]
