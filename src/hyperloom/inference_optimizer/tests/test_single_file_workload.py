# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-file patch boundaries and portable export contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from hyperloom.orchestrator.deployment.export import export_custom_inference
from hyperloom.orchestrator.framework.optimization_scope import (
    OptimizationScope,
    check_patch_scope,
    optimization_scope,
    scope_instructions,
)


def _make_workload(tmp_path, monkeypatch, module: str, function: str, scope: dict):
    root, bench, session = [tmp_path / name for name in ("model", "bench", "session")]
    for directory in (root, bench, session):
        directory.mkdir()
    hook = root / module
    hook.write_text(f"def {function}(model):\n    return model\n")
    (bench / "optimization_scope.json").write_text(json.dumps({**scope, "file": str(hook)}))
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(bench))
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(root))
    for args in (
        ["init", "-q"],
        ["add", hook.name],
        ["-c", "user.name=test", "-c", "user.email=test@local", "commit", "-qm", "baseline"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    state = {
        "framework": "custom",
        "framework_repo_path": str(root),
        "bypass_scripts_dir": str(bench),
        "current_best": {"tput": 100},
    }
    return root, hook, session, state


@pytest.fixture
def workload(tmp_path, monkeypatch):
    return _make_workload(tmp_path, monkeypatch, "hyperloom_optimize.py", "hyperloom_optimize", {"schema_version": 1})


@pytest.fixture
def named_workload(tmp_path, monkeypatch):
    scope = {"schema_version": 2, "function": "apply"}
    return _make_workload(tmp_path, monkeypatch, "encoder_opt.py", "apply", scope)


def patch(name):
    return f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-old\n+new\n"


def test_scope_allows_hook_only(workload, tmp_path):
    root, hook, _, _ = workload
    assert optimization_scope() == OptimizationScope(file=hook, function="hyperloom_optimize")
    check_patch_scope(root, [patch(hook.name)])
    for name in ("benchmark.py", "extra.py", "../transformers/model.py"):
        with pytest.raises(ValueError):
            check_patch_scope(root, [patch(name)])
    with pytest.raises(ValueError):
        check_patch_scope(tmp_path, [patch(hook.name)])


def test_launch_scope_check_accepts_the_declared_hook(workload, tmp_path):
    from hyperloom.orchestrator.actions.executors.bypass_scriptable import enforce_custom_scope

    root, hook, _, _ = workload
    assert optimization_scope() == OptimizationScope(file=hook, function="hyperloom_optimize")
    enforce_custom_scope("custom")
    enforce_custom_scope("sglang")


def test_launch_scope_check_refuses_an_out_of_boundary_root(workload, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors.bypass_scriptable import enforce_custom_scope

    _, hook, _, _ = workload
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tmp_path / "elsewhere"))
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(ValueError, match="permits changes only"):
        enforce_custom_scope("custom")


def test_integration_cannot_use_recorded_foreign_root(workload, tmp_path):
    from hyperloom.orchestrator.actions.executors.integrate_patch import _resolve_framework_root

    root, hook, _, _ = workload
    assert _resolve_framework_root(str(root), patch_texts=[patch(hook.name)], recorded_root=str(root)) == root
    assert _resolve_framework_root(str(root), patch_texts=[patch("extra.py")], recorded_root=str(root)) is None
    assert _resolve_framework_root(str(tmp_path), patch_texts=[patch(hook.name)], recorded_root=str(tmp_path)) is None


def test_kernel_direct_write_is_refused(workload, tmp_path):
    from hyperloom.orchestrator.kernel.apply_kernel_patch import apply_kernel_patch

    _, hook, _, _ = workload
    before = hook.read_bytes()
    result = apply_kernel_patch(patch_path=hook, target_file=hook, backup_root=tmp_path / "backup")
    assert result["error_class"] == "single_file_scope"
    assert hook.read_bytes() == before


def test_export_contains_only_module_and_metadata(workload):
    _, hook, session, state = workload
    result = export_custom_inference(session, state)
    assert result["status"] == "exported"
    assert result["reasons"] == []
    assert result["kind"] == "single_file"
    assert set(result["files"]) == {hook.name}
    assert {p.name for p in (session / "deployment").iterdir()} == {hook.name}
    assert json.loads((session / "reports/deployment.json").read_text())["kind"] == "single_file"
    assert (session / "deployment" / hook.name).read_bytes() == hook.read_bytes()


def test_named_module_and_function_export(named_workload):
    _, hook, session, state = named_workload
    state["current_best"]["extra_envs"] = {"HL_FUSE_ATTN": "1"}
    assert optimization_scope() == OptimizationScope(file=hook, function="apply")
    assert "define apply(model)" in scope_instructions()

    result = export_custom_inference(session, state)

    assert result["status"] == "exported"
    assert result["entrypoint"] == "encoder_opt.apply"
    assert {p.name for p in (session / "deployment").iterdir()} == {"encoder_opt.py"}
    assert "HL_FUSE_ATTN" in (session / "deployment" / "encoder_opt.py").read_text()


def test_export_refuses_a_module_without_its_declared_function(named_workload):
    root, hook, session, state = named_workload
    hook.write_text("def hyperloom_optimize(model):\n    return model\n")
    commit = ["-c", "user.name=test", "-c", "user.email=test@local", "commit", "-qam", "rename"]
    subprocess.run(["git", "-C", str(root), *commit], check=True)
    result = export_custom_inference(session, state)
    assert result["status"] != "exported"
    assert not (session / "deployment" / hook.name).exists()


@pytest.mark.parametrize(
    "scope",
    [
        {"schema_version": 2},
        {"schema_version": 2, "function": "not an identifier"},
        {"schema_version": 3, "function": "apply"},
    ],
)
def test_scope_refuses_an_invalid_declaration(tmp_path, monkeypatch, scope):
    _make_workload(tmp_path, monkeypatch, "encoder_opt.py", "apply", scope)
    with pytest.raises(ValueError, match="Invalid"):
        optimization_scope()


def test_dirty_source_does_not_export_as_accepted(workload):
    _, hook, session, state = workload
    hook.write_text("def hyperloom_optimize(model):\n    return None\n")
    result = export_custom_inference(session, state)
    assert result["status"] == "incomplete"
    assert not (session / "deployment" / hook.name).exists()


def test_export_bakes_accepted_switches_into_module(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"HL_FUSE_ATTN": "1", "HL_HOIST": "1"}

    result = export_custom_inference(session, state)

    exported = session / "deployment" / hook.name
    assert result["status"] == "exported"
    assert "HL_FUSE_ATTN" in exported.read_text()
    assert "HL_HOIST" in exported.read_text()
    assert hook.read_bytes() == b"def hyperloom_optimize(model):\n    return model\n"
    assert result["files"][hook.name] == hashlib.sha256(exported.read_bytes()).hexdigest()
    assert result["settings"]["base_sha256"] == hashlib.sha256(hook.read_bytes()).hexdigest()


def test_export_reports_untranslatable_settings(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}

    result = export_custom_inference(session, state)

    exported_text = (session / "deployment" / hook.name).read_text()
    assert result["status"] == "exported"
    assert {
        "name": "PYTORCH_HIP_ALLOC_CONF",
        "value": "expandable_segments:True",
        "reason": "pre_context_init",
    } in result["settings"]["untranslated"]
    assert "PYTORCH_HIP_ALLOC_CONF" not in exported_text
    assert len(result["warnings"]) == 1
    assert "PYTORCH_HIP_ALLOC_CONF" in result["warnings"][0]


def test_export_excludes_benchmark_control_envs(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"RUN_EVAL": "false", "CONC": "1"}

    result = export_custom_inference(session, state)

    exported_text = (session / "deployment" / hook.name).read_text()
    assert result["status"] == "exported"
    assert {"name": "RUN_EVAL", "value": "false", "reason": "benchmark_control"} in result["settings"]["excluded"]
    assert {"name": "CONC", "value": "1", "reason": "benchmark_control"} in result["settings"]["excluded"]
    assert "RUN_EVAL" not in exported_text
    assert "CONC" not in exported_text


def test_export_is_idempotent_across_closes(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"HL_FUSE_ATTN": "1"}

    first_result = export_custom_inference(session, state)
    first_bytes = (session / "deployment" / hook.name).read_bytes()
    second_result = export_custom_inference(session, state)
    second_bytes = (session / "deployment" / hook.name).read_bytes()

    assert first_result["status"] == "exported"
    assert second_result["status"] == "exported"
    assert first_result["files"] == second_result["files"]
    assert first_bytes == second_bytes


def test_export_refuses_credential_shaped_setting(workload):
    _, _, session, state = workload
    state["current_best"]["extra_envs"] = {"HF_TOKEN": "x"}

    result = export_custom_inference(session, state)

    assert result["status"] == "incomplete"
    assert "HF_TOKEN" in result["reasons"][0]


def test_pycache_does_not_block_export(workload):
    root, _, session, state = workload
    pycache = root / "__pycache__"
    pycache.mkdir()
    (pycache / "hyperloom_optimize.cpython-312.pyc").write_bytes(b"junk")

    result = export_custom_inference(session, state)

    assert result["status"] == "exported"


def test_scope_refuses_symlink(workload, tmp_path):
    _, hook, _, _ = workload
    other = tmp_path / "other.py"
    other.write_bytes(hook.read_bytes())
    hook.unlink()
    hook.symlink_to(other)
    with pytest.raises(ValueError, match="regular"):
        optimization_scope()


def test_specialist_inherits_scope_and_writable_cli_config(workload, monkeypatch):
    from hyperloom.orchestrator.actions.executors._ray_runtime import safe_runtime_env
    from hyperloom.orchestrator.specialists.subprocess_ import _build_specialist_env

    _, _, _, state = workload
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/scratch/run/claude")
    env = _build_specialist_env()
    assert env["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == state["bypass_scripts_dir"]
    assert env["CLAUDE_CONFIG_DIR"] == "/scratch/run/claude"
    worker_env = safe_runtime_env()["env_vars"]
    assert worker_env["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == state["bypass_scripts_dir"]
    assert worker_env["CLAUDE_CONFIG_DIR"] == "/scratch/run/claude"
