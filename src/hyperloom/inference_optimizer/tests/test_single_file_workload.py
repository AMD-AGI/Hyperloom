# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-file patch boundaries and portable export contracts."""

from __future__ import annotations

import json
import subprocess

import pytest

from hyperloom.inference_optimizer.deployment.export import export_custom_inference
from hyperloom.orchestrator.framework.optimization_scope import check_patch_scope, optimization_file


@pytest.fixture
def workload(tmp_path, monkeypatch):
    root, bench, session = [tmp_path / name for name in ("model", "bench", "session")]
    for directory in (root, bench, session):
        directory.mkdir()
    hook = root / "hyperloom_optimize.py"
    hook.write_text("def hyperloom_optimize(model):\n    return model\n")
    (bench / "optimization_scope.json").write_text(json.dumps({"schema_version": 1, "file": str(hook)}))
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


def patch(name):
    return f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-old\n+new\n"


def test_scope_allows_hook_only(workload, tmp_path):
    root, hook, _, _ = workload
    assert optimization_file() == hook
    check_patch_scope(root, [patch(hook.name)])
    for name in ("benchmark.py", "extra.py", "../transformers/model.py"):
        with pytest.raises(ValueError):
            check_patch_scope(root, [patch(name)])
    with pytest.raises(ValueError):
        check_patch_scope(tmp_path, [patch(hook.name)])


def test_integration_cannot_use_recorded_foreign_root(workload, tmp_path):
    from hyperloom.orchestrator.actions.executors.integrate_patch import _resolve_framework_root

    root, hook, _, _ = workload
    assert _resolve_framework_root(str(root), patch_texts=[patch(hook.name)], recorded_root=str(root)) == root
    assert _resolve_framework_root(str(root), patch_texts=[patch("extra.py")], recorded_root=str(root)) is None
    assert _resolve_framework_root(str(tmp_path), patch_texts=[patch(hook.name)], recorded_root=str(tmp_path)) is None


def test_kernel_direct_write_is_refused(workload, tmp_path):
    from hyperloom.agents.kernel.tools.apply_kernel_patch import apply_kernel_patch

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


def test_dirty_source_does_not_export_as_accepted(workload):
    _, hook, session, state = workload
    hook.write_text("def hyperloom_optimize(model):\n    return None\n")
    result = export_custom_inference(session, state)
    assert result["status"] == "incomplete"
    assert not (session / "deployment" / hook.name).exists()


def test_scope_refuses_symlink(workload, tmp_path):
    _, hook, _, _ = workload
    other = tmp_path / "other.py"
    other.write_bytes(hook.read_bytes())
    hook.unlink()
    hook.symlink_to(other)
    with pytest.raises(ValueError, match="regular"):
        optimization_file()


def test_specialist_inherits_scope_and_writable_cli_config(workload, monkeypatch):
    from hyperloom.agents.kernel.tools.backends.ray_runtime import safe_runtime_env
    from hyperloom.orchestrator.specialists.subprocess_ import _build_specialist_env

    _, _, _, state = workload
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/scratch/run/claude")
    env = _build_specialist_env()
    assert env["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == state["bypass_scripts_dir"]
    assert env["CLAUDE_CONFIG_DIR"] == "/scratch/run/claude"
    worker_env = safe_runtime_env()["env_vars"]
    assert worker_env["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == state["bypass_scripts_dir"]
    assert worker_env["CLAUDE_CONFIG_DIR"] == "/scratch/run/claude"
