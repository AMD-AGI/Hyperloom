# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep campaign-authorized assembly sources visible to the workspace guard."""

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from kernelforge.agent_backends.base import AgentRunSpec
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard, WorkspaceSafetyError


@pytest.fixture
def workspace(tmp_path):
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Forge Test"),
        ("config", "user.email", "forge@example.invalid"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    (tmp_path / "kernel.py").write_text("VALUE = 1\n")
    (tmp_path / "driver.py").write_text("ORACLE = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "baseline"], check=True, capture_output=True)
    return AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(tmp_path),
        target_files=["kernel.py"],
        driver_script="driver.py",
        commit_new_paths=["kernels/*.s"],
    )


@pytest.mark.parametrize("dirty_baseline", [False, True])
def test_allowed_assembly_survives_turn_and_is_counted(workspace, dirty_baseline):
    spec = replace(workspace, allow_dirty_baseline=dirty_baseline)
    guard = WorkspaceGuard(spec)
    guard.prepare()
    source = Path(spec.cwd) / "kernels/kernel.s"
    source.parent.mkdir()
    source.write_text("s_endpgm\n")

    assert guard.verify() == ["kernels/kernel.s"]
    assert guard.count_target_edits() == 1
    assert source.read_text() == "s_endpgm\n"

    resumed = WorkspaceGuard(replace(spec, allow_dirty_baseline=True, allow_dirty_targets=True))
    resumed.prepare()
    source.write_text("s_nop 0\ns_endpgm\n")
    assert resumed.verify() == ["kernels/kernel.s"]
    assert resumed.count_target_edits() == 1


@pytest.mark.parametrize("unexpected", ["helper.s", "kernels/nested/helper.s", "driver.py"])
def test_rejected_resume_restores_allowed_assembly_and_driver(workspace, unexpected):
    root = Path(workspace.cwd)
    source = root / "kernels/kernel.s"
    source.parent.mkdir()
    source.write_text("s_endpgm\n")
    guard = WorkspaceGuard(replace(workspace, allow_dirty_baseline=True))
    guard.prepare()
    source.write_text("s_nop 0\ns_endpgm\n")
    other = root / unexpected
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("unexpected\n")

    with pytest.raises(WorkspaceSafetyError):
        guard.verify()
    assert source.read_text() == "s_endpgm\n"
    assert (root / "driver.py").read_text() == "ORACLE = 1\n"
    if unexpected != "driver.py":
        assert not other.exists()


def test_allowlist_never_exempts_protected_source(workspace):
    spec = replace(workspace, commit_new_paths=["*.py"], protected_globs=["test_*.py"])
    guard = WorkspaceGuard(spec)
    guard.prepare()
    source = Path(spec.cwd) / "test_reference.py"
    source.write_text("ORACLE = 0\n")
    with pytest.raises(WorkspaceSafetyError, match="protected files created"):
        guard.verify()
    assert not source.exists()


def test_no_allowlist_still_rejects_new_assembly(workspace):
    guard = WorkspaceGuard(replace(workspace, commit_new_paths=[]))
    guard.prepare()
    source = Path(workspace.cwd) / "kernel.s"
    source.write_text("s_endpgm\n")
    with pytest.raises(WorkspaceSafetyError, match="new non-ignored files"):
        guard.verify()
    assert not source.exists()


def test_assembly_only_edit_after_keep_is_counted(workspace):
    root = Path(workspace.cwd)
    source = root / "kernels/kernel.s"
    source.parent.mkdir()
    source.write_text("s_endpgm\n")
    subprocess.run(["git", "-C", str(root), "add", "kernels/kernel.s"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "assembly route"], check=True, capture_output=True)
    guard = WorkspaceGuard(workspace)
    guard.prepare()
    source.write_text("s_nop 0\ns_endpgm\n")

    assert guard.verify() == ["kernels/kernel.s"]
    assert guard.count_target_edits() == 1
