# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep campaign-authorized assembly sources visible to the workspace guard."""

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from kernelforge.agent_backends.base import AgentRunSpec
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard, WorkspaceSafetyError
from kernelforge.workspace.edit_policy import FORGE_PROTECTED_PATHS, TOOL_OWNED_UNTRACKED_GLOBS, hide_forge_loop_output

OUTPUT_NOTE = "forge_experiments/orchestration/iter_001/note.md"


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
        protected_path_rules=FORGE_PROTECTED_PATHS,
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


def _implementer_session(spec: AgentRunSpec) -> WorkspaceGuard:
    """A guard shaped like an implementer session's, prepared before the session edits anything."""
    guard = WorkspaceGuard(
        replace(spec, allow_dirty_baseline=True, ignored_untracked_globs=list(TOOL_OWNED_UNTRACKED_GLOBS))
    )
    guard.prepare()
    return guard


def _edit_kernel_and_write_under_the_output_folder(root: Path) -> Path:
    (root / "kernel.py").write_text("VALUE = 2\n")
    note = root / OUTPUT_NOTE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("a note the loop wrote\n")
    return note


def test_a_file_in_the_hidden_output_folder_keeps_the_session(workspace):
    """The workspace ignores nothing of its own; the loop hiding its output folder is what keeps the file out."""
    root = Path(workspace.cwd)
    hide_forge_loop_output(root)
    guard = _implementer_session(workspace)
    note = _edit_kernel_and_write_under_the_output_folder(root)

    assert guard.verify() == ["kernel.py"]
    assert (root / "kernel.py").read_text() == "VALUE = 2\n"
    assert note.read_text() == "a note the loop wrote\n"


def test_a_file_in_a_visible_output_folder_rolls_the_session_back(workspace):
    root = Path(workspace.cwd)
    guard = _implementer_session(workspace)
    note = _edit_kernel_and_write_under_the_output_folder(root)

    with pytest.raises(WorkspaceSafetyError, match="new non-ignored files"):
        guard.verify()
    assert (root / "kernel.py").read_text() == "VALUE = 1\n"
    assert not note.exists()


def test_hidden_output_folder_exempts_no_new_source_file(workspace):
    root = Path(workspace.cwd)
    hide_forge_loop_output(root)
    guard = _implementer_session(workspace)
    _edit_kernel_and_write_under_the_output_folder(root)
    (root / "helper.py").write_text("UNEXPECTED = 1\n")

    with pytest.raises(WorkspaceSafetyError, match="new non-ignored files are unsupported: helper.py"):
        guard.verify()
    assert (root / "kernel.py").read_text() == "VALUE = 1\n"
    assert not (root / "helper.py").exists()
